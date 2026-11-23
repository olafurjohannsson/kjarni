use crate::WgpuContext;
use crate::gpu::GpuTensor;
use crate::tensor::DType;
use anyhow::Result;
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NormUniforms {
    m: u32, // batch_size * seq_len
    n: u32, // hidden_size
    eps: f32,
    is_bf16: u32,
}

/// Holds the weight tensor for the GpuRMSNorm operation
pub struct GpuRMSNormWeights {
    pub(crate) gamma: GpuTensor, // scale
}

impl GpuRMSNormWeights {
    pub fn new(gamma: GpuTensor) -> Result<Self> {
        assert_eq!(gamma.rank(), 1, "RMSNorm gamma must be 1D");
        Ok(Self { gamma })
    }
}

/// A GPU kernel for Root Mean Square Normalization (RMSNorm).
pub struct GpuRMSNorm {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    add_pipeline: wgpu::ComputePipeline,
    add_bind_group_layout: wgpu::BindGroupLayout,
    context: Arc<WgpuContext>,
    eps: f32,
}

impl GpuRMSNorm {
    pub fn new(context: &Arc<WgpuContext>, eps: f32) -> Self {
        let device = &context.device;
        let shader = device.create_shader_module(wgpu::include_wgsl!("./rms_norm.wgsl"));
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let write = wgpu::BufferBindingType::Storage { read_only: false };
        // Uniforms, input, gamma, output; the fused add also reads the addend
        // and writes the sum.
        let base = [
            entry(0, wgpu::BufferBindingType::Uniform),
            entry(1, read),
            entry(2, read),
            entry(3, write),
        ];
        let with_add = [
            base[0],
            base[1],
            base[2],
            base[3],
            entry(4, read),
            entry(5, write),
        ];
        let make = |label: &str, entries: &[wgpu::BindGroupLayoutEntry], entry_point: &str| {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });
            (layout, pipeline)
        };
        let (bind_group_layout, pipeline) = make("RMSNorm", &base, "main");
        let (add_bind_group_layout, add_pipeline) = make("Add + RMSNorm", &with_add, "add_main");
        Self {
            pipeline,
            bind_group_layout,
            add_pipeline,
            add_bind_group_layout,
            context: context.clone(),
            eps,
        }
    }

    /// Encodes the RMSNorm operation.
    pub fn encode(
        &self,
        encoder: &mut crate::gpu::GpuEncoder,
        weights: &GpuRMSNormWeights,
        input: &GpuTensor,
        output: &GpuTensor,
    ) {
        self.record(encoder, weights, input, None, output);
    }

    /// Records `sum = input + addend` and `output = rms_norm(sum)` in one
    /// dispatch: a residual add and the norm that follows it. The same values as
    /// an add and then [`Self::encode`].
    pub fn encode_add(
        &self,
        encoder: &mut crate::gpu::GpuEncoder,
        weights: &GpuRMSNormWeights,
        input: &GpuTensor,
        addend: &GpuTensor,
        sum: &GpuTensor,
        output: &GpuTensor,
    ) {
        assert_eq!(input.shape(), addend.shape(), "Add + RMSNorm addend shape");
        assert_eq!(input.shape(), sum.shape(), "Add + RMSNorm sum shape");
        self.record(encoder, weights, input, Some((addend, sum)), output);
    }

    fn record(
        &self,
        encoder: &mut crate::gpu::GpuEncoder,
        weights: &GpuRMSNormWeights,
        input: &GpuTensor,
        add: Option<(&GpuTensor, &GpuTensor)>,
        output: &GpuTensor,
    ) {
        let rank = input.rank();
        assert!(
            rank >= 2,
            "Input tensor must have at least rank 2 for RMSNorm."
        );
        let (rows, cols) = (
            input.shape()[..rank - 1].iter().product::<usize>(),
            input.shape()[rank - 1],
        );

        let is_bf16 = match weights.gamma.dtype() {
            DType::BF16 => 1u32,
            DType::F32 => 0u32,
            other => panic!("Unsupported gamma dtype: {:?}", other),
        };

        let uniforms = NormUniforms {
            m: rows as u32,
            n: cols as u32,
            eps: self.eps,
            is_bf16,
        };

        let uniform_buffer = self.context.uniform_buffer(&uniforms);
        let mut entries = vec![
            bind(0, &uniform_buffer),
            bind(1, input.buffer()),
            bind(2, weights.gamma.buffer()),
            bind(3, output.buffer()),
        ];
        let (layout, pipeline, label) = match add {
            Some((addend, sum)) => {
                entries.push(bind(4, addend.buffer()));
                entries.push(bind(5, sum.buffer()));
                (
                    &self.add_bind_group_layout,
                    &self.add_pipeline,
                    "Add + RMSNorm",
                )
            }
            None => (&self.bind_group_layout, &self.pipeline, "RMSNorm"),
        };
        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &entries,
            });
        crate::gpu_profile!(
            self.context,
            encoder,
            label,
            |pass: &mut wgpu::ComputePass<'_>| {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                crate::gpu_ops::utils::dispatch_1d(pass, rows as u32);
            }
        );
    }
}

fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

#[cfg(test)]
mod tests;
