use crate::WgpuContext;
use crate::gpu_ops::GpuTensor;
use anyhow::Result;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NormUniforms {
    m: u32, // batch_size * seq_len
    n: u32, // hidden_size
    eps: f32,
    _padding: u32,
}

/// Holds the weight tensor for the GpuRMSNorm operation.
/// LLaMA's RMSNorm does not have a bias term (beta).
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
    context: Arc<WgpuContext>,
    eps: f32,
}

impl GpuRMSNorm {
    pub fn new(context: &Arc<WgpuContext>, eps: f32) -> Self {
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("./rms_norm.wgsl"));
        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("RMSNorm Bind Group Layout"),
                    entries: &[
                        // Uniforms
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Input
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Gamma (weight)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("RMSNorm Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RMSNorm Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Self {
            pipeline,
            bind_group_layout,
            context: context.clone(),
            eps,
        }
    }

    /// Encodes the RMSNorm operation.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        weights: &GpuRMSNormWeights,
        input: &GpuTensor,
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

        let uniforms = NormUniforms {
            m: rows as u32,
            n: cols as u32,
            eps: self.eps,
            _padding: 0,
        };
        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("RMSNorm Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RMSNorm Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: weights.gamma.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output.buffer().as_entire_binding(),
                    },
                ],
            });

        // let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        //     label: Some("RMSNorm Pass"),
        //     timestamp_writes: None,
        // });
        let label = format!("RMSNorm");
        self.context
            .profiler
            .profile(encoder, &label, |compute_pass| {
                compute_pass.set_pipeline(&self.pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                // let workgroups = (rows as u32 + 255) / 256;
                // compute_pass.dispatch_workgroups(workgroups, 1, 1);
                let workgroups = rows as u32;

                compute_pass.dispatch_workgroups(workgroups, 1, 1);
            });
    }
}

#[cfg(test)]
mod tests;
