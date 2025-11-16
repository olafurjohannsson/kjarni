use crate::gpu_context::WgpuContext;
use crate::gpu_ops::{GpuTensor};
use crate::gpu_ops::primitives::matmul::GpuMatMul;
use anyhow::Result;
use std::sync::Arc;
use crate::gpu_ops::Kernel;

mod tests;

/// Holds the weight tensors for the GpuSwiGLUFFN operation.
pub struct GpuSwiGLUFFNWeights {
    pub(crate) gate_proj: GpuTensor,
    pub(crate) up_proj: GpuTensor,
    pub(crate) down_proj: GpuTensor,
}

impl GpuSwiGLUFFNWeights {
    pub fn new(gate_proj: GpuTensor, up_proj: GpuTensor, down_proj: GpuTensor) -> Result<Self> {
        assert_eq!(gate_proj.rank(), 2, "gate_proj weight must be 2D");
        assert_eq!(up_proj.rank(), 2, "up_proj weight must be 2D");
        assert_eq!(down_proj.rank(), 2, "down_proj weight must be 2D");
        Ok(Self { gate_proj, up_proj, down_proj })
    }
}

/// A GPU kernel for the SwiGLU Feed-Forward Network used in LLaMA.
pub struct GpuSwiGLUFFN {
    elementwise_pipeline: wgpu::ComputePipeline,
    elementwise_bind_group_layout: wgpu::BindGroupLayout,
    matmul: GpuMatMul,
    context: Arc<WgpuContext>,
}

impl GpuSwiGLUFFN {
    pub fn new(context: &Arc<WgpuContext>) -> Result<Self> {
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("./swiglu.wgsl"));
            
        let elementwise_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SwiGLU Elementwise Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                        wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                    ],
                });
        
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("SwiGLU Elementwise Pipeline Layout"),
                    bind_group_layouts: &[&elementwise_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let elementwise_pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SwiGLU Elementwise Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            elementwise_pipeline,
            elementwise_bind_group_layout,
            matmul: GpuMatMul::new(context),
            context: context.clone(),
        })
    }

    /// Encodes the SwiGLU FFN operation.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        weights: &GpuSwiGLUFFNWeights,
        input: &GpuTensor, // Assumes a 2D tensor [rows, hidden_size]
        output: &GpuTensor,
        temp: &mut super::attention::TempStorage,
    ) {
        let rank = input.rank();
        assert_eq!(rank, 2, "Input tensor for GpuSwiGLUFFN must be 2D");
        let (rows, _) = (input.shape()[0], input.shape()[1]);
        let intermediate_size = weights.up_proj.shape()[1];

        // 1. Gate projection: input @ gate_proj
        let gate_result = temp.get(vec![rows, intermediate_size]);
        self.matmul.encode(encoder, &[input, &weights.gate_proj], &gate_result);

        // 2. Up projection: input @ up_proj
        let up_result = temp.get(vec![rows, intermediate_size]);
        self.matmul.encode(encoder, &[input, &weights.up_proj], &up_result);

        // 3. Fused SiLU and element-wise multiply: SiLU(gate_result) * up_result
        let intermediate_activated = temp.get(vec![rows, intermediate_size]);
        self.encode_elementwise(encoder, &gate_result, &up_result, &intermediate_activated);

        // 4. Down projection: intermediate_activated @ down_proj
        self.matmul.encode(encoder, &[&intermediate_activated, &weights.down_proj], output);
    }

    /// Encodes the element-wise part of the SwiGLU operation.
    fn encode_elementwise(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gate_tensor: &GpuTensor,
        up_tensor: &GpuTensor,
        output: &GpuTensor,
    ) {
        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SwiGLU Elementwise Bind Group"),
            layout: &self.elementwise_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gate_tensor.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: up_tensor.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.buffer().as_entire_binding() },
            ],
        });
        
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("SwiGLU Elementwise Pass"), timestamp_writes: None });
        compute_pass.set_pipeline(&self.elementwise_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (output.num_elements() as u32 + 255) / 256;
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }
}