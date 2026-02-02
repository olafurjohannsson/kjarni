use crate::WgpuContext;
use crate::gpu::{GpuTensor};
use anyhow::Result;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NormUniforms {
    m: u32,
    n: u32,
    eps: f32,
    _padding: u32,
}

/// Holds the weight and bias tensors for the GpuLayerNorm operation.
pub struct GpuLayerNormWeights {
    pub(crate) gamma: GpuTensor, // scale
    pub(crate) beta: GpuTensor,  // bias
}

impl GpuLayerNormWeights {
    pub fn new(gamma: GpuTensor, beta: GpuTensor) -> Result<Self> {
        // Both must be 1D (one parameter per normalized feature)
        assert_eq!(gamma.rank(), 1, "LayerNorm gamma must be 1D");
        assert_eq!(beta.rank(), 1, "LayerNorm beta must be 1D");

        // gamma and beta must have the same shape
        assert_eq!(
            gamma.shape(),
            beta.shape(),
            "LayerNorm gamma and beta must have the same shape"
        );

        Ok(Self { gamma, beta })
    }
}

/// A GPU kernel for Layer Normalization.
pub struct GpuLayerNorm {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    context: Arc<WgpuContext>,
    eps: f32,
}

impl GpuLayerNorm {
    pub fn new(context: &Arc<WgpuContext>, eps: f32) -> Self {
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("./layer_norm.wgsl"));
        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("LayerNorm Bind Group Layout"),
                    entries: &[
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
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
                    label: Some("LayerNorm Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LayerNorm Pipeline"),
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

    /// Encodes the LayerNorm operation.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        weights: &GpuLayerNormWeights,
        input: &GpuTensor,
        output: &GpuTensor,
    ) {
        let rank = input.rank();
        assert!(
            rank >= 2,
            "Input tensor must have at least rank 2 for LayerNorm."
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
                    label: Some("LayerNorm Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("LayerNorm Bind Group"),
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
                        resource: weights.beta.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: output.buffer().as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LayerNorm Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (rows as u32 + 255) / 256;
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }
}

#[cfg(test)]
mod tests;
