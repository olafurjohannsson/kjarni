use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::gpu_context::WgpuContext;
use crate::gpu_ops::{DType, GpuTensor, Kernel};

mod tests;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AddUniforms {
    size: u32,
    _padding: [u32; 3],
}

/// A GPU kernel for element-wise addition of two tensors.
pub struct GpuAdd {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    context: Arc<WgpuContext>,
}

impl GpuAdd {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("add.wgsl"));
        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Add Bind Group Layout"),
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
                    label: Some("Add Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Add Pipeline"),
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
        }
    }
}

impl Kernel for GpuAdd {
    /// Encodes the add operation.
    ///
    /// # Arguments
    /// * `inputs`: A slice containing two `GpuTensor`s to be added.
    /// * `output`: The `GpuTensor` where the result will be written.
    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        inputs: &[&GpuTensor],
        output: &GpuTensor,
    ) {
        let a = inputs[0];
        let b = inputs[1];

        assert_eq!(
            a.shape(),
            b.shape(),
            "Input tensors must have the same shape for addition."
        );
        assert_eq!(
            a.shape(),
            output.shape(),
            "Output tensor must have the same shape as inputs."
        );

        let size = a.num_elements() as u32;

        let uniforms = AddUniforms {
            size,
            _padding: [0; 3],
        };
        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Add Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Add Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: a.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output.buffer().as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Add Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (size + 255) / 256;
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }
}
