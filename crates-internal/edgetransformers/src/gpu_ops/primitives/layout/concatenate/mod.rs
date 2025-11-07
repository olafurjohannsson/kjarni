use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::gpu_context::WgpuContext;
use crate::gpu_ops::GpuTensor;

mod tests;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ConcatUniforms {
    out_stride_0: u32,
    out_stride_1: u32,
    out_stride_2: u32,
    out_stride_3: u32,
    a_stride_0: u32,
    a_stride_1: u32,
    a_stride_2: u32,
    a_stride_3: u32,
    b_stride_0: u32,
    b_stride_1: u32,
    b_stride_2: u32,
    b_stride_3: u32,
    a_shape_0: u32,
    a_shape_1: u32,
    a_shape_2: u32,
    a_shape_3: u32,
    out_shape_0: u32,
    out_shape_1: u32,
    out_shape_2: u32,
    out_shape_3: u32,
    concat_axis: u32,
    _padding: [u32; 3], // Ensure alignment
}

/// A GPU kernel for concatenating two 4D tensors along a given axis.
pub struct GpuConcatenate {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    context: Arc<WgpuContext>,
}

// Helper to calculate strides for a 4D tensor
fn get_strides(shape: &[usize]) -> [u32; 4] {
    [
        (shape[1] * shape[2] * shape[3]) as u32,
        (shape[2] * shape[3]) as u32,
        shape[3] as u32,
        1,
    ]
}

impl GpuConcatenate {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("concatenate.wgsl"));
        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Concat Bind Group Layout"),
                    entries: &[
                        // a, b, output, uniforms
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
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
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Concat Pipeline"),
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

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        inputs: &[&GpuTensor],
        output: &GpuTensor,
        axis: usize,
    ) {
        let a = inputs[0];
        let b = inputs[1];

        // Basic validation
        assert_eq!(a.rank(), 4, "Concat kernel requires 4D tensors");
        assert_eq!(b.rank(), 4, "Concat kernel requires 4D tensors");
        assert!(axis < 4, "Concat axis out of bounds for 4D tensor");

        // Prepare uniform data
        let a_strides = get_strides(a.shape());
        let b_strides = get_strides(b.shape());
        let out_strides = get_strides(output.shape());
        let a_shape = a.shape();
        let out_shape = output.shape();

        let uniforms = ConcatUniforms {
            out_stride_0: out_strides[0],
            out_stride_1: out_strides[1],
            out_stride_2: out_strides[2],
            out_stride_3: out_strides[3],
            a_stride_0: a_strides[0],
            a_stride_1: a_strides[1],
            a_stride_2: a_strides[2],
            a_stride_3: a_strides[3],
            b_stride_0: b_strides[0],
            b_stride_1: b_strides[1],
            b_stride_2: b_strides[2],
            b_stride_3: b_strides[3],
            a_shape_0: a_shape[0] as u32,
            a_shape_1: a_shape[1] as u32,
            a_shape_2: a_shape[2] as u32,
            a_shape_3: a_shape[3] as u32,
            out_shape_0: out_shape[0] as u32,
            out_shape_1: out_shape[1] as u32,
            out_shape_2: out_shape[2] as u32,
            out_shape_3: out_shape[3] as u32,
            concat_axis: axis as u32,
            _padding: [0; 3],
        };

        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Concat Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Concat Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Concat Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_size = (8, 8, 8);
        let workgroups_x = (out_shape[3] as u32 + workgroup_size.0 - 1) / workgroup_size.0;
        let workgroups_y = (out_shape[2] as u32 + workgroup_size.1 - 1) / workgroup_size.1;
        let workgroups_z =
            (out_shape[0] * out_shape[1] + workgroup_size.2 - 1) / workgroup_size.2;

        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z as u32);
    }
}
