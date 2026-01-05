use crate::WgpuContext;
use crate::gpu_ops::GpuTensor;
use std::sync::Arc;
use wgpu::util::DeviceExt;

// Simplified for now, this would be part of a larger Kernel trait
pub struct GpuSlice {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    context: Arc<WgpuContext>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SliceUniforms {
    src_stride_b: u32,
    src_stride_h: u32,
    src_stride_s: u32,
    src_stride_d: u32,
    dst_stride_b: u32,
    dst_stride_h: u32,
    dst_stride_s: u32,
    dst_stride_d: u32,
    offset_b: u32,
    offset_h: u32,
    offset_s: u32,
    offset_d: u32,
    dst_shape_b: u32,
    dst_shape_h: u32,
    dst_shape_s: u32,
    dst_shape_d: u32,
}

impl GpuSlice {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("slice.wgsl"));
        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Slice Bind Group Layout"),
                    entries: &[
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
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
                    label: Some("Slice Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Slice Pipeline"),
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
        src: &GpuTensor,
        dst: &GpuTensor,
        offset: &[usize], // [B, H, S, D] offsets
    ) {
        // This implementation is specialized for 4D KV cache slicing for simplicity
        assert_eq!(src.rank(), 4, "This slice is specialized for 4D tensors.");
        assert_eq!(dst.rank(), 4, "This slice is specialized for 4D tensors.");

        let (src_b, src_h, src_s, src_d) = src.dims4();
        let (dst_b, dst_h, dst_s, dst_d) = dst.dims4();

        // Calculate standard C-contiguous strides for 4D tensors
        let src_stride_b = (src_h * src_s * src_d) as u32;
        let src_stride_h = (src_s * src_d) as u32;
        let src_stride_s = src_d as u32;
        let src_stride_d = 1;

        let dst_stride_b = (dst_h * dst_s * dst_d) as u32;
        let dst_stride_h = (dst_s * dst_d) as u32;
        let dst_stride_s = dst_d as u32;
        let dst_stride_d = 1;

        let uniforms = SliceUniforms {
            src_stride_b,
            src_stride_h,
            src_stride_s,
            src_stride_d,
            dst_stride_b,
            dst_stride_h,
            dst_stride_s,
            dst_stride_d,
            offset_b: offset[0] as u32,
            offset_h: offset[1] as u32,
            offset_s: offset[2] as u32,
            offset_d: offset[3] as u32,
            dst_shape_b: dst_b as u32,
            dst_shape_h: dst_h as u32,
            dst_shape_s: dst_s as u32,
            dst_shape_d: dst_d as u32,
        };

        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Slice Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Slice Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Slice Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_x = (dst_d as u32 + 15) / 16;
        let workgroup_y = (dst_s as u32 + 3) / 4;
        let workgroup_z = (dst_b * dst_h) as u32;
        compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, workgroup_z);
    }
}

#[cfg(test)]
mod tests;
