use crate::WgpuContext;
use crate::gpu::GpuTensor;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Extracts the CLS token (token index 0) from encoder hidden states.
///
/// Input:  [batch, seq_len, hidden_size]
/// Output: [batch, hidden_size]
pub struct GpuClsSlice {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    context: Arc<WgpuContext>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ClsSliceUniforms {
    batch_size: u32,
    seq_len: u32,     // Full sequence length in source
    hidden_size: u32, // Hidden size of encoder
    _padding: u32,
}

impl GpuClsSlice {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let shader = context.device.create_shader_module(
            wgpu::ShaderModuleDescriptor {
                label: Some("CLS Slice Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("./clsslice.wgsl").into()
                ),
            },
        );

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("CLS Slice Bind Group Layout"),
                    entries: &[
                        // Input: [batch, seq_len, hidden_size]
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage {
                                    read_only: true,
                                },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output: [batch, hidden_size]
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage {
                                    read_only: false,
                                },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Uniforms
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
                    label: Some("CLS Slice Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline =
            context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("CLS Slice Pipeline"),
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

    /// Extract CLS token: [batch, seq_len, hidden] → [batch, hidden]
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &GpuTensor, // [batch, seq_len, hidden_size]
        dst: &GpuTensor, // [batch, hidden_size]
    ) {
        assert_eq!(src.rank(), 3, "Source must be 3D: [batch, seq, hidden]");

        let batch_size = src.shape()[0];
        let seq_len = src.shape()[1];
        let hidden_size = src.shape()[2];

        let uniforms = ClsSliceUniforms {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            hidden_size: hidden_size as u32,
            _padding: 0,
        };

        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("CLS Slice Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group =
            self.context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("CLS Slice Bind Group"),
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

        let mut compute_pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CLS Slice Pass"),
                timestamp_writes: None,
            });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // One thread per (batch, hidden) element
        let workgroup_size = 256;
        let total_elements = (batch_size * hidden_size) as u32;
        let num_workgroups =
            (total_elements + workgroup_size - 1) / workgroup_size;

        compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
    }
}

#[cfg(test)]
mod tests;