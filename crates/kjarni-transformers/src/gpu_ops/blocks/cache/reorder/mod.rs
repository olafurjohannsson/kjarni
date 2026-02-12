use crate::WgpuContext;
use crate::gpu::GpuTensor;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroupEntry, CommandEncoder, ComputePipeline};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ReorderUniforms {
    num_beams: u32,
    num_heads: u32,
    capacity: u32,
    head_dim: u32,
    current_seq_len: u32,
    _padding: [u32; 3], // Matches pad1, pad2, pad3 in WGSL
}

#[derive(Clone)]
pub struct GpuReorderCache {
    pipeline: Arc<ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
    context: Arc<WgpuContext>,
}

impl GpuReorderCache {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let device = &context.device;

        // Ensure this string matches the WGSL above
        let shader_source = include_str!("./reorder.wgsl");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reorder Cache Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Reorder Cache Layout"),
            entries: &[
                // 0: Uniforms
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
                // 1: Indices (Read)
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
                // 2: Src Cache (Read)
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
                // 3: Dst Cache (Write)
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Reorder Cache Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reorder Cache Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            cache: None,
            compilation_options: Default::default(),
            entry_point: Some("main"),
        });

        Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
            context: context.clone(),
        }
    }

    pub fn encode(
        &self,
        encoder: &mut CommandEncoder,
        src_cache: &GpuTensor,
        dst_cache: &GpuTensor,
        indices: &GpuTensor,
        current_seq_len: usize,
    ) {
        let (num_beams, num_heads, capacity, head_dim) = src_cache.dims4();

        let uniforms = ReorderUniforms {
            num_beams: num_beams as u32,
            num_heads: num_heads as u32,
            capacity: capacity as u32,
            head_dim: head_dim as u32,
            current_seq_len: current_seq_len as u32,
            _padding: [0; 3],
        };

        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Reorder Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Reorder Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: indices.buffer().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: src_cache.buffer().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: dst_cache.buffer().as_entire_binding(),
                    },
                ],
            });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Reorder Cache Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch logic
        let wg_x = (head_dim as u32 + 15) / 16;
        let wg_y = (num_heads as u32 + 15) / 16;
        let wg_z = (num_beams * current_seq_len) as u32;

        pass.dispatch_workgroups(wg_x, wg_y, wg_z);
    }
}

#[cfg(test)]
mod tests;
