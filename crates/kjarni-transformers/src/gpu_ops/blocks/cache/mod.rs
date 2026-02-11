use crate::gpu::GpuTensor;
use crate::{WgpuContext, gpu_profile};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayout, CommandEncoder, ComputePipeline};

pub mod reorder;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct UpdateCacheUniforms {
    b: u32,       // batch_size
    s_new: u32,   // seq_len_new (number of new tokens)
    h: u32,       // num_heads
    d: u32,       // head_dim
    s_total: u32, // The full capacity of the cache
    offset: u32,  // The time offset to write at
    _padding: [u32; 2],
}

/// A specialized kernel to update the KV cache
#[derive(Clone)]
pub struct GpuUpdateCache {
    pipeline: Arc<ComputePipeline>,
    bind_group_layout: Arc<BindGroupLayout>,
    context: Arc<WgpuContext>,
}

impl GpuUpdateCache {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let (pipeline, bind_group_layout) = compile_update_cache_pipeline(context);
        Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
            context: context.clone(),
        }
    }

    pub fn encode(
        &self,
        encoder: &mut CommandEncoder,
        new_k: &GpuTensor,      // The new K tensor [B, S_new, H*D]
        new_v: &GpuTensor,      // The new V tensor [B, S_new, H*D]
        cache_k: &GpuTensor,    // The full K cache [B, H, S_total, D]
        cache_v: &GpuTensor,    // The full V cache [B, H, S_total, D]
        position_offset: usize, // Current length of the cache
    ) {
        let (b, s_new, hidden_size) = new_k.dims3();
        let h = cache_k.shape()[1];
        let s_total = cache_k.shape()[2];
        let d = cache_k.shape()[3];
        assert_eq!(
            hidden_size,
            h * d,
            "Hidden size must equal heads * head_dim"
        );

        let uniforms = UpdateCacheUniforms {
            b: b as u32,
            s_new: s_new as u32,
            h: h as u32,
            d: d as u32,
            s_total: s_total as u32,
            offset: position_offset as u32,
            _padding: [0; 2],
        };

        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Update Cache Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Update Cache Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: new_k.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: new_v.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: cache_k.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: cache_v.buffer().as_entire_binding(),
                    },
                ],
            });

        gpu_profile!(
            self.context,
            encoder,
            "UpdateCache",
            |pass: &mut wgpu::ComputePass<'_>| {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let workgroups_x = (d as u32 + 15) / 16;
                let workgroups_y = (s_new as u32 + 15) / 16;
                let workgroups_z = (b * h) as u32;
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
        );
    }
}

fn compile_update_cache_pipeline(context: &WgpuContext) -> (ComputePipeline, BindGroupLayout) {
    let device = &context.device;
    let shader = device.create_shader_module(wgpu::include_wgsl!("./kv_update_cache.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Update Cache Bind Group Layout"),
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

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Update Cache Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Update Cache Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipeline, bind_group_layout)
}

#[cfg(test)]
mod tests;
