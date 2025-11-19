

use std::sync::Arc;
use wgpu::util::DeviceExt;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::GpuTensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    batch_size: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    seq_len: u32,
    head_dim: u32,
    _p1: u32, _p2: u32, _p3: u32, // Padding
}

pub struct GpuRepeatKV {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    context: Arc<WgpuContext>,
}

impl GpuRepeatKV {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let device = &context.device;
        let shader = device.create_shader_module(wgpu::include_wgsl!("./repeat_kv.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RepeatKV Bind Group Layout"),
            entries: &[
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // Input KV Tensor
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // Output KV Tensor
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RepeatKV Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RepeatKV Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, bind_group_layout, context: context.clone() }
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_kv: &GpuTensor, // Shape: [B, num_kv_heads, S, D]
        output_kv: &GpuTensor, // Shape: [B, num_q_heads, S, D]
    ) {
        let (b, num_kv_heads, s, d) = input_kv.dims4();
        let num_q_heads = output_kv.shape()[1];

        let uniforms = Uniforms {
            batch_size: b as u32,
            num_q_heads: num_q_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            seq_len: s as u32,
            head_dim: d as u32,
            _p1: 0, _p2: 0, _p3: 0,
        };

        let uniform_buffer = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RepeatKV Uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RepeatKV Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input_kv.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_kv.buffer().as_entire_binding() },
            ],
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("RepeatKV Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (output_kv.num_elements() as u32 + 255) / 256;
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }
}


#[cfg(test)]
mod tests;