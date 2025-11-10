use crate::gpu_context::WgpuContext;
use crate::gpu_ops::GpuTensor;
use anyhow::Result;
use ndarray::Array2;
use std::sync::Arc;
use wgpu::util::DeviceExt;

mod tests;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RoPEUniforms {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    position_offset: u32,
    cos_stride: u32,
    sin_stride: u32,
    _padding: u32,
}

/// A GPU kernel for applying Rotary Position Embeddings (RoPE).
/// This is an in-place operation on the Q and K tensors.
pub struct GpuRoPE {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    context: Arc<WgpuContext>,
    // Precomputed frequency tables, stored on the GPU for fast access.
    cos_cache: GpuTensor,
    sin_cache: GpuTensor,
}

impl GpuRoPE {
    pub fn new(context: &Arc<WgpuContext>, cos_cache_cpu: &Array2<f32>, sin_cache_cpu: &Array2<f32>) -> Result<Self> {
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("./rope.wgsl"));

        // Upload the precomputed tables to the GPU
        let cos_cache = GpuTensor::from_ndarray(context, cos_cache_cpu)?;
        let sin_cache = GpuTensor::from_ndarray(context, sin_cache_cpu)?;

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("RoPE Bind Group Layout"),
                    entries: &[
                        // Uniforms
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                            count: None,
                        },
                        // Q tensor (read_write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                            count: None,
                        },
                        // K tensor (read_write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                            count: None,
                        },
                        // Cosine cache (read_only)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                            count: None,
                        },
                        // Sine cache (read_only)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("RoPE Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RoPE Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            pipeline,
            bind_group_layout,
            context: context.clone(),
            cos_cache,
            sin_cache,
        })
    }

    /// Encodes the RoPE operation. This is an IN-PLACE modification of Q and K.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q: &GpuTensor, // The Query tensor to modify [B, H, S, D]
        k: &GpuTensor, // The Key tensor to modify   [B, H, S, D]
        position_offset: usize,
    ) {
        let (batch_size, num_heads, seq_len, head_dim) = q.dims4();
        assert_eq!(k.shape(), q.shape(), "Q and K must have the same shape for RoPE");

        let uniforms = RoPEUniforms {
            batch_size: batch_size as u32,
            num_heads: num_heads as u32,
            seq_len: seq_len as u32,
            head_dim: head_dim as u32,
            position_offset: position_offset as u32,
            cos_stride: self.cos_cache.shape()[1] as u32,
            sin_stride: self.sin_cache.shape()[1] as u32,
            _padding: 0,
        };
        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("RoPE Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RoPE Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: q.buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: k.buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.cos_cache.buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.sin_cache.buffer().as_entire_binding() },
                ],
            });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("RoPE Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        
        // Dispatch one thread per dimension pair per sequence element per head per batch item.
        let workgroups_x = (head_dim as u32 / 2 + 15) / 16;
        let workgroups_y = seq_len as u32;
        let workgroups_z = (batch_size * num_heads) as u32;
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
    }
}