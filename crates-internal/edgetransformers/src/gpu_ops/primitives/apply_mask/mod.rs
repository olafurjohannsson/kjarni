use crate::gpu_context::WgpuContext;
use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayout, Buffer, CommandEncoder, ComputePipeline, include_wgsl};

use crate::gpu_ops::{GpuTensor};
use std::sync::Arc;



#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    batch_size: u32,
    num_heads: u32,
    query_len: u32,
    logical_key_len: u32,
    key_stride: u32,
    is_causal: u32,
    position_offset: u32,
    _padding: u32, // Rust structs need to be aligned to 16 bytes for WGSL
}

/// A kernel for applying causal and/or padding masks to an attention score tensor.
pub struct GpuApplyMask {
    pipeline: Arc<ComputePipeline>,
    bind_group_layout: Arc<BindGroupLayout>,
    context: Arc<WgpuContext>,
}

impl GpuApplyMask {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let (pipeline, bind_group_layout) = compile_apply_mask_pipeline(context);
        Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
            context: context.clone(),
        }
    }

    /// Encodes the masking operation. This is an in-place operation.
    pub fn encode(
        &self,
        encoder: &mut CommandEncoder,
        scores: &GpuTensor, // The [B, H, S_q, S_k_padded] score tensor
        mask: &GpuTensor,   // The [B, S_k_padded] mask tensor
        is_causal: bool,
        position_offset: u32,
        logical_key_len: u32,
    ) {
        let scores_shape = scores.shape();
        let mask_shape = mask.shape();

        let batch_size = scores_shape[0] as u32;
        let num_heads = scores_shape[1] as u32;
        let query_len = scores_shape[2] as u32;
        let key_stride = scores_shape[3] as u32; // Physical width of scores

        // Sanity check
        assert_eq!(key_stride, mask_shape[1] as u32, "Mask and Score physical widths must match");

        // let logical_key_len = position_offset + query_len;
        // self-attention = position_offset + query_len
        // cross-attention encoder_sequence_lngth

        let uniforms = Uniforms {
            batch_size,
            num_heads,
            query_len,
            logical_key_len,
            key_stride,
            is_causal: if is_causal { 1 } else { 0 },
            position_offset,
            _padding: 0,
        };

        run_internal_apply_mask(
            &self.context,
            encoder,
            &self.pipeline,
            &self.bind_group_layout,
            scores.buffer(),
            mask.buffer(),
            uniforms,
        );
    }
}

fn run_internal_apply_mask(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    bind_group_layout: &BindGroupLayout,
    scores: &Buffer,
    mask: &Buffer,
    uniforms: Uniforms,
) {
    let device = &context.device;
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Apply Mask Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Apply Mask Bind Group"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scores.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: mask.as_entire_binding(),
            },
        ],
    });

    // let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
    //     label: Some("Apply Mask Pass"),
    //     timestamp_writes: None,
    // });
    let label = format!("ApplyMask");
    context.profiler.profile(encoder, &label, |compute_pass| {
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_x = (uniforms.query_len + 15) / 16;
        let workgroup_y = (uniforms.key_stride + 15) / 16;
        let workgroup_z = uniforms.batch_size * uniforms.num_heads;
        compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, workgroup_z)
    });
}

pub fn compile_apply_mask_pipeline(context: &WgpuContext) -> (ComputePipeline, BindGroupLayout) {
    let device = &context.device;
    let shader = device.create_shader_module(include_wgsl!("./apply_mask.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Apply Mask Bind Group Layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Apply Mask Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Apply Mask Pipeline"),
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
