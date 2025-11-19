//! A GPU kernel for performing embedding lookups.
//!
//! This module translates a tensor of `u32` token IDs into a dense `f32` tensor
//! by looking up the corresponding vectors in an embedding table.

use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::gpu_context::WgpuContext;
use crate::gpu_ops::GpuTensor;


/// Uniforms to pass shape and size metadata to the lookup shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LookupUniforms {
    // Total number of elements in the output tensor (batch * seq * hidden)
    output_size: u32,
    // Strides for navigating the 3D output tensor from a 1D index
    output_batch_stride: u32, // = seq_len * hidden_size
    output_seq_stride: u32,   // = hidden_size
    // Metadata for the input tensors
    input_seq_len: u32,
    vocab_size: u32,
}

/// A GPU kernel for performing embedding lookups.
pub struct GpuLookup {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
    context: Arc<WgpuContext>,
}

impl GpuLookup {
    /// Creates a new `GpuLookup` module, compiling the lookup kernel.
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let (pipeline, bind_group_layout) = compile_lookup_pipeline(context);
        Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
            context: context.clone(),
        }
    }

    /// Encodes the lookup operation.
    ///
    /// output[i, j, k] = embedding_table[input_ids[i, j], k]
    ///
    /// # Panics
    /// Panics if tensor dimensions are incorrect.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        embedding_table: &GpuTensor, // The [vocab_size, hidden_size] f32 table
        input_ids: &GpuTensor,       // The [batch_size, seq_len] u32 IDs
        output: &GpuTensor,          // The [batch_size, seq_len, hidden_size] f32 output
    ) {
        // --- Validation ---
        assert_eq!(embedding_table.rank(), 2, "Embedding table must be 2D");
        assert_eq!(input_ids.rank(), 2, "Input IDs must be 2D");
        assert_eq!(output.rank(), 3, "Output must be 3D");
        assert_eq!(input_ids.dtype_as_str(), "u32", "Input IDs must be of type u32");

        let output_shape = output.shape();
        let batch_size = output_shape[0];
        let seq_len = output_shape[1];
        let hidden_size = output_shape[2];

        assert_eq!(batch_size, input_ids.shape()[0], "Batch dimensions must match");
        assert_eq!(seq_len, input_ids.shape()[1], "Sequence length dimensions must match");
        assert_eq!(hidden_size, embedding_table.shape()[1], "Hidden size dimensions must match");

        // --- Uniforms ---
        let uniforms = LookupUniforms {
            output_size: output.num_elements() as u32,
            output_batch_stride: (seq_len * hidden_size) as u32,
            output_seq_stride: hidden_size as u32,
            input_seq_len: input_ids.shape()[1] as u32,
            vocab_size: embedding_table.shape()[0] as u32,
        };
        let uniform_buffer = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lookup Uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // --- Bind Group ---
        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lookup Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: embedding_table.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: input_ids.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output.buffer().as_entire_binding() },
            ],
        });

        // --- Dispatch ---
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Lookup Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (uniforms.output_size + 255) / 256;
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }
}

/// Compiles the GPU pipeline for the lookup operation.
fn compile_lookup_pipeline(context: &WgpuContext) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader = context.device.create_shader_module(wgpu::include_wgsl!("lookup.wgsl"));
    let bind_group_layout = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Lookup Bind Group Layout"),
        entries: &[
            // Uniforms @binding(0)
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            // Embedding Table (f32) @binding(1)
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            // Input IDs (u32) @binding(2)
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            // Output (f32) @binding(3)
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Lookup Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Lookup Pipeline"),
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