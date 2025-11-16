use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::gpu_context::WgpuContext;
use crate::gpu_ops::{DType, GpuTensor, Kernel};

mod tests;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AddBroadcastOffsetUniforms {
    output_size: u32,
    b_row_offset: u32,
    seq_len: u32,
    hidden_size: u32,
    b_stride_0: u32, // <-- ADD THIS: Stride for the first dimension of b (rows)
    _padding: [u32; 3], // <-- ADD THIS: Ensure alignment
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AddUniforms {
    size: u32,
    _padding: [u32; 3],
}

/// A GPU kernel for element-wise addition of tensors.
/// Supports both element-wise and broadcast-with-offset addition.
pub struct GpuAdd {
    elementwise_pipeline: Arc<wgpu::ComputePipeline>,
    elementwise_layout: Arc<wgpu::BindGroupLayout>,
    broadcast_offset_pipeline: Arc<wgpu::ComputePipeline>,
    broadcast_offset_layout: Arc<wgpu::BindGroupLayout>,
    context: Arc<WgpuContext>,
}

impl GpuAdd {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let (ew_pipe, ew_layout) = compile_elementwise_pipeline(context);
        let (bo_pipe, bo_layout) = compile_broadcast_offset_pipeline(context);
        Self {
            elementwise_pipeline: Arc::new(ew_pipe),
            elementwise_layout: Arc::new(ew_layout),
            broadcast_offset_pipeline: Arc::new(bo_pipe),
            broadcast_offset_layout: Arc::new(bo_layout),
            context: context.clone(),
        }
    }

    /// Encodes standard element-wise addition: `output = a + b`.
    ///
    /// # Panics
    /// Panics if `a`, `b`, and `output` do not have the exact same shape.
    pub fn encode_elementwise(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuTensor,
        b: &GpuTensor,
        output: &GpuTensor,
    ) {
        assert_eq!(
            a.shape(),
            b.shape(),
            "Input tensors must have the same shape for element-wise addition."
        );
        assert_eq!(
            a.shape(),
            output.shape(),
            "Output tensor must have the same shape as inputs for element-wise addition."
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
                    label: Some("Add Element-wise Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Add Element-wise Bind Group"),
                layout: &self.elementwise_layout,
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
            label: Some("Add Element-wise Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.elementwise_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (size + 255) / 256;
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }
    /// Encodes broadcast addition with an offset: `output[i,j,k] = a[i,j,k] + b[j + offset, k]`.
    ///
    /// Designed for adding positional embeddings to word embeddings.
    ///
    /// # Panics
    /// Panics if the hidden dimensions of `a` and `b` do not match.
    pub fn encode_broadcast_offset(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuTensor,       // The 3D hidden_states tensor
        b: &GpuTensor,       // The 2D positional embedding table
        b_row_offset: usize, // The starting row in `b` (e.g., 2 for BART)
        output: &GpuTensor,
    ) {
        let a_shape = a.shape(); // [batch, seq_len, hidden_size]
        let b_shape = b.shape(); // [max_pos, hidden_size]

        assert_eq!(a.rank(), 3, "Input 'a' must be a 3D tensor.");
        assert_eq!(b.rank(), 2, "Input 'b' must be a 2D tensor.");
        assert_eq!(
            a_shape[2], b_shape[1],
            "Hidden dimensions of 'a' and 'b' must match."
        );
        assert_eq!(
            a.shape(),
            output.shape(),
            "Output tensor must have the same shape as input 'a'."
        );

        let uniforms = AddBroadcastOffsetUniforms {
            output_size: a.num_elements() as u32,
            b_row_offset: b_row_offset as u32,
            seq_len: a_shape[1] as u32,
            hidden_size: a_shape[2] as u32,
            b_stride_0: b_shape[1] as u32,
            _padding: [0; 3],
        };
        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Add Broadcast Offset Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Add Broadcast Offset Bind Group"),
                layout: &self.broadcast_offset_layout,
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
            label: Some("Add Broadcast Offset Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.broadcast_offset_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (uniforms.output_size + 255) / 256;
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
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
        assert_eq!(
            inputs.len(),
            2,
            "GpuAdd kernel requires exactly two input tensors."
        );
        let tensor_a: &GpuTensor = inputs[0];
        let tensor_b: &GpuTensor = inputs[1];
        self.encode_elementwise(encoder, tensor_a, tensor_b, output)
    }
}

fn compile_elementwise_pipeline(
    context: &WgpuContext,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
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
    let pipeline_layout = context
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
    (pipeline, bind_group_layout)
}

fn compile_broadcast_offset_pipeline(
    context: &WgpuContext,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader = context
        .device
        .create_shader_module(wgpu::include_wgsl!("add_broadcast_offset.wgsl"));
    let bind_group_layout =
        context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Add Broadcast Offset Bind Group Layout"),
                entries: &[
                    // Uniforms @binding(0)
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
                    // Input `a` (hidden_states) @binding(1)
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
                    // Input `b` (pos_embeddings) @binding(2)
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
                    // Output @binding(3)
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
    let pipeline_layout = context
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Add Broadcast Offset Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
    let pipeline = context
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Add Broadcast Offset Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    (pipeline, bind_group_layout)
}
