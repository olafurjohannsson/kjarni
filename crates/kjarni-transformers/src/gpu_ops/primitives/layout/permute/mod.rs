use crate::gpu_ops::{GpuTensor, Kernel};
use crate::{WgpuContext, gpu_profile};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayout, CommandEncoder, ComputePipeline};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PermuteUniforms {
    out_shape: [u32; 4],
    out_strides: [u32; 4],
    perm: [u32; 4],
}

/// A generic kernel for permuting the dimensions of a tensor.
pub struct GpuPermute {
    pipeline: Arc<ComputePipeline>,
    bind_group_layout: Arc<BindGroupLayout>,
    context: Arc<WgpuContext>,
}

impl GpuPermute {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let (pipeline, bind_group_layout) = compile_permute_pipeline(context);
        Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
            context: context.clone(),
        }
    }

    /// A specific `encode` for this kernel, as the generic `Kernel` trait is too simple.
    pub fn encode(
        &self,
        encoder: &mut CommandEncoder,
        input: &GpuTensor,
        output: &GpuTensor,
        perm: &[usize],
    ) {
        assert_eq!(
            input.rank(),
            perm.len(),
            "Permutation length must match tensor rank"
        );
        assert_eq!(
            input.rank(),
            output.rank(),
            "Input and output ranks must match"
        );

        let rank = input.rank();
        let mut out_shape_padded = [1; 4];
        let mut perm_padded = [0, 1, 2, 3];

        // Pad the shape and permutation to 4D for the shader
        for i in 0..rank {
            out_shape_padded[i] = output.shape()[i] as u32;
            perm_padded[i] = perm[i] as u32;
        }

        // Calculate strides for the output tensor
        let mut out_strides_padded = [0; 4];
        out_strides_padded[3] = 1;
        for i in (0..3).rev() {
            out_strides_padded[i] = out_strides_padded[i + 1] * out_shape_padded[i + 1];
        }

        let uniforms = PermuteUniforms {
            out_shape: out_shape_padded,
            out_strides: out_strides_padded,
            perm: perm_padded,
        };

        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Permute Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Permute Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer().as_entire_binding(),
                    },
                ],
            });

        gpu_profile!(
            self.context,
            encoder,
            "Permute",
            |pass: &mut wgpu::ComputePass<'_>| {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let workgroups = (output.num_elements() as u32 + 255) / 256;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        );
    }
}

fn compile_permute_pipeline(context: &WgpuContext) -> (ComputePipeline, BindGroupLayout) {
    let device = &context.device;
    let shader = device.create_shader_module(wgpu::include_wgsl!("./permute.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Permute Bind Group Layout"),
        entries: &[
            // @binding(0) var<uniform> uniforms: PermuteUniforms;
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
            // @binding(1) var<storage, read> input: array<f32>;
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
            // @binding(2) var<storage, read_write> output: array<f32>;
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
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Permute Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Permute Pipeline"),
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
