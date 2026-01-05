use crate::gpu_ops::GpuTensor;
use crate::{WgpuContext, gpu_profile};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayout, CommandEncoder, ComputePipeline};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ReshapeUniforms {
    b: u32,
    s: u32,
    h: u32,
    d: u32,
}

/// A specialized kernel to merge attention heads.
/// Reshapes [B, H, S, D] back into [B, S, H*D].
pub struct GpuUnreshape {
    pipeline: Arc<ComputePipeline>,
    bind_group_layout: Arc<BindGroupLayout>,
    context: Arc<WgpuContext>,
}

impl GpuUnreshape {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let (pipeline, bind_group_layout) = compile_unreshape_pipeline(context);
        Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
            context: context.clone(),
        }
    }

    pub fn encode(&self, encoder: &mut CommandEncoder, input: &GpuTensor, output: &GpuTensor) {
        // --- Assertions for correctness ---
        assert_eq!(
            input.rank(),
            4,
            "Unreshape input must be rank 4 [B, H, S, D]"
        );
        assert_eq!(
            output.rank(),
            3,
            "Unreshape output must be rank 3 [B, S, H]"
        );

        let (b, h, s, d) = input.dims4();
        let (out_b, out_s, out_h_d) = output.dims3();

        assert_eq!(b, out_b, "Batch dimensions must match");
        assert_eq!(s, out_s, "Sequence length dimensions must match");
        assert_eq!(h * d, out_h_d, "Hidden size must equal heads * head_dim");

        let uniforms = ReshapeUniforms {
            b: b as u32,
            s: s as u32,
            h: h as u32,
            d: d as u32,
        };

        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Unreshape Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Unreshape Bind Group"),
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
            "Unreshape",
            |pass: &mut wgpu::ComputePass<'_>| {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);

                // Dispatch a 3D grid that matches the shader's logic
                let workgroups_x = (s as u32 + 15) / 16;
                let workgroups_y = (h as u32 + 15) / 16;
                let workgroups_z = b as u32;
                pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }
        );
    }
}

fn compile_unreshape_pipeline(context: &WgpuContext) -> (ComputePipeline, BindGroupLayout) {
    let device = &context.device;
    // Make sure you have an `unreshape.wgsl` file at this location
    let shader = device.create_shader_module(wgpu::include_wgsl!("./unreshape.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Unreshape Bind Group Layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Unreshape Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Unreshape Pipeline"),
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
