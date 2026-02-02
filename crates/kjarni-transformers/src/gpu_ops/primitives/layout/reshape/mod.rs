use crate::WgpuContext;
use crate::gpu::GpuTensor;
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
    transpose_k: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}

/// A specialized kernel to reshape and optionally transpose for multi-head attention.
/// Splits [B, S, H*D] into [B, H, S, D] (for Q/V) or [B, H, D, S] (for K).
pub struct GpuReshape {
    pipeline: Arc<ComputePipeline>,
    bind_group_layout: Arc<BindGroupLayout>,
    context: Arc<WgpuContext>,
}

impl GpuReshape {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let (pipeline, bind_group_layout) = compile_reshape_pipeline(context);
        Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
            context: context.clone(),
        }
    }

    pub fn encode(
        &self,
        encoder: &mut CommandEncoder,
        input: &GpuTensor,
        output: &GpuTensor,
        transpose_k: bool,
    ) {
        // --- Assertions for correctness ---
        assert_eq!(input.rank(), 3, "Reshape input must be rank 3 [B, S, H]");
        assert_eq!(
            output.rank(),
            4,
            "Reshape output must be rank 4 [B, H, S, D] or [B, H, D, S]"
        );

        let (b, s, h_d) = input.dims3();
        let (out_b, out_h, out_s_or_d, out_d_or_s) = output.dims4();

        assert_eq!(b, out_b, "Batch dimensions must match");
        let h = out_h;
        let d = if transpose_k { out_s_or_d } else { out_d_or_s };
        let s_check = if transpose_k { out_d_or_s } else { out_s_or_d };
        assert_eq!(s, s_check, "Sequence length dimension mismatch");
        assert_eq!(h * d, h_d, "Hidden size must equal heads * head_dim");

        let uniforms = ReshapeUniforms {
            b: b as u32,
            s: s as u32,
            h: h as u32,
            d: d as u32,
            transpose_k: if transpose_k { 1 } else { 0 },
            _padding1: 0,
            _padding2: 0,
            _padding3: 0,
        };

        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Reshape Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Reshape Bind Group"),
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

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Reshape Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch a 3D grid that matches the shader's logic
        let workgroups_x = (s as u32 + 15) / 16;
        let workgroups_y = (h as u32 + 15) / 16;
        let workgroups_z = b as u32;
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
    }
}

fn compile_reshape_pipeline(context: &WgpuContext) -> (ComputePipeline, BindGroupLayout) {
    let device = &context.device;
    // Make sure you have a `reshape.wgsl` file at this location
    let shader = device.create_shader_module(wgpu::include_wgsl!("./reshape.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Reshape Bind Group Layout"),
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
        label: Some("Reshape Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Reshape Pipeline"),
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
