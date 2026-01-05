use crate::{WgpuContext, gpu_profile};
use crate::gpu_ops::GpuTensor;
use crate::tensor::DType;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LinearInfo {
    m: u32,
    k: u32,
    n: u32,
}

pub struct GpuLinearLayer {
    context: Arc<WgpuContext>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,

    // Pipelines
    gemv_f32: wgpu::ComputePipeline,
    gemv_bf16: wgpu::ComputePipeline,
    gemv_bf16_wide: wgpu::ComputePipeline, // <--- NEW
    bmm_f32: wgpu::ComputePipeline,
    bmm_bf16: wgpu::ComputePipeline,

    buffer: wgpu::Buffer,
}

impl GpuLinearLayer {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let device = &context.device;
        let shader = device.create_shader_module(wgpu::include_wgsl!("./linear.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Linear Layer Layout"),
            entries: &[
                // 0: Info
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
                // 1: A (Input)
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
                // 2: B_F32 (Weights)
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
                // 3: B_BF16 (Weights)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: Output
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
            label: Some("Linear Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let make_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Linear Dummy"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Self {
            context: context.clone(),
            bind_group_layout: Arc::new(bind_group_layout),
            gemv_f32: make_pipeline("gemv_f32"),
            gemv_bf16: make_pipeline("gemv_bf16"),
            gemv_bf16_wide: make_pipeline("gemv_bf16_wide"), // <--- Load new kernel
            bmm_f32: make_pipeline("bmm_f32"),
            bmm_bf16: make_pipeline("bmm_bf16"),
            buffer,
        }
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        weights: &GpuTensor,
        output: &GpuTensor,
    ) {
        let m = (input.num_elements() / input.shape().last().unwrap()) as u32;
        let k = weights.shape()[1] as u32;
        let n = weights.shape()[0] as u32;

        let is_bf16 = weights.dtype() == DType::BF16;
        let is_gemv = m == 1;

        // --- HEURISTIC SELECTION ---
        // For lm_head (N > 32000), standard GEMV is slow because of uncoalesced reads.
        // We use the "Wide" kernel which assigns a full workgroup to one output row.
        let use_wide_kernel = is_gemv && is_bf16 && n >= 128;

        let pipeline = if use_wide_kernel {
            &self.gemv_bf16_wide
        } else {
            match (is_gemv, is_bf16) {
                (true, true) => &self.gemv_bf16,
                (true, false) => &self.gemv_f32,
                (false, true) => &self.bmm_bf16,
                (false, false) => &self.bmm_f32,
            }
        };

        let uniforms = LinearInfo { m, k, n };
        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Linear Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let (b_f32, b_bf16) = if is_bf16 {
            (&self.buffer, weights.buffer())
        } else {
            (weights.buffer(), &self.buffer)
        };

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Linear BindGroup"),
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
                        resource: b_f32.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: b_bf16.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: output.buffer().as_entire_binding(),
                    },
                ],
            });

        let label = format!("Linear [{}x{} @ {}]", m, k, n);

        gpu_profile!(
            self.context,
            encoder,
            &label,
            |pass: &mut wgpu::ComputePass<'_>| {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);

                if use_wide_kernel {
                    // Dispatch 2D grid if N > 65535
                    let max_dim = 65535;
                    if n > max_dim {
                        let grid_x = max_dim;
                        let grid_y = (n + max_dim - 1) / max_dim;
                        pass.dispatch_workgroups(grid_x, grid_y, 1);
                    } else {
                        pass.dispatch_workgroups(n, 1, 1);
                    }
                } else if is_gemv {
                    // Standard GEMV: 1 Thread per Output Neuron
                    let groups = (n + 255) / 256;
                    pass.dispatch_workgroups(groups, 1, 1);
                } else {
                    // BMM: 2D Tiles
                    let groups_x = (n + 15) / 16;
                    let groups_y = (m + 15) / 16;
                    pass.dispatch_workgroups(groups_x, groups_y, 1);
                }
            }
        );
    }
}
