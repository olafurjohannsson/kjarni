use crate::gpu::GpuTensor;
use crate::tensor::DType;
use crate::{WgpuContext, gpu_profile};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LinearInfo {
    m: u32,
    k: u32,
    n: u32,
}

/// Row count from which the GEMMs use 128x128 tiles instead of 64x64. Measured
/// on an RTX A2000 at K=2048, N=8192: 64x64 is up to 1.5x faster at 16-100
/// rows, the two meet near 128, and 128x128 is 1.5x faster by 255 rows.
const GEMM_LARGE_TILE_MIN_ROWS: u32 = 128;

/// Row count below which the F32 and BF16 GEMMs take one thread per output instead
/// of a tile: two rows, a pair of chunks or beams stepping together. A tile costs the
/// same from 2 to 48 rows because reading the weights is what limits it. Measured on
/// an RTX A2000 at K=768, N=3072: 0.22 ms against 0.31 at two rows, level at three,
/// 0.38 against 0.31 at four.
const GEMM_FEW_ROWS: u32 = 3;

/// The most rows the wide GEMMs take, which is `WIDE_ROWS` in the shader. Their cost
/// rises with each row and a tile's does not, and the two meet about here.
const GEMM_WIDE_MAX_ROWS: u32 = 10;

/// The fewest outputs for a wide GEMM. Below this a workgroup per output is wasted.
const GEMM_WIDE_MIN_OUTPUTS: u32 = 128;

pub struct GpuLinearLayer {
    context: Arc<WgpuContext>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,

    // Pipelines
    gemv_f32: wgpu::ComputePipeline,
    gemv_bf16: wgpu::ComputePipeline,
    gemv_bf16_wide: wgpu::ComputePipeline,
    gemv_q4k: wgpu::ComputePipeline,
    gemm_q4k: wgpu::ComputePipeline,
    gemv_q6k: wgpu::ComputePipeline,
    gemm_q6k: wgpu::ComputePipeline,
    gemv_q8: wgpu::ComputePipeline,
    gemm_q8: wgpu::ComputePipeline,
    gemm_f32: wgpu::ComputePipeline,
    gemm64_f32: wgpu::ComputePipeline,
    gemm64_bf16: wgpu::ComputePipeline,
    gemm64_q4k: wgpu::ComputePipeline,
    gemm64_q6k: wgpu::ComputePipeline,
    gemm64_q8: wgpu::ComputePipeline,
    gemm_bf16: wgpu::ComputePipeline,
    gemm_wide_f32: wgpu::ComputePipeline,
    gemm_wide_bf16: wgpu::ComputePipeline,
    gemm_few_rows_f32: wgpu::ComputePipeline,
    gemm_few_rows_bf16: wgpu::ComputePipeline,

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
                // 10: B_Q8 (Weights, repacked to 36 bytes per block)
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 9: B_Q6K (Weights, packed and padded to 212 bytes per block)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: B_Q4K (Weights, packed)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
            gemv_bf16_wide: make_pipeline("gemv_bf16_wide"),
            gemv_q4k: make_pipeline("gemv_q4k"),
            gemm_q4k: make_pipeline("gemm_q4k"),
            gemv_q6k: make_pipeline("gemv_q6k"),
            gemm_q6k: make_pipeline("gemm_q6k"),
            gemv_q8: make_pipeline("gemv_q8"),
            gemm_q8: make_pipeline("gemm_q8"),
            gemm_f32: make_pipeline("gemm_f32"),
            gemm64_f32: make_pipeline("gemm64_f32"),
            gemm64_bf16: make_pipeline("gemm64_bf16"),
            gemm64_q4k: make_pipeline("gemm64_q4k"),
            gemm64_q6k: make_pipeline("gemm64_q6k"),
            gemm64_q8: make_pipeline("gemm64_q8"),
            gemm_bf16: make_pipeline("gemm_bf16"),
            gemm_wide_f32: make_pipeline("gemm_wide_f32"),
            gemm_wide_bf16: make_pipeline("gemm_wide_bf16"),
            gemm_few_rows_f32: make_pipeline("gemm_few_rows_f32"),
            gemm_few_rows_bf16: make_pipeline("gemm_few_rows_bf16"),
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

        // Q4_K stays packed: decode reads it in gemv_q4k, prefill dequantises it
        // tile by tile in gemm_q4k.
        let is_q4k = weights.dtype() == DType::Q4_K;
        let is_q6k = weights.dtype() == DType::Q6_K;
        let is_q8 = weights.dtype() == DType::Q8_0;
        // The GEMV kernels take a few rows as well, one set of workgroups per row.
        let few_quantized_rows = m <= GEMM_WIDE_MAX_ROWS;
        let use_q4k_kernel = few_quantized_rows && is_q4k;
        let use_q6k_kernel = few_quantized_rows && is_q6k;
        let use_q8_kernel = few_quantized_rows && is_q8;
        let use_wide_kernel = is_gemv && is_bf16 && n >= 128;

        // Below this many rows the 128x128 GEMM tile is mostly empty and the
        // 64x64 one is faster; above it the larger tile reuses more.
        let small_tile = m < GEMM_LARGE_TILE_MIN_ROWS;
        let is_float = matches!(weights.dtype(), DType::F32 | DType::BF16);
        let wide_gemm =
            is_float && !is_gemv && m <= GEMM_WIDE_MAX_ROWS && n >= GEMM_WIDE_MIN_OUTPUTS;
        let few_rows = is_float && !is_gemv && !wide_gemm && m < GEMM_FEW_ROWS;

        let pipeline = if use_q4k_kernel {
            &self.gemv_q4k
        } else if is_q4k {
            if small_tile {
                &self.gemm64_q4k
            } else {
                &self.gemm_q4k
            }
        } else if use_q6k_kernel {
            &self.gemv_q6k
        } else if is_q6k {
            if small_tile {
                &self.gemm64_q6k
            } else {
                &self.gemm_q6k
            }
        } else if use_q8_kernel {
            &self.gemv_q8
        } else if is_q8 {
            if small_tile {
                &self.gemm64_q8
            } else {
                &self.gemm_q8
            }
        } else if use_wide_kernel {
            &self.gemv_bf16_wide
        } else {
            match (is_gemv, is_bf16) {
                (true, true) => &self.gemv_bf16,
                (true, false) => &self.gemv_f32,
                (false, true) if wide_gemm => &self.gemm_wide_bf16,
                (false, false) if wide_gemm => &self.gemm_wide_f32,
                (false, true) if few_rows => &self.gemm_few_rows_bf16,
                (false, false) if few_rows => &self.gemm_few_rows_f32,
                (false, true) if small_tile => &self.gemm64_bf16,
                (false, true) => &self.gemm_bf16,
                (false, false) if small_tile => &self.gemm64_f32,
                (false, false) => &self.gemm_f32,
            }
        };

        let uniforms = LinearInfo { m, k, n };
        let uniform_buffer = self.context.uniform_buffer(&uniforms);

        let (b_f32, b_bf16, b_q4k, b_q6k, b_q8) = if is_q4k {
            let b = &self.buffer;
            (b, b, weights.buffer(), b, b)
        } else if is_q6k {
            let b = &self.buffer;
            (b, b, b, weights.buffer(), b)
        } else if is_q8 {
            let b = &self.buffer;
            (b, b, b, b, weights.buffer())
        } else if is_bf16 {
            let b = &self.buffer;
            (b, weights.buffer(), b, b, b)
        } else {
            let b = &self.buffer;
            (weights.buffer(), b, b, b, b)
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
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: b_q4k.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: b_q6k.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: b_q8.as_entire_binding(),
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

                if use_q4k_kernel || use_q6k_kernel || use_q8_kernel {
                    // Each workgroup produces 8 output rows for Q4_K and 4 otherwise.
                    let rows = n.div_ceil(if use_q4k_kernel { 8 } else { 4 });
                    let max_dim = 65535;
                    if rows > max_dim {
                        pass.dispatch_workgroups(max_dim, rows.div_ceil(max_dim), m);
                    } else {
                        pass.dispatch_workgroups(rows, 1, m);
                    }
                } else if use_wide_kernel || wide_gemm {
                    // Dispatch 2D grid if N > 65535
                    let max_dim = 65535;
                    if n > max_dim {
                        let grid_x = max_dim;
                        let grid_y = n.div_ceil(max_dim);
                        pass.dispatch_workgroups(grid_x, grid_y, 1);
                    } else {
                        pass.dispatch_workgroups(n, 1, 1);
                    }
                } else if is_gemv {
                    // Standard GEMV: 1 Thread per Output Neuron
                    let groups = n.div_ceil(256);
                    pass.dispatch_workgroups(groups, 1, 1);
                } else if few_rows {
                    crate::gpu_ops::utils::dispatch_1d(pass, (m * n).div_ceil(32));
                } else {
                    // The GEMMs: one 64x64 or 128x128 tile of outputs per workgroup.
                    let tile = if small_tile { 64 } else { 128 };
                    pass.dispatch_workgroups(n.div_ceil(tile), m.div_ceil(tile), 1);
                }
            }
        );
    }
}

#[cfg(test)]
mod tests;
