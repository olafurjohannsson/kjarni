//! GPU kernel for Layer Normalization with learnable scale and shift.
//!
//! # Overview
//!
//! Implements LayerNorm: `y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta`
//!
//! Used by encoder-based models (BERT, GPT-2, etc.) for normalizing activations
//! across the hidden dimension. Provides training stability and faster convergence.
//!
//! # Performance
//!
//! - **Current**: ~0.4ms for [128, 768] on RTX 3090
//! - **Critical Issue**: Only 1 thread per row active (255/256 threads idle)
//! - **Memory**: 4 passes over data (mean, variance, normalize, scale)
//!
//! # TODO
//!
//! - **CRITICAL**: Fix memory leak - uniform buffer allocated every `encode()` call (line 160)
//!   Should use uniform arena pattern like matmul does
//! - **CRITICAL**: Shader only uses 1/256 threads - see shader file for details
//! - Implement Welford's online algorithm for fused mean+variance
//! - Add BF16 support for gamma/beta weights
//! - Consider switching to RMSNorm for decoder-only models (20-30% faster)
//!
//! # See Also
//!
//! - [`layer_norm.wgsl`] — WGSL shader implementation
//! - [`super::rms_norm`] — Faster alternative without mean centering

use crate::gpu_ops::{GpuTensor, Kernel};
use crate::WgpuContext;
use anyhow::Result;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Uniform parameters passed to the LayerNorm shader.
///
/// # Fields
///
/// * `m` — Number of rows to normalize (batch * seq_len)
/// * `n` — Hidden dimension size
/// * `eps` — Epsilon for numerical stability (typically 1e-5 or 1e-6)
/// * `_padding` — Alignment padding for WebGPU uniform requirements
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NormUniforms {
    m: u32,
    n: u32,
    eps: f32,
    _padding: u32,
}

/// Holds the learnable scale (gamma) and shift (beta) parameters for LayerNorm.
///
/// Both gamma and beta must be 1D tensors with length equal to the hidden dimension.
/// They are applied element-wise after normalization:
/// `output = normalized * gamma + beta`
pub struct GpuLayerNormWeights {
    pub(crate) gamma: GpuTensor, // scale
    pub(crate) beta: GpuTensor,  // bias
}

impl GpuLayerNormWeights {
    /// Creates new LayerNorm weights from gamma (scale) and beta (shift) tensors.
    ///
    /// # Arguments
    ///
    /// * `gamma` — Scale parameter [hidden_size] in F32 or BF16
    /// * `beta` — Shift parameter [hidden_size] in F32 or BF16
    ///
    /// # Returns
    ///
    /// LayerNorm weights ready for GPU execution.
    ///
    /// # Panics
    ///
    /// Panics if gamma and beta have different shapes or are not 1D.
    pub fn new(gamma: GpuTensor, beta: GpuTensor) -> Result<Self> {
        // Both must be 1D (one parameter per normalized feature)
        assert_eq!(gamma.rank(), 1, "LayerNorm gamma must be 1D");
        assert_eq!(beta.rank(), 1, "LayerNorm beta must be 1D");

        // gamma and beta must have the same shape
        assert_eq!(
            gamma.shape(),
            beta.shape(),
            "LayerNorm gamma and beta must have the same shape"
        );

        Ok(Self { gamma, beta })
    }
}

/// GPU kernel for Layer Normalization.
///
/// Normalizes activations across the hidden dimension with learnable affine transform.
///
/// # Algorithm
///
/// For each token (row):
/// 1. Compute mean: μ = sum(x) / N
/// 2. Compute variance: σ² = sum((x - μ)²) / N
/// 3. Normalize: x_norm = (x - μ) / sqrt(σ² + eps)
/// 4. Apply affine: y = x_norm * gamma + beta
///
/// # Performance Note
///
/// Current implementation is inefficient:
/// - Only uses 1/256 threads per row (see shader TODO)
/// - Allocates uniform buffer every encode call (memory leak)
/// - Four sequential passes instead of fused computation
///
/// For decoder-only models, consider using RMSNorm instead (simpler and faster).
pub struct GpuLayerNorm {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    context: Arc<WgpuContext>,
    eps: f32,
}

impl GpuLayerNorm {
    /// Creates a new LayerNorm GPU kernel.
    ///
    /// # Arguments
    ///
    /// * `context` — WebGPU device context
    /// * `eps` — Epsilon for numerical stability (typically 1e-5 for F32, 1e-6 for BF16)
    ///
    /// # Returns
    ///
    /// Initialized LayerNorm kernel ready for encoding operations.
    pub fn new(context: &Arc<WgpuContext>, eps: f32) -> Self {
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("./layer_norm.wgsl"));
        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("LayerNorm Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
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
        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("LayerNorm Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LayerNorm Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Self {
            pipeline,
            bind_group_layout,
            context: context.clone(),
            eps,
        }
    }

    /// Encodes the LayerNorm operation into the command buffer.
    ///
    /// # Arguments
    ///
    /// * `encoder` — WebGPU command encoder to record GPU commands
    /// * `weights` — Learnable scale (gamma) and shift (beta) parameters
    /// * `input` — Input tensor [batch, seq_len, hidden_size] in F32
    /// * `output` — Output tensor (same shape as input) in F32
    ///
    /// # Panics
    ///
    /// Panics if input tensor rank is less than 2.
    ///
    /// # TODO
    ///
    /// - **CRITICAL**: Fix memory leak on line 160 - uniform buffer allocated every call
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        weights: &GpuLayerNormWeights,
        input: &GpuTensor,
        output: &GpuTensor,
    ) {
        let rank = input.rank();
        assert!(
            rank >= 2,
            "Input tensor must have at least rank 2 for LayerNorm."
        );
        let (rows, cols) = (
            input.shape()[..rank - 1].iter().product::<usize>(),
            input.shape()[rank - 1],
        );

        let uniforms = NormUniforms {
            m: rows as u32,
            n: cols as u32,
            eps: self.eps,
            _padding: 0,
        };
        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("LayerNorm Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("LayerNorm Bind Group"),
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
                        resource: weights.gamma.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: weights.beta.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: output.buffer().as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LayerNorm Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (rows as u32 + 255) / 256;
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }
}

#[cfg(test)]
mod tests;
