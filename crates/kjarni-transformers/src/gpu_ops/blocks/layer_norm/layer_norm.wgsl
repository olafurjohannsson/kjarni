//! Layer Normalization with learnable scale (gamma) and shift (beta).
//!
//! LayerNorm: y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
//! Used by BERT, GPT-2, and most transformer models for stabilizing training.
//!
//! # Algorithm
//!
//! For each row (token):
//! 1. Compute mean: μ = sum(x) / N
//! 2. Compute variance: σ² = sum((x - μ)²) / N
//! 3. Normalize: x_norm = (x - μ) / sqrt(σ² + eps)
//! 4. Apply affine: y = x_norm * gamma + beta
//!
//! # Performance
//!
//! - **Current**: ~0.4ms for [128, 768] on RTX 3090
//! - **Memory bound**: 4 passes over data (very inefficient)
//! - Sequential mean/variance computation
//!
//! # TODO (CRITICAL)
//!
//! - **MAJOR**: Current implementation processes each row with a SINGLE thread while
//!   using workgroup_size(256). This means 255/256 threads are idle! Should use
//!   shared memory reduction with all 256 threads cooperating on each row.
//! - Implement Welford's online algorithm (fused mean + variance in single pass)
//! - Add subgroup operations for warp-level reductions (4-8x speedup)
//! - Support BF16 gamma/beta for memory efficiency
//! - Profile numerical stability - current implementation may be prone to overflow
//!
//! # Limitations
//!
//! - Only supports F32 (no BF16 or quantized variants)
//! - No batching across rows (processes one row per workgroup)
//! - Hidden size must fit in single thread's sequential loop (no tiling)
//! - Inefficient for small hidden sizes (<512) due to lack of parallelism
//!
//! # Comparison to RMSNorm
//!
//! RMSNorm is faster because:
//! - No mean computation (one less pass)
//! - No bias term (one less read)
//! - Simpler normalization formula
//!
//! Consider using RMSNorm for inference if model supports it.
//!
//! # See Also
//!
//! - [`rms_norm.wgsl`] — Simplified normalization without mean centering
//! - [`crate::gpu_ops::blocks::layer_norm::GpuLayerNorm`] — Rust dispatch

/// Uniform parameters for layer normalization.
struct NormUniforms {
    /// Number of rows to normalize (batch * seq_len).
    m: u32,
    /// Elements per row (hidden dimension).
    n: u32,
    /// Epsilon for numerical stability (typically 1e-5 or 1e-6).
    eps: f32,
    /// Padding for 16-byte alignment.
    _padding1: f32
};

@group(0) @binding(0) var<uniform> uniforms: NormUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>; // Scale (weight)
@group(0) @binding(3) var<storage, read> beta: array<f32>;  // Shift (bias)
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// NOTE: Workgroup size is 256 but we only use 1 thread per row!
// This is extremely wasteful. Should refactor to use all threads cooperatively.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_idx = global_id.x;
    if (row_idx >= uniforms.m) {
        return;
    }

    let hidden_size = uniforms.n;
    let row_offset = row_idx * hidden_size;

    // TODO: These 3 loops should be fused using Welford's online algorithm:
    // - Single pass to compute mean AND variance
    // - Parallel reduction using shared memory
    // - Then parallel normalize pass

    // 1. Calculate mean
    // PERF: Sequential loop - should parallelize across threads
    var mean = 0.0;
    for (var i = 0u; i < hidden_size; i = i + 1u) {
        mean += input[row_offset + i];
    }
    mean /= f32(hidden_size);

    // 2. Calculate variance
    // PERF: Sequential loop - should parallelize across threads
    var variance = 0.0;
    for (var i = 0u; i < hidden_size; i = i + 1u) {
        let val = input[row_offset + i] - mean;
        variance += val * val;
    }
    variance /= f32(hidden_size);

    let inv_std = 1.0 / sqrt(variance + uniforms.eps);

    // 3. Normalize and apply gamma/beta
    // PERF: Sequential - could parallelize final pass
    for (var i = 0u; i < hidden_size; i = i + 1u) {
        let normalized = (input[row_offset + i] - mean) * inv_std;
        output[row_offset + i] = normalized * gamma[i] + beta[i];
    }
}