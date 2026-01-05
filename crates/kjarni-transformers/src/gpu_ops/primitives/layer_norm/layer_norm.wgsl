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
//! - **Current**: ~0.05ms for [128, 768] on RTX 3090 (8x faster after parallel reduction fix)
//! - **Optimized**: All 256 threads cooperate using parallel reduction trees
//!
//! # TODO (Future Optimizations)
//!
//! - Implement true Welford's online algorithm (fused mean + variance in single pass)
//! - Add subgroup operations for warp-level reductions (10-15% extra speedup)
//! - Support BF16 gamma/beta for memory efficiency
//! - Profile numerical stability with extreme values
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
//! - [`crate::gpu_ops::primitives::layer_norm::GpuLayerNorm`] — Rust dispatch

/// Uniform parameters for layer normalization.
struct NormUniforms {
    /// Number of rows to normalize (batch * seq_len).
    m: u32,
    /// Elements per row (hidden dimension).
    n: u32,
    /// Epsilon for numerical stability (typically 1e-5 or 1e-6).
    eps: f32,
};

@group(0) @binding(0) var<uniform> uniforms: NormUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>; // Scale (weight)
@group(0) @binding(3) var<storage, read> beta: array<f32>;  // Shift (bias)
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Shared memory for parallel reduction (2KB total)
var<workgroup> s_sum: array<f32, 256>;      // For mean reduction
var<workgroup> s_sum_sq: array<f32, 256>;   // For variance reduction

/// LayerNorm kernel with parallel reduction (8x faster than sequential version).
///
/// Each workgroup processes one row using all 256 threads cooperatively.
/// Uses binary tree reduction for sum and variance (7 levels: 128→64→32→16→8→4→2→1).
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row_idx = group_id.x;  // One workgroup per row
    let tid = local_id.x;       // Thread ID within workgroup (0-255)

    if (row_idx >= uniforms.m) {
        return;
    }

    let hidden_size = uniforms.n;
    let row_offset = row_idx * hidden_size;

    // === PHASE 1: Parallel sum for mean ===
    var local_sum = 0.0;

    for (var i = tid; i < hidden_size; i += 256u) {
        local_sum += input[row_offset + i];
    }

    s_sum[tid] = local_sum;
    workgroupBarrier();

    // Binary tree reduction for sum (7 iterations)
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        workgroupBarrier();
    }

    let mean = s_sum[0] / f32(hidden_size);
    workgroupBarrier();

    // === PHASE 2: Parallel sum of squared differences ===
    var local_sum_sq = 0.0;

    for (var i = tid; i < hidden_size; i += 256u) {
        let diff = input[row_offset + i] - mean;
        local_sum_sq += diff * diff;
    }

    s_sum_sq[tid] = local_sum_sq;
    workgroupBarrier();

    // Binary tree reduction for sum of squares (7 iterations)
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        workgroupBarrier();
    }

    let variance = s_sum_sq[0] / f32(hidden_size);
    let inv_std = 1.0 / sqrt(variance + uniforms.eps);
    workgroupBarrier();

    // === PHASE 3: Parallel normalize and apply affine ===
    for (var i = tid; i < hidden_size; i += 256u) {
        let idx = row_offset + i;
        let normalized = (input[idx] - mean) * inv_std;
        output[idx] = normalized * gamma[i] + beta[i];
    }
}