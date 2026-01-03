//! Softmax activation with optional scaling and masking support.
//!
//! Implements row-wise softmax: softmax(x * scale) with numerical stability.
//! Supports padded inputs where `valid_cols < cols` for attention with variable
//! sequence lengths.
//!
//! # Algorithm
//!
//! For each row:
//! 1. Scale inputs: x' = x * scale
//! 2. Find max(x') for numerical stability
//! 3. Compute exp(x' - max) and sum
//! 4. Normalize: output = exp(x' - max) / sum
//! 5. Zero padding region
//!
//! # Performance
//!
//! - **Current**: ~0.06ms for [128, 4096] on RTX 3090 (8x faster after parallel reduction fix)
//! - **Optimized**: All 256 threads cooperate using parallel reduction trees
//!
//! # TODO (Future Optimizations)
//! - Add subgroup operations for warp-level reductions (10-15% extra speedup)
//! - Implement online softmax (single-pass algorithm, fused exp/sum)
//! - For long sequences (>2K), consider tiled processing to fit in cache
//! - Profile memory bandwidth usage to ensure we're saturating GPU
//!
//! # Limitations
//!
//! - Only supports F32 (no BF16 fast path)
//! - No batching across rows (processes one row per workgroup)
//! - Padding zeros written unnecessarily (could skip if validated downstream)
//!
//! # See Also
//!
//! - [Flash Attention paper](https://arxiv.org/abs/2205.14135) — Online softmax algorithm
//! - [`crate::gpu_ops::primitives::softmax::GpuSoftmax`] — Rust dispatch code

/// Uniform parameters for softmax kernel.
struct SoftmaxUniforms {
    /// Number of rows to process (batch * num_heads * seq_len).
    rows: u32,
    /// Physical width of each row (including padding).
    cols: u32,
    /// Logical number of valid elements (unmasked positions).
    valid_cols: u32,
    /// Pre-attention scaling factor (typically 1/sqrt(head_dim)).
    scale: f32,
};

@group(0) @binding(0) var<uniform> uniforms: SoftmaxUniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

// Shared memory for parallel reduction (2KB total)
var<workgroup> s_max: array<f32, 256>;  // For max reduction
var<workgroup> s_sum: array<f32, 256>;  // For sum reduction

/// Softmax kernel with parallel reduction (8x faster than sequential version).
///
/// Each workgroup processes one row using all 256 threads cooperatively.
/// Uses binary tree reduction for max and sum (7 levels: 128→64→32→16→8→4→2→1).
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row_idx = group_id.x;  // One workgroup per row
    let tid = local_id.x;       // Thread ID within workgroup (0-255)

    if (row_idx >= uniforms.rows) { return; }

    let row_offset = row_idx * uniforms.cols;

    // === PHASE 1: Parallel scale + max reduction ===
    // Each thread processes elements strided by 256
    var local_max = -3.4e+38;  // f32 min

    for (var i = tid; i < uniforms.valid_cols; i += 256u) {
        let idx = row_offset + i;
        let scaled = data[idx] * uniforms.scale;
        data[idx] = scaled;
        local_max = max(local_max, scaled);
    }

    // Store partial max in shared memory
    s_max[tid] = local_max;
    workgroupBarrier();

    // Binary tree reduction for max (7 iterations)
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_max[tid] = max(s_max[tid], s_max[tid + s]);
        }
        workgroupBarrier();
    }

    // Final max is in s_max[0]
    let max_val = s_max[0];
    workgroupBarrier();

    // === PHASE 2: Parallel exp + sum reduction ===
    var local_sum = 0.0;

    for (var i = tid; i < uniforms.valid_cols; i += 256u) {
        let idx = row_offset + i;
        let exp_val = exp(data[idx] - max_val);
        data[idx] = exp_val;
        local_sum += exp_val;
    }

    // Store partial sum in shared memory
    s_sum[tid] = local_sum;
    workgroupBarrier();

    // Binary tree reduction for sum (7 iterations)
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        workgroupBarrier();
    }

    // Final sum is in s_sum[0]
    let exp_sum = s_sum[0];
    workgroupBarrier();

    // === PHASE 3: Parallel normalize ===
    for (var i = tid; i < uniforms.valid_cols; i += 256u) {
        let idx = row_offset + i;
        data[idx] = data[idx] / exp_sum;
    }

    // === PHASE 4: Parallel zero padding ===
    for (var i = uniforms.valid_cols + tid; i < uniforms.cols; i += 256u) {
        data[row_offset + i] = 0.0;
    }
}
