//! Softmax activation with optional scaling and masking support.
//!
//! Implements row-wise softmax: softmax(x * scale) with numerical stability.
//! Supports padded inputs where `valid_cols < cols` for attention with variable
//! sequence lengths.
//!
//! # Algorithm
//!
//! For each row:
//! 1. Load inputs and find max(x * scale) locally (reduces VRAM writes)
//! 2. Compute exp(x * scale - max) and sum, storing exp results
//! 3. Compute inverse sum: inv_sum = 1.0 / sum
//! 4. Normalize: output = exp_val * inv_sum
//! 5. Zero padding region
//!
//! # Performance
//!
//! - **Optimized**: ~0.05ms on high-end GPUs.
//! - **Memory**: Minimizes global memory roundtrips by re-computing scale locally.
//! - **Math**: Uses multiplication by inverse sum instead of repeated division.
//! - **Parallelism**: All 256 threads cooperate using parallel reduction trees.
//!
//! # TODO (Future Optimizations)
//! - Add subgroup operations for warp-level reductions (10-15% extra speedup)
//! - Implement online softmax (single-pass algorithm, fused exp/sum)
//! - For long sequences (>2K), consider tiled processing to fit in cache
//!
//! # Limitations
//!
//! - Only supports F32 (no BF16 fast path)
//! - No batching across rows (processes one row per workgroup)
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

/// Softmax kernel with parallel reduction.
///
/// Each workgroup processes one row using all 256 threads cooperatively.
/// Uses binary tree reduction for max and sum.
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row_idx = group_id.x;  // One workgroup per row
    let tid = local_id.x;      // Thread ID within workgroup (0-255)

    if (row_idx >= uniforms.rows) { return; }

    let row_offset = row_idx * uniforms.cols;

    // === PHASE 1: Parallel scale + max reduction ===
    // Optimization: We do NOT write the scaled value back to memory here.
    // It is faster to re-compute (val * scale) in Phase 2 than to pay
    // the VRAM write/read cost.
    var local_max = -3.402823e+38; // f32 min

    for (var i = tid; i < uniforms.valid_cols; i += 256u) {
        let idx = row_offset + i;
        let val = data[idx];
        local_max = max(local_max, val * uniforms.scale);
    }

    // Store partial max in shared memory
    s_max[tid] = local_max;
    workgroupBarrier();

    // Binary tree reduction for max (7 iterations: 128..1)
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
        // Optimization: Read original data and re-scale.
        let val = data[idx];
        let exp_val = exp((val * uniforms.scale) - max_val);
        
        // We write the exp_val here because we need it for Phase 3.
        // While writing is costly, re-computing exp() in Phase 3 is worse.
        data[idx] = exp_val;
        local_sum += exp_val;
    }

    // Store partial sum in shared memory
    s_sum[tid] = local_sum;
    workgroupBarrier();

    // Binary tree reduction for sum
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        workgroupBarrier();
    }

    // Final sum is in s_sum[0]
    let sum_val = s_sum[0];
    workgroupBarrier();

    // === PHASE 3: Parallel normalize ===
    // Optimization: Compute inverse once. Multiplication is faster than division.
    let inv_sum = 1.0 / sum_val;

    for (var i = tid; i < uniforms.valid_cols; i += 256u) {
        let idx = row_offset + i;
        // Read the exp_val we stored in Phase 2
        data[idx] = data[idx] * inv_sum;
    }

    // === PHASE 4: Parallel zero padding ===
    for (var i = uniforms.valid_cols + tid; i < uniforms.cols; i += 256u) {
        data[row_offset + i] = 0.0;
    }
}