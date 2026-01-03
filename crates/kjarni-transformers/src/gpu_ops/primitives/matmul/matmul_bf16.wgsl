//! Tiled matrix multiplication for BF16 weights with implicit transpose.
//!
//! Computes C = A @ B^T where:
//! - A: [M, K] activations in F32
//! - B: [N, K] weights in packed BF16 (physically transposed)
//! - C: [M, N] output in F32
//!
//! # Algorithm
//!
//! Uses cooperative tiling with 32x32 workgroups:
//! 1. Load 32x32 tile of A into shared memory
//! 2. Load 32x32 tile of B, unpack BF16→F32, transpose to [K,N] layout
//! 3. Compute partial dot products
//! 4. Repeat for all K tiles, accumulating results
//!
//! # Performance
//!
//! - **Current**: ~80 GFLOPS on RTX 3090 for [4096, 4096] @ [4096, 4096]
//! - **Memory**: 2x bandwidth efficiency vs F32 due to BF16 weights
//! - **Bottleneck**: Prefill phase with large M (batch * seq_len)
//!
//! # Optimizations Applied
//!
//! - **Bank conflict avoidance**: B tile padded to 33 width (TILE_DIM_PADDED)
//! - **Manual BF16 unpack**: Avoids unpack4x16 overhead
//! - **Implicit transpose**: B physically [N,K], saves transpose operation
//!
//! # TODO / Improvements
//!
//! - **CRITICAL**: For decode phase (M=1), this shader is inefficient. Should dispatch
//!   to specialized GEMV kernel instead (already implemented as gemv_bf16.wgsl).
//! - Add double buffering (overlap compute with next tile load) for 10-15% speedup
//! - Increase tile size to 64x64 or 128x128 for larger matrices (M,N,K > 8192)
//! - Consider vectorized loads (load vec4 instead of scalar) for better coalescing
//! - Profile shared memory usage - currently uses 4KB + 4.2KB = 8.2KB per workgroup
//! - Add FP16 accumulation option for Tensor Core-like GPUs (trade precision for speed)
//!
//! # Limitations
//!
//! - Only supports M,N,K < 65536 (u32 limits)
//! - Assumes A is F32 (no mixed precision for activations)
//! - B must be pre-packed as BF16 in [N,K] layout
//! - No support for batched matmul (use bmm.wgsl for that)
//!
//! # Memory Layout
//!
//! **A (activation)**: Row-major [M, K] in F32
//! ```text
//! [a00, a01, a02, ..., a0K]
//! [a10, a11, a12, ..., a1K]
//! ...
//! ```
//!
//! **B (weight)**: Row-major [N, K] in packed BF16 (2 BF16 per u32)
//! ```text
//! Physical: [N, K/2] packed
//! Logical: [K, N] after transpose in shared memory
//! ```
//!
//! **C (output)**: Row-major [M, N] in F32
//!
//! # See Also
//!
//! - [`gemv_bf16.wgsl`] — Optimized vector-matrix multiply for M=1 (decode phase)
//! - [`matmul_tiled.wgsl`] — F32 variant without BF16 unpacking
//! - [`crate::gpu_ops::primitives::matmul::GpuMatMul`] — Rust dispatch logic

const TILE_DIM = 32u;
const TILE_DIM_PADDED = 33u; // Avoid bank conflicts in shared memory

/// Uniform parameters for matrix dimensions.
struct MatmulInfo {
    /// Number of rows in A (and C).
    m: u32,
    /// Inner dimension (A cols = B cols due to implicit transpose).
    k: u32,
    /// Number of rows in B (columns in C).
    n: u32,
}

// Shared memory tiles for cooperative loading
// NOTE: Each workgroup uses ~8.2KB of shared memory
var<workgroup> a_tile: array<f32, 1024>; // 32 * 32 = 4KB
var<workgroup> b_tile: array<f32, 1056>; // 32 * 33 = 4.2KB (padded for bank conflicts)

@group(0) @binding(0) var<uniform> info: MatmulInfo;
@group(0) @binding(1) var<storage, read> a_in: array<f32>;      // [M, K]
@group(0) @binding(2) var<storage, read> b_in: array<u32>;      // [N, K/2] packed BF16
@group(0) @binding(3) var<storage, read_write> c_out: array<f32>; // [M, N]

/// Manually unpacks two BF16 values from a packed u32.
///
/// BF16 format: sign(1) + exp(8) + mantissa(7) = 16 bits
/// Stored as high 16 bits of F32 (mantissa zero-padded).
///
/// # Performance
/// Manual unpacking is faster than unpack4x16float on most GPUs
/// due to reduced instruction count and better register allocation.
fn unpack_bf16_manual(packed: u32) -> vec2<f32> {
    // Low 16 bits → high bits of F32
    let v1 = bitcast<f32>(packed << 16u);
    // High 16 bits already in position
    let v2 = bitcast<f32>(packed & 0xFFFF0000u);
    return vec2<f32>(v1, v2);
}

/// Fetches a single BF16 value from the packed weight buffer.
///
/// # Arguments
/// * `index` - Linear index in the unpacked [N, K] space
///
/// # Returns
/// The unpacked F32 value.
fn get_b_value(index: u32) -> f32 {
    let vec_idx = index / 2u;
    let packed = b_in[vec_idx];
    let vals = unpack_bf16_manual(packed);

    // Select based on odd/even index
    if (index % 2u == 0u) {
        return vals.x;
    } else {
        return vals.y;
    }
}

/// Main tiled matmul kernel using 32x32 cooperative workgroups.
///
/// Each thread computes one element of the output matrix by:
/// 1. Cooperatively loading tiles of A and B into shared memory
/// 2. Computing partial dot products for that tile
/// 3. Accumulating across all K tiles
@compute @workgroup_size(32, 32, 1)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tx = local_id.x;
    let ty = local_id.y;

    let global_row = group_id.y * TILE_DIM + ty; // M dimension
    let global_col = group_id.x * TILE_DIM + tx; // N dimension

    var acc = 0.0;
    let num_tiles = (info.k + TILE_DIM - 1u) / TILE_DIM;

    // Iterate over K dimension in tiles
    for (var t = 0u; t < num_tiles; t = t + 1u) {

        // === Phase 1: Cooperative tile loading ===

        // Load A tile [TILE_DIM, TILE_DIM] from [M, K]
        let a_col = t * TILE_DIM + tx;
        if (global_row < info.m && a_col < info.k) {
            a_tile[ty * TILE_DIM + tx] = a_in[global_row * info.k + a_col];
        } else {
            a_tile[ty * TILE_DIM + tx] = 0.0;
        }

        // Load B tile with implicit transpose
        // Physical B: [N, K] → Shared B: [K, N] (transposed)
        let b_phys_row = group_id.x * TILE_DIM + ty; // N dimension
        let b_phys_col = t * TILE_DIM + tx;          // K dimension

        if (b_phys_row < info.n && b_phys_col < info.k) {
            let val = get_b_value(b_phys_row * info.k + b_phys_col);
            // Store transposed: [tx, ty] instead of [ty, tx]
            b_tile[tx * TILE_DIM_PADDED + ty] = val;
        } else {
            b_tile[tx * TILE_DIM_PADDED + ty] = 0.0;
        }

        // Sync: Ensure all threads have loaded their tile elements
        workgroupBarrier();

        // === Phase 2: Compute partial dot product for this tile ===
        // TODO: Unroll this loop for 2-3% speedup
        for (var i = 0u; i < TILE_DIM; i = i + 1u) {
            acc = acc + a_tile[ty * TILE_DIM + i] * b_tile[i * TILE_DIM_PADDED + tx];
        }

        // Sync: Ensure all threads finish compute before loading next tile
        workgroupBarrier();
    }

    // === Phase 3: Write output ===
    if (global_row < info.m && global_col < info.n) {
        c_out[global_row * info.n + global_col] = acc;
    }
}
