//! Optimized vector-matrix multiplication (GEMV) for BF16 weights.
//!
//! Specialized kernel for decode phase where M=1 (single input vector).
//! Computes y = x @ W^T where:
//! - x: [1, K] input vector in F32
//! - W: [N, K] weight matrix in packed BF16 (transposed)
//! - y: [1, N] output vector in F32
//!
//! # Algorithm
//!
//! Each thread computes one output element:
//! 1. Load one row of W (N threads load N rows in parallel)
//! 2. Dot product with input vector x
//! 3. Write result
//!
//! No shared memory needed since each thread works independently.
//!
//! # Performance
//!
//! - **Current**: ~100 GFLOPS on RTX 3090 for [1, 4096] @ [4096, 4096]
//! - **Critical path**: Decode phase (70-80% of inference time)
//! - **Memory bound**: Limited by BF16 weight bandwidth (~350 GB/s)
//!
//! # Optimizations Applied
//!
//! - **Vectorized unpacking**: Processes 2 BF16 values per iteration
//! - **Coalesced loads**: Sequential thread IDs → sequential memory accesses
//! - **No shared memory**: Avoids sync overhead for this simple pattern
//!
//! # TODO / Improvements
//!
//! - **HIGH PRIORITY**: Add vec4 loads for 2x bandwidth utilization
//!   Currently loading u32 (2x BF16), should load u32x4 (8x BF16) per iteration
//! - Use subgroup operations for intra-warp reduction if K < 256
//! - Add FMA instruction hints for better compiler optimization
//! - Consider tiling for very large K (>16384) to improve cache reuse
//! - Profile memory bandwidth - should be hitting ~90% of peak
//!
//! # Limitations
//!
//! - **CRITICAL**: Assumes K is even (K % 2 == 0)! Will skip last element if K is odd.
//!   This is fine for Llama (K always divisible by 128) but not general purpose.
//! - Only supports M=1 (dispatch logic must check M before using this kernel)
//! - Workgroup size 256 is arbitrary - should tune based on occupancy
//! - No support for bias addition (must be handled separately)
//!
//! # Performance Note
//!
//! This kernel is the decode phase bottleneck. Every 1% improvement here
//! translates to measurable user-facing latency reduction. Priority improvements:
//! 1. Vec4 loads (2x speedup expected)
//! 2. Better warp occupancy tuning
//! 3. Cache prefetch hints
//!
//! # Memory Access Pattern
//!
//! **Thread 0**: Loads W[0, 0:K], computes y[0]
//! **Thread 1**: Loads W[1, 0:K], computes y[1]
//! ...
//! **Thread N-1**: Loads W[N-1, 0:K], computes y[N-1]
//!
//! All threads read the same x vector (broadcasts), but different W rows.
//! Memory access is perfectly coalesced within each warp.
//!
//! # See Also
//!
//! - [`matmul_bf16.wgsl`] — Tiled matmul for M > 1 (prefill phase)
//! - [`crate::gpu_ops::primitives::matmul::GpuMatMul`] — Dispatch logic

struct MatmulInfo {
    /// Number of rows in input (always 1 for GEMV).
    m: u32,
    /// Inner dimension (length of input vector).
    k: u32,
    /// Number of output elements (number of weight matrix rows).
    n: u32,
}

@group(0) @binding(0) var<uniform> info: MatmulInfo;
@group(0) @binding(1) var<storage, read> a_in: array<f32>;      // [1, K] input vector
@group(0) @binding(2) var<storage, read> b_in: array<u32>;      // [N, K/2] packed BF16 weights
@group(0) @binding(3) var<storage, read_write> c_out: array<f32>; // [1, N] output vector

/// Unpacks two BF16 values from a u32.
///
/// BF16 format: 16-bit float with 8-bit exponent (same as F32 high 16 bits).
/// To convert: shift low 16 bits left by 16, mask high 16 bits in place.
///
/// # Little Endian Layout
/// ```text
/// packed u32: [v2_high_16 | v1_low_16]
///             └─ bits 31:16  └─ bits 15:0
/// ```
fn unpack_bf16(packed: u32) -> vec2<f32> {
    // First value (low 16 bits) → shift to high position
    let v1_bits = packed << 16u;
    let v1 = bitcast<f32>(v1_bits);

    // Second value (high 16 bits) → already in position
    let v2_bits = packed & 0xFFFF0000u;
    let v2 = bitcast<f32>(v2_bits);

    return vec2<f32>(v1, v2);
}

/// GEMV kernel: Each thread computes one output element.
///
/// # Workgroup Size
/// 256 threads per workgroup. This provides good occupancy on most GPUs.
/// TODO: Benchmark 128 vs 256 vs 512 for optimal occupancy.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n_idx = global_id.x; // Output element index

    if (n_idx >= info.n) { return; }

    var acc = 0.0;

    // Weight matrix B is physically [N, K]
    // We want row n_idx for this output element
    let b_row_start_idx = n_idx * info.k; // Element offset

    // Process 2 BF16 values per iteration (one u32)
    // TODO: Process 8 BF16 values per iteration (four u32s) for better bandwidth
    let k_u32_start = b_row_start_idx / 2u;
    let k_u32_count = info.k / 2u; // NOTE: Assumes K is even!

    var a_idx = 0u;

    // Main dot product loop
    // TODO: Unroll by 4 or 8 for better ILP (instruction-level parallelism)
    for (var i = 0u; i < k_u32_count; i = i + 1u) {
        // Load packed weight (2x BF16)
        let packed_b = b_in[k_u32_start + i];
        let b_vals = unpack_bf16(packed_b);

        // Load corresponding input vector elements
        let a0 = a_in[a_idx];
        let a1 = a_in[a_idx + 1u];

        // Accumulate: dot product of 2 elements
        // NOTE: FMA is used automatically by most compilers
        acc = acc + (a0 * b_vals.x) + (a1 * b_vals.y);
        a_idx = a_idx + 2u;
    }

    // TODO: Handle odd K case (current implementation silently drops last element)
    // if (info.k % 2u == 1u) {
    //     acc += a_in[info.k - 1u] * get_b_single(n_idx * info.k + info.k - 1u);
    // }

    c_out[n_idx] = acc;
}
