//! RMS Normalization with learnable scale (gamma) and BF16 support.
//!
//! RMSNorm: x_norm = x / sqrt(mean(x²) + eps) * gamma
//! Simpler than LayerNorm (no mean subtraction, no bias), used by Llama/Mistral.
//!
//! # Algorithm
//!
//! 1. Parallel sum of squares across row (stride 256)
//! 2. Tree reduction to compute total sum
//! 3. Compute RMS = sqrt(sum / N + eps)
//! 4. Normalize and scale: output = input / RMS * gamma
//!
//! # Performance
//!
//! - **Current**: ~0.3ms for [128, 4096] on RTX 3090
//! - **Memory bound**: 3 passes over data (read input, write output, read gamma)
//! - Uses parallel reduction (good!) unlike softmax
//!
//! # TODO / Improvements
//!
//! - Fuse passes: Load input + gamma in single pass, compute RMS on-the-fly
//! - Add welford online algorithm for better numerical stability
//! - Support in-place operation (input == output buffer)
//! - Use subgroup operations for faster reduction on modern GPUs
//! - Profile shared memory bank conflicts in reduction tree
//!
//! # Limitations
//!
//! - Assumes N <= 256 * 256 = 65536 (max elements per row)
//! - Workgroup size 256 is hardcoded (should be tunable)
//! - Three memory passes is suboptimal (can be fused to 2)
//!
//! # See Also
//!
//! - [`layer_norm.wgsl`] — Full LayerNorm with mean centering

/// Uniform parameters for RMS normalization.
struct NormUniforms {
    /// Number of rows to normalize.
    m: u32,
    /// Elements per row (hidden dimension).
    n: u32,
    /// Epsilon for numerical stability (typically 1e-6).
    eps: f32,
    /// 0 = gamma is F32, 1 = gamma is packed BF16.
    is_bf16: u32,
}

@group(0) @binding(0) var<uniform> uniforms: NormUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<u32>;  // Polymorphic: F32 or BF16
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Shared memory for parallel reduction
// TODO: Could reduce to 128 elements if we unroll final iterations
var<workgroup> s_sum: array<f32, 256>;

/// Fetches gamma value, handling F32 or BF16 format.
fn get_gamma(index: u32) -> f32 {
    if (uniforms.is_bf16 == 1u) {
        // BF16: 2 values packed per u32
        let packed = gamma[index / 2u];
        if (index % 2u == 0u) {
            return bitcast<f32>(packed << 16u);
        } else {
            return bitcast<f32>(packed & 0xFFFF0000u);
        }
    } else {
        // F32: direct read
        return bitcast<f32>(gamma[index]);
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = group_id.x;
    let tid = local_id.x;

    if (row >= uniforms.m) { return; }

    let row_offset = row * uniforms.n;

    // === Phase 1: Parallel sum of squares ===
    // Each thread processes elements strided by 256
    // TODO: Vectorize loads (vec4) for better bandwidth
    var sum_sq = 0.0;
    for (var i = tid; i < uniforms.n; i += 256u) {
        let val = input[row_offset + i];
        sum_sq += val * val;
    }

    // === Phase 2: Tree reduction in shared memory ===
    s_sum[tid] = sum_sq;
    workgroupBarrier();

    // Binary tree reduction: 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
    // TODO: Use subgroup operations (subgroupAdd) for first few levels
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        workgroupBarrier();
    }

    // === Phase 3: Compute inverse RMS (single thread) ===
    if (tid == 0u) {
        let mean = s_sum[0] / f32(uniforms.n);
        s_sum[0] = 1.0 / sqrt(mean + uniforms.eps);
    }
    workgroupBarrier();

    let inv_rms = s_sum[0];

    // === Phase 4: Normalize and scale ===
    // TODO: Could fuse with Phase 1 using online algorithms
    for (var i = tid; i < uniforms.n; i += 256u) {
        let idx = row_offset + i;
        output[idx] = input[idx] * inv_rms * get_gamma(i);
    }
}
