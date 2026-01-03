//! Second feedforward layer (FC2): projects back to hidden dimension.
//!
//! Implements the second fully connected layer in encoder FFN blocks (BERT, GPT-2, T5).
//! Computes output = input @ W^T + bias, projecting from intermediate_size back to hidden_size.
//!
//! # Algorithm
//!
//! For each output element [m, n]:
//! 1. Compute dot product: val = dot(input[m, :], W[n, :])
//! 2. Add bias: output[m, n] = val + bias[n]
//!
//! No activation function (linear projection).
//!
//! # Performance
//!
//! - **Current**: ~0.5ms for [128, 3072] @ [768, 3072] on RTX 3090
//! - **Memory bound**: Dominated by weight reads
//! - **Simpler than FC1**: No activation computation overhead
//!
//! # FFN Architecture
//!
//! Standard encoder FFN has two layers:
//! 1. **FC1**: hidden_size → intermediate_size (typically 4x) + GELU
//! 2. **FC2**: intermediate_size → hidden_size (this layer, no activation)
//!
//! Example (BERT-base):
//! - FC1: [batch, 768] @ [3072, 768] + GELU → [batch, 3072]
//! - FC2: [batch, 3072] @ [768, 3072] → [batch, 768]
//!
//! # Memory Layout
//!
//! **Weight is transposed** to [N, K] for coalesced reads:
//! - Each thread reads a contiguous row for one output dimension
//! - Improves cache locality and bandwidth utilization
//! - Same optimization as FC1
//!
//! # Vectorization
//!
//! Uses vec4 loads with manual FMA (fused multiply-add):
//! - Processes 4 elements per iteration
//! - FMA provides better numerical accuracy and speed
//! - Remainder handled separately for non-divisible-by-4 dimensions
//!
//! # TODO / Improvements
//!
//! - Add BF16 weight support (2x memory bandwidth)
//! - Consider fusing with residual connection (FFN output + input)
//! - Profile whether vec4 loads actually help vs scalar loads
//! - Add dropout support if needed for training (currently inference-only)
//!
//! # Limitations
//!
//! - Only supports F32 (no BF16 or quantized weights)
//! - Assumes K is divisible by 4 for optimal vectorization (remainder handled)
//! - Workgroup size 512 may be too large for some GPUs (consider 256)
//! - No activation function (assumes linear projection only)
//!
//! # See Also
//!
//! - [`fc1.wgsl`] — First FFN layer with activation
//! - [`swiglu_fused.wgsl`] — Fused FFN for decoder architectures
//! - [BERT paper](https://arxiv.org/abs/1810.04805) — Original transformer FFN

/// Uniform parameters for FC2 operation.
struct FfnUniforms {
    /// Number of input rows (batch * seq_len).
    m: u32,
    /// Input dimension (intermediate_size).
    k: u32,
    /// Output dimension (hidden_size).
    n: u32,
    /// Padding for 16-byte alignment.
    _padding: u32,
};

@group(0) @binding(0) var<uniform> info: FfnUniforms;
// Weight is transposed to [N, K] for coalesced memory access
@group(0) @binding(1) var<storage, read> fc2_weight: array<f32>;
@group(0) @binding(2) var<storage, read> fc2_bias: array<f32>;
@group(0) @binding(3) var<storage, read> input: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(512, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    let total_outputs = info.m * info.n;
    if (idx >= total_outputs) {
        return;
    }

    let row = idx / info.n;
    let col = idx % info.n;

    var sum: f32 = 0.0;

    let input_offset = row * info.k;
    // --- MODIFICATION ---
    // The weight for a given output column `col` is now a contiguous row in the transposed matrix.
    // The offset to this row is `col * k`.
    let weight_offset = col * info.k;

    let k_vec = info.k / 4u;
    for (var t = 0u; t < k_vec; t = t + 1u) {
        let base = t * 4u;

        let input_vec = vec4<f32>(
            input[input_offset + base + 0u],
            input[input_offset + base + 1u],
            input[input_offset + base + 2u],
            input[input_offset + base + 3u]
        );

        // --- MODIFICATION ---
        // Read a contiguous vec4 from the transposed weight matrix. This is much faster.
        let weight_vec = vec4<f32>(
            fc2_weight[weight_offset + base + 0u],
            fc2_weight[weight_offset + base + 1u],
            fc2_weight[weight_offset + base + 2u],
            fc2_weight[weight_offset + base + 3u]
        );

        sum = fma(input_vec.x, weight_vec.x, sum);
        sum = fma(input_vec.y, weight_vec.y, sum);
        sum = fma(input_vec.z, weight_vec.z, sum);
        sum = fma(input_vec.w, weight_vec.w, sum);
    }

    // Remainder loop
    let remainder_start = k_vec * 4u;
    for (var kk = remainder_start; kk < info.k; kk = kk + 1u) {
        sum = fma(input[input_offset + kk], fc2_weight[weight_offset + kk], sum);
    }

    sum = sum + fc2_bias[col];
    output[row * info.n + col] = sum;
}