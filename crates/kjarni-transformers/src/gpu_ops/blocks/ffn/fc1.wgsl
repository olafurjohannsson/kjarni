//! First feedforward layer (FC1) with configurable activation function.
//!
//! Implements the first fully connected layer in encoder FFN blocks (BERT, GPT-2, T5).
//! Computes output = activation(input @ W^T + bias), expanding from hidden_size to intermediate_size.
//!
//! # Algorithm
//!
//! For each output element [m, n]:
//! 1. Compute linear: val = dot(input[m, :], W[n, :]) + bias[n]
//! 2. Apply activation: output[m, n] = activation(val)
//!
//! # Performance
//!
//! - **Current**: ~0.8ms for [128, 768] @ [3072, 768] + GELU on RTX 3090
//! - **Memory bound**: Dominated by weight reads
//! - **Activation overhead**: GELU ~2x slower than ReLU due to erf() computation
//!
//! # Activation Functions
//!
//! Supports three activation types via pipeline constant `COMPUTE_ACT_TYPE`:
//!
//! **GELU (0)**: Gaussian Error Linear Unit (default)
//! - Formula: 0.5 * x * (1 + erf(x / sqrt(2)))
//! - Used by BERT, GPT-2, GPT-3
//! - Smoother than ReLU, better gradients
//! - Expensive: requires erf approximation
//!
//! **GELU_NEW (1)**: Tanh approximation of GELU
//! - Formula: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
//! - Faster than standard GELU (avoids erf)
//! - Used by some implementations for speed
//!
//! **ReLU (2)**: Rectified Linear Unit
//! - Formula: max(0, x)
//! - Fastest activation
//! - Rarely used in modern transformers (dead neuron problem)
//!
//! # Memory Layout
//!
//! **Weight is transposed** to [N, K] for coalesced reads:
//! - Each thread reads a contiguous row for one output dimension
//! - Improves cache locality and bandwidth utilization
//!
//! # TODO / Improvements
//!
//! - Add BF16 weight support (2x memory bandwidth)
//! - Optimize erf approximation (currently uses Abramowitz & Stegun)
//! - Add SiLU activation option (used by some modern encoders)
//! - Consider fusing with layer norm (like fused SwiGLU)
//! - Profile whether vec4 loads actually help (currently used)
//!
//! # Limitations
//!
//! - Only supports F32 (no BF16 or quantized weights)
//! - Assumes K is divisible by 4 for vectorization (remainder handled separately)
//! - GELU erf approximation has ~1.5e-7 error (acceptable for inference)
//! - Workgroup size 512 may be too large for some GPUs (consider 256)
//!
//! # See Also
//!
//! - [`fc2.wgsl`] — Second FFN layer (projection back to hidden_size)
//! - [`swiglu.wgsl`] — SwiGLU activation for decoder architectures
//! - [GELU paper](https://arxiv.org/abs/1606.08415) — Original GELU publication

/// Uniform parameters for FC1 operation.
struct FfnUniforms {
    /// Number of input rows (batch * seq_len).
    m: u32,
    /// Input dimension (hidden_size).
    k: u32,
    /// Output dimension (intermediate_size, typically 4x hidden_size).
    n: u32,
    /// Padding for 16-byte alignment.
    _padding: u32,
};

/// Pipeline constant for activation function selection.
///
/// Set at pipeline creation time (compile-time specialization):
/// - 0: GELU (default, most accurate)
/// - 1: GELU_NEW (faster tanh approximation)
/// - 2: ReLU (fastest, rarely used)
@id(0) override COMPUTE_ACT_TYPE: u32 = 0;

@group(0) @binding(0) var<uniform> info: FfnUniforms;
// Weight is transposed to [N, K] for coalesced memory access
@group(0) @binding(1) var<storage, read> fc1_weight: array<f32>;
@group(0) @binding(2) var<storage, read> fc1_bias: array<f32>;
@group(0) @binding(3) var<storage, read> input: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// GELU (tanh approximation)
fn gelu_new(x: f32) -> f32 {
    let SQRT_2_OVER_PI: f32 = 0.7978845608;
    let GELU_COEFF: f32 = 0.044715;
    let x_cubed = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    return 0.5 * x * (1.0 + tanh(inner));
}

// Abramowitz and Stegun approximation (accurate to ~1.5e-7)
fn erf_approx(x: f32) -> f32 {
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;
    
    let sign = select(-1.0, 1.0, x >= 0.0);
    let abs_x = abs(x);
    let t = 1.0 / (1.0 + p * abs_x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * exp(-abs_x * abs_x);
    return sign * y;
}

fn gelu(x: f32) -> f32 {
    let SQRT_2_INV: f32 = 0.7071067811865476;  // 1/sqrt(2)
    return 0.5 * x * (1.0 + erf_approx(x * SQRT_2_INV));
}

@compute @workgroup_size(512, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_outputs = info.m * info.n;
    if (idx >= total_outputs) { return; }

    let row = idx / info.n;
    let col = idx % info.n;

    var sum: f32 = 0.0;

    let input_offset = row * info.k;
    // The weight for a given output column `col` is now a contiguous row.
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
        // Read a contiguous vec4 from the transposed weight matrix.
        let weight_vec = vec4<f32>(
            fc1_weight[weight_offset + base + 0u],
            fc1_weight[weight_offset + base + 1u],
            fc1_weight[weight_offset + base + 2u],
            fc1_weight[weight_offset + base + 3u]
        );
        sum = sum + dot(input_vec, weight_vec);
    }

    let remainder_start = k_vec * 4u;
    for (var kk = remainder_start; kk < info.k; kk = kk + 1u) {
        sum = sum + input[input_offset + kk] * fc1_weight[weight_offset + kk];
    }

    sum = sum + fc1_bias[col];
    //output[idx] = gelu(sum);
    // The driver will DELETE the branches that don't match the constant.
    if (COMPUTE_ACT_TYPE == 0u) {
        output[idx] = gelu(sum);
    } else if (COMPUTE_ACT_TYPE == 1u) {
        output[idx] = gelu_new(sum);
    } else if (COMPUTE_ACT_TYPE == 2u) { // Example
        output[idx] = max(0.0, sum);
    }
}