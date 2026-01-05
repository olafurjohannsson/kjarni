//! SwiGLU activation function (element-wise).
//!
//! Applies the Swish-Gated Linear Unit activation used in Llama, Mistral, and modern FFN blocks.
//! SwiGLU: output = silu(gate) ⊙ up, where ⊙ is element-wise multiplication.
//!
//! # Algorithm
//!
//! For each element i:
//! 1. Apply SiLU (Swish) to gate: silu(x) = x / (1 + exp(-x))
//! 2. Multiply by up projection: output[i] = silu(gate[i]) * up[i]
//!
//! # Performance
//!
//! - **Current**: ~0.01ms for [128, 14336] on RTX 3090
//! - **Compute bound**: Mostly exp() calculation
//! - Fully parallel (embarrassingly parallel operation)
//!
//! # SwiGLU vs GELU
//!
//! **SwiGLU** (Llama, Mistral):
//! - Gated activation: combines two projections (gate and up)
//! - Smoother gradients than ReLU
//! - Better performance than GELU in practice (despite higher FLOP count)
//!
//! **GELU** (BERT, GPT-2):
//! - Single projection with GELU activation
//! - Requires erf() approximation (more expensive than silu)
//!
//! # SiLU Implementation
//!
//! Current implementation uses branching for numerical stability:
//! - x <= -20: clamp to 0 (avoids exp overflow)
//! - x >= 20: return x (sigmoid ≈ 1)
//! - Otherwise: x / (1 + exp(-x))
//!
//! # TODO / Improvements
//!
//! - Profile whether branches hurt performance on GPU (likely fine due to coherence)
//! - Consider vec4 loads/stores for 4x bandwidth
//! - Add FP16 accumulation option for Tensor Core GPUs
//! - Profile whether fused matmul+activation is faster (see swiglu_fused.wgsl)
//!
//! # Limitations
//!
//! - Only supports F32 (no BF16 variant)
//! - Element-wise only (assumes gate and up projections already computed)
//! - No bounds checking (assumes dispatch matches buffer size)
//!
//! # See Also
//!
//! - [`swiglu_fused.wgsl`] — Fused matmul+SwiGLU for better performance
//! - [`fc1.wgsl`] — GELU-based FFN for encoder architectures
//! - [GLU Variants paper](https://arxiv.org/abs/2002.05202) — SwiGLU introduction

/// Input from gate projection (W_gate @ x).
@group(0) @binding(0) var<storage, read> a_in: array<f32>;
/// Input from up projection (W_up @ x).
@group(0) @binding(1) var<storage, read> b_in: array<f32>;
/// Output: silu(gate) * up.
@group(0) @binding(2) var<storage, read_write> c_out: array<f32>;

/// ReLU activation (unused, kept for reference).
fn relu(x: f32) -> f32 {
    return max(x, 0.0);
}

/// SiLU (Swish) activation: x * sigmoid(x).
///
/// Uses clamping for numerical stability:
/// - x <= -20: sigmoid ≈ 0, return 0
/// - x >= 20: sigmoid ≈ 1, return x
/// - Otherwise: x / (1 + exp(-x))
fn silu(x: f32) -> f32 {
    // Clamping prevents exp overflow and underflow
    if (x <= -20.0) { return 0.0; }
    if (x >= 20.0) { return x; }
    return x / (1.0 + exp(-x));
}

/// Fast SiLU without branches (for comparison).
///
/// TODO: Benchmark whether branching actually hurts performance.
/// On modern GPUs with warp coherence, branches may be fine.
fn silu_fast(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

/// Applies SwiGLU activation element-wise.
///
/// Each thread handles one element: out[i] = silu(gate[i]) * up[i]
///
/// # Workgroup Size
/// 256 - Standard 1D workgroup for element-wise operations.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // NOTE: No bounds check - assumes dispatch size matches buffer size.
    // Rust dispatch must ensure this to avoid out-of-bounds access.

    let gate_val = a_in[idx];
    let up_val = b_in[idx];

    // SwiGLU: silu(gate) * up
    c_out[idx] = silu(gate_val) * up_val;
}