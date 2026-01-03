//! KV head repetition for Grouped-Query Attention (GQA) and Multi-Query Attention (MQA).
//!
//! Expands KV heads by repeating each KV head multiple times to match the number
//! of query heads. This enables efficient attention variants that reduce KV cache size.
//!
//! # Algorithm
//!
//! For each output position [batch, q_head, seq, dim]:
//! 1. Compute which KV head to read from: kv_head = q_head / num_repeats
//! 2. Copy from input[batch, kv_head, seq, dim] to output
//!
//! Example (num_q_heads=8, num_kv_heads=2):
//! - Q heads [0,1,2,3] use KV head 0
//! - Q heads [4,5,6,7] use KV head 1
//!
//! # Performance
//!
//! - **Current**: ~0.1ms for [1, 8, 128, 128] → [1, 32, 128, 128] on RTX 3090
//! - **Memory bound**: Pure copy operation (read + write)
//! - Bandwidth: ~2x tensor size (one read, one write)
//!
//! # Use Cases
//!
//! **Grouped-Query Attention (GQA)**:
//! - Llama 2 70B: 8 KV heads, 64 Q heads (8:1 ratio)
//! - Saves 8x KV cache memory vs full MHA
//! - Minimal quality loss compared to full attention
//!
//! **Multi-Query Attention (MQA)**:
//! - num_kv_heads = 1 (extreme case, used by PaLM, Falcon)
//! - Maximum KV cache savings
//! - Slight quality degradation vs MHA
//!
//! # TODO / Improvements
//!
//! - Add vec4 loads/stores for 4x bandwidth (process 4 floats per thread)
//! - Consider fusing with attention matmul to eliminate this memory pass entirely
//! - Profile cache locality - scattered reads from KV heads may cause misses
//! - For decode (seq_len=1), consider specialized fast path
//!
//! # Limitations
//!
//! - Only supports F32 (no BF16 variant)
//! - Requires num_q_heads to be evenly divisible by num_kv_heads
//! - Pure memory copy (no computation savings, just cache savings)
//!
//! # Memory Savings
//!
//! **Example: Llama 2 70B with 4K context**:
//! - Full MHA: 64 heads × 4K tokens × 128 dim × 4 bytes = 128 MB per layer
//! - GQA (8 KV heads): 8 heads × 4K tokens × 128 dim × 4 bytes = 16 MB per layer
//! - Savings: 112 MB per layer × 80 layers = 8.7 GB total
//!
//! # See Also
//!
//! - [GQA paper](https://arxiv.org/abs/2305.13245) — Grouped-Query Attention
//! - [`crate::gpu_ops::primitives::repeat_kv::GpuRepeatKV`] — Rust dispatch

/// Uniform parameters for KV head repetition.
struct Uniforms {
    /// Batch size.
    batch_size: u32,
    /// Number of query heads (output).
    num_q_heads: u32,
    /// Number of key/value heads (input, typically < num_q_heads).
    num_kv_heads: u32,
    /// Sequence length (number of tokens).
    seq_len: u32,
    /// Head dimension.
    head_dim: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input_kv: array<f32>;      // [batch, num_kv_heads, seq_len, head_dim]
@group(0) @binding(2) var<storage, read_write> output_kv: array<f32>; // [batch, num_q_heads, seq_len, head_dim]

/// Repeats KV heads to match the number of query heads.
///
/// Each thread handles one element in the OUTPUT tensor and computes
/// which KV head to read from based on the query head index.
///
/// # Workgroup Size
/// 256 - Standard 1D workgroup. Could benefit from vectorization (vec4).
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    let output_size = uniforms.batch_size * uniforms.num_q_heads * uniforms.seq_len * uniforms.head_dim;

    if (output_idx >= output_size) {
        return;
    }

    // --- Deconstruct the output index to find the source index ---

    // 1. Find the coordinates in the 4D output tensor
    // Output shape: [batch, num_q_heads, seq_len, head_dim]
    let d = output_idx % uniforms.head_dim;
    let s = (output_idx / uniforms.head_dim) % uniforms.seq_len;
    let q_head = (output_idx / (uniforms.head_dim * uniforms.seq_len)) % uniforms.num_q_heads;
    let b = output_idx / (uniforms.head_dim * uniforms.seq_len * uniforms.num_q_heads);

    // 2. Calculate the corresponding source KV head index
    // Each KV head is repeated num_repeats times
    // Example: If num_q_heads=8, num_kv_heads=2, then num_repeats=4
    //   Q heads 0,1,2,3 → KV head 0
    //   Q heads 4,5,6,7 → KV head 1
    let num_repeats = uniforms.num_q_heads / uniforms.num_kv_heads;
    let kv_head = q_head / num_repeats;

    // 3. Reconstruct the 1D index into the input tensor
    // Input shape: [batch, num_kv_heads, seq_len, head_dim]
    let input_idx = b * (uniforms.num_kv_heads * uniforms.seq_len * uniforms.head_dim) +
                    kv_head * (uniforms.seq_len * uniforms.head_dim) +
                    s * uniforms.head_dim +
                    d;

    // 4. Perform the copy (broadcasting KV head to multiple Q heads)
    output_kv[output_idx] = input_kv[input_idx];
}