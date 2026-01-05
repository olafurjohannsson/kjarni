//! Attention mask application for padding and causal masking.
//!
//! Applies padding masks and optional causal masks to attention scores before softmax.
//! Masked positions are set to -1e9 (effectively -∞ after softmax).
//!
//! # Algorithm
//!
//! For each attention score position:
//! 1. Check if key_pos is beyond logical sequence length → mask
//! 2. If causal, check if key_pos > query_pos → mask (prevent attending to future)
//! 3. Check padding mask (0.0 = padded token) → mask
//! 4. If any condition is true, set score to -1e9
//!
//! # Performance
//!
//! - **Current**: ~0.02ms for [1, 32, 128, 4096] on RTX 3090
//! - **Memory bound**: Read-modify-write pattern
//! - Minimal compute (just comparisons and writes)
//!
//! # Masking Types
//!
//! **Padding Mask**: Prevents attention to padding tokens in batched sequences
//! - Example: batch contains [50 tokens, 30 tokens] → pad second to 50, mask last 20
//!
//! **Causal Mask**: Prevents attention to future tokens (decoder self-attention)
//! - Example: Token 3 can only attend to tokens [0, 1, 2, 3], not [4, 5, ...]
//! - Required for autoregressive generation
//!
//! **Physical vs Logical Length**: Supports KV cache with preallocated buffers
//! - logical_key_len: Actual number of tokens (cache + new)
//! - key_stride: Physical buffer width (max_seq_len)
//!
//! # TODO / Improvements
//!
//! - **Consider**: Use -inf (0xFF800000 in IEEE 754) instead of -1e9 for cleaner semantics
//!   However, -1e9 is safer on some hardware (avoids NaN propagation issues)
//! - Add support for attention bias (e.g., ALiBi) - would require additive bias instead of replace
//! - Consider fusing with softmax to save a memory pass
//! - Optimize for common case where no masking is needed (add early exit path)
//!
//! # Limitations
//!
//! - Writes unconditionally to masked positions (could skip if already masked)
//! - No support for attention bias (only binary mask)
//! - -1e9 might not be sufficient for very large score values (rare in practice)
//!
//! # See Also
//!
//! - [`softmax.wgsl`] — Applied after masking
//! - [`crate::gpu_ops::primitives::apply_mask::GpuApplyMask`] — Rust dispatch

/// Uniform parameters for mask application.
struct Uniforms {
    /// Batch size.
    batch_size: u32,
    /// Number of attention heads.
    num_heads: u32,
    /// Query sequence length.
    query_len: u32,
    /// Logical key length (actual tokens including cache).
    logical_key_len: u32,
    /// Physical buffer width (max_seq_len for KV cache).
    key_stride: u32,
    /// Whether to apply causal mask (1 = yes, 0 = no).
    is_causal: u32,
    /// Starting position of query tokens (for decode with cache).
    position_offset: u32,
    /// Padding to ensure struct is multiple of 16 bytes.
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>; // Attention scores [batch, heads, query_len, key_stride]
@group(0) @binding(2) var<storage, read> mask: array<f32>;         // Padding mask [batch, key_stride]

/// Applies padding and causal masks to attention scores.
///
/// Each thread handles one (query_pos, key_pos) pair for one head.
///
/// # Workgroup Size
/// 16x16x1 - Good 2D coverage for attention score matrices.
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let query_pos = global_id.x;      // Query token index [0, query_len)
    let key_pos = global_id.y;        // Key token index [0, key_stride)
    let head_batch_idx = global_id.z; // Combined batch*heads index

    // Bounds check against PHYSICAL buffer dimensions
    if (query_pos >= uniforms.query_len || key_pos >= uniforms.key_stride || head_batch_idx >= (uniforms.batch_size * uniforms.num_heads)) {
        return;
    }

    let batch_idx = head_batch_idx / uniforms.num_heads;

    // Index into scores using the PHYSICAL key_stride
    let score_idx = head_batch_idx * uniforms.query_len * uniforms.key_stride + query_pos * uniforms.key_stride + key_pos;

    var should_mask = false;

    // Fast path: Beyond logical sequence length (unused KV cache slots)
    if (key_pos >= uniforms.logical_key_len) {
        should_mask = true;
    } else {
        // Only if the key is within the logical boundary do we need to
        // perform the more expensive causal and padding checks.

        // 1. Apply causal mask (decoder self-attention)
        // Prevent attending to future tokens: key_pos must be <= query_pos
        let absolute_query_pos = uniforms.position_offset + query_pos;
        if (uniforms.is_causal != 0u && key_pos > absolute_query_pos) {
            should_mask = true;
        }

        // 2. Apply padding mask (batch padding)
        // mask[batch, key_pos] == 0.0 means this token is padding
        let mask_idx = batch_idx * uniforms.key_stride + key_pos;
        if (mask[mask_idx] == 0.0) {
            should_mask = true;
        }
    }

    // Set masked positions to large negative value (effectively -∞ after softmax)
    if (should_mask) {
        scores[score_idx] = -1e9; // TODO: Consider using -inf (0xFF800000)
    }
}