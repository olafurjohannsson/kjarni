//! Token embedding lookup (F32 only).
//!
//! Converts token IDs to dense vector representations by looking up pre-trained embeddings.
//! This is the first operation in all transformer models (BERT, GPT, Llama, etc.).
//!
//! # Algorithm
//!
//! For each output position [batch, seq, hidden]:
//! 1. Read token ID: token_id = input_ids[batch, seq]
//! 2. Look up embedding: output = embedding_table[token_id, :]
//! 3. Copy embedding vector to output
//!
//! # Performance
//!
//! - **Current**: ~0.05ms for [128, 128] token IDs on RTX 3090
//! - **Memory bound**: Scattered reads from embedding table
//! - **Cache friendly**: Common tokens (e.g., "the", "a") cached in L2
//!
//! # Embedding Tables
//!
//! **Size examples**:
//! - BERT-base: 30,522 vocab × 768 hidden = 23M params (~92 MB in F32)
//! - GPT-2: 50,257 vocab × 768 hidden = 38M params (~154 MB in F32)
//! - Llama 2: 32,000 vocab × 4096 hidden = 131M params (~524 MB in F32)
//!
//! **Memory footprint**: Embedding tables are often the largest single tensor
//! in small models. BF16 reduces by 2x (see lookup2.wgsl).
//!
//! # Input Validation
//!
//! Handles out-of-bounds token IDs gracefully:
//! - If token_id >= vocab_size, writes 0.0 vector
//! - Prevents GPU crashes from malformed input
//! - Production code should validate tokens on CPU first
//!
//! # TODO / Improvements
//!
//! - **DEPRECATED**: This shader should be replaced by lookup2.wgsl which supports BF16
//! - Add vec4 loads for 4x bandwidth when hidden_size is divisible by 4
//! - Consider adding position embeddings fusion (like BERT)
//! - Profile cache hit rate for common tokens
//!
//! # Limitations
//!
//! - Only supports F32 embeddings (no BF16 support, use lookup2.wgsl instead)
//! - Scattered memory access pattern (cache-unfriendly for large vocab)
//! - No support for learned position embeddings (must add separately)
//! - Assumes token IDs fit in u32 (max vocab 4B, sufficient for all models)
//!
//! # See Also
//!
//! - [`lookup2.wgsl`] — BF16-capable embedding lookup (preferred)
//! - [`crate::gpu_ops::primitives::lookup::GpuLookup`] — Rust dispatch

/// Uniform parameters for embedding lookup.
struct LookupUniforms {
    /// Total number of output elements (batch * seq * hidden).
    output_size: u32,
    /// Stride for batch dimension in output.
    output_batch_stride: u32,
    /// Stride for sequence dimension (== hidden_size).
    output_seq_stride: u32,
    /// Input sequence length.
    input_seq_len: u32,
    /// Vocabulary size (number of tokens).
    vocab_size: u32,
};

// TODO: Deprecated - use lookup2.wgsl which supports BF16
@group(0) @binding(0) var<uniform> uniforms: LookupUniforms;
@group(0) @binding(1) var<storage, read> embedding_table: array<f32>; // [vocab_size, hidden_size]
@group(0) @binding(2) var<storage, read> input_ids: array<u32>;      // [batch, seq_len]
@group(0) @binding(3) var<storage, read_write> output: array<f32>;    // [batch, seq_len, hidden_size]

/// Performs token embedding lookup for F32 embeddings.
///
/// Each thread handles one output element (one dimension of one token's embedding).
///
/// # Workgroup Size
/// 256 - Standard 1D workgroup for element-wise operations.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;

    // Boundary check
    if (output_idx >= uniforms.output_size) {
        return;
    }

    // --- 1. Decode the 1D output index into 3D coordinates ---
    // Output shape: [batch, seq_len, hidden_size]
    let batch_idx = output_idx / uniforms.output_batch_stride;
    let temp = output_idx % uniforms.output_batch_stride;
    let seq_idx = temp / uniforms.output_seq_stride;
    let hidden_idx = temp % uniforms.output_seq_stride;

    // --- 2. Find the source token ID ---
    // Input shape: [batch, seq_len]
    let input_id_idx = batch_idx * uniforms.input_seq_len + seq_idx;
    let token_id = input_ids[input_id_idx];

    // --- 3. Validate and look up embedding ---
    // Bounds check: handle out-of-vocabulary tokens gracefully
    if (token_id >= uniforms.vocab_size) {
        // Out-of-bounds token ID - write zero vector
        // NOTE: This should be rare in production (validate on CPU first)
        output[output_idx] = 0.0;
        return;
    }

    // Calculate index into embedding table [vocab_size, hidden_size]
    let source_idx = token_id * uniforms.output_seq_stride + hidden_idx;

    // --- 4. Perform the lookup (scattered read) ---
    output[output_idx] = embedding_table[source_idx];
}