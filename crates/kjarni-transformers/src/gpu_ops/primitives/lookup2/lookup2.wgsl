//! Token embedding lookup with BF16 support.
//!
//! Converts token IDs to dense vector representations, supporting both F32 and BF16 embeddings.
//! BF16 reduces memory footprint by 2x with minimal quality loss.
//!
//! # Algorithm
//!
//! For each output position [batch, seq, hidden]:
//! 1. Read token ID: token_id = input_ids[batch, seq]
//! 2. Look up embedding from table (F32 or BF16)
//! 3. If BF16, unpack to F32 on-the-fly
//! 4. Write F32 output
//!
//! # Performance
//!
//! - **F32 mode**: ~0.05ms for [128, 128] token IDs on RTX 3090
//! - **BF16 mode**: ~0.04ms (slightly faster due to 2x bandwidth)
//! - **Memory savings**: BF16 reduces embedding table size by 2x
//!
//! # BF16 Format
//!
//! BF16 (Brain Floating Point 16):
//! - Uses high 16 bits of F32 (1 sign + 8 exp + 7 mantissa)
//! - Lower 16 bits are zeros
//! - Fast conversion: just shift bits
//! - Maintains F32 dynamic range (unlike FP16)
//!
//! **Packing**: Two BF16 values per u32
//! ```text
//! u32: [value1_high_16 | value0_low_16]
//! ```
//!
//! # Memory Savings Examples
//!
//! **Llama 2 7B**:
//! - F32: 32K vocab × 4K hidden × 4 bytes = 512 MB
//! - BF16: 32K vocab × 4K hidden × 2 bytes = 256 MB
//! - **Savings**: 256 MB per model
//!
//! **GPT-2**:
//! - F32: 50K vocab × 768 hidden × 4 bytes = 154 MB
//! - BF16: 50K vocab × 768 hidden × 2 bytes = 77 MB
//! - **Savings**: 77 MB
//!
//! # Quality Impact
//!
//! BF16 embeddings have negligible quality loss:
//! - Embeddings are noisy by nature (learned from data)
//! - 7-bit mantissa is sufficient for most NLP tasks
//! - Production models (Llama, GPT-4) use BF16 embeddings
//!
//! # TODO / Improvements
//!
//! - Add vec4 loads for 4x bandwidth when hidden_size divisible by 4
//! - Consider fusing position embedding addition
//! - Profile cache hit rate for common tokens in BF16 mode
//! - Add INT8 quantization support for even smaller embeddings
//!
//! # Limitations
//!
//! - BF16 mode assumes hidden_size is even (2 values per u32)
//! - Scattered memory access (cache-unfriendly for large vocab)
//! - No learned position embeddings (must add separately)
//! - Output is always F32 (no BF16 output option)
//!
//! # See Also
//!
//! - [`lookup.wgsl`] — F32-only variant (deprecated)
//! - [`crate::gpu_ops::primitives::lookup2::GpuLookup2`] — Rust dispatch

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
    /// Hidden dimension size (needed for BF16 indexing).
    hidden_size: u32,
    /// Data type: 0 = F32, 1 = BF16.
    is_bf16: u32,
    /// Padding for 16-byte alignment.
    _padding: u32,
};

@group(0) @binding(0) var<uniform> uniforms: LookupUniforms;
@group(0) @binding(1) var<storage, read> embedding_table: array<u32>; // Polymorphic: F32 or packed BF16
@group(0) @binding(2) var<storage, read> input_ids: array<u32>;      // [batch, seq_len]
@group(0) @binding(3) var<storage, read_write> output: array<f32>;    // [batch, seq_len, hidden_size]

/// Unpacks the low 16 bits of a u32 as BF16 to F32.
///
/// BF16 low 16 bits → shift left 16 → F32 upper 16 bits.
fn unpack_bf16_low(packed: u32) -> f32 {
    let bits = packed << 16u;
    return bitcast<f32>(bits);
}

/// Unpacks the high 16 bits of a u32 as BF16 to F32.
///
/// BF16 high 16 bits already in correct position → mask and convert.
fn unpack_bf16_high(packed: u32) -> f32 {
    let bits = packed & 0xFFFF0000u;
    return bitcast<f32>(bits);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;

    if (output_idx >= uniforms.output_size) {
        return;
    }

    // Decode 1D output index into 3D coordinates
    let batch_idx = output_idx / uniforms.output_batch_stride;
    let temp = output_idx % uniforms.output_batch_stride;
    let seq_idx = temp / uniforms.output_seq_stride;
    let hidden_idx = temp % uniforms.output_seq_stride;

    // Find the source token ID
    let input_id_idx = batch_idx * uniforms.input_seq_len + seq_idx;
    let token_id = input_ids[input_id_idx];

    if (token_id >= uniforms.vocab_size) {
        output[output_idx] = 0.0;
        return;
    }

    // Calculate source index and read value
    if (uniforms.is_bf16 == 1u) {
        // BF16 path: 2 values packed per u32
        let element_idx = token_id * uniforms.hidden_size + hidden_idx;
        let u32_idx = element_idx / 2u;
        let is_high = (element_idx % 2u) == 1u;
        
        let packed = embedding_table[u32_idx];
        
        if (is_high) {
            output[output_idx] = unpack_bf16_high(packed);
        } else {
            output[output_idx] = unpack_bf16_low(packed);
        }
    } else {
        // F32 path: direct read (reinterpret u32 as f32)
        let source_idx = token_id * uniforms.hidden_size + hidden_idx;
        output[output_idx] = bitcast<f32>(embedding_table[source_idx]);
    }
}