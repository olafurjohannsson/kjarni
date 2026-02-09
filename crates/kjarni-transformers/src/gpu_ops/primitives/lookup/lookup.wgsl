//! Token embedding lookup (F32 only).

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
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;

    // Boundary check
    if (output_idx >= uniforms.output_size) {
        return;
    }

    // Output shape: [batch, seq_len, hidden_size]
    let batch_idx = output_idx / uniforms.output_batch_stride;
    let temp = output_idx % uniforms.output_batch_stride;
    let seq_idx = temp / uniforms.output_seq_stride;
    let hidden_idx = temp % uniforms.output_seq_stride;

    // Find the source token ID
    let input_id_idx = batch_idx * uniforms.input_seq_len + seq_idx;
    let token_id = input_ids[input_id_idx];

    // Validate and look up embedding
    if (token_id >= uniforms.vocab_size) {
        output[output_idx] = 0.0;
        return;
    }

    // Calculate index into embedding table [vocab_size, hidden_size]
    let source_idx = token_id * uniforms.output_seq_stride + hidden_idx;
    output[output_idx] = embedding_table[source_idx];
}