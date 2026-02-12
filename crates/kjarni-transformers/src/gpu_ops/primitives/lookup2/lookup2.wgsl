//! Token embedding lookup with BF16 support

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

/// Unpacks the low 16 bits of a u32 as BF16 to F32
fn unpack_bf16_low(packed: u32) -> f32 {
    let bits = packed << 16u;
    return bitcast<f32>(bits);
}

/// Unpacks the high 16 bits of a u32 as BF16 to F32
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
        // F32 
        let source_idx = token_id * uniforms.hidden_size + hidden_idx;
        output[output_idx] = bitcast<f32>(embedding_table[source_idx]);
    }
}