// lookup.wgsl - Supports both F32 and BF16 embedding tables

struct LookupUniforms {
    output_size: u32,
    output_batch_stride: u32,
    output_seq_stride: u32,
    input_seq_len: u32,
    vocab_size: u32,
    hidden_size: u32,  // Added: needed for BF16 indexing
    is_bf16: u32,      // Added: 0 = F32, 1 = BF16
    _padding: u32,
};

@group(0) @binding(0) var<uniform> uniforms: LookupUniforms;
@group(0) @binding(1) var<storage, read> embedding_table: array<u32>; // Can hold f32 or packed bf16
@group(0) @binding(2) var<storage, read> input_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Unpack a single BF16 value from a u32
// BF16 is the upper 16 bits of an F32, so we shift left by 16
fn unpack_bf16_low(packed: u32) -> f32 {
    // Low 16 bits -> shift to upper 16 bits, lower 16 become zeros
    let bits = packed << 16u;
    return bitcast<f32>(bits);
}

fn unpack_bf16_high(packed: u32) -> f32 {
    // High 16 bits are already in position, mask off low bits
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