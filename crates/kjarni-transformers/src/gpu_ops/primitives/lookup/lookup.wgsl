struct LookupUniforms {
    output_size: u32,
    output_batch_stride: u32,
    output_seq_stride: u32,
    input_seq_len: u32,
    vocab_size: u32,
};

@group(0) @binding(0) var<uniform> uniforms: LookupUniforms;
@group(0) @binding(1) var<storage, read> embedding_table: array<f32>;
@group(0) @binding(2) var<storage, read> input_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;

    // Boundary check
    if (output_idx >= uniforms.output_size) {
        return;
    }

    // --- 1. Decode the 1D output index into 3D coordinates ---
    let batch_idx = output_idx / uniforms.output_batch_stride;
    let temp = output_idx % uniforms.output_batch_stride;
    let seq_idx = temp / uniforms.output_seq_stride;
    let hidden_idx = temp % uniforms.output_seq_stride;

    // --- 2. Find the source token ID ---
    let input_id_idx = batch_idx * uniforms.input_seq_len + seq_idx;
    let token_id = input_ids[input_id_idx];

    // --- 3. Find the source value in the embedding table ---
    // Optional: Add a bounds check for safety, though valid inputs shouldn't fail.
    if (token_id >= uniforms.vocab_size) {
        // Handle out-of-bounds, e.g., write 0. Or just let it panic on GPU if that's preferred.
        output[output_idx] = 0.0;
        return;
    }
    let source_idx = token_id * uniforms.output_seq_stride + hidden_idx;

    // --- 4. Perform the copy ---
    output[output_idx] = embedding_table[source_idx];
}