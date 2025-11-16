struct Uniforms {
    // Input shape: [batch, num_kv_heads, seq_len, head_dim]
    // Output shape: [batch, num_q_heads, seq_len, head_dim]
    batch_size: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    seq_len: u32,
    head_dim: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input_kv: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_kv: array<f32>;

// We launch one thread per element in the OUTPUT tensor.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    let output_size = uniforms.batch_size * uniforms.num_q_heads * uniforms.seq_len * uniforms.head_dim;

    if (output_idx >= output_size) {
        return;
    }

    // --- Deconstruct the output index to find the source index ---

    // 1. Find the coordinates in the 4D output tensor
    let d = output_idx % uniforms.head_dim;
    let s = (output_idx / uniforms.head_dim) % uniforms.seq_len;
    let q_head = (output_idx / (uniforms.head_dim * uniforms.seq_len)) % uniforms.num_q_heads;
    let b = output_idx / (uniforms.head_dim * uniforms.seq_len * uniforms.num_q_heads);

    // 2. Calculate the corresponding source KV head index
    let num_repeats = uniforms.num_q_heads / uniforms.num_kv_heads;
    let kv_head = q_head / num_repeats;

    // 3. Reconstruct the 1D index into the input tensor
    let input_idx = b * (uniforms.num_kv_heads * uniforms.seq_len * uniforms.head_dim) +
                    kv_head * (uniforms.seq_len * uniforms.head_dim) +
                    s * uniforms.head_dim +
                    d;

    // 4. Perform the copy
    output_kv[output_idx] = input_kv[input_idx];
}