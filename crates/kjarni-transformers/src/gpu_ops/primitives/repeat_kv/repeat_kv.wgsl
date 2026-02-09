//! KV head repetition for Grouped-Query Attention (GQA) and Multi-Query Attention (MQA).

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
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;
    let output_size = uniforms.batch_size * uniforms.num_q_heads * uniforms.seq_len * uniforms.head_dim;

    if (output_idx >= output_size) {
        return;
    }

    // Output shape: [batch, num_q_heads, seq_len, head_dim]
    let d = output_idx % uniforms.head_dim;
    let s = (output_idx / uniforms.head_dim) % uniforms.seq_len;
    let q_head = (output_idx / (uniforms.head_dim * uniforms.seq_len)) % uniforms.num_q_heads;
    let b = output_idx / (uniforms.head_dim * uniforms.seq_len * uniforms.num_q_heads);

    // Each KV head is repeated num_repeats times
    let num_repeats = uniforms.num_q_heads / uniforms.num_kv_heads;
    let kv_head = q_head / num_repeats;

    // Input shape: [batch, num_kv_heads, seq_len, head_dim]
    let input_idx = b * (uniforms.num_kv_heads * uniforms.seq_len * uniforms.head_dim) +
                    kv_head * (uniforms.seq_len * uniforms.head_dim) +
                    s * uniforms.head_dim +
                    d;

    // broadcastKV head to multiple Q heads
    output_kv[output_idx] = input_kv[input_idx];
}