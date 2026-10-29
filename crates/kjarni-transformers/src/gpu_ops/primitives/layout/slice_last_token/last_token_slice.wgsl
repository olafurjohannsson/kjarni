@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

struct Uniforms {
    batch_size: u32,
    seq_len: u32,
    vocab_size: u32,
    _padding: u32,
}
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    // Launched over a 2D grid when one dimension would exceed 65535
    // workgroups; see gpu_ops::utils::grid_1d.
    let idx = global_id.y * grid.x * 256u + global_id.x;
    let total_elements = uniforms.batch_size * uniforms.vocab_size;
    
    if (idx >= total_elements) {
        return;
    }

    // Map linear index to [batch, vocab]
    let batch_idx = idx / uniforms.vocab_size;
    let vocab_idx = idx % uniforms.vocab_size;

    // Source index: [batch, LAST_TOKEN, vocab]
    let last_token_idx = uniforms.seq_len - 1u;
    let src_idx = batch_idx * uniforms.seq_len * uniforms.vocab_size +
                  last_token_idx * uniforms.vocab_size +
                  vocab_idx;

    // Destination index: [batch, vocab] (flattened)
    dst[idx] = src[src_idx];
}