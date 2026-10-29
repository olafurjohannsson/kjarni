@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

struct Uniforms {
    batch_size: u32,
    seq_len: u32,
    hidden_size: u32,
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
    let total = uniforms.batch_size * uniforms.hidden_size;

    if (idx >= total) {
        return;
    }

    // Map linear index -> [batch, hidden]
    let batch_idx = idx / uniforms.hidden_size;
    let hidden_idx = idx % uniforms.hidden_size;

    // CLS token = token index 0
    let src_idx =
        batch_idx * uniforms.seq_len * uniforms.hidden_size +
        hidden_idx;

    dst[idx] = src[src_idx];
}
