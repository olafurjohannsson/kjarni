//! The output gate of a gated attention layer: each attended value times the
//! sigmoid of the gate the query projection computed beside it.

struct Info {
    rows: u32,
    heads: u32,
    head_dim: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> qg: array<f32>;            // [B * S, H * 2D]
@group(0) @binding(2) var<storage, read_write> attended: array<f32>; // [B * S, H * D]

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let idx = (wg.y * grid.x + wg.x) * 64u + lid.x;
    let width = info.heads * info.head_dim;
    if (idx >= info.rows * width) { return; }
    let row = idx / width;
    let head = (idx % width) / info.head_dim;
    let dim = idx % info.head_dim;
    let g = qg[row * 2u * width + head * 2u * info.head_dim + info.head_dim + dim];
    attended[idx] = attended[idx] * (1.0 / (1.0 + exp(-g)));
}
