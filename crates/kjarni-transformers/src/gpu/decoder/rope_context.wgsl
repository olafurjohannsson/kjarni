//! Attention output: softmaxed scores times the values, read from the KV store in
//! place and written with the heads already merged.

struct Info {
    batch: u32,
    seq: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    /// Absolute position of the first query: the last key it may see.
    position_offset: u32,
    /// Keys per head the store has room for.
    key_stride: u32,
    /// Keys in the store, the width of the scores.
    keys: u32,
    mask_stride: u32,
    mask_rows: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> weights: array<f32>; // [B * H * S, keys]
@group(0) @binding(2) var<storage, read> v_store: array<f32>; // [B, KV, capacity, D]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [B, S, H * D]

/// One thread per output element. The keys after the query carry no weight, so
/// they are not visited.
@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let idx = (wg.y * grid.x + wg.x) * 32u + lid.x;
    let d = info.head_dim;
    let rows = info.batch * info.heads * info.seq;
    if (idx >= rows * d) { return; }

    let row = idx / d;
    let dim = idx % d;
    let query = row % info.seq;
    let batch_head = row / info.seq;
    let batch = batch_head / info.heads;
    let head = batch_head % info.heads;

    let kv = head / (info.heads / info.kv_heads);
    let v_base = (batch * info.kv_heads + kv) * info.key_stride * d + dim;
    let weight_base = row * info.keys;
    let visible = min(info.keys, info.position_offset + query + 1u);

    var total = 0.0;
    for (var key = 0u; key < visible; key++) {
        total += weights[weight_base + key] * v_store[v_base + key * d];
    }
    output[(batch * info.seq + query) * info.heads * d + head * d + dim] = total;
}
