//! Attention scores, q . k for every query and every key it may see, read from
//! the KV store in place.

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
    /// Row width of the padding mask.
    mask_stride: u32,
    /// Rows of the padding mask: 1 is shared by every batch row.
    mask_rows: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> q: array<f32>;       // [B, H, S, D]
@group(0) @binding(2) var<storage, read> k_store: array<f32>; // [B, KV, capacity, D]
@group(0) @binding(3) var<storage, read> mask: array<f32>;    // [mask_rows, mask_stride]
@group(0) @binding(4) var<storage, read_write> scores: array<f32>; // [B * H * S, keys]

/// What the softmax takes for a key that is not there.
const MASKED: f32 = -1e9;

/// One thread per score. Each query head reads the KV head it shares under
/// grouped-query attention.
@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let idx = (wg.y * grid.x + wg.x) * 32u + lid.x;
    let rows = info.batch * info.heads * info.seq;
    if (idx >= rows * info.keys) { return; }

    let row = idx / info.keys;
    let key = idx % info.keys;
    let query = row % info.seq;
    let batch_head = row / info.seq;
    let batch = batch_head / info.heads;
    let head = batch_head % info.heads;

    let mask_row = select(batch, 0u, info.mask_rows == 1u);
    if (key > info.position_offset + query || mask[mask_row * info.mask_stride + key] == 0.0) {
        scores[idx] = MASKED;
        return;
    }

    let d = info.head_dim;
    let kv = head / (info.heads / info.kv_heads);
    let q_base = row * d;
    let k_base = ((batch * info.kv_heads + kv) * info.key_stride + key) * d;

    var total = 0.0;
    for (var i = 0u; i < d; i++) {
        total += q[q_base + i] * k_store[k_base + i];
    }
    scores[idx] = total;
}
