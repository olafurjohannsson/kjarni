//! Attention for one query per sequence, second half: merges the chunks of
//! rope_decode_chunks.wgsl into the softmax-weighted sum of values over every
//! key, written with the heads merged.

struct Info {
    batch: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    position_offset: u32,
    key_stride: u32,
    keys: u32,
    mask_stride: u32,
    mask_rows: u32,
    chunks: u32,
    scale: f32,
    _pad: u32,
}

const MASKED: f32 = -1e30;

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> stats: array<f32>;    // [B, H, chunks, 2]
@group(0) @binding(2) var<storage, read> partial: array<f32>;  // [B, H, chunks, D]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [B, 1, H * D]

/// Workgroup (head, batch); one thread per dimension.
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let head = wg.x;
    let batch = wg.y;
    let dim = lid.x;
    let d = info.head_dim;
    if (dim >= d) { return; }
    let row = batch * info.heads + head;

    var largest = MASKED;
    for (var c = 0u; c < info.chunks; c++) {
        largest = max(largest, stats[(row * info.chunks + c) * 2u]);
    }
    var total = 0.0;
    var acc = 0.0;
    for (var c = 0u; c < info.chunks; c++) {
        let at = (row * info.chunks + c) * 2u;
        let weight = select(exp(stats[at] - largest), 0.0, stats[at] <= MASKED);
        total += stats[at + 1u] * weight;
        acc += partial[(row * info.chunks + c) * d + dim] * weight;
    }
    output[row * d + dim] = select(acc / total, 0.0, total == 0.0);
}
