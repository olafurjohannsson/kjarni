//! Attention output: softmaxed scores times the values, read from the KV store in
//! place and written with the heads already merged.

// The KV store's element type: f32, or f16 in the variant that
// `crate::gpu::kv_shader` builds for a half-precision cache.
alias KV = f32;

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
    /// The first query this dispatch covers, and how many: long prompts are
    /// attended in pieces so the scores fit in one buffer binding.
    query_start: u32,
    queries: u32,
}

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> weights: array<f32>; // [B * H * queries, keys]
@group(0) @binding(2) var<storage, read> v_store: array<KV>; // [B, KV, capacity, D]
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
    let rows = info.batch * info.heads * info.queries;
    if (idx >= rows * d) { return; }

    let row = idx / d;
    let dim = idx % d;
    let query = info.query_start + row % info.queries;
    let batch_head = row / info.queries;
    let batch = batch_head / info.heads;
    let head = batch_head % info.heads;

    let kv = head / (info.heads / info.kv_heads);
    let v_base = (batch * info.kv_heads + kv) * info.key_stride * d + dim;
    let weight_base = row * info.keys;
    let visible = min(info.keys, info.position_offset + query + 1u);

    var total = 0.0;
    for (var key = 0u; key < visible; key++) {
        total += weights[weight_base + key] * f32(v_store[v_base + key * d]);
    }
    output[(batch * info.seq + query) * info.heads * d + head * d + dim] = total;
}
