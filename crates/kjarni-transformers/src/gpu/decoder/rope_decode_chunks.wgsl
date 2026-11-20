//! Attention for one query per sequence, as in decoding, first half: the keys
//! are split into chunks of CHUNK, and each workgroup takes one chunk of one KV
//! head for every query head that shares it. It writes the chunk's largest
//! score, its sum of exponentials and its exponential-weighted sum of values;
//! rope_decode_combine.wgsl merges the chunks.
//!
//! One thread per output walks every key in rope_context.wgsl, which at a few
//! thousand keys of context is a long serial loop on few threads. Here the
//! keys are spread over workgroups, and each key and value row is read once
//! for its group of query heads.

struct Info {
    batch: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    /// Absolute position of the query: the last key it may see.
    position_offset: u32,
    /// Keys per head the store has room for.
    key_stride: u32,
    /// Keys in the store.
    keys: u32,
    mask_stride: u32,
    /// Rows of the padding mask: 1 is shared by every batch row.
    mask_rows: u32,
    chunks: u32,
    scale: f32,
    _pad: u32,
}

const CHUNK: u32 = 128u;
const THREADS: u32 = 256u;
const MAX_GROUP: u32 = 8u;
const MAX_HEAD_DIM: u32 = 256u;
const MASKED: f32 = -1e30;

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [B, H, 1, D]
@group(0) @binding(2) var<storage, read> k_store: array<f32>;  // [B, KV, capacity, D]
@group(0) @binding(3) var<storage, read> v_store: array<f32>;  // [B, KV, capacity, D]
@group(0) @binding(4) var<storage, read> mask: array<f32>;     // [mask_rows, mask_stride]
@group(0) @binding(5) var<storage, read_write> stats: array<f32>;  // [B, H, chunks, 2]: max, sum
@group(0) @binding(6) var<storage, read_write> partial: array<f32>; // [B, H, chunks, D]

var<workgroup> scores: array<f32, 1024>;    // [MAX_GROUP][CHUNK]
var<workgroup> reduce: array<f32, THREADS>;
var<workgroup> maxes: array<f32, MAX_GROUP>;

/// Workgroup (chunk, KV head, batch).
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let chunk = wg.x;
    let kv = wg.y;
    let batch = wg.z;
    let t = lid.x;
    let d = info.head_dim;
    let group = info.heads / info.kv_heads;
    let first_key = chunk * CHUNK;
    let kv_base = (batch * info.kv_heads + kv) * info.key_stride * d;
    let mask_row = select(batch, 0u, info.mask_rows == 1u);

    // Scores, one per (query head of the group, key of the chunk).
    for (var idx = t; idx < group * CHUNK; idx += THREADS) {
        let g = idx / CHUNK;
        let key = first_key + idx % CHUNK;
        var s = MASKED;
        if (key < info.keys && key <= info.position_offset
                && mask[mask_row * info.mask_stride + key] != 0.0) {
            let q_base = (batch * info.heads + kv * group + g) * d;
            let k_base = kv_base + key * d;
            var dot = 0.0;
            for (var i = 0u; i < d; i += 4u) {
                dot += q[q_base + i] * k_store[k_base + i]
                    + q[q_base + i + 1u] * k_store[k_base + i + 1u]
                    + q[q_base + i + 2u] * k_store[k_base + i + 2u]
                    + q[q_base + i + 3u] * k_store[k_base + i + 3u];
            }
            s = dot * info.scale;
        }
        scores[g * CHUNK + idx % CHUNK] = s;
    }
    workgroupBarrier();

    // The largest score of each query head: 32 threads per head.
    let lane = t % 32u;
    let mine = t / 32u;
    var largest = MASKED;
    if (mine < group) {
        for (var key = lane; key < CHUNK; key += 32u) {
            largest = max(largest, scores[mine * CHUNK + key]);
        }
    }
    reduce[t] = largest;
    workgroupBarrier();
    if (lane == 0u && mine < group) {
        var m = MASKED;
        for (var i = 0u; i < 32u; i++) {
            m = max(m, reduce[t + i]);
        }
        maxes[mine] = m;
    }
    workgroupBarrier();

    // Exponentials in place, and their sum per head.
    var total = 0.0;
    if (mine < group) {
        let m = maxes[mine];
        for (var key = lane; key < CHUNK; key += 32u) {
            let s = scores[mine * CHUNK + key];
            let e = select(exp(s - m), 0.0, s <= MASKED);
            scores[mine * CHUNK + key] = e;
            total += e;
        }
    }
    reduce[t] = total;
    workgroupBarrier();
    if (lane == 0u && mine < group) {
        var sum = 0.0;
        for (var i = 0u; i < 32u; i++) {
            sum += reduce[t + i];
        }
        let at = ((batch * info.heads + kv * group + mine) * info.chunks + chunk) * 2u;
        stats[at] = maxes[mine];
        stats[at + 1u] = sum;
    }

    // Weighted values, one output per (query head, dimension).
    let keys_here = min(CHUNK, select(0u, info.keys - first_key, info.keys > first_key));
    for (var idx = t; idx < group * d; idx += THREADS) {
        let g = idx / d;
        let dim = idx % d;
        var acc = 0.0;
        for (var key = 0u; key < keys_here; key++) {
            acc += scores[g * CHUNK + key] * v_store[kv_base + (first_key + key) * d + dim];
        }
        partial[((batch * info.heads + kv * group + g) * info.chunks + chunk) * d + dim] = acc;
    }
}
