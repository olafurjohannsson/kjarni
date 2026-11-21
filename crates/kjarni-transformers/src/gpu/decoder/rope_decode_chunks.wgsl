//! Attention for one query per sequence, as in decoding, first half: the keys
//! are split into chunks of CHUNK, and each workgroup takes one chunk of one KV
//! head for every query head that shares it. It writes the chunk's largest
//! score, its sum of exponentials and its exponential-weighted sum of values;
//! rope_decode_combine.wgsl merges the chunks.
//!
//! Four threads share a key: each reads every fourth sixteen-byte piece of
//! its row and scores it against every query head of the group, held in
//! workgroup memory, and the four parts are added up. The values are then
//! summed four dimensions per thread.

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

const CHUNK: u32 = 32u;
const THREADS: u32 = 128u;
/// Threads sharing one key's dot products.
const PARTS: u32 = 4u; // THREADS / CHUNK
const MAX_GROUP: u32 = 8u;
/// Threads per query head when reducing over the chunk.
const LANES: u32 = 16u; // THREADS / MAX_GROUP
const MASKED: f32 = -1e30;

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> q: array<vec4<f32>>;       // [B, H, 1, D]
@group(0) @binding(2) var<storage, read> k_store: array<vec4<f32>>; // [B, KV, capacity, D]
@group(0) @binding(3) var<storage, read> v_store: array<vec4<f32>>; // [B, KV, capacity, D]
@group(0) @binding(4) var<storage, read> mask: array<f32>;          // [mask_rows, mask_stride]
@group(0) @binding(5) var<storage, read_write> stats: array<f32>;        // [B, H, chunks, 2]: max, sum
@group(0) @binding(6) var<storage, read_write> partial: array<vec4<f32>>; // [B, H, chunks, D]

/// The group's queries, at most MAX_GROUP heads of 256 dimensions.
var<workgroup> q_s: array<vec4<f32>, 512>;
var<workgroup> parts: array<f32, 1024>; // [PARTS][MAX_GROUP][CHUNK]
var<workgroup> scores: array<f32, 256>; // [MAX_GROUP][CHUNK]
var<workgroup> reduce: array<f32, THREADS>;
var<workgroup> maxes: array<f32, MAX_GROUP>;

/// Workgroup (chunk, KV head, batch); threads t, t + CHUNK, ... share key t.
@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let chunk = wg.x;
    let kv = wg.y;
    let batch = wg.z;
    let t = lid.x;
    let d4 = info.head_dim / 4u;
    let group = info.heads / info.kv_heads;
    let first_key = chunk * CHUNK;
    let kv_row0 = (batch * info.kv_heads + kv) * info.key_stride;
    let q_row0 = batch * info.heads + kv * group;

    for (var i = t; i < group * d4; i += THREADS) {
        q_s[i] = q[q_row0 * d4 + i];
    }
    workgroupBarrier();

    let slot = t % CHUNK;
    let part = t / CHUNK;
    let key = first_key + slot;
    let mask_row = select(batch, 0u, info.mask_rows == 1u);
    let visible = key < info.keys && key <= info.position_offset
        && mask[mask_row * info.mask_stride + min(key, info.mask_stride - 1u)] != 0.0;
    var dots: array<f32, MAX_GROUP>;
    if (visible) {
        let k_base = (kv_row0 + key) * d4;
        for (var i = part; i < d4; i += PARTS) {
            let k4 = k_store[k_base + i];
            for (var g = 0u; g < MAX_GROUP; g++) {
                if (g < group) {
                    dots[g] += dot(q_s[g * d4 + i], k4);
                }
            }
        }
    }
    for (var g = 0u; g < group; g++) {
        parts[(part * MAX_GROUP + g) * CHUNK + slot] = dots[g];
    }
    workgroupBarrier();
    for (var idx = t; idx < group * CHUNK; idx += THREADS) {
        let g = idx / CHUNK;
        let k = idx % CHUNK;
        var sum = 0.0;
        for (var p = 0u; p < PARTS; p++) {
            sum += parts[(p * MAX_GROUP + g) * CHUNK + k];
        }
        let kk = first_key + k;
        let seen = kk < info.keys && kk <= info.position_offset
            && mask[mask_row * info.mask_stride + min(kk, info.mask_stride - 1u)] != 0.0;
        scores[g * CHUNK + k] = select(MASKED, sum * info.scale, seen);
    }
    workgroupBarrier();

    // The largest score and the sum of exponentials of each query head:
    // LANES threads per head.
    let mine = t / LANES;
    let lane = t % LANES;
    var largest = MASKED;
    if (mine < group) {
        for (var k = lane; k < CHUNK; k += LANES) {
            largest = max(largest, scores[mine * CHUNK + k]);
        }
    }
    reduce[t] = largest;
    workgroupBarrier();
    if (lane == 0u && mine < group) {
        var m = MASKED;
        for (var i = 0u; i < LANES; i++) {
            m = max(m, reduce[t + i]);
        }
        maxes[mine] = m;
    }
    workgroupBarrier();
    var total = 0.0;
    if (mine < group) {
        let m = maxes[mine];
        for (var k = lane; k < CHUNK; k += LANES) {
            let s = scores[mine * CHUNK + k];
            let e = select(exp(s - m), 0.0, s <= MASKED);
            scores[mine * CHUNK + k] = e;
            total += e;
        }
    }
    reduce[t] = total;
    workgroupBarrier();
    if (lane == 0u && mine < group) {
        var sum = 0.0;
        for (var i = 0u; i < LANES; i++) {
            sum += reduce[t + i];
        }
        let at = ((q_row0 + mine) * info.chunks + chunk) * 2u;
        stats[at] = maxes[mine];
        stats[at + 1u] = sum;
    }

    // Weighted values, four dimensions of one query head per output.
    let keys_here = min(CHUNK, select(0u, info.keys - first_key, info.keys > first_key));
    for (var idx = t; idx < group * d4; idx += THREADS) {
        let g = idx / d4;
        let i = idx % d4;
        var acc = vec4<f32>(0.0);
        for (var k = 0u; k < keys_here; k++) {
            acc += scores[g * CHUNK + k] * v_store[(kv_row0 + first_key + k) * d4 + i];
        }
        partial[((q_row0 + g) * info.chunks + chunk) * d4 + i] = acc;
    }
}
