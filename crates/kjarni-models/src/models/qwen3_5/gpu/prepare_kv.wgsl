//! Keys and values of a gated attention layer: per-head RMSNorm and rotary on
//! the keys, then both written into the KV store.

struct Info {
    rows: u32,
    seq: u32,
    kv_heads: u32,
    head_dim: u32,
    rotary_half: u32,
    position_offset: u32,
    rope_stride: u32,
    eps: f32,
    /// Where the first new token goes in the store.
    write_offset: u32,
    /// Keys per head the store has room for.
    key_stride: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> k_in: array<f32>;      // [B * S, KV * D]
@group(0) @binding(2) var<storage, read> v_in: array<f32>;      // [B * S, KV * D]
@group(0) @binding(3) var<storage, read> norm: array<f32>;      // [D]
@group(0) @binding(4) var<storage, read> cos_table: array<f32>;
@group(0) @binding(5) var<storage, read> sin_table: array<f32>;
@group(0) @binding(6) var<storage, read_write> k_store: array<f32>; // [B, KV, capacity, D]
@group(0) @binding(7) var<storage, read_write> v_store: array<f32>;

/// One thread per (row, kv head).
@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let idx = (wg.y * grid.x + wg.x) * 32u + lid.x;
    if (idx >= info.rows * info.kv_heads) { return; }
    let row = idx / info.kv_heads;
    let head = idx % info.kv_heads;
    let token = row % info.seq;
    let batch = row / info.seq;
    let d = info.head_dim;

    let src = row * info.kv_heads * d + head * d;
    var sum = 0.0;
    for (var i = 0u; i < d; i++) {
        let x = k_in[src + i];
        sum += x * x;
    }
    let inv = 1.0 / sqrt(sum / f32(d) + info.eps);

    let dst = ((batch * info.kv_heads + head) * info.key_stride + info.write_offset + token) * d;
    let table = (info.position_offset + token) * info.rope_stride;
    let half = info.rotary_half;
    for (var i = 0u; i < half; i++) {
        let x0 = k_in[src + i] * inv * norm[i];
        let x1 = k_in[src + i + half] * inv * norm[i + half];
        let c = cos_table[table + i];
        let s = sin_table[table + i];
        k_store[dst + i] = x0 * c - x1 * s;
        k_store[dst + i + half] = x0 * s + x1 * c;
    }
    for (var i = 2u * half; i < d; i++) {
        k_store[dst + i] = k_in[src + i] * inv * norm[i];
    }
    for (var i = 0u; i < d; i++) {
        v_store[dst + i] = v_in[src + i];
    }
}
