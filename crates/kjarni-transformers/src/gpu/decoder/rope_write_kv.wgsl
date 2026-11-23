//! Rotates a step's queries and keys, and writes keys and values straight into
//! the KV store.

// The KV store's element type: f32, or f16 in the variant that
// `crate::gpu::kv_shader` builds for a half-precision cache.
alias KV = f32;

struct Info {
    batch: u32,
    seq: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    /// Absolute position of the first new token, which selects its rotation.
    position_offset: u32,
    /// Where the first new token goes in the store. Equal to `position_offset`
    /// with a KV cache, 0 without one.
    write_offset: u32,
    /// Keys per head the store has room for.
    key_stride: u32,
    /// Row width of the rotation tables.
    rope_stride: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> q_in: array<f32>;      // [B, S, H * D]
@group(0) @binding(2) var<storage, read> k_in: array<f32>;      // [B, S, KV * D]
@group(0) @binding(3) var<storage, read> v_in: array<f32>;      // [B, S, KV * D]
@group(0) @binding(4) var<storage, read> cos_table: array<f32>; // [positions, D / 2]
@group(0) @binding(5) var<storage, read> sin_table: array<f32>; // [positions, D / 2]
@group(0) @binding(6) var<storage, read_write> q_out: array<f32>;   // [B, H, S, D]
@group(0) @binding(7) var<storage, read_write> k_store: array<KV>; // [B, KV, capacity, D]
@group(0) @binding(8) var<storage, read_write> v_store: array<KV>; // [B, KV, capacity, D]

/// One thread per (batch, token, head). The first `kv_heads` heads also carry a
/// key and a value.
/// A thread per rotated pair: dimensions i and i + head_dim / 2 of one head of
/// one token, of the query and, for the first kv_heads heads, of the key and
/// value.
@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let d = info.head_dim;
    let half = d / 2u;
    let idx = (wg.y * grid.x + wg.x) * 32u + lid.x;
    if (idx >= info.batch * info.seq * info.heads * half) { return; }

    let i = idx % half;
    let item = idx / half;
    let head = item % info.heads;
    let token = (item / info.heads) % info.seq;
    let batch = item / (info.heads * info.seq);

    let table = (info.position_offset + token) * info.rope_stride + i;
    let c = cos_table[table];
    let s = sin_table[table];

    let q_src = (batch * info.seq + token) * info.heads * d + head * d + i;
    let q_dst = ((batch * info.heads + head) * info.seq + token) * d + i;
    let q0 = q_in[q_src];
    let q1 = q_in[q_src + half];
    q_out[q_dst] = q0 * c - q1 * s;
    q_out[q_dst + half] = q1 * c + q0 * s;

    if (head >= info.kv_heads) { return; }

    let kv_src = (batch * info.seq + token) * info.kv_heads * d + head * d + i;
    let kv_dst = ((batch * info.kv_heads + head) * info.key_stride + info.write_offset + token) * d + i;
    let k0 = k_in[kv_src];
    let k1 = k_in[kv_src + half];
    k_store[kv_dst] = KV(k0 * c - k1 * s);
    k_store[kv_dst + half] = KV(k1 * c + k0 * s);
    v_store[kv_dst] = KV(v_in[kv_src]);
    v_store[kv_dst + half] = KV(v_in[kv_src + half]);
}
