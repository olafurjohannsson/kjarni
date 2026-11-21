//! Queries of a gated attention layer: per-head RMSNorm, then rotary on the
//! leading `rotary_half * 2` dimensions, written head-major for the scores.

struct Info {
    rows: u32,
    seq: u32,
    heads: u32,
    head_dim: u32,
    rotary_half: u32,
    position_offset: u32,
    rope_stride: u32,
    eps: f32,
}

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> qg: array<f32>;        // [B * S, H * 2D], query then gate per head
@group(0) @binding(2) var<storage, read> norm: array<f32>;      // [D]
@group(0) @binding(3) var<storage, read> cos_table: array<f32>; // [positions, rope_stride]
@group(0) @binding(4) var<storage, read> sin_table: array<f32>;
@group(0) @binding(5) var<storage, read_write> q_out: array<f32>; // [B, H, S, D]

const THREADS: u32 = 64u;
var<workgroup> partial: array<f32, THREADS>;

/// A workgroup per (row, head); its threads split the head's dimensions.
@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let idx = wg.y * grid.x + wg.x;
    if (idx >= info.rows * info.heads) { return; }
    let t = lid.x;
    let row = idx / info.heads;
    let head = idx % info.heads;
    let token = row % info.seq;
    let batch = row / info.seq;
    let d = info.head_dim;

    let src = row * info.heads * 2u * d + head * 2u * d;
    var sum = 0.0;
    for (var i = t; i < d; i += THREADS) {
        let x = qg[src + i];
        sum += x * x;
    }
    partial[t] = sum;
    workgroupBarrier();
    for (var s = THREADS / 2u; s > 0u; s >>= 1u) {
        if (t < s) {
            partial[t] += partial[t + s];
        }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(partial[0] / f32(d) + info.eps);

    let dst = ((batch * info.heads + head) * info.seq + token) * d;
    let table = (info.position_offset + token) * info.rope_stride;
    let half = info.rotary_half;
    for (var i = t; i < half; i += THREADS) {
        let x0 = qg[src + i] * inv * norm[i];
        let x1 = qg[src + i + half] * inv * norm[i + half];
        let c = cos_table[table + i];
        let s = sin_table[table + i];
        q_out[dst + i] = x0 * c - x1 * s;
        q_out[dst + i + half] = x0 * s + x1 * c;
    }
    for (var i = 2u * half + t; i < d; i += THREADS) {
        q_out[dst + i] = qg[src + i] * inv * norm[i];
    }
}
