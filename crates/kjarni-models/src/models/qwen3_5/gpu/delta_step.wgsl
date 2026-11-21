//! Gated DeltaNet's delta rule for one token, as in decoding: delta.wgsl's
//! update with the key dimension split across threads.
//!   S <- g S;  delta = beta (v - Sᵀk);  S <- S + k deltaᵀ;  o = Sᵀq
//!
//! A workgroup takes COLS value columns of one head. Each column is shared by
//! GROUPS threads, each holding every GROUPS-th row of it in registers; the two
//! sums over the key dimension, Sᵀk and Sᵀq, are added up across them in
//! workgroup memory. For one token that is eight times the threads of
//! delta.wgsl, and no column in private memory.

struct Info {
    batch: u32,
    seq: u32,
    k_heads: u32,
    v_heads: u32,
    key_dim: u32,
    value_dim: u32,
    tiled: u32,
    capture: u32,
    channels: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
}

const COLS: u32 = 32u;
const GROUPS: u32 = 8u;
const MAX_KEY_DIM: u32 = 128u;
/// Rows of a column per thread.
const PER: u32 = 16u; // MAX_KEY_DIM / GROUPS

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;            // [B, C]: q, k, v after the convolution
@group(0) @binding(2) var<storage, read> a: array<f32>;              // [B, Hv]
@group(0) @binding(3) var<storage, read> b: array<f32>;              // [B, Hv]
@group(0) @binding(4) var<storage, read> params: array<f32>;         // [2 * Hv]: -exp(A_log), then dt_bias
@group(0) @binding(5) var<storage, read_write> state: array<f32>;    // [B, Hv, Dk, Dv]
@group(0) @binding(6) var<storage, read_write> captured: array<f32>; // [B, Hv, Dk, Dv]
@group(0) @binding(7) var<storage, read_write> out: array<f32>;      // [B, Hv * Dv]

var<workgroup> q_s: array<f32, MAX_KEY_DIM>;
var<workgroup> k_s: array<f32, MAX_KEY_DIM>;
var<workgroup> partial: array<f32, 256>; // [GROUPS][COLS]
var<workgroup> deltas: array<f32, COLS>;

fn softplus(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    return log(1.0 + exp(x));
}

/// Workgroup (column block, value head, batch); thread (group, column).
@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let head = wg.y;
    let batch = wg.z;
    let col = lid.x % COLS;
    let group = lid.x / COLS;
    let j = wg.x * COLS + col;
    let dk = info.key_dim;
    let dv = info.value_dim;
    let owns = j < dv;
    let kh = select(head / (info.v_heads / info.k_heads), head % info.k_heads, info.tiled == 1u);
    let key_width = info.k_heads * dk;
    let row = batch * info.channels;

    if (lid.x < dk) {
        q_s[lid.x] = qkv[row + kh * dk + lid.x];
        k_s[lid.x] = qkv[row + key_width + kh * dk + lid.x];
    }
    workgroupBarrier();

    var q_sum = 0.0;
    var k_sum = 0.0;
    for (var i = 0u; i < dk; i++) {
        q_sum += q_s[i] * q_s[i];
        k_sum += k_s[i] * k_s[i];
    }
    let q_inv = info.scale / sqrt(q_sum + 1e-6);
    let k_inv = 1.0 / sqrt(k_sum + 1e-6);
    let beta = 1.0 / (1.0 + exp(-b[batch * info.v_heads + head]));
    let g = exp(params[head] * softplus(a[batch * info.v_heads + head] + params[info.v_heads + head]));

    let state_base = ((batch * info.v_heads + head) * dk) * dv + j;
    var s: array<f32, PER>;
    var kv = 0.0;
    for (var m = 0u; m < PER; m++) {
        let i = group + m * GROUPS;
        if (owns && i < dk) {
            let decayed = state[state_base + i * dv] * g;
            s[m] = decayed;
            kv += decayed * k_s[i];
        }
    }
    partial[lid.x] = kv;
    workgroupBarrier();
    if (group == 0u && owns) {
        var total = 0.0;
        for (var p = 0u; p < GROUPS; p++) {
            total += partial[p * COLS + col];
        }
        let v = qkv[row + 2u * key_width + head * dv + j];
        deltas[col] = (v - total * k_inv) * beta;
    }
    workgroupBarrier();

    let delta = deltas[col] * k_inv;
    var o = 0.0;
    for (var m = 0u; m < PER; m++) {
        let i = group + m * GROUPS;
        if (owns && i < dk) {
            let updated = s[m] + k_s[i] * delta;
            state[state_base + i * dv] = updated;
            if (info.capture == 1u) {
                captured[state_base + i * dv] = updated;
            }
            o += updated * q_s[i];
        }
    }
    // Every thread has read its Sᵀk part; the buffer takes the Sᵀq parts.
    workgroupBarrier();
    partial[lid.x] = o;
    workgroupBarrier();
    if (group == 0u && owns) {
        var total = 0.0;
        for (var p = 0u; p < GROUPS; p++) {
            total += partial[p * COLS + col];
        }
        out[batch * info.v_heads * dv + head * dv + j] = total * q_inv;
    }
}
