//! Gated DeltaNet's delta rule. Per head and token:
//!   S <- g S;  delta = beta (v - Sᵀk);  S <- S + k deltaᵀ;  o = Sᵀq
//! with q and k L2-normalised and q scaled by 1/sqrt(key_dim).
//!
//! Every column of S, one value dimension, evolves on its own given k, q, g
//! and beta, so a thread owns a column for the whole sequence and keeps it in
//! private memory. A workgroup takes COLS columns of one head and shares the
//! normalised q and k of each token.

struct Info {
    batch: u32,
    seq: u32,
    k_heads: u32,
    v_heads: u32,
    key_dim: u32,
    value_dim: u32,
    /// Value head h reads key head h % k_heads when 1 (llama.cpp's order), or
    /// h / (v_heads / k_heads) when 0 (Hugging Face's).
    tiled: u32,
    /// Also keep the state as it stood after this many tokens; 0 for none.
    capture: u32,
    channels: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
}

const COLS: u32 = 32u;
const MAX_KEY_DIM: u32 = 128u;

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;            // [B * S, C]: q, k, v after the convolution
@group(0) @binding(2) var<storage, read> a: array<f32>;              // [B * S, Hv]
@group(0) @binding(3) var<storage, read> b: array<f32>;              // [B * S, Hv]
@group(0) @binding(4) var<storage, read> params: array<f32>;         // [2 * Hv]: -exp(A_log), then dt_bias
@group(0) @binding(5) var<storage, read_write> state: array<f32>;    // [B, Hv, Dk, Dv]
@group(0) @binding(6) var<storage, read_write> captured: array<f32>; // [B, Hv, Dk, Dv]
@group(0) @binding(7) var<storage, read_write> out: array<f32>;      // [B * S, Hv * Dv]

var<workgroup> q_s: array<f32, MAX_KEY_DIM>;
var<workgroup> k_s: array<f32, MAX_KEY_DIM>;
var<workgroup> q_sq: array<f32, COLS>;
var<workgroup> k_sq: array<f32, COLS>;

fn softplus(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    return log(1.0 + exp(x));
}

/// Workgroup (column block, value head, batch).
@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let head = wg.y;
    let batch = wg.z;
    let j = wg.x * COLS + lid.x;
    let dk = info.key_dim;
    let dv = info.value_dim;
    let owns = j < dv;
    let kh = select(head / (info.v_heads / info.k_heads), head % info.k_heads, info.tiled == 1u);
    let key_width = info.k_heads * dk;
    let state_base = ((batch * info.v_heads + head) * dk) * dv + j;

    var s: array<f32, MAX_KEY_DIM>;
    if (owns) {
        for (var i = 0u; i < dk; i++) {
            s[i] = state[state_base + i * dv];
        }
    }

    let neg_a = params[head];
    let dt_bias = params[info.v_heads + head];

    for (var t = 0u; t < info.seq; t++) {
        let r = batch * info.seq + t;
        let row = r * info.channels;

        var q_part = 0.0;
        var k_part = 0.0;
        for (var i = lid.x; i < dk; i += COLS) {
            let qi = qkv[row + kh * dk + i];
            let ki = qkv[row + key_width + kh * dk + i];
            q_s[i] = qi;
            k_s[i] = ki;
            q_part += qi * qi;
            k_part += ki * ki;
        }
        q_sq[lid.x] = q_part;
        k_sq[lid.x] = k_part;
        workgroupBarrier();

        var q_sum = 0.0;
        var k_sum = 0.0;
        for (var p = 0u; p < COLS; p++) {
            q_sum += q_sq[p];
            k_sum += k_sq[p];
        }
        let q_inv = 1.0 / sqrt(q_sum + 1e-6);
        let k_inv = 1.0 / sqrt(k_sum + 1e-6);

        let beta = 1.0 / (1.0 + exp(-b[r * info.v_heads + head]));
        let g = exp(neg_a * softplus(a[r * info.v_heads + head] + dt_bias));

        if (owns) {
            var kv = 0.0;
            for (var i = 0u; i < dk; i++) {
                s[i] *= g;
                kv += s[i] * (k_s[i] * k_inv);
            }
            let v = qkv[row + 2u * key_width + head * dv + j];
            let delta = (v - kv) * beta;
            var o = 0.0;
            for (var i = 0u; i < dk; i++) {
                s[i] += (k_s[i] * k_inv) * delta;
                o += s[i] * (q_s[i] * q_inv * info.scale);
            }
            out[r * info.v_heads * dv + head * dv + j] = o;
            if (t + 1u == info.capture) {
                for (var i = 0u; i < dk; i++) {
                    captured[state_base + i * dv] = s[i];
                }
            }
        }
        // q_s and k_s are rewritten for the next token.
        workgroupBarrier();
    }

    if (owns) {
        for (var i = 0u; i < dk; i++) {
            state[state_base + i * dv] = s[i];
        }
    }
}
