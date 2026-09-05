//! Fused SwiGLU: combines matmul and activation into a single kernel.

/// Uniform parameters for fused SwiGLU operation.
struct Info {
    /// Number of input rows (batch * seq_len).
    M: u32,
    /// Input dimension (hidden_size).
    K: u32,
    /// Output dimension (intermediate_size).
    N: u32,
}

@group(0) @binding(0) var<uniform> info: Info;
@group(0) @binding(1) var<storage, read> input: array<f32>;       // [M, K]

// BF16 Weights (Packed u32)
@group(0) @binding(2) var<storage, read> gate_w_bf16: array<u32>; // [N, K/2]
@group(0) @binding(3) var<storage, read> up_w_bf16: array<u32>;   // [N, K/2]

@group(0) @binding(4) var<storage, read_write> output: array<f32>; // [M, N]

// F32 Weights
@group(0) @binding(5) var<storage, read> gate_w_f32: array<f32>;  // [N, K]
@group(0) @binding(6) var<storage, read> up_w_f32: array<f32>;    // [N, K]

// Packed Q4_K weights, read as they sit in the GGUF. Gate and up together are the
// largest block of quantised weight in the model, so leaving these expanded would
// undo most of the point of a packed GPU path.
@group(0) @binding(7) var<storage, read> gate_w_q4k: array<u32>;
@group(0) @binding(8) var<storage, read> up_w_q4k: array<u32>;

// Shared Memory Cache for GEMV (M=1)
// Size 8192 floats = 32KB. Supports hidden_size up to 8192 (Llama-70B).
// TODO: Reduce to 4096 (16KB) for better compatibility
var<workgroup> sh_input: array<f32, 8192>;

/// Reduction buffers for parallel gate projection computation.
var<workgroup> wg_sum_gate: array<f32, 256>;
/// Reduction buffers for parallel up projection computation.
var<workgroup> wg_sum_up: array<f32, 256>;

/// Unpacks two BF16 values from a packed u32.
fn unpack_bf16(packed: u32) -> vec2<f32> {
    let lo = bitcast<f32>(packed << 16u);
    let hi = bitcast<f32>(packed & 0xFFFF0000u);
    return vec2<f32>(lo, hi);
}

/// SiLU (Swish) activation: x * sigmoid(x) = x / (1 + exp(-x)).
fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

// WIDE Fused GEMV (BF16)
@compute @workgroup_size(256)
fn fused_gemv_bf16(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let n = wg_id.x; // Output neuron index
    let tid = local_id.x;

    if (n >= info.N) { return; }

    let k_pairs = info.K / 2u;
    let weight_base = n * k_pairs;
    
    var partial_gate = 0.0;
    var partial_up = 0.0;

    // Stride by 256
    for (var k = tid; k < k_pairs; k = k + 256u) {
        // Read Input (Broadcast read from L2)
        let in0 = input[k * 2u];
        let in1 = input[k * 2u + 1u];
        
        // Read Weights (Coalesced global read)
        let g_packed = gate_w_bf16[weight_base + k];
        let u_packed = up_w_bf16[weight_base + k];
        
        let g_vec = unpack_bf16(g_packed);
        let u_vec = unpack_bf16(u_packed);
        
        partial_gate += in0 * g_vec.x + in1 * g_vec.y;
        partial_up   += in0 * u_vec.x + in1 * u_vec.y;
    }

    wg_sum_gate[tid] = partial_gate;
    wg_sum_up[tid] = partial_up;
    workgroupBarrier();

    // Parallel Reduction
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sum_gate[tid] += wg_sum_gate[tid + s];
            wg_sum_up[tid]   += wg_sum_up[tid + s];
        }
        workgroupBarrier();
    }

    // Activation & Write
    if (tid == 0u) {
        let gate_val = wg_sum_gate[0];
        let up_val = wg_sum_up[0];
        output[n] = silu(gate_val) * up_val;
    }
}


@compute @workgroup_size(16, 16)
fn fused_bmm_bf16(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x; // Output dim
    let m = id.y; // Batch/Seq dim
    
    if (m >= info.M || n >= info.N) { return; }
    
    var gate_sum = 0.0;
    var up_sum = 0.0;
    
    let input_offset = m * info.K;
    let k_pairs = info.K / 2u;
    let weight_offset = n * k_pairs;
    
    // Naive global read for BMM (Prefill phase is compute bound anyway)
    for (var k = 0u; k < k_pairs; k = k + 1u) {
        let in0 = input[input_offset + k * 2u];
        let in1 = input[input_offset + k * 2u + 1u];
        
        let g_vec = unpack_bf16(gate_w_bf16[weight_offset + k]);
        let u_vec = unpack_bf16(up_w_bf16[weight_offset + k]);
        
        gate_sum += in0 * g_vec.x + in1 * g_vec.y;
        up_sum   += in0 * u_vec.x + in1 * u_vec.y;
    }
    
    output[m * info.N + n] = silu(gate_sum) * up_sum;
}

@compute @workgroup_size(256)
fn fused_gemv_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let n = global_id.x;

    // Collaborative Load
    let num_tiles = (info.K + 255u) / 256u;
    for (var i = 0u; i < num_tiles; i = i + 1u) {
        let load_idx = i * 256u + tid;
        if (load_idx < info.K) {
            sh_input[load_idx] = input[load_idx];
        }
    }

    workgroupBarrier();

    if (n >= info.N) { return; }
    
    var gate_sum = 0.0;
    var up_sum = 0.0;
    
    let weight_offset = n * info.K;
    
    // Compute
    for (var k = 0u; k < info.K; k = k + 1u) {
        let val = sh_input[k];
        gate_sum += val * gate_w_f32[weight_offset + k];
        up_sum   += val * up_w_f32[weight_offset + k];
    }
    
    output[n] = silu(gate_sum) * up_sum;
}

@compute @workgroup_size(16, 16)
fn fused_bmm_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    let m = id.y;
    
    if (m >= info.M || n >= info.N) { return; }
    
    var gate_sum = 0.0;
    var up_sum = 0.0;
    
    let input_offset = m * info.K;
    let weight_offset = n * info.K;
    
    for (var k = 0u; k < info.K; k = k + 1u) {
        let val = input[input_offset + k];
        gate_sum += val * gate_w_f32[weight_offset + k];
        up_sum   += val * up_w_f32[weight_offset + k];
    }
    
    output[m * info.N + n] = silu(gate_sum) * up_sum;
}

// ---------------------------------------------------------------------------
// Fused SwiGLU over packed Q4_K weights.
//
// One 144-byte block covers 256 weights: d (f16) and dmin (f16) in word 0,
// scales[12] in words 1..3, then qs[128] with two 4-bit weights per byte. The
// scale words are passed into the helpers so one set of helpers serves both the
// gate and the up buffer; WGSL has no way to write it generically over storage.
// ---------------------------------------------------------------------------

const Q4K_WORDS: u32 = 36u;

fn q4k_sbyte(s0: u32, s1: u32, s2: u32, i: u32) -> u32 {
    var w = s0;
    if (i >= 8u) { w = s2; } else if (i >= 4u) { w = s1; }
    return (w >> ((i & 3u) * 8u)) & 0xFFu;
}

// Mirrors `get_scale_min_k4`. Sub-blocks 0-3 are plain 6-bit values; 4-7 are
// reassembled from a low nibble plus the top two bits of an earlier byte.
fn q4k_sm(s0: u32, s1: u32, s2: u32, j: u32) -> vec2<f32> {
    if (j < 4u) {
        return vec2<f32>(
            f32(q4k_sbyte(s0, s1, s2, j) & 63u),
            f32(q4k_sbyte(s0, s1, s2, j + 4u) & 63u),
        );
    }
    let sc = (q4k_sbyte(s0, s1, s2, j + 4u) & 0xFu)
           | ((q4k_sbyte(s0, s1, s2, j - 4u) >> 6u) << 4u);
    let m  = (q4k_sbyte(s0, s1, s2, j + 4u) >> 4u)
           | ((q4k_sbyte(s0, s1, s2, j) >> 6u) << 4u);
    return vec2<f32>(f32(sc), f32(m));
}

// Dequantises one qs word (8 weights) from one of the two packed buffers into two
// vec4 lanes, low nibbles and high nibbles, already scaled and offset.
//
// It returns the weights instead of a dot product so the caller can load the input
// activations once and reuse them across several output rows. That reuse is what
// makes the kernel fast: it moves a quarter of the bytes the BF16 path does, so it
// is ALU- and latency-bound rather than bandwidth-bound.
struct Q4KPair {
    lo: vec4<f32>,
    hi: vec4<f32>,
}

fn q4k_word_weights(base: u32, wi: u32, is_gate: bool) -> Q4KPair {
    var d_word = 0u;
    var s0 = 0u;
    var s1 = 0u;
    var s2 = 0u;
    var qword = 0u;
    if (is_gate) {
        d_word = gate_w_q4k[base];
        s0 = gate_w_q4k[base + 1u];
        s1 = gate_w_q4k[base + 2u];
        s2 = gate_w_q4k[base + 3u];
        qword = gate_w_q4k[base + 4u + wi];
    } else {
        d_word = up_w_q4k[base];
        s0 = up_w_q4k[base + 1u];
        s1 = up_w_q4k[base + 2u];
        s2 = up_w_q4k[base + 3u];
        qword = up_w_q4k[base + 4u + wi];
    }

    let dd = unpack2x16float(d_word);
    let j = wi / 8u;
    let sm1 = q4k_sm(s0, s1, s2, j * 2u);
    let sm2 = q4k_sm(s0, s1, s2, j * 2u + 1u);

    let lo_n = vec4<f32>(unpack4xU8(qword & 0x0F0F0F0Fu));
    let hi_n = vec4<f32>(unpack4xU8((qword >> 4u) & 0x0F0F0F0Fu));

    var out: Q4KPair;
    out.lo = lo_n * (dd.x * sm1.x) - vec4<f32>(dd.y * sm1.y);
    out.hi = hi_n * (dd.x * sm2.x) - vec4<f32>(dd.y * sm2.y);
    return out;
}

const Q4K_ROWS_PER_WG: u32 = 4u;

// Two accumulators per row (gate and up) for the four rows a workgroup owns.
//
// 64 threads, not 256: at K=3072 a row is 384 qs words, so with 256 threads each one
// did about 1.5 words and then paid an eight-step barrier reduction that cost as much
// as the arithmetic. Fewer, busier threads shorten the reduction and lengthen the work.
var<workgroup> wg_q4k: array<vec4<f32>, 64>;
var<workgroup> wg_q4k_b: array<vec4<f32>, 64>;

@compute @workgroup_size(64)
fn fused_gemv_q4k(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row0 = wg_id.x * Q4K_ROWS_PER_WG;
    let tid = local_id.x;
    if (row0 >= info.N) { return; }

    let blocks_per_row = info.K / 256u;
    let words_per_row = blocks_per_row * Q4K_WORDS;
    let total_words = blocks_per_row * 32u;
    // (gate0, up0, gate1, up1) and (gate2, up2, gate3, up3)
    var acc = vec4<f32>(0.0);
    var acc_b = vec4<f32>(0.0);

    for (var w = tid; w < total_words; w = w + 64u) {
        let blk = w / 32u;
        let wi = w % 32u;
        let l0 = blk * 256u + (wi / 8u) * 64u + (wi % 8u) * 4u;

        let a_lo = vec4<f32>(input[l0], input[l0 + 1u], input[l0 + 2u], input[l0 + 3u]);
        let a_hi = vec4<f32>(
            input[l0 + 32u], input[l0 + 33u], input[l0 + 34u], input[l0 + 35u],
        );
        let base = row0 * words_per_row + blk * Q4K_WORDS;

        let g0 = q4k_word_weights(base, wi, true);
        let u0 = q4k_word_weights(base, wi, false);
        acc.x += dot(a_lo, g0.lo) + dot(a_hi, g0.hi);
        acc.y += dot(a_lo, u0.lo) + dot(a_hi, u0.hi);

        if (row0 + 1u < info.N) {
            let b1 = base + words_per_row;
            let g1 = q4k_word_weights(b1, wi, true);
            let u1 = q4k_word_weights(b1, wi, false);
            acc.z += dot(a_lo, g1.lo) + dot(a_hi, g1.hi);
            acc.w += dot(a_lo, u1.lo) + dot(a_hi, u1.hi);
        }
        if (row0 + 2u < info.N) {
            let b2 = base + words_per_row * 2u;
            let g2 = q4k_word_weights(b2, wi, true);
            let u2 = q4k_word_weights(b2, wi, false);
            acc_b.x += dot(a_lo, g2.lo) + dot(a_hi, g2.hi);
            acc_b.y += dot(a_lo, u2.lo) + dot(a_hi, u2.hi);
        }
        if (row0 + 3u < info.N) {
            let b3 = base + words_per_row * 3u;
            let g3 = q4k_word_weights(b3, wi, true);
            let u3 = q4k_word_weights(b3, wi, false);
            acc_b.z += dot(a_lo, g3.lo) + dot(a_hi, g3.hi);
            acc_b.w += dot(a_lo, u3.lo) + dot(a_hi, u3.hi);
        }
    }

    wg_q4k[tid] = acc;
    wg_q4k_b[tid] = acc_b;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_q4k[tid] += wg_q4k[tid + s];
            wg_q4k_b[tid] += wg_q4k_b[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let r = wg_q4k[0];
        let rb = wg_q4k_b[0];
        output[row0] = silu(r.x) * r.y;
        if (row0 + 1u < info.N) { output[row0 + 1u] = silu(r.z) * r.w; }
        if (row0 + 2u < info.N) { output[row0 + 2u] = silu(rb.x) * rb.y; }
        if (row0 + 3u < info.N) { output[row0 + 3u] = silu(rb.z) * rb.w; }
    }
}

// Batched form: grid is (n, m), one workgroup per output element.
@compute @workgroup_size(256)
fn fused_bmm_q4k(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let n = wg_id.x;
    let row = wg_id.y;
    let tid = local_id.x;
    if (n >= info.N || row >= info.M) { return; }

    let blocks_per_row = info.K / 256u;
    let row_base = n * blocks_per_row * Q4K_WORDS;
    let total_words = blocks_per_row * 32u;
    let input_offset = row * info.K;

    var pg = 0.0;
    var pu = 0.0;
    for (var w = tid; w < total_words; w = w + 256u) {
        let blk = w / 32u;
        let wi = w % 32u;
        let base = row_base + blk * Q4K_WORDS;
        let l0 = input_offset + blk * 256u + (wi / 8u) * 64u + (wi % 8u) * 4u;
        let a_lo = vec4<f32>(input[l0], input[l0 + 1u], input[l0 + 2u], input[l0 + 3u]);
        let a_hi = vec4<f32>(
            input[l0 + 32u], input[l0 + 33u], input[l0 + 34u], input[l0 + 35u],
        );
        let g = q4k_word_weights(base, wi, true);
        let u = q4k_word_weights(base, wi, false);
        pg += dot(a_lo, g.lo) + dot(a_hi, g.hi);
        pu += dot(a_lo, u.lo) + dot(a_hi, u.hi);
    }

    wg_sum_gate[tid] = pg;
    wg_sum_up[tid] = pu;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sum_gate[tid] += wg_sum_gate[tid + s];
            wg_sum_up[tid]   += wg_sum_up[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        output[row * info.N + n] = silu(wg_sum_gate[0]) * wg_sum_up[0];
    }
}
