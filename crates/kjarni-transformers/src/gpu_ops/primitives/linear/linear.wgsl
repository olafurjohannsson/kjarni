struct LinearInfo {
    M: u32, // Batch * Seq
    K: u32, // Hidden Size
    N: u32, // Vocab Size
}

@group(0) @binding(0) var<uniform> info: LinearInfo;
@group(0) @binding(1) var<storage, read> Input: array<f32>;

// We use two bindings for weights. Only one is active per pipeline.
@group(0) @binding(2) var<storage, read> W_F32: array<f32>;  // [N, K]
@group(0) @binding(3) var<storage, read> W_BF16: array<u32>; // [N, K/2]

@group(0) @binding(4) var<storage, read_write> Output: array<f32>; // [M, N]

// Q4_K weights, read as raw words. A block is 144 bytes covering 256 weights:
// d (f16), dmin (f16), scales[12], qs[128]. Kept packed so decode streams a
// quarter of what the BF16 path streams.
@group(0) @binding(5) var<storage, read> W_Q4K: array<u32>;

// Q6_K weights. A block is 210 bytes -- ql[128], qh[64], scales[16], d -- which is
// not a multiple of four, so the upload pads each block to 212 (53 words) and the
// shader can index words directly instead of straddling boundaries.
@group(0) @binding(9) var<storage, read> W_Q6K: array<u32>;

// Q8_0 weights. The on-disk block is 34 bytes -- d (f16) then qs[32] -- which
// leaves `qs` two bytes out of word alignment, so the upload repacks each block
// into 9 words: d alone in word 0, then the 32 values as eight clean words. That
// costs 5.9% of the buffer and buys an aligned `unpack4xI8` per word.
@group(0) @binding(10) var<storage, read> W_Q8: array<u32>;

// Reduction cache for Wide kernel
var<workgroup> wg_sum: array<f32, 256>;

// Four row accumulators per thread, reduced together in one barrier chain.
//
// The GEMV kernels run 64 threads rather than 256. At K=3072 a row is only 384 qs
// words, so with 256 threads each one did about 1.5 words of work and then paid an
// eight-step barrier reduction; the reduction cost as much as the arithmetic. Fewer,
// busier threads move that ratio the right way.
var<workgroup> wg_sum4: array<vec4<f32>, 64>;

fn unpack_bf16(packed: u32) -> vec2<f32> {
    let x = bitcast<f32>(packed << 16u);
    let y = bitcast<f32>(packed & 0xFFFF0000u);
    return vec2<f32>(x, y);
}

// Wide GEMV (BF16)
@compute @workgroup_size(256)
fn gemv_bf16_wide(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32> 
) {
    let n = wg_id.y * grid.x + wg_id.x; 
    
    let tid = local_id.x;

    if (n >= info.N) { return; }
    
    let k_pairs = info.K / 2u;
    let weight_base = n * k_pairs;
    var partial_sum = 0.0;

    for (var k = tid; k < k_pairs; k = k + 256u) {
        let in0 = Input[k * 2u];
        let in1 = Input[k * 2u + 1u];
        
        let w_packed = W_BF16[weight_base + k];
        let w_vec = unpack_bf16(w_packed);
        
        partial_sum += in0 * w_vec.x + in1 * w_vec.y;
    }

    wg_sum[tid] = partial_sum;
    workgroupBarrier();

    // Tree Reduction
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sum[tid] += wg_sum[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        Output[n] = wg_sum[0];
    }
}

// Standard GEMV (BF16) - 1 Thread per Output
@compute @workgroup_size(256)
fn gemv_bf16(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    if (n >= info.N) { return; }

    var sum = vec4<f32>(0.0);
    let k_pairs = info.K / 2u;
    let weight_offset = n * k_pairs;

    // Process 4 pairs (8 floats) per iteration
    var k = 0u;
    for (; k + 3u < k_pairs; k += 4u) {
        let w0 = unpack_bf16(W_BF16[weight_offset + k]);
        let w1 = unpack_bf16(W_BF16[weight_offset + k + 1u]);
        let w2 = unpack_bf16(W_BF16[weight_offset + k + 2u]);
        let w3 = unpack_bf16(W_BF16[weight_offset + k + 3u]);

        // Input index is k*2
        let idx = k * 2u;
        let in0 = Input[idx];      let in1 = Input[idx + 1u];
        let in2 = Input[idx + 2u]; let in3 = Input[idx + 3u];
        let in4 = Input[idx + 4u]; let in5 = Input[idx + 5u];
        let in6 = Input[idx + 6u]; let in7 = Input[idx + 7u];

        sum.x += in0 * w0.x + in1 * w0.y;
        sum.y += in2 * w1.x + in3 * w1.y;
        sum.z += in4 * w2.x + in5 * w2.y;
        sum.w += in6 * w3.x + in7 * w3.y;
    }

    // Cleanup tail
    var tail_sum = 0.0;
    for (; k < k_pairs; k += 1u) {
        let w = unpack_bf16(W_BF16[weight_offset + k]);
        let in0 = Input[k * 2u];
        let in1 = Input[k * 2u + 1u];
        tail_sum += in0 * w.x + in1 * w.y;
    }

    Output[n] = sum.x + sum.y + sum.z + sum.w + tail_sum;
}

// BMM (BF16) - 2D Tiled

// GEMV (F32)
@compute @workgroup_size(256)
fn gemv_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    if (n >= info.N) { return; }

    var sum = 0.0;
    let weight_offset = n * info.K;
    
    for (var k = 0u; k < info.K; k = k + 1u) {
        sum += Input[k] * W_F32[weight_offset + k];
    }
    Output[n] = sum;
}


// ---------------------------------------------------------------------------
// Wide GEMM for a few rows: the wide GEMV with one accumulator per row.
//
// A workgroup computes one output column for every row, so the weights are read once
// and the cost follows the row count. A tile costs the same from 2 to 48 rows. On an
// RTX A2000, across a 28-layer decoder, this takes 47 ms at two rows, 96 at five and
// 163 at nine, against 70, 169 and 187 for the tiles.
// ---------------------------------------------------------------------------
const WIDE_ROWS = 10u;
const WIDE_THREADS = 64u;

var<workgroup> wg_rows: array<array<f32, WIDE_ROWS>, WIDE_THREADS>;

fn gemm_wide_finish(tid: u32, n: u32, acc: ptr<function, array<f32, WIDE_ROWS>>) {
    for (var r = 0u; r < info.M; r++) {
        wg_rows[tid][r] = (*acc)[r];
    }
    workgroupBarrier();

    for (var s = WIDE_THREADS / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            for (var r = 0u; r < info.M; r++) {
                wg_rows[tid][r] += wg_rows[tid + s][r];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        for (var r = 0u; r < info.M; r++) {
            Output[r * info.N + n] = wg_rows[0][r];
        }
    }
}

@compute @workgroup_size(64)
fn gemm_wide_bf16(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let n = wg_id.y * grid.x + wg_id.x;
    if (n >= info.N) { return; }

    let tid = local_id.x;
    let k_pairs = info.K / 2u;
    let weight_base = n * k_pairs;
    var acc: array<f32, WIDE_ROWS>;

    for (var k = tid; k < k_pairs; k += WIDE_THREADS) {
        let w = unpack_bf16(W_BF16[weight_base + k]);
        for (var r = 0u; r < info.M; r++) {
            let base = r * info.K + k * 2u;
            acc[r] += Input[base] * w.x + Input[base + 1u] * w.y;
        }
    }

    gemm_wide_finish(tid, n, &acc);
}

@compute @workgroup_size(64)
fn gemm_wide_f32(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let n = wg_id.y * grid.x + wg_id.x;
    if (n >= info.N) { return; }

    let tid = local_id.x;
    let weight_base = n * info.K;
    var acc: array<f32, WIDE_ROWS>;

    for (var k = tid; k < info.K; k += WIDE_THREADS) {
        let w = W_F32[weight_base + k];
        for (var r = 0u; r < info.M; r++) {
            acc[r] += Input[r * info.K + k] * w;
        }
    }

    gemm_wide_finish(tid, n, &acc);
}

// ---------------------------------------------------------------------------
// GEMM for two rows: one thread per output, reading both operands directly.
//
// A tile's time is the time to read the weights, whatever the row count, so it is
// the right kernel from three rows up. At two, this is 1.4x faster on an RTX A2000.
// Each thread walks one row of weights, consecutive threads take consecutive columns.
// ---------------------------------------------------------------------------
@compute @workgroup_size(32)
fn gemm_few_rows_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let idx = (wg.y * grid.x + wg.x) * 32u + lid.x;
    if (idx >= info.M * info.N) { return; }
    let row = idx / info.N;
    let col = idx % info.N;
    let a_base = row * info.K;
    let w_base = col * info.K;

    var acc = 0.0;
    for (var k = 0u; k < info.K; k++) {
        acc += Input[a_base + k] * W_F32[w_base + k];
    }
    Output[idx] = acc;
}

/// The same over BF16 weights, packed two to a word as [N, K/2].
@compute @workgroup_size(32)
fn gemm_few_rows_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let idx = (wg.y * grid.x + wg.x) * 32u + lid.x;
    if (idx >= info.M * info.N) { return; }
    let row = idx / info.N;
    let col = idx % info.N;
    let a_base = row * info.K;
    let w_base = col * info.K / 2u;

    var acc = 0.0;
    for (var pair = 0u; pair < info.K / 2u; pair++) {
        let w = unpack_bf16(W_BF16[w_base + pair]);
        acc += Input[a_base + pair * 2u] * w.x + Input[a_base + pair * 2u + 1u] * w.y;
    }
    Output[idx] = acc;
}

// ---------------------------------------------------------------------------
// Tiled GEMM, 64x64 tiles: the same kernel with 4x4 outputs per thread.
//
// For fewer than a few hundred rows (short prompts, small encoder batches) the
// 128x128 tile below leaves most of its rows empty; this one wastes less and
// measured up to 1.5x faster there. GpuLinearLayer picks between them by M.
// ---------------------------------------------------------------------------
const GEMM64_BM: u32 = 64u;
const GEMM64_BN: u32 = 64u;
const GEMM64_BK: u32 = 16u;

var<workgroup> gemm64_a: array<f32, 1024>; // [GEMM64_BK][GEMM64_BM]
var<workgroup> gemm64_w: array<f32, 1024>; // [GEMM64_BK][GEMM64_BN]

@compute @workgroup_size(16, 16)
fn gemm64_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    gemm64_tile(wg, lid, GEMM64_F32);
}

/// The same GEMM over BF16 weights, packed two to a word as [N, K/2]: decoder
/// prefill at full precision. Only the weight load differs.
@compute @workgroup_size(16, 16)
fn gemm64_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    gemm64_tile(wg, lid, GEMM64_BF16);
}

/// The same GEMM over packed Q4_K weights: decoder prefill from a GGUF. The
/// weight tile is dequantised as it is loaded, so the multiply is the F32 one.
@compute @workgroup_size(16, 16)
fn gemm64_q4k(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    gemm64_tile(wg, lid, GEMM64_Q4K);
}

/// The same GEMM over packed Q6_K weights, which a Q4_K_M file uses for some
/// of its tensors (attn_v and half the ffn_down).
@compute @workgroup_size(16, 16)
fn gemm64_q6k(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    gemm64_tile(wg, lid, GEMM64_Q6K);
}

/// The same GEMM over Q8_0 weights, which a Q4_K_M file can use for its FFN and
/// attention projections (Q4_0 and Q5_0 arrive transcoded to Q8_0).
@compute @workgroup_size(16, 16)
fn gemm64_q8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    gemm64_tile(wg, lid, GEMM64_Q8);
}

const GEMM64_F32: u32 = 0u;
const GEMM64_BF16: u32 = 1u;
const GEMM64_Q4K: u32 = 2u;
const GEMM64_Q6K: u32 = 3u;
const GEMM64_Q8: u32 = 4u;

/// Loads the weight tile for k0..k0+16 from Q8_0 blocks.
///
/// A block is 32 weights of one row under one scale, so a 16-wide slice starting
/// at a multiple of 16 lies in one block as four qs words. 64 rows x 4 words is
/// one word per thread, each giving four weights.
fn gemm64_load_w_q8(tid: u32, col0: u32, k0: u32) {
    let r = tid / 4u;
    let q = tid % 4u;
    let gn = col0 + r;
    var vals = vec4<f32>(0.0);
    if (gn < info.N) {
        let gk = k0 + q * 4u;
        vals = q8_unit_weights((gn * (info.K / 32u) + gk / 32u) * Q8_WORDS, (gk % 32u) / 4u);
    }
    let kk = q * 4u;
    gemm64_w[kk * GEMM64_BN + r] = vals.x;
    gemm64_w[(kk + 1u) * GEMM64_BN + r] = vals.y;
    gemm64_w[(kk + 2u) * GEMM64_BN + r] = vals.z;
    gemm64_w[(kk + 3u) * GEMM64_BN + r] = vals.w;
}

/// Loads the weight tile for k0..k0+16 from Q6_K blocks.
///
/// A block is 256 weights as two halves of 128, each in four quarters of 32
/// (see q6k_unit_weights). A 16-wide slice starting at a multiple of 16 lies in
/// one quarter and shares one scale: four (ql, qh) word positions, one per
/// thread, each giving four weights.
fn gemm64_load_w_q6k(tid: u32, col0: u32, k0: u32) {
    let r = tid / 4u;
    let q = tid % 4u;
    let gn = col0 + r;
    var vals = vec4<f32>(0.0);
    if (gn < info.N) {
        let gk = k0 + q * 4u;
        let blk = gk / 256u;
        let t = gk % 256u;
        let i = t / 128u;         // half-block
        let p = t % 128u;
        let quarter = p / 32u;
        let jg = (p % 32u) / 4u;  // word within the quarter

        let base = (gn * (info.K / 256u) + blk) * Q6K_WORDS;
        let scale = unpack2x16float(W_Q6K[base + 52u]).x
            * q6k_scale(base, i * 8u + quarter * 2u + jg / 4u);
        let qlw = W_Q6K[base + i * 16u + jg + select(0u, 8u, (quarter & 1u) == 1u)];
        let qhw = W_Q6K[base + 32u + i * 8u + jg];

        // As in q6k_unit_weights: quarters 0 and 1 take the low nibbles of
        // their ql word, 2 and 3 the high, each with its two bits of qh.
        var low = qlw & 0x0F0F0F0Fu;
        if (quarter >= 2u) { low = (qlw >> 4u) & 0x0F0F0F0Fu; }
        let high = ((qhw >> (quarter * 2u)) & 0x03030303u) << 4u;
        vals = (vec4<f32>(unpack4xU8(low | high)) - vec4<f32>(32.0)) * scale;
    }
    let kk = q * 4u;
    gemm64_w[kk * GEMM64_BN + r] = vals.x;
    gemm64_w[(kk + 1u) * GEMM64_BN + r] = vals.y;
    gemm64_w[(kk + 2u) * GEMM64_BN + r] = vals.z;
    gemm64_w[(kk + 3u) * GEMM64_BN + r] = vals.w;
}

/// Loads the weight tile for k0..k0+16 from Q4_K blocks.
///
/// A block is 256 weights of one row, and within it each run of 32 shares a
/// sub-block scale and min (see q4k_word_weights). A 16-wide slice starting at
/// a multiple of 16 therefore sits inside one block and one sub-block, as four
/// qs words. 64 rows x 4 words is one word per thread: each dequantises four
/// weights with a single scale lookup.
fn gemm64_load_w_q4k(tid: u32, col0: u32, k0: u32) {
    let r = tid / 4u;
    let q = tid % 4u;
    let gn = col0 + r;
    var vals = vec4<f32>(0.0);
    if (gn < info.N) {
        let gk = k0 + q * 4u;
        let blk = gk / 256u;
        let t = gk % 256u;
        let g = t / 64u;          // which 64: one qs word range of 8
        let within = t % 64u;
        let hi = within >= 32u;   // high nibbles belong to the second sub-block
        let wi = g * 8u + (within % 32u) / 4u;

        let base = (gn * (info.K / 256u) + blk) * Q4K_WORDS;
        let dd = unpack2x16float(W_Q4K[base]);
        let sub = g * 2u + select(0u, 1u, hi);
        let sm = q4k_scale_min(W_Q4K[base + 1u], W_Q4K[base + 2u], W_Q4K[base + 3u], sub);
        let qword = W_Q4K[base + 4u + wi];
        let nib = select(qword & 0x0F0F0F0Fu, (qword >> 4u) & 0x0F0F0F0Fu, hi);
        vals = vec4<f32>(unpack4xU8(nib)) * (dd.x * sm.x) - vec4<f32>(dd.y * sm.y);
    }
    let kk = q * 4u;
    gemm64_w[kk * GEMM64_BN + r] = vals.x;
    gemm64_w[(kk + 1u) * GEMM64_BN + r] = vals.y;
    gemm64_w[(kk + 2u) * GEMM64_BN + r] = vals.z;
    gemm64_w[(kk + 3u) * GEMM64_BN + r] = vals.w;
}

/// One 64x64 output tile. `mode` is a constant at each call site, so the
/// branches in the weight load fold away.
fn gemm64_tile(wg: vec3<u32>, lid: vec3<u32>, mode: u32) {
    let row0 = wg.y * GEMM64_BM;
    let col0 = wg.x * GEMM64_BN;
    let tid = lid.y * 16u + lid.x;

    // Four rows of four columns each, as vec4 so they stay in registers: an
    // indexed array here is spilled to local memory and runs at half the speed
    // of the untiled kernel.
    var c0 = vec4<f32>(0.0);
    var c1 = vec4<f32>(0.0);
    var c2 = vec4<f32>(0.0);
    var c3 = vec4<f32>(0.0);

    for (var k0 = 0u; k0 < info.K; k0 += GEMM64_BK) {
        // Each tile is 64 x 16 = 1024 values; 256 threads load four each.
        // Consecutive threads take consecutive k, which is contiguous in both
        // operands, so the loads coalesce.
        for (var l = 0u; l < 4u; l++) {
            let idx = tid + l * 256u;
            let r = idx / GEMM64_BK;
            let kk = idx % GEMM64_BK;
            let gk = k0 + kk;

            let gr = row0 + r;
            var a = 0.0;
            if (gr < info.M && gk < info.K) { a = Input[gr * info.K + gk]; }
            gemm64_a[kk * GEMM64_BM + r] = a;

            if (mode == GEMM64_Q4K || mode == GEMM64_Q6K || mode == GEMM64_Q8) { continue; }
            let gn = col0 + r;
            var w = 0.0;
            if (gn < info.N && gk < info.K) {
                if (mode == GEMM64_BF16) {
                    // K is even for packed weights, so a row's pairs never
                    // straddle two rows.
                    let pair = unpack_bf16(W_BF16[(gn * info.K + gk) / 2u]);
                    w = select(pair.x, pair.y, (gk & 1u) == 1u);
                } else {
                    w = W_F32[gn * info.K + gk];
                }
            }
            gemm64_w[kk * GEMM64_BN + r] = w;
        }
        if (mode == GEMM64_Q4K) { gemm64_load_w_q4k(tid, col0, k0); }
        if (mode == GEMM64_Q6K) { gemm64_load_w_q6k(tid, col0, k0); }
        if (mode == GEMM64_Q8) { gemm64_load_w_q8(tid, col0, k0); }
        workgroupBarrier();

        for (var kk = 0u; kk < GEMM64_BK; kk++) {
            let ab = kk * GEMM64_BM + lid.y;
            let wb = kk * GEMM64_BN + lid.x;
            let w = vec4<f32>(gemm64_w[wb], gemm64_w[wb + 16u], gemm64_w[wb + 32u], gemm64_w[wb + 48u]);
            c0 += gemm64_a[ab] * w;
            c1 += gemm64_a[ab + 16u] * w;
            c2 += gemm64_a[ab + 32u] * w;
            c3 += gemm64_a[ab + 48u] * w;
        }
        workgroupBarrier();
    }

    gemm64_store_row(row0 + lid.y, col0 + lid.x, c0);
    gemm64_store_row(row0 + lid.y + 16u, col0 + lid.x, c1);
    gemm64_store_row(row0 + lid.y + 32u, col0 + lid.x, c2);
    gemm64_store_row(row0 + lid.y + 48u, col0 + lid.x, c3);
}

/// Writes one thread's four outputs of row `r`, at columns c, c+16, c+32, c+48.
fn gemm64_store_row(r: u32, c: u32, v: vec4<f32>) {
    if (r >= info.M) { return; }
    let base = r * info.N;
    if (c < info.N) { Output[base + c] = v.x; }
    if (c + 16u < info.N) { Output[base + c + 16u] = v.y; }
    if (c + 32u < info.N) { Output[base + c + 32u] = v.z; }
    if (c + 48u < info.N) { Output[base + c + 48u] = v.w; }
}

// ---------------------------------------------------------------------------
// Tiled GEMM: Output[M, N] = Input[M, K] * W[N, K]^T, W in F32, BF16, Q4_K, Q6_K or Q8_0
//
// The batched path for every weight type: encoders, seq2seq and decoder
// prefill all land here. One output per thread, reading both operands from
// global memory for every k, runs at about 150 GFLOP/s on an RTX A2000.
// Here a 16x16 workgroup computes a 128x128 tile, 8x8 outputs per thread, from
// 16-wide slices of both operands staged in workgroup memory, so each value
// loaded from global memory is used 128 times and each workgroup-memory read
// feeds eight multiply-adds.
//
// Threads take rows ly, ly+16, ..., ly+112 and the same stride in columns, so a
// row of threads reads consecutive workgroup-memory words and writes
// consecutive outputs. Out-of-range rows, columns and k are loaded as zero and
// never stored, so any M, N and K work. Packed weights are dequantised as their
// tile is loaded, so the multiply is the same for every type.
//
// The 64 accumulators are 16 named vec4s. An indexed array here is spilled to
// local memory and ran at half the speed of the untiled kernel.
// ---------------------------------------------------------------------------
const GEMM_BM: u32 = 128u;
const GEMM_BN: u32 = 128u;
const GEMM_BK: u32 = 16u;

var<workgroup> gemm_a: array<f32, 2048>; // [GEMM_BK][GEMM_BM]
var<workgroup> gemm_w: array<f32, 2048>; // [GEMM_BK][GEMM_BN]

@compute @workgroup_size(16, 16)
fn gemm_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    gemm_tile(wg, lid, GEMM_F32);
}

/// The same GEMM over BF16 weights, packed two to a word as [N, K/2]: decoder
/// prefill at full precision. Only the weight load differs.
@compute @workgroup_size(16, 16)
fn gemm_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    gemm_tile(wg, lid, GEMM_BF16);
}

/// The same GEMM over packed Q4_K weights: decoder prefill from a GGUF.
@compute @workgroup_size(16, 16)
fn gemm_q4k(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    gemm_tile(wg, lid, GEMM_Q4K);
}

/// The same GEMM over packed Q6_K weights, which a Q4_K_M file uses for some
/// of its tensors (attn_v and half the ffn_down).
@compute @workgroup_size(16, 16)
fn gemm_q6k(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    gemm_tile(wg, lid, GEMM_Q6K);
}

/// The same GEMM over Q8_0 weights; see gemm64_q8.
@compute @workgroup_size(16, 16)
fn gemm_q8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    gemm_tile(wg, lid, GEMM_Q8);
}

const GEMM_F32: u32 = 0u;
const GEMM_BF16: u32 = 1u;
const GEMM_Q4K: u32 = 2u;
const GEMM_Q6K: u32 = 3u;
const GEMM_Q8: u32 = 4u;

/// Loads the weight tile for k0..k0+16 from Q8_0 blocks: four qs words per row,
/// so 128 rows x 4 words is two words per thread. See gemm64_load_w_q8.
fn gemm_load_w_q8(tid: u32, col0: u32, k0: u32) {
    let q = tid % 4u;
    for (var l = 0u; l < 2u; l++) {
        let r = (tid + l * 256u) / 4u;
        let gn = col0 + r;
        var vals = vec4<f32>(0.0);
        if (gn < info.N) {
            let gk = k0 + q * 4u;
            vals = q8_unit_weights((gn * (info.K / 32u) + gk / 32u) * Q8_WORDS, (gk % 32u) / 4u);
        }
        let kk = q * 4u;
        gemm_w[kk * GEMM_BN + r] = vals.x;
        gemm_w[(kk + 1u) * GEMM_BN + r] = vals.y;
        gemm_w[(kk + 2u) * GEMM_BN + r] = vals.z;
        gemm_w[(kk + 3u) * GEMM_BN + r] = vals.w;
    }
}

/// Loads the weight tile for k0..k0+16 from Q6_K blocks.
///
/// A block is 256 weights as two halves of 128, each in four quarters of 32
/// (see q6k_unit_weights). A 16-wide slice starting at a multiple of 16 lies in
/// one quarter and shares one scale: four (ql, qh) word positions per row. 128
/// rows x 4 words is two words per thread, each giving four weights.
fn gemm_load_w_q6k(tid: u32, col0: u32, k0: u32) {
    let q = tid % 4u;
    for (var l = 0u; l < 2u; l++) {
        let r = (tid + l * 256u) / 4u;
        let gn = col0 + r;
        var vals = vec4<f32>(0.0);
        if (gn < info.N) {
            let gk = k0 + q * 4u;
            let blk = gk / 256u;
            let t = gk % 256u;
            let i = t / 128u;         // half-block
            let p = t % 128u;
            let quarter = p / 32u;
            let jg = (p % 32u) / 4u;  // word within the quarter

            let base = (gn * (info.K / 256u) + blk) * Q6K_WORDS;
            let scale = unpack2x16float(W_Q6K[base + 52u]).x
                * q6k_scale(base, i * 8u + quarter * 2u + jg / 4u);
            let qlw = W_Q6K[base + i * 16u + jg + select(0u, 8u, (quarter & 1u) == 1u)];
            let qhw = W_Q6K[base + 32u + i * 8u + jg];

            // As in q6k_unit_weights: quarters 0 and 1 take the low nibbles of
            // their ql word, 2 and 3 the high, each with its two bits of qh.
            var low = qlw & 0x0F0F0F0Fu;
            if (quarter >= 2u) { low = (qlw >> 4u) & 0x0F0F0F0Fu; }
            let high = ((qhw >> (quarter * 2u)) & 0x03030303u) << 4u;
            vals = (vec4<f32>(unpack4xU8(low | high)) - vec4<f32>(32.0)) * scale;
        }
        let kk = q * 4u;
        gemm_w[kk * GEMM_BN + r] = vals.x;
        gemm_w[(kk + 1u) * GEMM_BN + r] = vals.y;
        gemm_w[(kk + 2u) * GEMM_BN + r] = vals.z;
        gemm_w[(kk + 3u) * GEMM_BN + r] = vals.w;
    }
}

/// Loads the weight tile for k0..k0+16 from Q4_K blocks.
///
/// A block is 256 weights of one row, and within it each run of 32 shares a
/// sub-block scale and min (see q4k_word_weights). A 16-wide slice starting at
/// a multiple of 16 therefore sits inside one block and one sub-block, as four
/// qs words. 128 rows x 4 words is two words per thread, each dequantising four
/// weights with a single scale lookup.
fn gemm_load_w_q4k(tid: u32, col0: u32, k0: u32) {
    let q = tid % 4u;
    for (var l = 0u; l < 2u; l++) {
        let r = (tid + l * 256u) / 4u;
        let gn = col0 + r;
        var vals = vec4<f32>(0.0);
        if (gn < info.N) {
            let gk = k0 + q * 4u;
            let blk = gk / 256u;
            let t = gk % 256u;
            let g = t / 64u;          // which 64: one qs word range of 8
            let within = t % 64u;
            let hi = within >= 32u;   // high nibbles belong to the second sub-block
            let wi = g * 8u + (within % 32u) / 4u;

            let base = (gn * (info.K / 256u) + blk) * Q4K_WORDS;
            let dd = unpack2x16float(W_Q4K[base]);
            let sub = g * 2u + select(0u, 1u, hi);
            let sm = q4k_scale_min(W_Q4K[base + 1u], W_Q4K[base + 2u], W_Q4K[base + 3u], sub);
            let qword = W_Q4K[base + 4u + wi];
            let nib = select(qword & 0x0F0F0F0Fu, (qword >> 4u) & 0x0F0F0F0Fu, hi);
            vals = vec4<f32>(unpack4xU8(nib)) * (dd.x * sm.x) - vec4<f32>(dd.y * sm.y);
        }
        let kk = q * 4u;
        gemm_w[kk * GEMM_BN + r] = vals.x;
        gemm_w[(kk + 1u) * GEMM_BN + r] = vals.y;
        gemm_w[(kk + 2u) * GEMM_BN + r] = vals.z;
        gemm_w[(kk + 3u) * GEMM_BN + r] = vals.w;
    }
}

/// One 128x128 output tile. `mode` is a constant at each call site, so the
/// branches in the weight load fold away.
fn gemm_tile(wg: vec3<u32>, lid: vec3<u32>, mode: u32) {
    let row0 = wg.y * GEMM_BM;
    let col0 = wg.x * GEMM_BN;
    let tid = lid.y * 16u + lid.x;

    var c0l = vec4<f32>(0.0);
    var c0r = vec4<f32>(0.0);
    var c1l = vec4<f32>(0.0);
    var c1r = vec4<f32>(0.0);
    var c2l = vec4<f32>(0.0);
    var c2r = vec4<f32>(0.0);
    var c3l = vec4<f32>(0.0);
    var c3r = vec4<f32>(0.0);
    var c4l = vec4<f32>(0.0);
    var c4r = vec4<f32>(0.0);
    var c5l = vec4<f32>(0.0);
    var c5r = vec4<f32>(0.0);
    var c6l = vec4<f32>(0.0);
    var c6r = vec4<f32>(0.0);
    var c7l = vec4<f32>(0.0);
    var c7r = vec4<f32>(0.0);

    for (var k0 = 0u; k0 < info.K; k0 += GEMM_BK) {
        // Each tile is 128 x 16 = 2048 values; 256 threads load eight each.
        // Consecutive threads take consecutive k, which is contiguous in both
        // operands, so the loads coalesce.
        for (var l = 0u; l < 8u; l++) {
            let idx = tid + l * 256u;
            let r = idx / GEMM_BK;
            let kk = idx % GEMM_BK;
            let gk = k0 + kk;

            let gr = row0 + r;
            var a = 0.0;
            if (gr < info.M && gk < info.K) { a = Input[gr * info.K + gk]; }
            gemm_a[kk * GEMM_BM + r] = a;

            if (mode == GEMM_Q4K || mode == GEMM_Q6K || mode == GEMM_Q8) { continue; }
            let gn = col0 + r;
            var w = 0.0;
            if (gn < info.N && gk < info.K) {
                if (mode == GEMM_BF16) {
                    // K is even for packed weights, so a row's pairs never
                    // straddle two rows.
                    let pair = unpack_bf16(W_BF16[(gn * info.K + gk) / 2u]);
                    w = select(pair.x, pair.y, (gk & 1u) == 1u);
                } else {
                    w = W_F32[gn * info.K + gk];
                }
            }
            gemm_w[kk * GEMM_BN + r] = w;
        }
        if (mode == GEMM_Q4K) { gemm_load_w_q4k(tid, col0, k0); }
        if (mode == GEMM_Q6K) { gemm_load_w_q6k(tid, col0, k0); }
        if (mode == GEMM_Q8) { gemm_load_w_q8(tid, col0, k0); }
        workgroupBarrier();

        for (var kk = 0u; kk < GEMM_BK; kk++) {
            let ab = kk * GEMM_BM + lid.y;
            let wb = kk * GEMM_BN + lid.x;
            let wl = vec4<f32>(gemm_w[wb], gemm_w[wb + 16u], gemm_w[wb + 32u], gemm_w[wb + 48u]);
            let wr = vec4<f32>(gemm_w[wb + 64u], gemm_w[wb + 80u], gemm_w[wb + 96u], gemm_w[wb + 112u]);
            let a0 = gemm_a[ab + 0u];
            c0l += a0 * wl;
            c0r += a0 * wr;
            let a1 = gemm_a[ab + 16u];
            c1l += a1 * wl;
            c1r += a1 * wr;
            let a2 = gemm_a[ab + 32u];
            c2l += a2 * wl;
            c2r += a2 * wr;
            let a3 = gemm_a[ab + 48u];
            c3l += a3 * wl;
            c3r += a3 * wr;
            let a4 = gemm_a[ab + 64u];
            c4l += a4 * wl;
            c4r += a4 * wr;
            let a5 = gemm_a[ab + 80u];
            c5l += a5 * wl;
            c5r += a5 * wr;
            let a6 = gemm_a[ab + 96u];
            c6l += a6 * wl;
            c6r += a6 * wr;
            let a7 = gemm_a[ab + 112u];
            c7l += a7 * wl;
            c7r += a7 * wr;
        }
        workgroupBarrier();
    }

    gemm_store_row(row0 + lid.y + 0u, col0 + lid.x, c0l, c0r);
    gemm_store_row(row0 + lid.y + 16u, col0 + lid.x, c1l, c1r);
    gemm_store_row(row0 + lid.y + 32u, col0 + lid.x, c2l, c2r);
    gemm_store_row(row0 + lid.y + 48u, col0 + lid.x, c3l, c3r);
    gemm_store_row(row0 + lid.y + 64u, col0 + lid.x, c4l, c4r);
    gemm_store_row(row0 + lid.y + 80u, col0 + lid.x, c5l, c5r);
    gemm_store_row(row0 + lid.y + 96u, col0 + lid.x, c6l, c6r);
    gemm_store_row(row0 + lid.y + 112u, col0 + lid.x, c7l, c7r);
}

/// Writes one thread's eight outputs of row `r`, at columns c, c+16, ..., c+112.
fn gemm_store_row(r: u32, c: u32, lo: vec4<f32>, hi: vec4<f32>) {
    if (r >= info.M) { return; }
    let base = r * info.N;
    if (c < info.N) { Output[base + c] = lo.x; }
    if (c + 16u < info.N) { Output[base + c + 16u] = lo.y; }
    if (c + 32u < info.N) { Output[base + c + 32u] = lo.z; }
    if (c + 48u < info.N) { Output[base + c + 48u] = lo.w; }
    if (c + 64u < info.N) { Output[base + c + 64u] = hi.x; }
    if (c + 80u < info.N) { Output[base + c + 80u] = hi.y; }
    if (c + 96u < info.N) { Output[base + c + 96u] = hi.z; }
    if (c + 112u < info.N) { Output[base + c + 112u] = hi.w; }
}

// ---------------------------------------------------------------------------
// Q4_K GEMV
//
// The BF16 kernels above require the weights to be expanded before upload, which
// costs 3.4x the memory traffic per token and, on a 12GB card, the difference
// between a 3B model fitting and spilling to host memory. This kernel reads the
// Q4_K blocks as they sit in the GGUF.
//
// Layout of one 144-byte block, as 36 u32 words:
//   word  0      : d (f16) in the low half, dmin (f16) in the high half
//   words 1..3   : scales[12], six-bit scales and mins for the 8 sub-blocks
//   words 4..35  : qs[128], two 4-bit weights per byte
//
// Dequantisation matches `dequantize_q4_k_block` on the CPU side exactly: each
// group of 64 weights takes its low nibbles from one sub-block scale/min pair and
// its high nibbles from the next.
// ---------------------------------------------------------------------------

const Q4K_WORDS: u32 = 36u; // 144 bytes per block

// The three scale words are passed by value, never re-read. Indexing the storage
// buffer per extracted byte costs a memory op each time and the compiler will not
// hoist it out, which measured 4x slower than the BF16 kernel on the same weights.
fn q4k_sbyte(s0: u32, s1: u32, s2: u32, i: u32) -> u32 {
    var w = s0;
    if (i >= 8u) { w = s2; } else if (i >= 4u) { w = s1; }
    return (w >> ((i & 3u) * 8u)) & 0xFFu;
}

// Mirrors `get_scale_min_k4`: sub-blocks 0-3 are plain 6-bit values, 4-7 are
// reassembled from a low nibble plus the top two bits of an earlier byte.
fn q4k_scale_min(s0: u32, s1: u32, s2: u32, j: u32) -> vec2<f32> {
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

// Dequantises one qs word (8 weights) into two vec4 lanes: the low nibbles and the
// high nibbles, each already scaled and offset.
//
// It returns the weights rather than a dot product so the caller can load the input
// activations once and reuse them across several output rows. That reuse is the
// point: the kernel moves a quarter of the bytes the BF16 kernel does in the same
// wall time, so it is ALU- and latency-bound, and giving each thread more rows is
// the only change measured to help.
struct Q4KPair {
    lo: vec4<f32>,
    hi: vec4<f32>,
}

fn q4k_word_weights(base: u32, wi: u32) -> Q4KPair {
    let dd = unpack2x16float(W_Q4K[base]);
    let s0 = W_Q4K[base + 1u];
    let s1 = W_Q4K[base + 2u];
    let s2 = W_Q4K[base + 3u];
    let qword = W_Q4K[base + 4u + wi];

    let j = wi / 8u;
    let sm1 = q4k_scale_min(s0, s1, s2, j * 2u);
    let sm2 = q4k_scale_min(s0, s1, s2, j * 2u + 1u);

    // One mask over the whole word extracts four nibbles at once.
    let lo_n = vec4<f32>(unpack4xU8(qword & 0x0F0F0F0Fu));
    let hi_n = vec4<f32>(unpack4xU8((qword >> 4u) & 0x0F0F0F0Fu));

    var out: Q4KPair;
    out.lo = lo_n * (dd.x * sm1.x) - vec4<f32>(dd.y * sm1.y);
    out.hi = hi_n * (dd.x * sm2.x) - vec4<f32>(dd.y * sm2.y);
    return out;
}

// Output rows each GEMV workgroup computes. Shared by the Q4_K and Q6_K kernels:
// the input activations are loaded once and reused across all of them, which is what
// took these kernels from 12% of the card's bandwidth to competitive with BF16.
// Measured: 4 beats 1, and 8 is no better than 4 once register pressure bites.
const ROWS_PER_WG: u32 = 4u;

@compute @workgroup_size(64)
fn gemv_q4k(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>
) {
    let row0 = (wg_id.y * grid.x + wg_id.x) * ROWS_PER_WG;
    let tid = local_id.x;
    if (row0 >= info.N) { return; }

    let blocks_per_row = info.K / 256u;
    let words_per_row = blocks_per_row * Q4K_WORDS;
    let total_words = blocks_per_row * 32u;

    var p0 = 0.0;
    var p1 = 0.0;
    var p2 = 0.0;
    var p3 = 0.0;

    for (var w = tid; w < total_words; w = w + 64u) {
        let blk = w / 32u;
        let wi = w % 32u;
        let l0 = blk * 256u + (wi / 8u) * 64u + (wi % 8u) * 4u;

        // Loaded once, used by every row this workgroup owns.
        let a_lo = vec4<f32>(Input[l0], Input[l0 + 1u], Input[l0 + 2u], Input[l0 + 3u]);
        let a_hi = vec4<f32>(
            Input[l0 + 32u], Input[l0 + 33u], Input[l0 + 34u], Input[l0 + 35u],
        );
        let blk_off = blk * Q4K_WORDS + wi;
        let base = row0 * words_per_row + blk * Q4K_WORDS;

        let w0 = q4k_word_weights(base, wi);
        p0 += dot(a_lo, w0.lo) + dot(a_hi, w0.hi);
        if (row0 + 1u < info.N) {
            let w1 = q4k_word_weights(base + words_per_row, wi);
            p1 += dot(a_lo, w1.lo) + dot(a_hi, w1.hi);
        }
        if (row0 + 2u < info.N) {
            let w2 = q4k_word_weights(base + words_per_row * 2u, wi);
            p2 += dot(a_lo, w2.lo) + dot(a_hi, w2.hi);
        }
        if (row0 + 3u < info.N) {
            let w3 = q4k_word_weights(base + words_per_row * 3u, wi);
            p3 += dot(a_lo, w3.lo) + dot(a_hi, w3.hi);
        }
    }

    wg_sum4[tid] = vec4<f32>(p0, p1, p2, p3);
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sum4[tid] += wg_sum4[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let r = wg_sum4[0];
        Output[row0] = r.x;
        if (row0 + 1u < info.N) { Output[row0 + 1u] = r.y; }
        if (row0 + 2u < info.N) { Output[row0 + 2u] = r.z; }
        if (row0 + 3u < info.N) { Output[row0 + 3u] = r.w; }
    }
}

// ---------------------------------------------------------------------------
// Q8_0 GEMV
//
// Repacked block layout, 9 u32 words:
//   word    0 : d (f16) in the low half
//   words 1..8: qs[32], four signed 8-bit values per word
//
// A block covers 32 weights, so each of the 8 work units per block is one word
// and four weights. Dequantisation is `d * qs`, matching `dequantize_q8_0_block`.
// ---------------------------------------------------------------------------

const Q8_WORDS: u32 = 9u;

fn q8_unit_weights(base: u32, jg: u32) -> vec4<f32> {
    let d = unpack2x16float(W_Q8[base]).x;
    return vec4<f32>(unpack4xI8(W_Q8[base + 1u + jg])) * d;
}

@compute @workgroup_size(64)
fn gemv_q8(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>
) {
    let row0 = (wg_id.y * grid.x + wg_id.x) * ROWS_PER_WG;
    let tid = local_id.x;
    if (row0 >= info.N) { return; }

    let blocks_per_row = info.K / 32u;
    let words_per_row = blocks_per_row * Q8_WORDS;
    let total_units = blocks_per_row * 8u;

    var p0 = 0.0;
    var p1 = 0.0;
    var p2 = 0.0;
    var p3 = 0.0;

    for (var u = tid; u < total_units; u = u + 64u) {
        let blk = u / 8u;
        let jg = u % 8u;
        let ob = blk * 32u + jg * 4u;

        // Loaded once, reused by every row this workgroup owns.
        let a = vec4<f32>(Input[ob], Input[ob + 1u], Input[ob + 2u], Input[ob + 3u]);
        let base = row0 * words_per_row + blk * Q8_WORDS;

        p0 += dot(a, q8_unit_weights(base, jg));
        if (row0 + 1u < info.N) {
            p1 += dot(a, q8_unit_weights(base + words_per_row, jg));
        }
        if (row0 + 2u < info.N) {
            p2 += dot(a, q8_unit_weights(base + words_per_row * 2u, jg));
        }
        if (row0 + 3u < info.N) {
            p3 += dot(a, q8_unit_weights(base + words_per_row * 3u, jg));
        }
    }

    wg_sum4[tid] = vec4<f32>(p0, p1, p2, p3);
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sum4[tid] += wg_sum4[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let r = wg_sum4[0];
        Output[row0] = r.x;
        if (row0 + 1u < info.N) { Output[row0 + 1u] = r.y; }
        if (row0 + 2u < info.N) { Output[row0 + 2u] = r.z; }
        if (row0 + 3u < info.N) { Output[row0 + 3u] = r.w; }
    }
}


// Q6_K GEMV
//
// Padded block layout, 53 u32 words:
//   words  0..31 : ql[128], the low 4 bits of each weight
//   words 32..47 : qh[64],  the high 2 bits, four weights per byte
//   words 48..51 : scales[16], signed 8-bit
//   word     52  : d (f16) in the low half
//
// Each work unit covers four consecutive `j` within one half-block, which is 16
// weights drawn from one ql word, one ql word 32 bytes on, and one qh word. Four
// consecutive `j` always share the same scale index, so the scales are read once.
// Unpacking matches `dequantize_q6_k_block` exactly.
// ---------------------------------------------------------------------------

const Q6K_WORDS: u32 = 53u;

fn q6k_scale(base: u32, idx: u32) -> f32 {
    let word = W_Q6K[base + 48u + (idx >> 2u)];
    let b = (word >> ((idx & 3u) * 8u)) & 0xFFu;
    return f32(i32(b << 24u) >> 24u); // scales are signed
}

struct Q6KQuad {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
    d: vec4<f32>,
}

fn q6k_unit_weights(base: u32, i: u32, jg: u32) -> Q6KQuad {
    let d = unpack2x16float(W_Q6K[base + 52u]).x;
    let scb = i * 8u + jg / 4u;
    let s0 = d * q6k_scale(base, scb);
    let s1 = d * q6k_scale(base, scb + 2u);
    let s2 = d * q6k_scale(base, scb + 4u);
    let s3 = d * q6k_scale(base, scb + 6u);

    let qlw = W_Q6K[base + i * 16u + jg];
    let qlw2 = W_Q6K[base + i * 16u + jg + 8u];
    let qhw = W_Q6K[base + 32u + i * 8u + jg];

    // Each mask keeps the two high bits in place for its quarter, so the OR lands
    // them directly on top of the low nibble without any per-byte shifting.
    let n0 = unpack4xU8((qlw & 0x0F0F0F0Fu) | ((qhw & 0x03030303u) << 4u));
    let n1 = unpack4xU8((qlw2 & 0x0F0F0F0Fu) | ((qhw & 0x0C0C0C0Cu) << 2u));
    let n2 = unpack4xU8(((qlw >> 4u) & 0x0F0F0F0Fu) | (qhw & 0x30303030u));
    let n3 = unpack4xU8(((qlw2 >> 4u) & 0x0F0F0F0Fu) | ((qhw & 0xC0C0C0C0u) >> 2u));

    let bias = vec4<f32>(32.0);
    var o: Q6KQuad;
    o.a = (vec4<f32>(n0) - bias) * s0;
    o.b = (vec4<f32>(n1) - bias) * s1;
    o.c = (vec4<f32>(n2) - bias) * s2;
    o.d = (vec4<f32>(n3) - bias) * s3;
    return o;
}

@compute @workgroup_size(64)
fn gemv_q6k(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>
) {
    let row0 = (wg_id.y * grid.x + wg_id.x) * ROWS_PER_WG;
    let tid = local_id.x;
    if (row0 >= info.N) { return; }

    let blocks_per_row = info.K / 256u;
    let words_per_row = blocks_per_row * Q6K_WORDS;
    let total_units = blocks_per_row * 16u;

    var p0 = 0.0;
    var p1 = 0.0;
    var p2 = 0.0;
    var p3 = 0.0;

    for (var u = tid; u < total_units; u = u + 64u) {
        let blk = u / 16u;
        let ub = u % 16u;
        let i = ub / 8u;
        let jg = ub % 8u;
        let ob = blk * 256u + i * 128u + jg * 4u;

        // Loaded once, reused by every row this workgroup owns.
        let a0 = vec4<f32>(Input[ob], Input[ob + 1u], Input[ob + 2u], Input[ob + 3u]);
        let a1 = vec4<f32>(Input[ob + 32u], Input[ob + 33u], Input[ob + 34u], Input[ob + 35u]);
        let a2 = vec4<f32>(Input[ob + 64u], Input[ob + 65u], Input[ob + 66u], Input[ob + 67u]);
        let a3 = vec4<f32>(Input[ob + 96u], Input[ob + 97u], Input[ob + 98u], Input[ob + 99u]);

        let base = row0 * words_per_row + blk * Q6K_WORDS;

        let w0 = q6k_unit_weights(base, i, jg);
        p0 += dot(a0, w0.a) + dot(a1, w0.b) + dot(a2, w0.c) + dot(a3, w0.d);
        if (row0 + 1u < info.N) {
            let w1 = q6k_unit_weights(base + words_per_row, i, jg);
            p1 += dot(a0, w1.a) + dot(a1, w1.b) + dot(a2, w1.c) + dot(a3, w1.d);
        }
        if (row0 + 2u < info.N) {
            let w2 = q6k_unit_weights(base + words_per_row * 2u, i, jg);
            p2 += dot(a0, w2.a) + dot(a1, w2.b) + dot(a2, w2.c) + dot(a3, w2.d);
        }
        if (row0 + 3u < info.N) {
            let w3 = q6k_unit_weights(base + words_per_row * 3u, i, jg);
            p3 += dot(a0, w3.a) + dot(a1, w3.b) + dot(a2, w3.c) + dot(a3, w3.d);
        }
    }

    wg_sum4[tid] = vec4<f32>(p0, p1, p2, p3);
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sum4[tid] += wg_sum4[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let r = wg_sum4[0];
        Output[row0] = r.x;
        if (row0 + 1u < info.N) { Output[row0 + 1u] = r.y; }
        if (row0 + 2u < info.N) { Output[row0 + 2u] = r.z; }
        if (row0 + 3u < info.N) { Output[row0 + 3u] = r.w; }
    }
}
