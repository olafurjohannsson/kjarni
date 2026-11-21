// Quantised GEMVs on int8 activations.
//
// `quantize_q8` turns each row of activations into int8 in blocks of 32, one
// scale a block, with the sums of each half block kept for the weights' minimums.
// The GEMVs then multiply four weights by four activations at a time with
// dot4I8Packed. The result differs from the f32 kernels by the activations'
// rounding, about 0.5% relative, as llama.cpp's int8 path does.

struct LinearInfo {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0) var<uniform> info: LinearInfo;
@group(0) @binding(1) var<storage, read> Input: array<f32>;            // [M, K]
@group(0) @binding(2) var<storage, read> W_Q4K4: array<vec4<u32>>;     // Q4_K blocks, 9 vec4 each
@group(0) @binding(3) var<storage, read_write> Act: array<u32>;        // [M, K/4] int8 x4
@group(0) @binding(4) var<storage, read_write> Aux: array<vec4<f32>>;  // [M, K/32] scale, sum of first 16, sum of last 16
@group(0) @binding(5) var<storage, read_write> Output: array<f32>;     // [M, N]

const QUANTIZE_THREADS: u32 = 256u;

var<workgroup> qa_max: array<f32, QUANTIZE_THREADS>;
var<workgroup> qa_sum: array<f32, QUANTIZE_THREADS>;
var<workgroup> qa_val: array<f32, QUANTIZE_THREADS>;

// A thread per activation, eight blocks of 32 a workgroup. K is a multiple of
// 256, so no block spans two rows.
@compute @workgroup_size(QUANTIZE_THREADS)
fn quantize_q8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>
) {
    let t = lid.x;
    let total = info.M * info.K;
    let i = (wg.y * grid.x + wg.x) * QUANTIZE_THREADS + t;
    let live = i < total;
    let x = select(0.0, Input[min(i, total - 1u)], live);
    qa_val[t] = x;
    qa_max[t] = abs(x);
    qa_sum[t] = x;
    workgroupBarrier();
    for (var s = 8u; s > 0u; s >>= 1u) {
        if ((t % 16u) < s) {
            qa_max[t] = max(qa_max[t], qa_max[t + s]);
            qa_sum[t] += qa_sum[t + s];
        }
        workgroupBarrier();
    }
    let first = t - t % 32u;
    let amax = max(qa_max[first], qa_max[first + 16u]);
    let scale = amax / 127.0;
    let inv = select(0.0, 1.0 / scale, scale > 0.0);
    if (live && t % 4u == 0u) {
        let v = vec4<f32>(qa_val[t], qa_val[t + 1u], qa_val[t + 2u], qa_val[t + 3u]);
        Act[i / 4u] = pack4xI8(vec4<i32>(round(v * inv)));
    }
    if (live && t % 32u == 0u) {
        Aux[i / 32u] = vec4<f32>(scale, qa_sum[t], qa_sum[t + 16u], 0.0);
    }
}

const Q4K_VEC4S: u32 = 9u;
const ROWS_PER_WG: u32 = 8u;
const MAX_THREADS: u32 = 128u;

var<workgroup> red_a: array<vec4<f32>, MAX_THREADS>;
var<workgroup> red_b: array<vec4<f32>, MAX_THREADS>;

// The scale and minimum of sub-blocks 2j and 2j+1, as d*scale, dmin*min,
// d*scale, dmin*min. See q4k_header_scales in linear.wgsl for the layout.
fn q4k_scales(h: vec4<u32>, j: u32) -> vec4<f32> {
    let dd = unpack2x16float(h.x);
    let first = j < 2u;
    let scales = select(
        (h.w & 0x0F0F0F0Fu) | ((h.y >> 2u) & 0x30303030u),
        h.y & 0x3F3F3F3Fu,
        first,
    );
    let mins = select(
        ((h.w >> 4u) & 0x0F0F0F0Fu) | ((h.z >> 2u) & 0x30303030u),
        h.z & 0x3F3F3F3Fu,
        first,
    );
    let shift = (j & 1u) * 16u;
    let sc = scales >> shift;
    let mn = mins >> shift;
    return vec4<f32>(
        f32(sc & 0xFFu) * dd.x,
        f32(mn & 0xFFu) * dd.y,
        f32((sc >> 8u) & 0xFFu) * dd.x,
        f32((mn >> 8u) & 0xFFu) * dd.y,
    );
}

fn low_dot(q: vec4<u32>, a: vec4<u32>) -> i32 {
    let m = 0x0F0F0F0Fu;
    return dot4I8Packed(q.x & m, a.x) + dot4I8Packed(q.y & m, a.y)
        + dot4I8Packed(q.z & m, a.z) + dot4I8Packed(q.w & m, a.w);
}

fn high_dot(q: vec4<u32>, a: vec4<u32>) -> i32 {
    let m = 0x0F0F0F0Fu;
    return dot4I8Packed((q.x >> 4u) & m, a.x) + dot4I8Packed((q.y >> 4u) & m, a.y)
        + dot4I8Packed((q.z >> 4u) & m, a.z) + dot4I8Packed((q.w >> 4u) & m, a.w);
}

struct Chunk {
    a_lo: vec4<u32>,
    a_hi: vec4<u32>,
    s_lo: f32,
    s_hi: f32,
    sum_lo: f32,
    sum_hi: f32,
    j: u32,
    at: u32,
}

// One row's share of a chunk: 16 weights of sub-block 2j and the 16 of 2j+1
// that sit in the same bytes.
fn q4k_row(base: u32, c: Chunk) -> f32 {
    let h = q4k_scales(W_Q4K4[base], c.j);
    let q = W_Q4K4[base + c.at];
    return h.x * c.s_lo * f32(low_dot(q, c.a_lo)) - h.y * c.sum_lo
        + h.z * c.s_hi * f32(high_dot(q, c.a_hi)) - h.w * c.sum_hi;
}

// Eight rows a workgroup, as gemv_q4k in linear.wgsl. Each chunk is one vec4
// of a block's quantised weights: 16 bytes holding 16 weights of two sub-blocks.
fn gemv_q4k_rows(wg: vec3<u32>, grid: vec3<u32>, tid: u32, threads: u32) {
    let row0 = (wg.y * grid.x + wg.x) * ROWS_PER_WG;
    if (row0 >= info.N) { return; }
    let blocks = info.K / 256u;
    let per_row = blocks * Q4K_VEC4S;
    let last = info.N - 1u;
    let ra = vec4<u32>(row0, min(row0 + 1u, last), min(row0 + 2u, last), min(row0 + 3u, last)) * per_row;
    let rb = vec4<u32>(min(row0 + 4u, last), min(row0 + 5u, last), min(row0 + 6u, last), min(row0 + 7u, last)) * per_row;
    let act0 = wg.z * (info.K / 4u);
    let aux0 = wg.z * (info.K / 32u);

    var acc_a = vec4<f32>(0.0);
    var acc_b = vec4<f32>(0.0);
    for (var chunk = tid; chunk < blocks * 8u; chunk += threads) {
        let blk = chunk / 8u;
        let j = (chunk % 8u) / 2u;
        let half = chunk % 2u;
        let p = blk * 256u + j * 64u;
        let w4 = act0 + (p + half * 16u) / 4u;
        let x_lo = Aux[aux0 + p / 32u];
        let x_hi = Aux[aux0 + p / 32u + 1u];
        var c: Chunk;
        c.a_lo = vec4<u32>(Act[w4], Act[w4 + 1u], Act[w4 + 2u], Act[w4 + 3u]);
        c.a_hi = vec4<u32>(Act[w4 + 8u], Act[w4 + 9u], Act[w4 + 10u], Act[w4 + 11u]);
        c.s_lo = x_lo.x;
        c.s_hi = x_hi.x;
        c.sum_lo = select(x_lo.y, x_lo.z, half == 1u);
        c.sum_hi = select(x_hi.y, x_hi.z, half == 1u);
        c.j = j;
        c.at = 1u + chunk % 8u;
        let off = blk * Q4K_VEC4S;
        acc_a += vec4<f32>(q4k_row(ra.x + off, c), q4k_row(ra.y + off, c), q4k_row(ra.z + off, c), q4k_row(ra.w + off, c));
        acc_b += vec4<f32>(q4k_row(rb.x + off, c), q4k_row(rb.y + off, c), q4k_row(rb.z + off, c), q4k_row(rb.w + off, c));
    }

    red_a[tid] = acc_a;
    red_b[tid] = acc_b;
    workgroupBarrier();
    for (var s = threads / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            red_a[tid] += red_a[tid + s];
            red_b[tid] += red_b[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let a = red_a[0];
        let b = red_b[0];
        let out0 = wg.z * info.N + row0;
        for (var r = 0u; r < 4u; r++) {
            if (row0 + r < info.N) { Output[out0 + r] = a[r]; }
            if (row0 + 4u + r < info.N) { Output[out0 + 4u + r] = b[r]; }
        }
    }
}

@compute @workgroup_size(32)
fn gemv_q4k_q8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>
) {
    gemv_q4k_rows(wg, grid, lid.x, 32u);
}

// For few outputs from long rows: four warps share the eight rows.
@compute @workgroup_size(MAX_THREADS)
fn gemv_q4k_q8_split(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>
) {
    gemv_q4k_rows(wg, grid, lid.x, MAX_THREADS);
}
