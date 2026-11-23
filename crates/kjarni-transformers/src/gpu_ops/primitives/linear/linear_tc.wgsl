// Quantised GEMMs on tensor cores, for prefill.
//
// Compiled after linear.wgsl, with `enable f16` and
// `enable wgpu_cooperative_matrix` before both, so these kernels use its
// bindings and its four-weight dequantisers. Each step converts a 64 x 32 tile
// of activations and a 64 x 32 tile of weights to f16 in workgroup memory, and
// four warps multiply them in 16 x 16 x 16 cooperative matrices with f32
// accumulators. Each warp owns a 32 x 32 quarter of the 64 x 64 output tile.
//
// The result differs from the f32 GEMMs by rounding activations and weights to
// f16, as llama.cpp's tensor-core path does. Activations beyond f16's range are
// clamped to it.

const TC_BM: u32 = 64u;
const TC_BN: u32 = 64u;
const TC_BK: u32 = 32u;
// Row stride of the f16 tiles in workgroup memory, padded past TC_BK so the
// rows a cooperative load reads fall in different banks.
const TC_LD: u32 = 40u;
const TC_THREADS: u32 = 128u;
const TC_F16_MAX: f32 = 65504.0;

var<workgroup> tc_a: array<f16, 2560>;   // [TC_BM][TC_LD]
var<workgroup> tc_w: array<f16, 2560>;   // [TC_BN][TC_LD]
var<workgroup> tc_out: array<f32, 1024>; // a 16 x 16 tile per warp, for edge tiles

fn tc_weights(mode: u32, n: u32, k: u32) -> vec4<f32> {
    if (mode == GEMM_Q4K) { return q4k_four(n, k); }
    if (mode == GEMM_Q5K) { return q5k_four(n, k); }
    if (mode == GEMM_Q6K) { return q6k_four(n, k); }
    return q8_four(n, k);
}

fn tc_half4(v: vec4<f32>) -> vec4<f16> {
    return vec4<f16>(clamp(v, vec4<f32>(-TC_F16_MAX), vec4<f32>(TC_F16_MAX)));
}

// The cooperative stores take their stride as a parameter: naga 30's SPIR-V
// writer fails on a stride that is a literal or a uniform's field.
fn tc_store_out(c: coop_mat16x16<f32, C>, at: u32, stride: u32) {
    coopStoreT(c, &Output[at], stride);
}

fn tc_store_scratch(c: coop_mat16x16<f32, C>, at: u32, stride: u32) {
    coopStoreT(c, &tc_out[at], stride);
}

// Writes one warp's 16 x 16 tile at (r0, c0) through tc_out, dropping the
// rows and columns past the output's edge.
fn tc_store_edge(c: coop_mat16x16<f32, C>, warp: u32, lane: u32, r0: u32, c0: u32) {
    tc_store_scratch(c, warp * 256u, 16u);
    workgroupBarrier();
    for (var e = 0u; e < 8u; e++) {
        let idx = lane + e * 32u;
        let r = r0 + idx / 16u;
        let col = c0 + idx % 16u;
        if (r < info.M && col < info.N) {
            Output[r * info.N + col] = tc_out[warp * 256u + idx];
        }
    }
    workgroupBarrier();
}

fn tc_tile(wg: vec3<u32>, tid: u32, mode: u32) {
    let row0 = wg.y * TC_BM;
    let col0 = wg.x * TC_BN;
    let warp = tid / 32u;
    let lane = tid % 32u;
    let wm = (warp / 2u) * 32u;
    let wn = (warp % 2u) * 32u;

    var c00: coop_mat16x16<f32, C>;
    var c01: coop_mat16x16<f32, C>;
    var c10: coop_mat16x16<f32, C>;
    var c11: coop_mat16x16<f32, C>;

    for (var k0 = 0u; k0 < info.K; k0 += TC_BK) {
        // 64 rows x 32 values is 512 groups of four in each operand, four per
        // thread. Eight consecutive threads cover one row's 32 values.
        for (var l = 0u; l < 4u; l++) {
            let idx = tid + l * TC_THREADS;
            let r = idx / 8u;
            let q = (idx % 8u) * 4u;

            let gr = row0 + r;
            var a = vec4<f32>(0.0);
            if (gr < info.M) { a = Input4[(gr * info.K + k0 + q) / 4u]; }
            let ah = tc_half4(a);
            let at = r * TC_LD + q;
            tc_a[at] = ah.x;
            tc_a[at + 1u] = ah.y;
            tc_a[at + 2u] = ah.z;
            tc_a[at + 3u] = ah.w;

            let gn = col0 + r;
            var w = vec4<f32>(0.0);
            if (gn < info.N) { w = tc_weights(mode, gn, k0 + q); }
            let wh = vec4<f16>(w);
            tc_w[at] = wh.x;
            tc_w[at + 1u] = wh.y;
            tc_w[at + 2u] = wh.z;
            tc_w[at + 3u] = wh.w;
        }
        workgroupBarrier();

        for (var kk = 0u; kk < TC_BK; kk += 16u) {
            // A is row-major [m][k]; the weight tile [n][k] is B = W^T in
            // column-major order.
            let a0 = coopLoadT<coop_mat16x16<f16, A>>(&tc_a[wm * TC_LD + kk], TC_LD);
            let a1 = coopLoadT<coop_mat16x16<f16, A>>(&tc_a[(wm + 16u) * TC_LD + kk], TC_LD);
            let b0 = coopLoad<coop_mat16x16<f16, B>>(&tc_w[wn * TC_LD + kk], TC_LD);
            let b1 = coopLoad<coop_mat16x16<f16, B>>(&tc_w[(wn + 16u) * TC_LD + kk], TC_LD);
            c00 = coopMultiplyAdd(a0, b0, c00);
            c01 = coopMultiplyAdd(a0, b1, c01);
            c10 = coopMultiplyAdd(a1, b0, c10);
            c11 = coopMultiplyAdd(a1, b1, c11);
        }
        workgroupBarrier();
    }

    let s00 = c00;
    let s01 = c01;
    let s10 = c10;
    let s11 = c11;
    let r0 = row0 + wm;
    let c0 = col0 + wn;
    if (row0 + TC_BM <= info.M && col0 + TC_BN <= info.N) {
        tc_store_out(s00, r0 * info.N + c0, info.N);
        tc_store_out(s01, r0 * info.N + c0 + 16u, info.N);
        tc_store_out(s10, (r0 + 16u) * info.N + c0, info.N);
        tc_store_out(s11, (r0 + 16u) * info.N + c0 + 16u, info.N);
    } else {
        tc_store_edge(s00, warp, lane, r0, c0);
        tc_store_edge(s01, warp, lane, r0, c0 + 16u);
        tc_store_edge(s10, warp, lane, r0 + 16u, c0);
        tc_store_edge(s11, warp, lane, r0 + 16u, c0 + 16u);
    }
}

@compute @workgroup_size(TC_THREADS)
fn tc_gemm_q4k(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    tc_tile(wg, tid, GEMM_Q4K);
}

@compute @workgroup_size(TC_THREADS)
fn tc_gemm_q5k(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    tc_tile(wg, tid, GEMM_Q5K);
}

@compute @workgroup_size(TC_THREADS)
fn tc_gemm_q6k(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    tc_tile(wg, tid, GEMM_Q6K);
}

@compute @workgroup_size(TC_THREADS)
fn tc_gemm_q8(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    tc_tile(wg, tid, GEMM_Q8);
}
