//! WASM SIMD128 kernels for matrix operations.

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

#[cfg(target_arch = "wasm32")]
use crate::cpu::kernels::q_common::BlockQ8_0;

/// Computes dot product of two f32 slices using WASM SIMD128.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn wasm_dot_product(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: the caller guarantees the slices are valid and
    // correctly aligned; this body only reads through them.
    unsafe {
        let n = a.len();
        debug_assert_eq!(n, b.len());

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut sum0 = f32x4_splat(0.0);
        let mut sum1 = f32x4_splat(0.0);
        let mut sum2 = f32x4_splat(0.0);
        let mut sum3 = f32x4_splat(0.0);

        let mut i = 0;

        // Main loop: 16 elements per iteration
        while i + 16 <= n {
            let a0 = v128_load(a_ptr.add(i) as *const v128);
            let a1 = v128_load(a_ptr.add(i + 4) as *const v128);
            let a2 = v128_load(a_ptr.add(i + 8) as *const v128);
            let a3 = v128_load(a_ptr.add(i + 12) as *const v128);

            let b0 = v128_load(b_ptr.add(i) as *const v128);
            let b1 = v128_load(b_ptr.add(i + 4) as *const v128);
            let b2 = v128_load(b_ptr.add(i + 8) as *const v128);
            let b3 = v128_load(b_ptr.add(i + 12) as *const v128);

            sum0 = f32x4_add(sum0, f32x4_mul(a0, b0));
            sum1 = f32x4_add(sum1, f32x4_mul(a1, b1));
            sum2 = f32x4_add(sum2, f32x4_mul(a2, b2));
            sum3 = f32x4_add(sum3, f32x4_mul(a3, b3));

            i += 16;
        }

        // Handle 4-element chunks
        while i + 4 <= n {
            let a0 = v128_load(a_ptr.add(i) as *const v128);
            let b0 = v128_load(b_ptr.add(i) as *const v128);
            sum0 = f32x4_add(sum0, f32x4_mul(a0, b0));
            i += 4;
        }

        // Combine accumulators
        sum0 = f32x4_add(f32x4_add(sum0, sum1), f32x4_add(sum2, sum3));

        // Horizontal sum
        let mut result = f32x4_extract_lane::<0>(sum0)
            + f32x4_extract_lane::<1>(sum0)
            + f32x4_extract_lane::<2>(sum0)
            + f32x4_extract_lane::<3>(sum0);

        // Scalar remainder
        while i < n {
            result += *a_ptr.add(i) * *b_ptr.add(i);
            i += 1;
        }

        result
    }
}

/// Reduces a 4 lane vector to one float with a shuffle tree.
///
/// Two shuffles and two adds beats four `extract_lane` calls plus three scalar
/// adds: fewer operations, and the tree halves the dependency chain.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn hsum_f32x4(v: v128) -> f32 {
    let swapped = i32x4_shuffle::<2, 3, 0, 1>(v, v);
    let s = f32x4_add(v, swapped);
    let swapped2 = i32x4_shuffle::<1, 0, 3, 2>(s, s);
    f32x4_extract_lane::<0>(f32x4_add(s, swapped2))
}

/// Computes a [4 rows x 3 columns] block of the output.
///
/// The dot-product form computes one output at a time: two loads for every
/// multiply-add, so the kernel is bound by loads rather than arithmetic. Holding
/// twelve accumulators and reusing each weight vector across four input rows,
/// and each input row across three weight vectors, makes it seven loads for
/// twelve multiply-adds instead.
///
/// The shape is chosen to fit a 16 register file, which is what SIMD128 maps to
/// on the hosts that matter: twelve accumulators, three weight vectors live, and
/// one input vector streaming through is exactly 16. Holding the four input
/// vectors as well would be 19 and would spill.
///
/// SIMD128 has no fused multiply-add, so this is a mul and an add where the AVX2
/// equivalent issues one instruction. The load reduction still applies in full.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn wasm_block_4x3(
    out_ptr: *mut f32,
    out_stride: usize,
    a_ptr: *const f32,
    b_ptr: *const f32,
    k: usize,
) {
    unsafe {
        let z = f32x4_splat(0.0);
        let (mut c00, mut c01, mut c02) = (z, z, z);
        let (mut c10, mut c11, mut c12) = (z, z, z);
        let (mut c20, mut c21, mut c22) = (z, z, z);
        let (mut c30, mut c31, mut c32) = (z, z, z);

        let mut a0 = a_ptr;
        let mut a1 = a_ptr.add(k);
        let mut a2 = a_ptr.add(2 * k);
        let mut a3 = a_ptr.add(3 * k);
        let mut b0 = b_ptr;
        let mut b1 = b_ptr.add(k);
        let mut b2 = b_ptr.add(2 * k);

        let mut rem = k;
        while rem >= 4 {
            let w0 = v128_load(b0 as *const v128);
            let w1 = v128_load(b1 as *const v128);
            let w2 = v128_load(b2 as *const v128);

            let av = v128_load(a0 as *const v128);
            c00 = f32x4_add(c00, f32x4_mul(av, w0));
            c01 = f32x4_add(c01, f32x4_mul(av, w1));
            c02 = f32x4_add(c02, f32x4_mul(av, w2));

            let av = v128_load(a1 as *const v128);
            c10 = f32x4_add(c10, f32x4_mul(av, w0));
            c11 = f32x4_add(c11, f32x4_mul(av, w1));
            c12 = f32x4_add(c12, f32x4_mul(av, w2));

            let av = v128_load(a2 as *const v128);
            c20 = f32x4_add(c20, f32x4_mul(av, w0));
            c21 = f32x4_add(c21, f32x4_mul(av, w1));
            c22 = f32x4_add(c22, f32x4_mul(av, w2));

            let av = v128_load(a3 as *const v128);
            c30 = f32x4_add(c30, f32x4_mul(av, w0));
            c31 = f32x4_add(c31, f32x4_mul(av, w1));
            c32 = f32x4_add(c32, f32x4_mul(av, w2));

            a0 = a0.add(4);
            a1 = a1.add(4);
            a2 = a2.add(4);
            a3 = a3.add(4);
            b0 = b0.add(4);
            b1 = b1.add(4);
            b2 = b2.add(4);
            rem -= 4;
        }

        let mut r = [
            [hsum_f32x4(c00), hsum_f32x4(c01), hsum_f32x4(c02)],
            [hsum_f32x4(c10), hsum_f32x4(c11), hsum_f32x4(c12)],
            [hsum_f32x4(c20), hsum_f32x4(c21), hsum_f32x4(c22)],
            [hsum_f32x4(c30), hsum_f32x4(c31), hsum_f32x4(c32)],
        ];

        // Tail when k is not a multiple of 4.
        let aa = [a0, a1, a2, a3];
        let bb = [b0, b1, b2];
        for i in 0..rem {
            for (row, ap) in aa.iter().enumerate() {
                let av = *ap.add(i);
                for (col, bp) in bb.iter().enumerate() {
                    r[row][col] += av * *bp.add(i);
                }
            }
        }

        for (row, vals) in r.iter().enumerate() {
            let dst = out_ptr.add(row * out_stride);
            *dst = vals[0];
            *dst.add(1) = vals[1];
            *dst.add(2) = vals[2];
        }
    }
}

/// Vector-matrix multiply against Q8_0 weights, using SIMD128.
///
/// The wasm build had no quantised kernel at all, so every `.kjq` model that
/// stays block-quantised, which is what the browser chat ships, ran through the
/// scalar fallback: one multiply and one add per weight, no vectorisation.
///
/// Q8_0 stores one f16 scale per 32 int8 weights. Dequantising is an extend from
/// i8 to i32 and a convert to f32; the scale is folded in once per block rather
/// than once per weight. Four accumulators are rotated so no add waits on the one
/// before it, which is what a single accumulator would force.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn wasm_matmul_vec_q8_0(
    out_chunk: &mut [f32],
    a_ptr: *const f32,
    b_blocks: &[BlockQ8_0],
    k: usize,
) {
    unsafe {
        let blocks_per_row = k / 32;

        for (i, out_val) in out_chunk.iter_mut().enumerate() {
            let row = &b_blocks[i * blocks_per_row..(i + 1) * blocks_per_row];

            let mut acc0 = f32x4_splat(0.0);
            let mut acc1 = f32x4_splat(0.0);
            let mut acc2 = f32x4_splat(0.0);
            let mut acc3 = f32x4_splat(0.0);

            let mut a = a_ptr;
            for block in row {
                let d = f32x4_splat(block.d.to_f32());
                let q = block.qs.as_ptr();

                // 32 int8 weights arrive as two 16 byte loads, each widening to
                // four f32x4 vectors.
                for half in 0..2 {
                    let raw = v128_load(q.add(half * 16) as *const v128);
                    let lo = i16x8_extend_low_i8x16(raw);
                    let hi = i16x8_extend_high_i8x16(raw);

                    let w0 = f32x4_mul(f32x4_convert_i32x4(i32x4_extend_low_i16x8(lo)), d);
                    let w1 = f32x4_mul(f32x4_convert_i32x4(i32x4_extend_high_i16x8(lo)), d);
                    let w2 = f32x4_mul(f32x4_convert_i32x4(i32x4_extend_low_i16x8(hi)), d);
                    let w3 = f32x4_mul(f32x4_convert_i32x4(i32x4_extend_high_i16x8(hi)), d);

                    let base = a.add(half * 16);
                    acc0 = f32x4_add(acc0, f32x4_mul(v128_load(base as *const v128), w0));
                    acc1 = f32x4_add(acc1, f32x4_mul(v128_load(base.add(4) as *const v128), w1));
                    acc2 = f32x4_add(acc2, f32x4_mul(v128_load(base.add(8) as *const v128), w2));
                    acc3 = f32x4_add(acc3, f32x4_mul(v128_load(base.add(12) as *const v128), w3));
                }

                a = a.add(32);
            }

            let sum = f32x4_add(f32x4_add(acc0, acc1), f32x4_add(acc2, acc3));
            *out_val = hsum_f32x4(sum);
        }
    }
}

/// Computes C = A @ B^T using WASM SIMD128.
/// A is [m, k], B is [n, k] (row-major, transposed), C is [m, n].
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn wasm_matmul_2d(out: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize) {
    // SAFETY: the caller guarantees the slices are valid and
    // correctly aligned; this body only reads through them.
    unsafe {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), n * k);
        debug_assert_eq!(out.len(), m * n);

        let mut row = 0;
        while row + 4 <= m {
            let a_ptr = a.as_ptr().add(row * k);
            let mut col = 0;
            while col + 3 <= n {
                wasm_block_4x3(
                    out.as_mut_ptr().add(row * n + col),
                    n,
                    a_ptr,
                    b.as_ptr().add(col * k),
                    k,
                );
                col += 3;
            }
            // Columns past a multiple of three.
            while col < n {
                let b_row = &b[col * k..(col + 1) * k];
                for r in 0..4 {
                    let a_row = &a[(row + r) * k..(row + r + 1) * k];
                    out[(row + r) * n + col] = wasm_dot_product(a_row, b_row);
                }
                col += 1;
            }
            row += 4;
        }

        // Rows past a multiple of four.
        while row < m {
            let a_row = &a[row * k..(row + 1) * k];
            for col in 0..n {
                let b_row = &b[col * k..(col + 1) * k];
                out[row * n + col] = wasm_dot_product(a_row, b_row);
            }
            row += 1;
        }
    }
}

/// Computes C = A @ B (not transposed) using WASM SIMD128.
/// A is [m, k], B is [k, n] (row-major), C is [m, n].
/// Transposes B internally then delegates to wasm_matmul_2d.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[allow(dead_code, reason = "reachable only on some targets")]
pub unsafe fn wasm_matmul_2d_nn(
    out: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);

    let mut b_t = vec![0.0f32; n * k];
    for i in 0..k {
        for j in 0..n {
            b_t[j * k + i] = b[i * n + j];
        }
    }

    // SAFETY: the caller guarantees the slices are valid and correctly sized;
    // b_t is built here and is correct by construction.
    unsafe {
        wasm_matmul_2d(out, a, &b_t, m, n, k);
    }
}
