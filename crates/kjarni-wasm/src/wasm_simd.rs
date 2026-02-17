// wasm_simd.rs — WASM SIMD128 kernels for encoder matmul
//
// Two matmul variants:
//   wasm_matmul_2d:    C = A @ B^T  (B is (n, k) row-major — weight matrix layout)
//   wasm_matmul_2d_nn: C = A @ B    (B is (k, n) row-major — standard matmul)

use std::arch::wasm32::*;

/// Dot product of two f32 slices using WASM SIMD128.
/// 4x unrolled (16 floats per iteration) to hide latency.
#[target_feature(enable = "simd128")]
pub unsafe fn wasm_dot_product(a: &[f32], b: &[f32]) -> f32 {
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

/// C = A @ B^T where A is (m, k), B is (n, k), C is (m, n).
/// B rows are contiguous — each output is a dot product of an A row with a B row.
/// This is the layout for weight matrices stored as (out_features, in_features).
#[target_feature(enable = "simd128")]
pub unsafe fn wasm_matmul_2d(
    out: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(out.len(), m * n);

    for row in 0..m {
        let a_row = &a[row * k..(row + 1) * k];
        for col in 0..n {
            let b_row = &b[col * k..(col + 1) * k];
            out[row * n + col] = wasm_dot_product(a_row, b_row);
        }
    }
}

#[target_feature(enable = "simd128")]
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

    // Now use the fast A @ B^T path
    wasm_matmul_2d(out, a, &b_t, m, n, k);
}