//! AVX2/FMA accelerated kernels for F32 weights.

#![allow(unsafe_code)]
use super::common::hsum_ps_avx;
use std::arch::x86_64::*;

/// Computes a [4 x 3] block matrix multiplication.
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn matmul_block_4x3_f32(
    out_ptr: *mut f32,       // Pointer to Output [row, col]
    out_stride: usize,       // Output stride (usually hidden_dim)
    a_ptr: *const f32,       // Input buffer (4 rows)
    b_ptr: *const f32,       // Weight buffer (3 rows)
    k: usize,                // Hidden dim
    bias_ptr: *const f32,    // Optional bias pointer (len 3)
) { unsafe {
    let zero = _mm256_setzero_ps();
    let (mut c00, mut c01, mut c02) = (zero, zero, zero);
    let (mut c10, mut c11, mut c12) = (zero, zero, zero);
    let (mut c20, mut c21, mut c22) = (zero, zero, zero);
    let (mut c30, mut c31, mut c32) = (zero, zero, zero);

    let mut a0_ptr = a_ptr;
    let mut a1_ptr = a_ptr.add(k);
    let mut a2_ptr = a_ptr.add(2 * k);
    let mut a3_ptr = a_ptr.add(3 * k);

    let mut b0_ptr = b_ptr;
    let mut b1_ptr = b_ptr.add(k);
    let mut b2_ptr = b_ptr.add(2 * k);

    let mut n = k;

    while n >= 8 {
        let a0 = _mm256_loadu_ps(a0_ptr);
        let a1 = _mm256_loadu_ps(a1_ptr);
        let a2 = _mm256_loadu_ps(a2_ptr);
        let a3 = _mm256_loadu_ps(a3_ptr);

        let w0 = _mm256_loadu_ps(b0_ptr);
        let w0 = _mm256_loadu_ps(b0_ptr);
        c00 = _mm256_fmadd_ps(a0, w0, c00);
        c10 = _mm256_fmadd_ps(a1, w0, c10);
        c20 = _mm256_fmadd_ps(a2, w0, c20);
        c30 = _mm256_fmadd_ps(a3, w0, c30);

        let w1 = _mm256_loadu_ps(b1_ptr);
        c01 = _mm256_fmadd_ps(a0, w1, c01);
        c11 = _mm256_fmadd_ps(a1, w1, c11);
        c21 = _mm256_fmadd_ps(a2, w1, c21);
        c31 = _mm256_fmadd_ps(a3, w1, c31);

        let w2 = _mm256_loadu_ps(b2_ptr);
        c02 = _mm256_fmadd_ps(a0, w2, c02);
        c12 = _mm256_fmadd_ps(a1, w2, c12);
        c22 = _mm256_fmadd_ps(a2, w2, c22);
        c32 = _mm256_fmadd_ps(a3, w2, c32);

        // Advance
        a0_ptr = a0_ptr.add(8);
        a1_ptr = a1_ptr.add(8);
        a2_ptr = a2_ptr.add(8);
        a3_ptr = a3_ptr.add(8);
        b0_ptr = b0_ptr.add(8);
        b1_ptr = b1_ptr.add(8);
        b2_ptr = b2_ptr.add(8);
        n -= 8;
    }

    // Horizontal Sums
    let mut r00 = hsum_ps_avx(c00); let mut r01 = hsum_ps_avx(c01); let mut r02 = hsum_ps_avx(c02);
    let mut r10 = hsum_ps_avx(c10); let mut r11 = hsum_ps_avx(c11); let mut r12 = hsum_ps_avx(c12);
    let mut r20 = hsum_ps_avx(c20); let mut r21 = hsum_ps_avx(c21); let mut r22 = hsum_ps_avx(c22);
    let mut r30 = hsum_ps_avx(c30); let mut r31 = hsum_ps_avx(c31); let mut r32 = hsum_ps_avx(c32);

    // Remainder loop (scalar)
    while n > 0 {
        let va0 = *a0_ptr; let va1 = *a1_ptr; let va2 = *a2_ptr; let va3 = *a3_ptr;
        let wb0 = *b0_ptr; let wb1 = *b1_ptr; let wb2 = *b2_ptr;
        
        r00 += va0 * wb0; r01 += va0 * wb1; r02 += va0 * wb2;
        r10 += va1 * wb0; r11 += va1 * wb1; r12 += va1 * wb2;
        r20 += va2 * wb0; r21 += va2 * wb1; r22 += va2 * wb2;
        r30 += va3 * wb0; r31 += va3 * wb1; r32 += va3 * wb2;

        a0_ptr = a0_ptr.add(1); a1_ptr = a1_ptr.add(1); a2_ptr = a2_ptr.add(1); a3_ptr = a3_ptr.add(1);
        b0_ptr = b0_ptr.add(1); b1_ptr = b1_ptr.add(1); b2_ptr = b2_ptr.add(1);
        n -= 1;
    }

    // Add bias ONCE per output (not 8x via broadcast!)
    if !bias_ptr.is_null() {
        let b0 = *bias_ptr;
        let b1 = *bias_ptr.add(1);
        let b2 = *bias_ptr.add(2);
        
        r00 += b0; r10 += b0; r20 += b0; r30 += b0;
        r01 += b1; r11 += b1; r21 += b1; r31 += b1;
        r02 += b2; r12 += b2; r22 += b2; r32 += b2;
    }

    // Write to Output [Batch, Out]
    // Row 0
    *out_ptr = r00;
    *out_ptr.add(1) = r01;
    *out_ptr.add(2) = r02;

    // Row 1
    let dst1 = out_ptr.add(out_stride);
    *dst1 = r10;
    *dst1.add(1) = r11;
    *dst1.add(2) = r12;

    // Row 2
    let dst2 = out_ptr.add(2 * out_stride);
    *dst2 = r20;
    *dst2.add(1) = r21;
    *dst2.add(2) = r22;

    // Row 3
    let dst3 = out_ptr.add(3 * out_stride);
    *dst3 = r30;
    *dst3.add(1) = r31;
    *dst3.add(2) = r32;
}}

/// Computes a vector-matrix multiplication (vec @ mat) for F32 weights
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn matmul_vec_f32(
    out_chunk: &mut [f32],
    a_ptr: *const f32,
    b_row_start: *const f32,
    k: usize,
) {
    let mut b_row_ptr = b_row_start;
    unsafe {
        for val in out_chunk.iter_mut() {
            let mut a_chunk_ptr = a_ptr;
            let mut b_chunk_ptr = b_row_ptr;

            let mut sum0 = _mm256_setzero_ps();
            let mut sum1 = _mm256_setzero_ps();
            let mut sum2 = _mm256_setzero_ps();
            let mut sum3 = _mm256_setzero_ps();

            let mut n = k;
            while n >= 32 {
                let a0 = _mm256_loadu_ps(a_chunk_ptr);
                let a1 = _mm256_loadu_ps(a_chunk_ptr.add(8));
                let a2 = _mm256_loadu_ps(a_chunk_ptr.add(16));
                let a3 = _mm256_loadu_ps(a_chunk_ptr.add(24));

                // For F32, loading weights is a direct operation.
                let b0 = _mm256_loadu_ps(b_chunk_ptr);
                let b1 = _mm256_loadu_ps(b_chunk_ptr.add(8));
                let b2 = _mm256_loadu_ps(b_chunk_ptr.add(16));
                let b3 = _mm256_loadu_ps(b_chunk_ptr.add(24));

                sum0 = _mm256_fmadd_ps(a0, b0, sum0);
                sum1 = _mm256_fmadd_ps(a1, b1, sum1);
                sum2 = _mm256_fmadd_ps(a2, b2, sum2);
                sum3 = _mm256_fmadd_ps(a3, b3, sum3);

                a_chunk_ptr = a_chunk_ptr.add(32);
                b_chunk_ptr = b_chunk_ptr.add(32);
                n -= 32;
            }

            sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
            let mut sum = hsum_ps_avx(sum0);

            while n > 0 {
                sum += *a_chunk_ptr * *b_chunk_ptr;
                a_chunk_ptr = a_chunk_ptr.add(1);
                b_chunk_ptr = b_chunk_ptr.add(1);
                n -= 1;
            }

            *val = sum;
            b_row_ptr = b_row_ptr.add(k);
        }
    }
}


#[cfg(test)]
mod simd_matmul_tests {
    use super::*;
    use std::time::Instant;

    fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    fn scalar_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let a_row = &a[row * k..(row + 1) * k];
                let b_row = &b[col * k..(col + 1) * k];
                c[row * n + col] = scalar_dot(a_row, b_row);
            }
        }
        c
    }

    fn scalar_matmul_bias(a: &[f32], b: &[f32], bias: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = scalar_matmul(a, b, m, k, n);
        for row in 0..m {
            for col in 0..n {
                c[row * n + col] += bias[col];
            }
        }
        c
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn mean_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        sum / a.len() as f32
    }

    /// Generate deterministic test data
    fn make_input(rows: usize, cols: usize, seed: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| ((i + seed) % 1000) as f32 * 0.001 - 0.5)
            .collect()
    }

    fn make_bias(n: usize, val: f32) -> Vec<f32> {
        (0..n).map(|i| val + i as f32 * 0.001).collect()
    }

    #[test]
    fn test_matmul_vec_f32_tiny() {
        // k=4, n=4 - minimal case
        let k = 4;
        let n = 4;
        let a = make_input(1, k, 0);
        let b = make_input(n, k, 100);

        let expected = scalar_matmul(&a, &b, 1, k, n);
        let mut actual = vec![0.0f32; n];

        unsafe {
            matmul_vec_f32(&mut actual, a.as_ptr(), b.as_ptr(), k);
        }

        let diff = max_diff(&expected, &actual);
        println!("\n=== matmul_vec_f32 (k={}, n={}) ===", k, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-6, "Tiny test failed. Max diff: {}", diff);
    }

    #[test]
    fn test_matmul_vec_f32_small() {
        // k=32, n=64 - fits in cache, tests main loop
        let k = 32;
        let n = 64;
        let a = make_input(1, k, 0);
        let b = make_input(n, k, 100);

        let expected = scalar_matmul(&a, &b, 1, k, n);
        let mut actual = vec![0.0f32; n];

        unsafe {
            matmul_vec_f32(&mut actual, a.as_ptr(), b.as_ptr(), k);
        }

        let diff = max_diff(&expected, &actual);
        println!("\n=== matmul_vec_f32 (k={}, n={}) ===", k, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Small test failed. Max diff: {}", diff);
    }

    #[test]
    fn test_matmul_vec_f32_medium() {
        // k=128, n=256 - realistic hidden dim
        let k = 128;
        let n = 256;
        let a = make_input(1, k, 0);
        let b = make_input(n, k, 100);

        let expected = scalar_matmul(&a, &b, 1, k, n);
        let mut actual = vec![0.0f32; n];

        unsafe {
            matmul_vec_f32(&mut actual, a.as_ptr(), b.as_ptr(), k);
        }

        let diff = max_diff(&expected, &actual);
        println!("\n=== matmul_vec_f32 (k={}, n={}) ===", k, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-4, "Medium test failed. Max diff: {}", diff);
    }

    #[test]
    fn test_matmul_vec_f32_large() {
        // k=384, n=1536 - MiniLM FFN dimensions
        let k = 384;
        let n = 1536;
        let a = make_input(1, k, 0);
        let b = make_input(n, k, 100);

        let expected = scalar_matmul(&a, &b, 1, k, n);
        let mut actual = vec![0.0f32; n];

        unsafe {
            matmul_vec_f32(&mut actual, a.as_ptr(), b.as_ptr(), k);
        }

        let diff = max_diff(&expected, &actual);
        println!("\n=== matmul_vec_f32 (k={}, n={}) ===", k, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-4, "Large test failed. Max diff: {}", diff);
    }

    #[test]
    fn test_matmul_vec_f32_non_aligned() {
        // k=37, n=41 - not divisible by 8 or 32, tests remainder loops
        let k = 37;
        let n = 41;
        let a = make_input(1, k, 0);
        let b = make_input(n, k, 100);

        let expected = scalar_matmul(&a, &b, 1, k, n);
        let mut actual = vec![0.0f32; n];

        unsafe {
            matmul_vec_f32(&mut actual, a.as_ptr(), b.as_ptr(), k);
        }

        let diff = max_diff(&expected, &actual);
        println!("\n=== matmul_vec_f32 NON-ALIGNED (k={}, n={}) ===", k, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Non-aligned test failed. Max diff: {}", diff);
    }

    #[test]
    fn test_block_4x3_tiny_no_bias() {
        // k=4, 4 tokens, 3 outputs - minimal case
        let k = 4;
        let m = 4; // tokens
        let n = 3; // outputs (exactly what the kernel produces)

        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = scalar_matmul(&a, &b, m, k, n);
        let mut actual = vec![0.0f32; m * n];

        unsafe {
            matmul_block_4x3_f32(
                actual.as_mut_ptr(),
                n,              // out_stride
                a.as_ptr(),
                b.as_ptr(),
                k,
                std::ptr::null(), // no bias
            );
        }

        let diff = max_diff(&expected, &actual);
        println!("\n=== matmul_block_4x3 NO BIAS (k={}, m={}, n={}) ===", k, m, n);
        println!("Expected: {:?}", &expected[..12.min(expected.len())]);
        println!("Actual:   {:?}", &actual[..12.min(actual.len())]);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-6, "Tiny no-bias test failed. Max diff: {}", diff);
    }

    #[test]
    fn test_block_4x3_tiny_with_bias() {
        // k=4, 4 tokens, 3 outputs, WITH bias
        let k = 4;
        let m = 4;
        let n = 3;

        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = scalar_matmul_bias(&a, &b, &bias, m, k, n);
        let mut actual = vec![0.0f32; m * n];

        unsafe {
            matmul_block_4x3_f32(
                actual.as_mut_ptr(),
                n,
                a.as_ptr(),
                b.as_ptr(),
                k,
                bias.as_ptr(),
            );
        }

        let diff = max_diff(&expected, &actual);
        println!("\n=== matmul_block_4x3 WITH BIAS (k={}, m={}, n={}) ===", k, m, n);
        println!("Bias: {:?}", &bias);
        println!("Expected: {:?}", &expected[..12.min(expected.len())]);
        println!("Actual:   {:?}", &actual[..12.min(actual.len())]);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-6, "Tiny with-bias test failed. Max diff: {}", diff);
    }

    #[test]
    fn test_block_4x3_small_no_bias() {
        // k=32 - tests main SIMD loop (32 / 8 = 4 iterations)
        let k = 32;
        let m = 4;
        let n = 3;

        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = scalar_matmul(&a, &b, m, k, n);
        let mut actual = vec![0.0f32; m * n];

        unsafe {
            matmul_block_4x3_f32(
                actual.as_mut_ptr(),
                n,
                a.as_ptr(),
                b.as_ptr(),
                k,
                std::ptr::null(),
            );
        }

        let diff = max_diff(&expected, &actual);
        println!("\n=== matmul_block_4x3 (k={}, m={}, n={}) ===", k, m, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Small test failed. Max diff: {}", diff);
    }

    #[test]
    fn test_block_4x3_small_with_bias() {
        let k = 32;
        let m = 4;
        let n = 3;

        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = scalar_matmul_bias(&a, &b, &bias, m, k, n);
        let mut actual = vec![0.0f32; m * n];

        unsafe {
            matmul_block_4x3_f32(
                actual.as_mut_ptr(),
                n,
                a.as_ptr(),
                b.as_ptr(),
                k,
                bias.as_ptr(),
            );
        }

        let diff = max_diff(&expected, &actual);
        println!("\n=== matmul_block_4x3 WITH BIAS (k={}, m={}, n={}) ===", k, m, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Small with-bias test failed. Max diff: {}", diff);
    }

    #[test]
    fn test_block_4x3_medium_no_bias() {
        // k=128 - the dimension that was failing in integration tests
        let k = 128;
        let m = 4;
        let n = 3;

        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = scalar_matmul(&a, &b, m, k, n);
        let mut actual = vec![0.0f32; m * n];

        unsafe {
            matmul_block_4x3_f32(
                actual.as_mut_ptr(),
                n,
                a.as_ptr(),
                b.as_ptr(),
                k,
                std::ptr::null(),
            );
        }

        let diff = max_diff(&expected, &actual);
        println!("\n=== matmul_block_4x3 MEDIUM (k={}, m={}, n={}) ===", k, m, n);
        println!("Expected: {:?}", &expected[..6]);
        println!("Actual:   {:?}", &actual[..6]);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-4, "Medium test failed. Max diff: {}", diff);
    }

    #[test]
    fn test_block_4x3_medium_with_bias() {
        let k = 128;
        let m = 4;
        let n = 3;

        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = scalar_matmul_bias(&a, &b, &bias, m, k, n);
        let mut actual = vec![0.0f32; m * n];

        unsafe {
            matmul_block_4x3_f32(
                actual.as_mut_ptr(),
                n,
                a.as_ptr(),
                b.as_ptr(),
                k,
                bias.as_ptr(),
            );
        }

        let diff = max_diff(&expected, &actual);
        println!("\n=== matmul_block_4x3 MEDIUM WITH BIAS (k={}, m={}, n={}) ===", k, m, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-4, "Medium with-bias test failed. Max diff: {}", diff);
    }

    #[test]
    fn test_block_4x3_large_no_bias() {
        // k=384 - MiniLM hidden dim
        let k = 384;
        let m = 4;
        let n = 3;

        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = scalar_matmul(&a, &b, m, k, n);
        let mut actual = vec![0.0f32; m * n];

        unsafe {
            matmul_block_4x3_f32(
                actual.as_mut_ptr(),
                n,
                a.as_ptr(),
                b.as_ptr(),
                k,
                std::ptr::null(),
            );
        }

        let diff = max_diff(&expected, &actual);
        println!("\n=== matmul_block_4x3 LARGE (k={}, m={}, n={}) ===", k, m, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-4, "Large test failed. Max diff: {}", diff);
    }

    #[test]
    fn test_block_4x3_non_aligned_k() {
        // k=37 - not divisible by 8, tests remainder loop
        let k = 37;
        let m = 4;
        let n = 3;

        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = scalar_matmul(&a, &b, m, k, n);
        let mut actual = vec![0.0f32; m * n];

        unsafe {
            matmul_block_4x3_f32(
                actual.as_mut_ptr(),
                n,
                a.as_ptr(),
                b.as_ptr(),
                k,
                std::ptr::null(),
            );
        }

        let diff = max_diff(&expected, &actual);
        println!("\n=== matmul_block_4x3 NON-ALIGNED K (k={}) ===", k);
        println!("Expected: {:?}", &expected);
        println!("Actual:   {:?}", &actual);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Non-aligned k test failed. Max diff: {}", diff);
    }

    #[test]
    fn test_block_4x3_non_aligned_k_with_bias() {
        // k=37 with bias
        let k = 37;
        let m = 4;
        let n = 3;

        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = scalar_matmul_bias(&a, &b, &bias, m, k, n);
        let mut actual = vec![0.0f32; m * n];

        unsafe {
            matmul_block_4x3_f32(
                actual.as_mut_ptr(),
                n,
                a.as_ptr(),
                b.as_ptr(),
                k,
                bias.as_ptr(),
            );
        }

        let diff = max_diff(&expected, &actual);
        println!("\n=== matmul_block_4x3 NON-ALIGNED K WITH BIAS (k={}) ===", k);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Non-aligned k with-bias test failed. Max diff: {}", diff);
    }

    #[test]
    fn test_block_4x3_output_stride() {
        let k = 32;
        let m = 4;
        let n_kernel = 3;  // kernel produces 3 outputs
        let n_total = 10;  // but output buffer is wider

        let a = make_input(m, k, 0);
        let b = make_input(n_kernel, k, 100);

        // Expected: compute just the 3 outputs per row
        let expected_small = scalar_matmul(&a, &b, m, k, n_kernel);

        // Actual: write to a wider buffer with stride = n_total
        let mut actual = vec![0.0f32; m * n_total];

        unsafe {
            matmul_block_4x3_f32(
                actual.as_mut_ptr(),  // start at column 0
                n_total,              // stride is full width
                a.as_ptr(),
                b.as_ptr(),
                k,
                std::ptr::null(),
            );
        }

        // Extract the first 3 columns from each row
        let mut actual_extracted = Vec::with_capacity(m * n_kernel);
        for row in 0..m {
            for col in 0..n_kernel {
                actual_extracted.push(actual[row * n_total + col]);
            }
        }

        let diff = max_diff(&expected_small, &actual_extracted);
        println!("\n=== matmul_block_4x3 OUTPUT STRIDE (stride={}) ===", n_total);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Output stride test failed. Max diff: {}", diff);
    }
}