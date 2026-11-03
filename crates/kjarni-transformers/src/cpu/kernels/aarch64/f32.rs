#![allow(unsafe_code)]
use std::arch::aarch64::*;

/// Computes a [4 x 3] tile: four input rows against three weight rows.
///
/// Each iteration does 12 multiply-adds from 7 loads. All 12 accumulators, the three
/// weight vectors and the input vector fit in NEON's 32 registers.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn matmul_block_4x3_f32_neon(
    out_ptr: *mut f32,
    out_stride: usize,
    a_ptr: *const f32,
    b_ptr: *const f32,
    k: usize,
    bias_ptr: *const f32,
) {
    let mut acc = [[vdupq_n_f32(0.0); 3]; 4];
    let mut i = 0;

    // SAFETY: the caller guarantees `a_ptr` points to 4 rows and `b_ptr` to 3 rows of
    // `k` floats, and that `out_ptr` can take 3 values on each of 4 rows `out_stride`
    // apart. `bias_ptr` is null or points to 3 floats.
    unsafe {
        while i + 4 <= k {
            let w0 = vld1q_f32(b_ptr.add(i));
            let w1 = vld1q_f32(b_ptr.add(k + i));
            let w2 = vld1q_f32(b_ptr.add(2 * k + i));
            for (row, c) in acc.iter_mut().enumerate() {
                let av = vld1q_f32(a_ptr.add(row * k + i));
                c[0] = vfmaq_f32(c[0], av, w0);
                c[1] = vfmaq_f32(c[1], av, w1);
                c[2] = vfmaq_f32(c[2], av, w2);
            }
            i += 4;
        }

        for (row, c) in acc.iter().enumerate() {
            for (col, lane) in c.iter().enumerate() {
                let mut sum = vaddvq_f32(*lane);
                for j in i..k {
                    sum += *a_ptr.add(row * k + j) * *b_ptr.add(col * k + j);
                }
                if !bias_ptr.is_null() {
                    sum += *bias_ptr.add(col);
                }
                *out_ptr.add(row * out_stride + col) = sum;
            }
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn matmul_vec_f32_neon(
    out_chunk: &mut [f32],
    a_ptr: *const f32,
    b_row_start: *const f32,
    k: usize,
) {
    // SAFETY: the caller guarantees `a_ptr` points to `k` floats and the weights to
    // `out_chunk.len()` rows of `k` values.
    unsafe {
        let mut b_row_ptr = b_row_start;
        for val in out_chunk.iter_mut() {
            let mut a_chunk_ptr = a_ptr;
            let mut b_chunk_ptr = b_row_ptr;
            // Four sums so the multiply-adds do not wait on each other's results.
            let mut sum0 = vdupq_n_f32(0.0);
            let mut sum1 = vdupq_n_f32(0.0);
            let mut sum2 = vdupq_n_f32(0.0);
            let mut sum3 = vdupq_n_f32(0.0);
            let mut n = k;

            while n >= 16 {
                let a0 = vld1q_f32(a_chunk_ptr);
                let a1 = vld1q_f32(a_chunk_ptr.add(4));
                let a2 = vld1q_f32(a_chunk_ptr.add(8));
                let a3 = vld1q_f32(a_chunk_ptr.add(12));

                let b0 = vld1q_f32(b_chunk_ptr);
                let b1 = vld1q_f32(b_chunk_ptr.add(4));
                let b2 = vld1q_f32(b_chunk_ptr.add(8));
                let b3 = vld1q_f32(b_chunk_ptr.add(12));

                sum0 = vfmaq_f32(sum0, a0, b0);
                sum1 = vfmaq_f32(sum1, a1, b1);
                sum2 = vfmaq_f32(sum2, a2, b2);
                sum3 = vfmaq_f32(sum3, a3, b3);

                a_chunk_ptr = a_chunk_ptr.add(16);
                b_chunk_ptr = b_chunk_ptr.add(16);
                n -= 16;
            }

            let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
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
mod tests {
    use super::*;
    use crate::cpu::kernels::scalar::matmul_vec_f32_scalar;

    #[test]
    fn matches_hand_computed_value() {
        // 19 elements covers the 16-wide loop and the 3-element tail: 2 * sum(0..19).
        let a: Vec<f32> = (0..19).map(|i| i as f32).collect();
        let b = vec![2.0f32; 19];
        let mut out = [0.0f32];
        unsafe { matmul_vec_f32_neon(&mut out, a.as_ptr(), b.as_ptr(), 19) };
        assert_eq!(out, [342.0]);
    }

    #[test]
    fn matches_scalar_across_rows() {
        let (k, rows) = (70, 5);
        let a: Vec<f32> = (0..k).map(|i| (i as f32 - 30.0) * 0.1).collect();
        let b: Vec<f32> = (0..k * rows)
            .map(|i| ((i * 17 % 23) as f32 - 11.0) * 0.2)
            .collect();
        let mut expected = vec![0.0f32; rows];
        let mut actual = vec![0.0f32; rows];
        matmul_vec_f32_scalar(&mut expected, &a, &b, k);
        unsafe { matmul_vec_f32_neon(&mut actual, a.as_ptr(), b.as_ptr(), k) };
        for (e, g) in expected.iter().zip(&actual) {
            assert!((e - g).abs() <= e.abs().max(1.0) * 1e-5, "{e} vs {g}");
        }
    }

    #[test]
    fn block_4x3_matches_vector_kernel_with_bias_and_stride() {
        // k = 19 covers the wide loop and the tail, and stride 5 leaves gaps to check.
        let k = 19;
        let a: Vec<f32> = (0..4 * k).map(|i| (i as f32 - 30.0) * 0.1).collect();
        let b: Vec<f32> = (0..3 * k)
            .map(|i| ((i * 17 % 23) as f32 - 11.0) * 0.2)
            .collect();
        let bias = [1.0f32, -2.0, 3.0];
        let mut out = vec![99.0f32; 4 * 5];
        unsafe {
            matmul_block_4x3_f32_neon(
                out.as_mut_ptr(),
                5,
                a.as_ptr(),
                b.as_ptr(),
                k,
                bias.as_ptr(),
            )
        };

        for row in 0..4 {
            let mut expected = [0.0f32; 3];
            matmul_vec_f32_scalar(&mut expected, &a[row * k..(row + 1) * k], &b, k);
            for col in 0..3 {
                let (e, g) = (expected[col] + bias[col], out[row * 5 + col]);
                assert!(
                    (e - g).abs() <= e.abs().max(1.0) * 1e-5,
                    "[{row},{col}] {e} vs {g}"
                );
            }
            assert_eq!(&out[row * 5 + 3..row * 5 + 5], [99.0, 99.0]);
        }
    }

    #[test]
    fn block_4x3_matches_hand_computed_value() {
        // All ones over k = 5, so each output is 5, with no bias.
        let a = vec![1.0f32; 20];
        let b = vec![1.0f32; 15];
        let mut out = [0.0f32; 12];
        unsafe {
            matmul_block_4x3_f32_neon(
                out.as_mut_ptr(),
                3,
                a.as_ptr(),
                b.as_ptr(),
                5,
                std::ptr::null(),
            )
        };
        assert_eq!(out, [5.0; 12]);
    }
}
