use std::arch::x86_64::*;

use crate::kernels::x86::common::hsum_ps_avx;

/// RMS Norm optimized for AVX2.
///
/// # Arguments
/// * `x` - Input/Output vector (in-place).
/// * `w` - Weight vector (same size as x).
/// * `epsilon` - Small constant (e.g. 1e-5).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn rms_norm_avx2(x: &mut [f32], w: &[f32], epsilon: f32) {
    unsafe {
        let len = x.len();
        let mut sum_sq_vec = _mm256_setzero_ps();

        // 1. Calculate Sum of Squares
        let mut i = 0;
        while i + 8 <= len {
            let v = _mm256_loadu_ps(x.as_ptr().add(i));
            sum_sq_vec = _mm256_fmadd_ps(v, v, sum_sq_vec);
            i += 8;
        }

        // Reduce sum
        let sum_sq = hsum_ps_avx(sum_sq_vec) + x[i..].iter().map(|v| v * v).sum::<f32>();

        // 2. Calculate RMS scale
        let mean = sum_sq / len as f32;
        let scale = 1.0 / (mean + epsilon).sqrt();
        let scale_vec = _mm256_set1_ps(scale);

        // 3. Apply Scale and Weight
        i = 0;
        while i + 8 <= len {
            let x_ptr = x.as_mut_ptr().add(i);
            let w_ptr = w.as_ptr().add(i);

            let v_x = _mm256_loadu_ps(x_ptr);
            let v_w = _mm256_loadu_ps(w_ptr);

            // out = x * scale * w
            let out = _mm256_mul_ps(_mm256_mul_ps(v_x, scale_vec), v_w);

            _mm256_storeu_ps(x_ptr, out);
            i += 8;
        }

        // Remainder
        for j in i..len {
            x[j] = x[j] * scale * w[j];
        }
    }
}
