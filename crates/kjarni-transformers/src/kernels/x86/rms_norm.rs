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

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar reference RMSNorm (1D, in-place)
    fn rms_norm_scalar(x: &mut [f32], w: &[f32], epsilon: f32) {
        assert_eq!(x.len(), w.len());

        let len = x.len();
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let mean = sum_sq / len as f32;
        let scale = 1.0 / (mean + epsilon).sqrt();

        for i in 0..len {
            x[i] = x[i] * scale * w[i];
        }
    }

    /// AVX2-enabled test body.
    /// Must be isolated due to `#[target_feature]`.
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn run_rms_norm_avx2_vs_scalar() {
        let len = 257; // deliberately NOT a multiple of 8
        let epsilon = 1e-6;

        // Input vector with non-trivial values
        let mut x_scalar: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.01) - 1.3)
            .collect();

        let mut x_avx2 = x_scalar.clone();

        // Weight vector
        let w: Vec<f32> = (0..len)
            .map(|i| 1.0 + (i as f32 * 0.001))
            .collect();

        // Scalar reference
        rms_norm_scalar(&mut x_scalar, &w, epsilon);

        // AVX2 kernel
        unsafe { rms_norm_avx2(&mut x_avx2, &w, epsilon) };

        // Compare
        for i in 0..len {
            let diff = (x_scalar[i] - x_avx2[i]).abs();
            assert!(
                diff < 1e-5,
                "RMSNorm AVX2 mismatch at index {}: scalar={} avx2={} diff={}",
                i,
                x_scalar[i],
                x_avx2[i],
                diff
            );
        }
    }

    /// Safe test entry point.
    #[test]
    fn test_rms_norm_avx2_matches_scalar() {
        if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
        {
            unsafe {
                run_rms_norm_avx2_vs_scalar();
            }
        } else {
            eprintln!("skipping RMSNorm AVX2 test: CPU lacks AVX2+FMA");
        }
    }
}