//! Rotary Position Embeddings (RoPE) kernel

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Applies RoPE rotation to a single head vector in-place
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn rotate_half_avx2(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let head_dim = x.len();
    let half_dim = head_dim / 2;

    debug_assert_eq!(cos.len(), head_dim);
    debug_assert_eq!(sin.len(), head_dim);
    debug_assert!(head_dim % 2 == 0);
    unsafe {
        let x0_ptr = x.as_ptr();
        let x1_ptr = x.as_ptr().add(half_dim);
        let out0_ptr = x.as_mut_ptr();
        let out1_ptr = x.as_mut_ptr().add(half_dim);

        // Process 8 elements at a time
        let simd_iterations = half_dim / 8;
        let remainder = half_dim % 8;

        for i in 0..simd_iterations {
            let offset = i * 8;

            // Load x[0:half] and x[half:end]
            let x0 = _mm256_loadu_ps(x0_ptr.add(offset));
            let x1 = _mm256_loadu_ps(x1_ptr.add(offset));

            // Load cos and sin (first half only, same values mirrored in cache)
            let cos_vec = _mm256_loadu_ps(cos.as_ptr().add(offset));
            let sin_vec = _mm256_loadu_ps(sin.as_ptr().add(offset));

            // Compute rotations using FMA:
            // out0 = x0 * cos - x1 * sin = fmsub(x0, cos, x1 * sin)
            // out1 = x0 * sin + x1 * cos = fmadd(x0, sin, x1 * cos)
            let x1_sin = _mm256_mul_ps(x1, sin_vec);
            let out0 = _mm256_fmsub_ps(x0, cos_vec, x1_sin);

            let x1_cos = _mm256_mul_ps(x1, cos_vec);
            let out1 = _mm256_fmadd_ps(x0, sin_vec, x1_cos);

            // Store results
            _mm256_storeu_ps(out0_ptr.add(offset), out0);
            _mm256_storeu_ps(out1_ptr.add(offset), out1);
        }

        // Handle remainder (if half_dim not divisible by 8)
        if remainder > 0 {
            let offset = simd_iterations * 8;
            for i in 0..remainder {
                let idx = offset + i;
                let x0_val = *x0_ptr.add(idx);
                let x1_val = *x1_ptr.add(idx);
                let cos_val = cos[idx];
                let sin_val = sin[idx];

                *out0_ptr.add(idx) = x0_val * cos_val - x1_val * sin_val;
                *out1_ptr.add(idx) = x0_val * sin_val + x1_val * cos_val;
            }
        }
    }
}

/// Applies RoPE rotation to a batch of head vectors
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn rotate_4d_avx2(
    x: &mut [f32],
    cos_cache: &[f32],  // Flattened [max_seq_len, head_dim]
    sin_cache: &[f32],
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    cache_stride: usize,  // head_dim (row stride in cache)
    position_offset: usize,
) { unsafe {
    let _half_dim = head_dim / 2;

    // x is [batch, heads, seq, head_dim] in row-major
    // Stride calculations
    let head_stride = seq_len * head_dim;
    let batch_stride = num_heads * head_stride;

    for b in 0..batch_size {
        for h in 0..num_heads {
            for s in 0..seq_len {
                let pos = position_offset + s;

                // Calculate offset into x
                let x_offset = b * batch_stride + h * head_stride + s * head_dim;
                let x_slice = &mut x[x_offset..x_offset + head_dim];

                // Calculate offset into cache
                let cache_offset = pos * cache_stride;
                let cos_slice = &cos_cache[cache_offset..cache_offset + head_dim];
                let sin_slice = &sin_cache[cache_offset..cache_offset + head_dim];

                // Apply rotation to this head
                rotate_half_avx2(x_slice, cos_slice, sin_slice);
            }
        }
    }
}}

/// Scalar fallback for non-AVX2 systems or testing.
pub fn rotate_half_scalar(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let head_dim = x.len();
    let half_dim = head_dim / 2;

    for i in 0..half_dim {
        let x0 = x[i];
        let x1 = x[i + half_dim];
        let c = cos[i];
        let s = sin[i];

        x[i] = x0 * c - x1 * s;
        x[i + half_dim] = x0 * s + x1 * c;
    }
}

/// Dispatches to AVX2 or scalar based on CPU features.
pub fn rotate_half(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                rotate_half_avx2(x, cos, sin);
            }
            return;
        }
    }

    rotate_half_scalar(x, cos, sin);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rotate_half_reference(x: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
        let head_dim = x.len();
        let half_dim = head_dim / 2;
        let mut out = x.to_vec();

        for i in 0..half_dim {
            let x0 = x[i];
            let x1 = x[i + half_dim];
            let c = cos[i];
            let s = sin[i];

            out[i] = x0 * c - x1 * s;
            out[i + half_dim] = x0 * s + x1 * c;
        }

        out
    }

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn vecs_approx_eq(a: &[f32], b: &[f32], eps: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, eps))
    }

    #[test]
    fn test_rotate_half_scalar_matches_reference() {
        let head_dim = 64;
        let half_dim = head_dim / 2;

        let x: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1).collect();
        let cos: Vec<f32> = (0..head_dim).map(|i| ((i as f32) * 0.05).cos()).collect();
        let sin: Vec<f32> = (0..head_dim).map(|i| ((i as f32) * 0.05).sin()).collect();

        // Reference
        let expected = rotate_half_reference(&x, &cos, &sin);

        // Scalar
        let mut actual = x.clone();
        rotate_half_scalar(&mut actual, &cos, &sin);

        assert!(
            vecs_approx_eq(&expected, &actual, 1e-6),
            "Scalar doesn't match reference"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_rotate_half_avx2_matches_reference() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test - not supported on this CPU");
            return;
        }

        for head_dim in [16, 32, 64, 128, 256] {
            let x: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1).collect();
            let cos: Vec<f32> = (0..head_dim).map(|i| ((i as f32) * 0.05).cos()).collect();
            let sin: Vec<f32> = (0..head_dim).map(|i| ((i as f32) * 0.05).sin()).collect();

            let expected = rotate_half_reference(&x, &cos, &sin);

            let mut actual = x.clone();
            unsafe {
                rotate_half_avx2(&mut actual, &cos, &sin);
            }

            assert!(
                vecs_approx_eq(&expected, &actual, 1e-5),
                "AVX2 doesn't match reference for head_dim={}.\nExpected: {:?}\nActual: {:?}",
                head_dim,
                &expected[..8],
                &actual[..8]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_rotate_half_avx2_with_remainder() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        // head_dim = 20 -> half_dim = 10 -> not divisible by 8
        let head_dim = 20;
        let x: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1).collect();
        let cos: Vec<f32> = (0..head_dim).map(|i| ((i as f32) * 0.05).cos()).collect();
        let sin: Vec<f32> = (0..head_dim).map(|i| ((i as f32) * 0.05).sin()).collect();

        let expected = rotate_half_reference(&x, &cos, &sin);

        let mut actual = x.clone();
        unsafe {
            rotate_half_avx2(&mut actual, &cos, &sin);
        }

        assert!(
            vecs_approx_eq(&expected, &actual, 1e-5),
            "AVX2 remainder handling failed"
        );
    }

    #[test]
    fn test_rotate_half_dispatch() {
        let head_dim = 64;
        let x: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1).collect();
        let cos: Vec<f32> = (0..head_dim).map(|i| ((i as f32) * 0.05).cos()).collect();
        let sin: Vec<f32> = (0..head_dim).map(|i| ((i as f32) * 0.05).sin()).collect();

        let expected = rotate_half_reference(&x, &cos, &sin);

        let mut actual = x.clone();
        rotate_half(&mut actual, &cos, &sin);

        assert!(
            vecs_approx_eq(&expected, &actual, 1e-5),
            "Dispatch function failed"
        );
    }

    #[test]
    fn test_rotation_preserves_magnitude() {
        let head_dim = 8;
        let x = vec![3.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]; // [3, 1] in first half, [4, 0] in second
        let angle = std::f32::consts::PI / 4.0; // 45 degrees
        let cos = vec![angle.cos(); head_dim];
        let sin = vec![angle.sin(); head_dim];

        let mut rotated = x.clone();
        rotate_half_scalar(&mut rotated, &cos, &sin);

        // Check magnitude preserved for first pair
        let mag_before = (x[0] * x[0] + x[4] * x[4]).sqrt();
        let mag_after = (rotated[0] * rotated[0] + rotated[4] * rotated[4]).sqrt();

        assert!(
            approx_eq(mag_before, mag_after, 1e-5),
            "Magnitude not preserved: {} vs {}",
            mag_before,
            mag_after
        );
    }

    #[test]
    fn test_rotation_by_zero_is_identity() {
        let head_dim = 16;
        let x: Vec<f32> = (0..head_dim).map(|i| i as f32).collect();
        let cos = vec![1.0; head_dim]; // cos(0) = 1
        let sin = vec![0.0; head_dim]; // sin(0) = 0

        let mut rotated = x.clone();
        rotate_half(&mut rotated, &cos, &sin);

        assert!(
            vecs_approx_eq(&x, &rotated, 1e-6),
            "Rotation by 0 should be identity"
        );
    }

    #[test]
    fn test_rotation_by_90_degrees() {
        let head_dim = 8;
        let half_dim = head_dim / 2;

        // x = [1, 0, 0, 0] in first half, [0, 0, 0, 0] in second half
        let mut x = vec![0.0; head_dim];
        x[0] = 1.0;

        let angle = std::f32::consts::PI / 2.0; // 90 degrees
        let cos = vec![angle.cos(); head_dim]; // ~0
        let sin = vec![angle.sin(); head_dim]; // ~1

        rotate_half(&mut x, &cos, &sin);

        assert!(approx_eq(x[0], 0.0, 1e-5), "x[0] should be ~0, got {}", x[0]);
        assert!(
            approx_eq(x[half_dim], 1.0, 1e-5),
            "x[half] should be ~1, got {}",
            x[half_dim]
        );
    }
}
