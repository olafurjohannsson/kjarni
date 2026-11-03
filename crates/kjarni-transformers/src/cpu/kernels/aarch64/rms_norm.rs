#![allow(unsafe_code)]
use std::arch::aarch64::*;

/// RMS norm of one row, in place: `x * w / sqrt(mean(x^2) + epsilon)`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rms_norm_neon(x: &mut [f32], w: &[f32], epsilon: f32) {
    let len = x.len();
    assert_eq!(len, w.len());

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut chunks = x.chunks_exact(8);
    for c in &mut chunks {
        // SAFETY: the chunk holds 8 floats.
        let (v0, v1) = unsafe { (vld1q_f32(c.as_ptr()), vld1q_f32(c.as_ptr().add(4))) };
        sum0 = vfmaq_f32(sum0, v0, v0);
        sum1 = vfmaq_f32(sum1, v1, v1);
    }
    let tail: f32 = chunks.remainder().iter().map(|v| v * v).sum();
    let sum_sq = vaddvq_f32(vaddq_f32(sum0, sum1)) + tail;

    let scale = 1.0 / (sum_sq / len as f32 + epsilon).sqrt();
    let scale_v = vdupq_n_f32(scale);

    let mut i = 0;
    while i + 4 <= len {
        // SAFETY: i + 4 <= len, and `x` and `w` have the same length.
        unsafe {
            let p = x.as_mut_ptr().add(i);
            let out = vmulq_f32(
                vmulq_f32(vld1q_f32(p), scale_v),
                vld1q_f32(w.as_ptr().add(i)),
            );
            vst1q_f32(p, out);
        }
        i += 4;
    }
    for j in i..len {
        x[j] = x[j] * scale * w[j];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_hand_computed_value() {
        // mean(x^2) = 4 over 9 elements, so with epsilon 0 the scale is 1/2.
        let mut x = [2.0f32; 9];
        let w = [3.0f32; 9];
        unsafe { rms_norm_neon(&mut x, &w, 0.0) };
        assert_eq!(x, [3.0f32; 9]);
    }

    #[test]
    fn matches_scalar_formula() {
        let len = 257;
        let w: Vec<f32> = (0..len).map(|i| 0.5 + i as f32 * 0.01).collect();
        let mut x: Vec<f32> = (0..len).map(|i| (i as f32 - 100.0) * 0.1).collect();
        let sum: f32 = x.iter().map(|v| v * v).sum();
        let scale = 1.0 / (sum / len as f32 + 1e-6).sqrt();
        let expected: Vec<f32> = x.iter().zip(&w).map(|(v, w)| v * scale * w).collect();
        unsafe { rms_norm_neon(&mut x, &w, 1e-6) };
        for (e, g) in expected.iter().zip(&x) {
            assert!((e - g).abs() <= e.abs().max(1.0) * 1e-5, "{e} vs {g}");
        }
    }
}
