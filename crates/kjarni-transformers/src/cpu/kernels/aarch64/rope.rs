#![allow(unsafe_code)]
use std::arch::aarch64::*;

/// Applies the RoPE rotation to one head vector in place. `cos` and `sin` hold at
/// least the first half of the head dimension.
#[target_feature(enable = "neon")]
fn rotate_half_neon(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = x.len() / 2;
    let (x0s, x1s) = x.split_at_mut(half);
    let mut i = 0;

    while i + 4 <= half {
        // SAFETY: i + 4 <= half, the length of `x0s` and at most that of the others.
        unsafe {
            let (p0, p1) = (x0s.as_mut_ptr().add(i), x1s.as_mut_ptr().add(i));
            let (c, s) = (vld1q_f32(cos[i..].as_ptr()), vld1q_f32(sin[i..].as_ptr()));
            let (v0, v1) = (vld1q_f32(p0), vld1q_f32(p1));
            // Same terms fused as the AVX2 kernel, so both round the same way.
            vst1q_f32(p0, vfmaq_f32(vnegq_f32(vmulq_f32(v1, s)), v0, c));
            vst1q_f32(p1, vfmaq_f32(vmulq_f32(v1, c), v0, s));
        }
        i += 4;
    }

    while i < half {
        let (v0, v1) = (x0s[i], x1s[i]);
        x0s[i] = v0 * cos[i] - v1 * sin[i];
        x1s[i] = v0 * sin[i] + v1 * cos[i];
        i += 1;
    }
}

/// Applies RoPE to `x`, laid out as [batch, heads, seq, head_dim], using caches laid
/// out as [max_seq_len, cache_stride].
#[target_feature(enable = "neon")]
pub(crate) fn rotate_4d_neon(
    x: &mut [f32],
    cos_cache: &[f32],
    sin_cache: &[f32],
    seq_len: usize,
    head_dim: usize,
    cache_stride: usize,
    position_offset: usize,
) {
    if head_dim == 0 {
        return;
    }
    for (i, row) in x.chunks_exact_mut(head_dim).enumerate() {
        let pos = position_offset + i % seq_len;
        let at = pos * cache_stride;
        rotate_half_neon(
            row,
            &cos_cache[at..at + head_dim],
            &sin_cache[at..at + head_dim],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarter_turn_matches_hand_computed_value() {
        // cos 0 and sin 1 send (x0, x1) to (-x1, x0).
        let mut x: Vec<f32> = (1..=10).map(|i| i as f32).collect();
        let cos = vec![0.0f32; 10];
        let sin = vec![1.0f32; 10];
        unsafe { rotate_half_neon(&mut x, &cos, &sin) };
        assert_eq!(x, [-6., -7., -8., -9., -10., 1., 2., 3., 4., 5.]);
    }

    #[test]
    fn matches_exact_rotation_on_many_values() {
        // Compared in f64 against the exact rotation, tail included (head_dim 22).
        let n = 22;
        let cos: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).cos()).collect();
        let sin: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 1.3 - 9.0) * 0.7).collect();
        let mut got = x.clone();
        unsafe { rotate_half_neon(&mut got, &cos, &sin) };
        for i in 0..n / 2 {
            let (a, b) = (x[i] as f64, x[i + n / 2] as f64);
            let (c, s) = (cos[i] as f64, sin[i] as f64);
            assert!((got[i] as f64 - (a * c - b * s)).abs() < 2e-6, "x0 at {i}");
            assert!(
                (got[i + n / 2] as f64 - (a * s + b * c)).abs() < 2e-6,
                "x1 at {i}"
            );
        }
    }

    #[test]
    fn uses_the_cache_row_of_each_position() {
        // Two heads, two positions, head_dim 2. The cache rows are identity at position
        // 3 and a quarter turn at position 4.
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut cos = vec![0.0f32; 2 * 5];
        let mut sin = vec![0.0f32; 2 * 5];
        cos[6..8].fill(1.0);
        sin[8..10].fill(1.0);
        unsafe { rotate_4d_neon(&mut x, &cos, &sin, 2, 2, 2, 3) };
        assert_eq!(x, [1.0, 2.0, -4.0, 3.0, 5.0, 6.0, -8.0, 7.0]);
    }
}
