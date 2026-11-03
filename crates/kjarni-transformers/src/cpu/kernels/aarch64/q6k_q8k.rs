#![allow(unsafe_code)]
use crate::cpu::kernels::q_common::{BlockQ6_K, BlockQ8_K, QK_K};
use std::arch::aarch64::*;

/// Sums the products of 16 6-bit weights, centred on zero, and 16 int8 activations.
///
/// A centred weight fits in an int8 and each product fits in an int16.
#[inline]
#[target_feature(enable = "neon")]
fn dot_centred(w: uint8x16_t, q: int8x16_t) -> i32 {
    let w = vreinterpretq_s8_u8(vsubq_u8(w, vdupq_n_u8(32)));
    let sum = vpaddlq_s16(vmull_s8(vget_low_s8(w), vget_low_s8(q)));
    vaddvq_s32(vpadalq_s16(sum, vmull_high_s8(w, q)))
}

/// Computes the dot product of Q6_K weights and a Q8_K activation row using NEON.
///
/// All the arithmetic before the final scale is on integers, so the result is
/// identical to `vec_dot_q6k_q8k_scalar`, bit for bit.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vec_dot_q6k_q8k_neon(
    n: usize,
    w_blocks: &[BlockQ6_K],
    q_blocks: &[BlockQ8_K],
) -> f32 {
    let num_blocks = n / QK_K;
    let low4 = vdupq_n_u8(0x0F);
    let low2 = vdupq_n_u8(0x03);
    let mut acc = 0.0f32;

    for i in 0..num_blocks {
        let w = &w_blocks[i];
        let q = &q_blocks[i];
        let mut sum_qs = 0i32;

        // Two halves of 128 weights, each with two groups of 16 positions per stream.
        for half in 0..2 {
            for group in 0..2 {
                let at = group * 16;
                // SAFETY: half < 2 and group < 2, so `at + 16` stays within 32 for `qh`
                // (32 bytes per half), `32 + at + 16` within 64 for `ql` (64 per half),
                // and `96 + at + 16` within 128 for `qs` (128 per half).
                let (ql0, ql1, qh, q0, q1, q2, q3) = unsafe {
                    let ql = w.ql.as_ptr().add(half * 64);
                    let qs = q.qs.as_ptr().add(half * 128 + at);
                    (
                        vld1q_u8(ql.add(at)),
                        vld1q_u8(ql.add(32 + at)),
                        vld1q_u8(w.qh.as_ptr().add(half * 32 + at)),
                        vld1q_s8(qs),
                        vld1q_s8(qs.add(32)),
                        vld1q_s8(qs.add(64)),
                        vld1q_s8(qs.add(96)),
                    )
                };

                // Each `qh` byte holds the top two bits of four weights.
                let w0 = vorrq_u8(vandq_u8(ql0, low4), vshlq_n_u8::<4>(vandq_u8(qh, low2)));
                let w1 = vorrq_u8(
                    vandq_u8(ql1, low4),
                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(qh), low2)),
                );
                let w2 = vorrq_u8(
                    vshrq_n_u8::<4>(ql0),
                    vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(qh), low2)),
                );
                let w3 = vorrq_u8(vshrq_n_u8::<4>(ql1), vshlq_n_u8::<4>(vshrq_n_u8::<6>(qh)));

                let sc = &w.scales[half * 8 + group..];
                sum_qs += sc[0] as i32 * dot_centred(w0, q0);
                sum_qs += sc[2] as i32 * dot_centred(w1, q1);
                sum_qs += sc[4] as i32 * dot_centred(w2, q2);
                sum_qs += sc[6] as i32 * dot_centred(w3, q3);
            }
        }

        acc += w.d.to_f32() * q.d * sum_qs as f32;
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::kernels::scalar::vec_dot_q6k_q8k_scalar;
    use half::f16;

    fn q8k_block(seed: usize) -> BlockQ8_K {
        let mut qs = [0i8; 256];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = match i % 7 {
                0 => -127,
                1 => 127,
                _ => ((seed * 13 + i * 29) % 255) as u8 as i8,
            };
        }
        BlockQ8_K {
            d: 0.01 + seed as f32 * 0.001,
            qs,
            bsums: [0; 16],
        }
    }

    fn q6k_block(seed: usize) -> BlockQ6_K {
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        let mut scales = [0i8; 16];
        for (i, v) in ql.iter_mut().enumerate() {
            *v = ((seed * 7 + i * 11) % 256) as u8;
        }
        for (i, v) in qh.iter_mut().enumerate() {
            *v = ((seed * 3 + i * 23) % 256) as u8;
        }
        for (i, v) in scales.iter_mut().enumerate() {
            *v = ((seed + i * 5) % 40) as i8 - 20;
        }
        BlockQ6_K {
            ql,
            qh,
            scales,
            d: f16::from_f32(0.03),
        }
    }

    #[test]
    fn matches_hand_computed_value() {
        // ql 0 and qh 0 give the raw weight 0, which centres to -32. Scale 1, every
        // activation 1, and d 0.5 and 0.5: 256 * -32 * 0.25 = -2048.
        let w = BlockQ6_K {
            ql: [0; 128],
            qh: [0; 64],
            scales: [1; 16],
            d: f16::from_f32(0.5),
        };
        let q = BlockQ8_K {
            d: 0.5,
            qs: [1; 256],
            bsums: [16; 16],
        };
        let out = unsafe { vec_dot_q6k_q8k_neon(QK_K, &[w], std::slice::from_ref(&q)) };
        assert_eq!(out, -2048.0);
    }

    #[test]
    fn identical_to_scalar() {
        for blocks in [1, 2, 5] {
            let w: Vec<_> = (0..blocks).map(q6k_block).collect();
            let q: Vec<_> = (0..blocks).map(q8k_block).collect();
            let n = blocks * QK_K;
            let expected = vec_dot_q6k_q8k_scalar(n, &w, &q);
            let actual = unsafe { vec_dot_q6k_q8k_neon(n, &w, &q) };
            assert_eq!(
                expected.to_bits(),
                actual.to_bits(),
                "{expected} vs {actual}"
            );
        }
    }

    #[test]
    fn extreme_weights_keep_their_sign() {
        // The top raw weight, 63, centres to 31; every bit of `ql` and `qh` is set.
        let mut w = q6k_block(2);
        w.ql = [0xFF; 128];
        w.qh = [0xFF; 64];
        let mut q = q8k_block(1);
        q.qs = [-127; 256];
        let expected = vec_dot_q6k_q8k_scalar(QK_K, &[w], std::slice::from_ref(&q));
        let actual = unsafe { vec_dot_q6k_q8k_neon(QK_K, &[w], std::slice::from_ref(&q)) };
        assert_eq!(expected.to_bits(), actual.to_bits());
    }
}
