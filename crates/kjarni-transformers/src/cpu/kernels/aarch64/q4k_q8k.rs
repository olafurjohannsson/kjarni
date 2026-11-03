#![allow(unsafe_code)]
use crate::cpu::kernels::{
    dequantize::get_scale_min_k4,
    q_common::{BlockQ4_K, BlockQ8_K, QK_K},
};
use std::arch::aarch64::*;

/// Adds the products of 16 4-bit weights and 16 int8 activations to `acc`.
///
/// A nibble fits in an int8 and each product fits in an int16.
#[inline]
#[target_feature(enable = "neon")]
fn dot_nibbles(acc: int32x4_t, w: uint8x16_t, q: int8x16_t) -> int32x4_t {
    let w = vreinterpretq_s8_u8(w);
    let acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(w), vget_low_s8(q)));
    vpadalq_s16(acc, vmull_high_s8(w, q))
}

/// Computes the dot product of Q4_K weights and a Q8_K activation row using NEON.
///
/// All the arithmetic before the final scale is on integers, so the result is
/// identical to `vec_dot_q4k_q8k_scalar`, bit for bit.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vec_dot_q4k_q8k_neon(
    n: usize,
    w_blocks: &[BlockQ4_K],
    q_blocks: &[BlockQ8_K],
) -> f32 {
    let num_blocks = n / QK_K;
    let mask = vdupq_n_u8(0x0F);
    let mut acc = 0.0f32;

    for i in 0..num_blocks {
        let w = &w_blocks[i];
        let q = &q_blocks[i];
        let d = w.d.to_f32();
        let dmin = w.dmin.to_f32();

        let mut sum_qs = 0i32;
        let mut sum_mins = 0i32;

        for j in 0..4 {
            let is = j * 2;
            let (sc1, m1) = get_scale_min_k4(is, &w.scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, &w.scales);

            // SAFETY: j < 4, and `w.qs` holds 128 bytes and `q.qs` holds 256, so the
            // 32 bytes at j * 32 and the 64 bytes at j * 64 are in bounds.
            let (w0, w1, q0, q1, q2, q3) = unsafe {
                let w_ptr = w.qs.as_ptr().add(j * 32);
                let q_ptr = q.qs.as_ptr().add(j * 64);
                (
                    vld1q_u8(w_ptr),
                    vld1q_u8(w_ptr.add(16)),
                    vld1q_s8(q_ptr),
                    vld1q_s8(q_ptr.add(16)),
                    vld1q_s8(q_ptr.add(32)),
                    vld1q_s8(q_ptr.add(48)),
                )
            };

            // Low nibbles belong to the first sub-block, high nibbles to the second.
            let zero = vdupq_n_s32(0);
            let low = dot_nibbles(
                dot_nibbles(zero, vandq_u8(w0, mask), q0),
                vandq_u8(w1, mask),
                q1,
            );
            let high = dot_nibbles(
                dot_nibbles(zero, vshrq_n_u8::<4>(w0), q2),
                vshrq_n_u8::<4>(w1),
                q3,
            );

            sum_qs += vaddvq_s32(low) * (sc1 as i32);
            sum_qs += vaddvq_s32(high) * (sc2 as i32);

            let isum1 = q.bsums[is * 2] as i32 + q.bsums[is * 2 + 1] as i32;
            sum_mins += isum1 * (m1 as i32);
            let isum2 = q.bsums[(is + 1) * 2] as i32 + q.bsums[(is + 1) * 2 + 1] as i32;
            sum_mins += isum2 * (m2 as i32);
        }

        acc += q.d * d * (sum_qs as f32) - q.d * dmin * (sum_mins as f32);
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::kernels::scalar::vec_dot_q4k_q8k_scalar;
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
        let mut bsums = [0i16; 16];
        for (b, sum) in bsums.iter_mut().enumerate() {
            *sum = qs[b * 16..(b + 1) * 16].iter().map(|&x| x as i16).sum();
        }
        BlockQ8_K {
            d: 0.01 + seed as f32 * 0.001,
            qs,
            bsums,
        }
    }

    fn q4k_block(seed: usize) -> BlockQ4_K {
        let mut qs = [0u8; 128];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = ((seed * 7 + i * 11) % 256) as u8;
        }
        let mut scales = [0u8; 12];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = ((seed + i * 19) % 256) as u8;
        }
        BlockQ4_K {
            d: f16::from_f32(0.02),
            dmin: f16::from_f32(0.01),
            scales,
            qs,
        }
    }

    #[test]
    fn identical_to_scalar() {
        for blocks in [1, 2, 5] {
            let w: Vec<_> = (0..blocks).map(q4k_block).collect();
            let q: Vec<_> = (0..blocks).map(q8k_block).collect();
            let n = blocks * QK_K;
            let expected = vec_dot_q4k_q8k_scalar(n, &w, &q);
            let actual = unsafe { vec_dot_q4k_q8k_neon(n, &w, &q) };
            assert_eq!(
                expected.to_bits(),
                actual.to_bits(),
                "{expected} vs {actual}"
            );
        }
    }

    #[test]
    fn extreme_nibbles_do_not_overflow_or_change_sign() {
        // 15 * -127 is the largest product. All 256 elements with scale 0 would hide
        // it, so the scale bytes are set to a non-zero pattern by q4k_block.
        let mut w = q4k_block(3);
        w.qs = [0xFF; 128];
        let mut q = q8k_block(1);
        q.qs = [-127; 256];
        q.bsums = [-127 * 16; 16];
        let expected = vec_dot_q4k_q8k_scalar(QK_K, &[w], std::slice::from_ref(&q));
        let actual = unsafe { vec_dot_q4k_q8k_neon(QK_K, &[w], std::slice::from_ref(&q)) };
        assert_eq!(expected.to_bits(), actual.to_bits());
    }
}
