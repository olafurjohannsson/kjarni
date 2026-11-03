#![allow(unsafe_code)]
use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) unsafe fn matmul_vec_bf16_neon(
    out_chunk: &mut [f32],
    a_ptr: *const f32,
    b_row_start: *const u16,
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

                let b_u16_0 = vld1q_u16(b_chunk_ptr);
                let b_u16_1 = vld1q_u16(b_chunk_ptr.add(8));

                let b0_u32 = vmovl_u16(vget_low_u16(b_u16_0));
                let b1_u32 = vmovl_high_u16(b_u16_0);
                let b2_u32 = vmovl_u16(vget_low_u16(b_u16_1));
                let b3_u32 = vmovl_high_u16(b_u16_1);

                let shift = vdupq_n_s32(16);
                let b0_f = vreinterpretq_f32_u32(vshlq_u32(b0_u32, shift));
                let b1_f = vreinterpretq_f32_u32(vshlq_u32(b1_u32, shift));
                let b2_f = vreinterpretq_f32_u32(vshlq_u32(b2_u32, shift));
                let b3_f = vreinterpretq_f32_u32(vshlq_u32(b3_u32, shift));

                sum0 = vfmaq_f32(sum0, a0, b0_f);
                sum1 = vfmaq_f32(sum1, a1, b1_f);
                sum2 = vfmaq_f32(sum2, a2, b2_f);
                sum3 = vfmaq_f32(sum3, a3, b3_f);

                a_chunk_ptr = a_chunk_ptr.add(16);
                b_chunk_ptr = b_chunk_ptr.add(16);
                n -= 16;
            }

            let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
            while n > 0 {
                sum += *a_chunk_ptr * f32::from_bits((*b_chunk_ptr as u32) << 16);
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
    use crate::cpu::kernels::scalar::matmul_vec_bf16_scalar;

    #[test]
    fn matches_hand_computed_value() {
        // 0x4000 is 2.0 in bf16. 19 elements covers the wide loop and the tail.
        let a: Vec<f32> = (0..19).map(|i| i as f32).collect();
        let b = vec![0x4000u16; 19];
        let mut out = [0.0f32];
        unsafe { matmul_vec_bf16_neon(&mut out, a.as_ptr(), b.as_ptr(), 19) };
        assert_eq!(out, [342.0]);
    }

    #[test]
    fn matches_scalar_across_rows() {
        let (k, rows) = (70, 5);
        let a: Vec<f32> = (0..k).map(|i| (i as f32 - 30.0) * 0.1).collect();
        let b: Vec<u16> = (0..k * rows)
            .map(|i| half::bf16::from_f32((i % 23) as f32 * 0.2 - 2.0).to_bits())
            .collect();
        let mut expected = vec![0.0f32; rows];
        let mut actual = vec![0.0f32; rows];
        matmul_vec_bf16_scalar(&mut expected, &a, &b, k);
        unsafe { matmul_vec_bf16_neon(&mut actual, a.as_ptr(), b.as_ptr(), k) };
        for (e, g) in expected.iter().zip(&actual) {
            assert!((e - g).abs() <= e.abs().max(1.0) * 1e-5, "{e} vs {g}");
        }
    }
}
