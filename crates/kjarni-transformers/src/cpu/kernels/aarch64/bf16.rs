#![allow(unsafe_code)]
use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) unsafe fn matmul_vec_bf16_neon(
    out_chunk: &mut [f32],
    a_ptr: *const f32,
    b_row_start: *const u16,
    k: usize,
) {
    let mut b_row_ptr = b_row_start;
    for val in out_chunk.iter_mut() {
        let mut a_chunk_ptr = a_ptr;
        let mut b_chunk_ptr = b_row_ptr;
        let mut sum_v = vdupq_n_f32(0.0);
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

            sum_v = vfmaq_f32(sum_v, a0, b0_f);
            sum_v = vfmaq_f32(sum_v, a1, b1_f);
            sum_v = vfmaq_f32(sum_v, a2, b2_f);
            sum_v = vfmaq_f32(sum_v, a3, b3_f);

            a_chunk_ptr = a_chunk_ptr.add(16);
            b_chunk_ptr = b_chunk_ptr.add(16);
            n -= 16;
        }

        let mut sum = vaddvq_f32(sum_v);
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