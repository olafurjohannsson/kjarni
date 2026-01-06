#![allow(unsafe_code)]
use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) unsafe fn matmul_vec_f32_neon(
    out_chunk: &mut [f32],
    a_ptr: *const f32,
    b_row_start: *const f32,
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

            let b0 = vld1q_f32(b_chunk_ptr);
            let b1 = vld1q_f32(b_chunk_ptr.add(4));
            let b2 = vld1q_f32(b_chunk_ptr.add(8));
            let b3 = vld1q_f32(b_chunk_ptr.add(12));

            sum_v = vfmaq_f32(sum_v, a0, b0);
            sum_v = vfmaq_f32(sum_v, a1, b1);
            sum_v = vfmaq_f32(sum_v, a2, b2);
            sum_v = vfmaq_f32(sum_v, a3, b3);

            a_chunk_ptr = a_chunk_ptr.add(16);
            b_chunk_ptr = b_chunk_ptr.add(16);
            n -= 16;
        }

        let mut sum = vaddvq_f32(sum_v);
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