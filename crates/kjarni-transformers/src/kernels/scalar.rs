//! Scalar, hardware-agnostic kernel implementations.
//!
//! This module provides the baseline computation kernels that are guaranteed to work
//! on any architecture. They serve as the reference implementation for correctness.

use crate::kernels::{
    dequantize::{dequantize_q4_k_block, dequantize_q6_k_block, get_scale_min_k4},
    q_common::{BlockQ6_K, BlockQ8_K},
};

use super::q_common::{BlockQ4_K, BlockQ8_0, QK_K};

/// Scalar implementation of vector-matrix multiplication for BF16 weights.
pub(crate) fn matmul_vec_bf16_scalar(out_chunk: &mut [f32], a: &[f32], b_rows: &[u16], k: usize) {
    for (i, out_val) in out_chunk.iter_mut().enumerate() {
        let b_row = &b_rows[i * k..(i + 1) * k];
        let sum: f32 = a
            .iter()
            .zip(b_row.iter())
            .map(|(&a_val, &b_val)| a_val * f32::from_bits((b_val as u32) << 16))
            .sum();
        *out_val = sum;
    }
}

/// Scalar implementation of vector-matrix multiplication for F32 weights.
pub(crate) fn matmul_vec_f32_scalar(out_chunk: &mut [f32], a: &[f32], b_rows: &[f32], k: usize) {
    for (i, out_val) in out_chunk.iter_mut().enumerate() {
        let b_row = &b_rows[i * k..(i + 1) * k];
        let sum: f32 = a.iter().zip(b_row.iter()).map(|(&x, &y)| x * y).sum();
        *out_val = sum;
    }
}

/// Scalar implementation of vector-matrix multiplication for Q8_0 weights.
pub(crate) fn matmul_vec_q8_0_scalar(
    out_chunk: &mut [f32],
    a: &[f32],
    b_blocks: &[BlockQ8_0],
    k: usize,
) {
    let k_per_block = std::mem::size_of_val(&b_blocks[0].qs);
    for (i, out_val) in out_chunk.iter_mut().enumerate() {
        let row_blocks = &b_blocks[i * (k / k_per_block)..];
        let mut sum = 0.0f32;
        for (block_idx, a_chunk) in a.chunks_exact(k_per_block).enumerate() {
            let block = &row_blocks[block_idx];
            let d = block.d.to_f32();
            let block_sum: f32 = a_chunk
                .iter()
                .zip(block.qs.iter())
                .map(|(&a_val, &q_val)| a_val * (q_val as f32 * d))
                .sum();
            sum += block_sum;
        }
        *out_val = sum;
    }
}

// pub(crate) fn matmul_vec_q4_k_scalar(
pub fn matmul_vec_q4_k_scalar(out_chunk: &mut [f32], a: &[f32], b_blocks: &[BlockQ4_K], k: usize) {
    for (i, out_val) in out_chunk.iter_mut().enumerate() {
        let row_blocks = &b_blocks[i * (k / QK_K)..];
        let mut sum = 0.0f32;
        let mut temp_w = [0.0f32; QK_K];

        for (block_idx, a_chunk) in a.chunks_exact(QK_K).enumerate() {
            let b = &row_blocks[block_idx];
            dequantize_q4_k_block(b, &mut temp_w);
            for j in 0..QK_K {
                sum += a_chunk[j] * temp_w[j];
            }
        }
        *out_val = sum;
    }
}

// pub(crate) fn matmul_vec_q6_k_scalar(
pub fn matmul_vec_q6_k_scalar(out_chunk: &mut [f32], a: &[f32], b_blocks: &[BlockQ6_K], k: usize) {
    for (i, out_val) in out_chunk.iter_mut().enumerate() {
        let row_blocks = &b_blocks[i * (k / QK_K)..];
        let mut sum = 0.0f32;
        let mut temp_w = [0.0f32; QK_K];

        for (block_idx, a_chunk) in a.chunks_exact(QK_K).enumerate() {
            let b = &row_blocks[block_idx];
            dequantize_q6_k_block(b, &mut temp_w);
            for j in 0..QK_K {
                sum += a_chunk[j] * temp_w[j];
            }
        }
        *out_val = sum;
    }
}

/// Computes dot product of Q4_K weights and Q8_K input using integer math.
pub fn vec_dot_q4k_q8k_scalar(
    n: usize,               
    w_blocks: &[BlockQ4_K], 
    q_blocks: &[BlockQ8_K], 
) -> f32 {
    let num_blocks = n / QK_K;
    let mut sumf = 0.0f32;

    for i in 0..num_blocks {
        let w = &w_blocks[i];
        let q = &q_blocks[i];

        let d = w.d.to_f32();
        let dmin = w.dmin.to_f32();
        
        let mut sum_qs = 0;
        let mut sum_mins = 0;

        let mut is = 0; 
        let mut q_idx = 0;

        for _j in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, &w.scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, &w.scales);

            // Calculate base index for Input (q.qs)
            // is goes 0, 2, 4, 6.
            // Each 'is' step represents 32 elements.
            // So q_offset should be is * 32.
            let q_offset = is * 32;

            // --- Sub-block 1 ---
            let mut sum_q1 = 0;
            for l in 0..32 {
                let w_q = (w.qs[q_idx + l] & 0xF) as i32;
                // Corrected Index: q_offset + l
                let i_q = q.qs[q_offset + l] as i32;
                sum_q1 += w_q * i_q;
            }
            sum_qs += sum_q1 * (sc1 as i32);
            
            let isum1 = q.bsums[is * 2] as i32 + q.bsums[is * 2 + 1] as i32;
            sum_mins += isum1 * (m1 as i32);

            // --- Sub-block 2 ---
            let mut sum_q2 = 0;
            for l in 0..32 {
                let w_q = (w.qs[q_idx + l] >> 4) as i32;
                // Corrected Index: q_offset + 32 + l
                let i_q = q.qs[q_offset + 32 + l] as i32;
                sum_q2 += w_q * i_q;
            }
            sum_qs += sum_q2 * (sc2 as i32);

            let isum2 = q.bsums[(is + 1) * 2] as i32 + q.bsums[(is + 1) * 2 + 1] as i32;
            sum_mins += isum2 * (m2 as i32);

            q_idx += 32;
            is += 2;
        }

        sumf += q.d * d * (sum_qs as f32) - q.d * dmin * (sum_mins as f32);
    }

    sumf
}



pub fn vec_dot_q6k_q8k_scalar(
    n: usize,
    w_blocks: &[BlockQ6_K],
    q_blocks: &[BlockQ8_K],
) -> f32 {
    let num_blocks = n / QK_K;
    let mut sumf = 0.0f32;

    for i in 0..num_blocks {
        let w = &w_blocks[i];
        let q = &q_blocks[i];
        let d = w.d.to_f32();

        let mut sum_qs = 0;
        let mut sum_mins = 0; // Q6_K doesn't use mins in the same way, but let's see.

        // Q6_K: value = d * scale * (q - 32)
        // dot = sum( d * scale * (q_w - 32) * q_i * q_d )
        //     = d * q_d * sum( scale * (q_w * q_i - 32 * q_i) )
        
        // We iterate 2 halves
        for j in 0..2 {
            let ql = &w.ql[j * 64..];
            let qh = &w.qh[j * 32..];
            let sc = &w.scales[j * 8..];
            
            // Input offset
            let q_offset = j * 128;

            for k in 0..32 {
                let is = k / 16; // scale index offset (0 or 1)
                
                let qh_val = qh[k];
                let sc0 = sc[is] as i32;
                let sc1 = sc[is + 2] as i32;
                let sc2 = sc[is + 4] as i32;
                let sc3 = sc[is + 6] as i32;

                // 4 elements per k
                // El 0
                let q0_w = ((ql[k] & 0xF) as i32) | (((qh_val & 0x03) as i32) << 4);
                let q0_i = q.qs[q_offset + k] as i32;
                sum_qs += sc0 * (q0_w * q0_i - 32 * q0_i);

                // El 1
                let q1_w = ((ql[k + 32] & 0xF) as i32) | (((qh_val & 0x0C) as i32) << 2);
                let q1_i = q.qs[q_offset + k + 32] as i32;
                sum_qs += sc1 * (q1_w * q1_i - 32 * q1_i);

                // El 2
                let q2_w = ((ql[k] >> 4) as i32) | (((qh_val & 0x30) as i32));
                let q2_i = q.qs[q_offset + k + 64] as i32;
                sum_qs += sc2 * (q2_w * q2_i - 32 * q2_i);

                // El 3
                let q3_w = ((ql[k + 32] >> 4) as i32) | (((qh_val & 0xC0) as i32) >> 2);
                let q3_i = q.qs[q_offset + k + 96] as i32;
                sum_qs += sc3 * (q3_w * q3_i - 32 * q3_i);
            }
        }
        sumf += d * q.d * sum_qs as f32;
    }
    sumf
}