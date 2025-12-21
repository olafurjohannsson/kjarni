
//! Scalar, hardware-agnostic kernel implementations.
//!
//! This module provides the baseline computation kernels that are guaranteed to work
//! on any architecture. They serve as the reference implementation for correctness.

use super::q_common::{BlockQ4_K, BlockQ8_0, QK_K, QS_K};

/// Scalar implementation of vector-matrix multiplication for BF16 weights.
pub(crate) fn matmul_vec_bf16_scalar(
    out_chunk: &mut [f32],
    a: &[f32],
    b_rows: &[u16],
    k: usize,
) {
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
pub(crate) fn matmul_vec_f32_scalar(
    out_chunk: &mut [f32],
    a: &[f32],
    b_rows: &[f32],
    k: usize,
) {
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
            // CORRECTED: Convert f16 scale to f32 *before* the loop.
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

/// Scalar implementation of vector-matrix multiplication for Q4_K weights.
pub(crate) fn matmul_vec_q4_k_scalar(
    out_chunk: &mut [f32],
    a: &[f32],
    b_blocks: &[BlockQ4_K],
    k: usize,
) {
    for (i, out_val) in out_chunk.iter_mut().enumerate() {
        let row_blocks = &b_blocks[i * (k / QK_K)..];
        let mut sum = 0.0f32;

        for (block_idx, a_chunk) in a.chunks_exact(QK_K).enumerate() {
            let block = &row_blocks[block_idx];
            // CORRECTED: Convert super-block scales to f32 once.
            let d = block.d.to_f32();
            let dmin = block.dmin.to_f32();
            let mut block_sum = 0.0f32;

            // A Q4_K block has 8 sub-blocks of 32 elements each.
            for j in 0..QK_K / QS_K { // j iterates from 0 to 7
                // CORRECTED: Unpack the 4-bit scale and 4-bit min from the `scales` array.
                let scale_and_min_byte = if j < 4 {
                    block.scales[j]
                } else {
                    block.scales[j + 4]
                };
                
                let sub_block_scale = d * (scale_and_min_byte & 0x0F) as f32;
                let sub_block_min = dmin * (scale_and_min_byte >> 4) as f32;

                let q_idx_base = j * (QS_K / 2); // Each sub-block uses 16 bytes of `qs`
                let a_idx_base = j * QS_K;

                for l in 0..QS_K / 2 { // l iterates from 0 to 15
                    let q_byte = block.qs[q_idx_base + l];
                    
                    // Unpack two 4-bit values from the byte
                    let q1 = q_byte & 0x0F;
                    let q2 = q_byte >> 4;

                    // Dequantize and accumulate
                    block_sum += a_chunk[a_idx_base + l * 2] * (q1 as f32 * sub_block_scale + sub_block_min);
                    block_sum += a_chunk[a_idx_base + l * 2 + 1] * (q2 as f32 * sub_block_scale + sub_block_min);
                }
            }
            sum += block_sum;
        }
        *out_val = sum;
    }
}