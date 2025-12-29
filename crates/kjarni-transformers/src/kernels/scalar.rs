//! Scalar, hardware-agnostic kernel implementations.
//!
//! This module provides the baseline computation kernels that are guaranteed to work
//! on any architecture. They serve as the reference implementation for correctness.

use crate::kernels::{dequantize::{dequantize_q4_k_block, dequantize_q6_k_block}, q_common::BlockQ6_K};

use super::q_common::{BlockQ4_K, BlockQ8_0, QK_K};

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
pub fn matmul_vec_q4_k_scalar(
    out_chunk: &mut [f32],
    a: &[f32],
    b_blocks: &[BlockQ4_K],
    k: usize,
) {
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
pub fn matmul_vec_q6_k_scalar(
    out_chunk: &mut [f32],
    a: &[f32],
    b_blocks: &[BlockQ6_K],
    k: usize,
) {
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
