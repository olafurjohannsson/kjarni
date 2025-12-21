
//! Scalar, hardware-agnostic kernel implementations.
//!
//! This module provides the baseline computation kernels that are guaranteed to work
//! on any architecture. They serve as the reference implementation for correctness.

use crate::kernels::q_common::BlockQ6_K;

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

pub(crate) fn matmul_vec_q4_k_scalar(out_chunk: &mut [f32], a: &[f32], b_blocks: &[BlockQ4_K], k: usize) {
    for (i, out_val) in out_chunk.iter_mut().enumerate() {
        let row_blocks = &b_blocks[i * (k / QK_K)..];
        let mut sum = 0.0f32;
        let mut temp_w = [0.0f32; QK_K];

        for (block_idx, a_chunk) in a.chunks_exact(QK_K).enumerate() {
            let b = &row_blocks[block_idx];
            dequantize_q4_k(b, &mut temp_w);
            for j in 0..QK_K {
                sum += a_chunk[j] * temp_w[j];
            }
        }
        *out_val = sum;
    }
}

pub(crate) fn matmul_vec_q6_k_scalar(out_chunk: &mut [f32], a: &[f32], b_blocks: &[BlockQ6_K], k: usize) {
    for (i, out_val) in out_chunk.iter_mut().enumerate() {
        let row_blocks = &b_blocks[i * (k / QK_K)..];
        let mut sum = 0.0f32;
        let mut temp_w = [0.0f32; QK_K];

        for (block_idx, a_chunk) in a.chunks_exact(QK_K).enumerate() {
            let b = &row_blocks[block_idx];
            dequantize_q6_k(b, &mut temp_w);
            for j in 0..QK_K {
                sum += a_chunk[j] * temp_w[j];
            }
        }
        *out_val = sum;
    }
}

fn dequantize_q4_k(b: &BlockQ4_K, out: &mut [f32]) {
    let d = b.d.to_f32();
    let dmin = b.dmin.to_f32();
    let mut is = 0;
    for j in 0..QK_K / 64 {
        let sc = d * (b.scales[is] & 0xF) as f32;
        let m = dmin * (b.scales[is] >> 4) as f32;
        for l in 0..32 {
            out[j * 64 + l] = sc * (b.qs[j * 32 + l] & 0xF) as f32 - m;
        }
        is += 1;
        let sc = d * (b.scales[is] & 0xF) as f32;
        let m = dmin * (b.scales[is] >> 4) as f32;
        for l in 0..32 {
            out[j * 64 + 32 + l] = sc * (b.qs[j * 32 + l] >> 4) as f32 - m;
        }
        is += 1;
    }
}

fn dequantize_q6_k(b: &BlockQ6_K, out: &mut [f32]) {
    let d = b.d.to_f32();
    for j in 0..QK_K / 64 {
        let base_ql = j * 32;
        let base_qh = j * 16;
        let sc_base = j * 4;
        
        for l in 0..32 {
            let ql = b.ql[base_ql + l];
            let qh = (b.qh[base_qh + l % 16] >> (2 * (l / 16))) & 3;
            let q = (ql as i8 | ((qh as i8) << 4)) - 32;
            out[j * 64 + l] = d * q as f32 * b.scales[sc_base + l / 16] as f32;
            
            let ql_h = b.ql[base_ql + l] >> 4;
            let q = (ql_h as i8 | ((qh as i8) << 4)) - 32; // This is a simplified bit mapping
            out[j * 64 + 32 + l] = d * q as f32 * b.scales[sc_base + 2 + l / 16] as f32;
        }
    }
}


// #[cfg(target_arch = "x86_64")]
// #[target_feature(enable = "avx2", enable = "fma")]
// pub unsafe fn matmul_vec_q4_k_avx2(out: &mut [f32], a: &[f32], b: &[BlockQ4_K], k: usize) {
//     let mut temp_w = [0.0f32; 256];
//     for (i, out_val) in out.iter_mut().enumerate() {
//         let row_blocks = &b[i * (k / 256)..];
//         let mut sum_vec = _mm256_setzero_ps();

//         for (block_idx, a_chunk) in a.chunks_exact(256).enumerate() {
//             let block = &row_blocks[block_idx];
//             dequantize_q4_k(block, &mut temp_w); // Use the scalar dequantizer for simplicity

//             let mut a_ptr = a_chunk.as_ptr();
//             let mut w_ptr = temp_w.as_ptr();
            
//             for _ in 0..32 { // 256 / 8 = 32
//                 let av = _mm256_loadu_ps(a_ptr);
//                 let wv = _mm256_loadu_ps(w_ptr);
//                 sum_vec = _mm256_fmadd_ps(av, wv, sum_vec);
//                 a_ptr = a_ptr.add(8);
//                 w_ptr = w_ptr.add(8);
//             }
//         }
//         *out_val = hsum_avx(sum_vec);
//     }
// }



pub fn dequantize_q4_k_block(b: &BlockQ4_K, out: &mut [f32]) {
    let d = b.d.to_f32();
    let dmin = b.dmin.to_f32();
    let mut is = 0;
    for j in 0..4 { // 4 super-blocks
        let sc = d * (b.scales[is] & 0xF) as f32;
        let m = dmin * (b.scales[is] >> 4) as f32;
        for l in 0..32 {
            out[j * 64 + l] = sc * (b.qs[j * 32 + l] & 0xF) as f32 - m;
        }
        is += 1;
        let sc = d * (b.scales[is] & 0xF) as f32;
        let m = dmin * (b.scales[is] >> 4) as f32;
        for l in 0..32 {
            out[j * 64 + 32 + l] = sc * (b.qs[j * 32 + l] >> 4) as f32 - m;
        }
        is += 1;
    }
}

pub fn dequantize_q6_k_block(b: &BlockQ6_K, out: &mut [f32]) {
    let d = b.d.to_f32();
    
    // Process 256 elements in two 128-element halves
    for i in 0..2 {
        let off_ql = i * 64;   // 0 or 64
        let off_qh = i * 32;   // 0 or 32
        let off_sc = i * 8;    // 0 or 8
        let off_out = i * 128; // 0 or 128

        for j in 0..32 {
            // High bits byte h contains 2 bits for each of 4 elements
            let h = b.qh[off_qh + j];
            
            // Reassemble the 6-bit values
            // Elements j and j+32 (lower nibbles)
            let q0 = ((b.ql[off_ql + j] & 0xF) | ((h & 0x03) << 4)) as i8 - 32;
            let q1 = ((b.ql[off_ql + j + 32] & 0xF) | ((h & 0x0C) << 2)) as i8 - 32;
            
            // Elements j and j+32 (upper nibbles)
            let q2 = ((b.ql[off_ql + j] >> 4) | ((h & 0x30) << 0)) as i8 - 32;
            let q3 = ((b.ql[off_ql + j + 32] >> 4) | ((h & 0xC0) >> 2)) as i8 - 32;

            // Apply global scale and sub-block scales
            // Q6_K has 16 scales (8 per half). Each scale covers 16 elements.
            let sc = &b.scales; // Use the full array to avoid slice boundary issues
            
            out[off_out + j]      = d * (q0 as f32) * (sc[off_sc + j / 16] as f32);
            out[off_out + j + 32] = d * (q1 as f32) * (sc[off_sc + 2 + j / 16] as f32);
            out[off_out + j + 64] = d * (q2 as f32) * (sc[off_sc + 4 + j / 16] as f32);
            out[off_out + j + 96] = d * (q3 as f32) * (sc[off_sc + 6 + j / 16] as f32);
        }
    }
}