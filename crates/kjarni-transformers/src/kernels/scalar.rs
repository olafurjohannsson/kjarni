//! Scalar (non-SIMD) kernel implementations for matrix operations.
//!
//! This module provides portable, hardware-agnostic computation kernels that work
//! on any CPU architecture. These kernels serve two purposes:
//!
//! 1. **Fallback implementations** when SIMD extensions (AVX2, NEON) are unavailable.
//! 2. **Reference implementations** for validating correctness of optimized kernels.
//!
//! # Overview
//!
//! All kernels compute vector-matrix products of the form `out = A @ B^T` where:
//! - `A` is a single input vector (F32)
//! - `B` is a weight matrix in various formats (F32, BF16, Q8_0, Q4_K, Q6_K)
//! - `out` is the output vector (F32)
//!
//! For quantized formats, the kernels handle on-the-fly dequantization during
//! the dot product computation.
//!
//! # Performance
//!
//! These kernels prioritize correctness and portability over speed. For production
//! workloads on x86_64, use the AVX2+FMA kernels in [`crate::kernels::x86`]. On
//! aarch64, use the NEON kernels in [`crate::kernels::aarch64`].
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::kernels::scalar::matmul_vec_f32_scalar;
//!
//! let input = vec![1.0f32; 2048];
//! let weights = vec![0.5f32; 2048 * 4096]; // [4096, 2048] flattened
//! let mut output = vec![0.0f32; 4096];
//!
//! matmul_vec_f32_scalar(&mut output, &input, &weights, 2048);
//! ```
//!
//! # See Also
//!
//! - [`crate::kernels::x86`] — AVX2+FMA optimized kernels.
//! - [`crate::kernels::aarch64`] — ARM NEON optimized kernels.
//! - [`crate::ops::matmul`] — High-level dispatchers that select the best kernel.

use crate::kernels::{
    dequantize::{dequantize_q4_k_block, dequantize_q6_k_block, get_scale_min_k4},
    q_common::{BlockQ6_K, BlockQ8_K},
};

use super::q_common::{BlockQ4_K, BlockQ8_0, QK_K};

/// Computes vector-matrix product for F32 input and BF16 weights.
///
/// Performs `out[i] = dot(a, b_rows[i])` for each output element, converting
/// BF16 weights to F32 on-the-fly using bit manipulation.
///
/// # Arguments
///
/// * `out_chunk` - Output slice to write results into.
/// * `a` - Input vector of length `k`.
/// * `b_rows` - Flattened weight matrix stored as raw u16 (BF16 bit patterns).
///   Shape is `[out_chunk.len(), k]` in row-major order.
/// * `k` - Number of input features (inner dimension).
///
/// # BF16 Conversion
///
/// BF16 to F32 conversion is done by left-shifting the 16-bit value by 16 bits,
/// effectively placing the BF16 mantissa and exponent in the upper bits of an
/// F32 representation: `f32::from_bits((bf16_bits as u32) << 16)`.
pub(crate) fn matmul_vec_bf16_scalar(out_chunk: &mut [f32], a: &[f32], b_rows: &[u16], k: usize) {
    for (i, out_val) in out_chunk.iter_mut().enumerate() {
        // Extract this output's weight row
        let b_row = &b_rows[i * k..(i + 1) * k];

        // Compute dot product with BF16->F32 conversion
        let sum: f32 = a
            .iter()
            .zip(b_row.iter())
            .map(|(&a_val, &b_val)| {
                // BF16 to F32: shift left by 16 bits to align exponent/mantissa
                a_val * f32::from_bits((b_val as u32) << 16)
            })
            .sum();

        *out_val = sum;
    }
}

/// Computes vector-matrix product for F32 input and F32 weights.
///
/// Performs `out[i] = dot(a, b_rows[i])` for each output element.
/// This is the simplest kernel with no format conversion.
///
/// # Arguments
///
/// * `out_chunk` - Output slice to write results into.
/// * `a` - Input vector of length `k`.
/// * `b_rows` - Flattened weight matrix of shape `[out_chunk.len(), k]`.
/// * `k` - Number of input features (inner dimension).
pub(crate) fn matmul_vec_f32_scalar(out_chunk: &mut [f32], a: &[f32], b_rows: &[f32], k: usize) {
    for (i, out_val) in out_chunk.iter_mut().enumerate() {
        // Extract this output's weight row
        let b_row = &b_rows[i * k..(i + 1) * k];

        // Simple dot product
        let sum: f32 = a.iter().zip(b_row.iter()).map(|(&x, &y)| x * y).sum();
        *out_val = sum;
    }
}

/// Computes vector-matrix product for F32 input and Q8_0 quantized weights.
///
/// Performs `out[i] = dot(a, dequant(b_blocks[i]))` for each output element.
/// Q8_0 weights are dequantized on-the-fly: `value = scale * q` where `q` is
/// the int8 quantized value and `scale` is the per-block F16 scale factor.
///
/// # Arguments
///
/// * `out_chunk` - Output slice to write results into.
/// * `a` - Input vector of length `k`.
/// * `b_blocks` - Q8_0 blocks containing quantized weights.
/// * `k` - Number of input features (must be a multiple of 32).
///
/// # Q8_0 Format
///
/// Each Q8_0 block contains:
/// - `d`: F16 scale factor
/// - `qs`: 32 int8 quantized values
///
/// Dequantization: `value[i] = d * qs[i]`
pub(crate) fn matmul_vec_q8_0_scalar(
    out_chunk: &mut [f32],
    a: &[f32],
    b_blocks: &[BlockQ8_0],
    k: usize,
) {
    // Q8_0 block size: 32 elements per block
    let k_per_block = std::mem::size_of_val(&b_blocks[0].qs);
    let blocks_per_row = k / k_per_block;

    for (i, out_val) in out_chunk.iter_mut().enumerate() {
        // Get blocks for this output row
        let row_blocks = &b_blocks[i * blocks_per_row..];
        let mut sum = 0.0f32;

        // Process each block: dequantize and accumulate dot product
        for (block_idx, a_chunk) in a.chunks_exact(k_per_block).enumerate() {
            let block = &row_blocks[block_idx];
            let scale = block.d.to_f32();

            // Dot product with on-the-fly dequantization
            let block_sum: f32 = a_chunk
                .iter()
                .zip(block.qs.iter())
                .map(|(&a_val, &q_val)| a_val * (q_val as f32 * scale))
                .sum();

            sum += block_sum;
        }

        *out_val = sum;
    }
}

/// Computes vector-matrix product for F32 input and Q4_K quantized weights.
///
/// Performs `out[i] = dot(a, dequant(b_blocks[i]))` for each output element.
/// Q4_K is a 4-bit K-quant format with 256 elements per block. Weights are
/// fully dequantized into a temporary buffer before computing the dot product.
///
/// # Arguments
///
/// * `out_chunk` - Output slice to write results into.
/// * `a` - Input vector of length `k`.
/// * `b_blocks` - Q4_K blocks containing quantized weights.
/// * `k` - Number of input features (must be a multiple of 256).
///
/// # Q4_K Format
///
/// Each Q4_K block contains 256 4-bit quantized values with multiple scale
/// factors for improved accuracy. See [`crate::kernels::dequantize::dequantize_q4_k_block`]
/// for the dequantization algorithm.
///
/// # Performance
///
/// This implementation fully dequantizes each block before the dot product,
/// which is simpler but slower than the integer-only approach used in
/// [`vec_dot_q4k_q8k_scalar`].
pub fn matmul_vec_q4_k_scalar(out_chunk: &mut [f32], a: &[f32], b_blocks: &[BlockQ4_K], k: usize) {
    let blocks_per_row = k / QK_K;

    for (i, out_val) in out_chunk.iter_mut().enumerate() {
        // Get blocks for this output row
        let row_blocks = &b_blocks[i * blocks_per_row..];
        let mut sum = 0.0f32;

        // Temporary buffer for dequantized block values
        let mut temp_w = [0.0f32; QK_K];

        // Process each block: dequantize fully, then compute dot product
        for (block_idx, a_chunk) in a.chunks_exact(QK_K).enumerate() {
            let b = &row_blocks[block_idx];
            dequantize_q4_k_block(b, &mut temp_w);

            // Accumulate dot product
            for j in 0..QK_K {
                sum += a_chunk[j] * temp_w[j];
            }
        }

        *out_val = sum;
    }
}

/// Computes vector-matrix product for F32 input and Q6_K quantized weights.
///
/// Performs `out[i] = dot(a, dequant(b_blocks[i]))` for each output element.
/// Q6_K is a 6-bit K-quant format with 256 elements per block. Weights are
/// fully dequantized into a temporary buffer before computing the dot product.
///
/// # Arguments
///
/// * `out_chunk` - Output slice to write results into.
/// * `a` - Input vector of length `k`.
/// * `b_blocks` - Q6_K blocks containing quantized weights.
/// * `k` - Number of input features (must be a multiple of 256).
///
/// # Q6_K Format
///
/// Each Q6_K block contains 256 6-bit quantized values. The 6-bit values are
/// stored as 4-bit low parts (`ql`) and 2-bit high parts (`qh`), requiring
/// bit manipulation to reconstruct. See [`crate::kernels::dequantize::dequantize_q6_k_block`]
/// for the dequantization algorithm.
///
/// # See Also
///
/// - [`vec_dot_q6k_q8k_scalar`] — Integer-only dot product for Q6_K × Q8_K.
pub fn matmul_vec_q6_k_scalar(out_chunk: &mut [f32], a: &[f32], b_blocks: &[BlockQ6_K], k: usize) {
    let blocks_per_row = k / QK_K;

    for (i, out_val) in out_chunk.iter_mut().enumerate() {
        // Get blocks for this output row
        let row_blocks = &b_blocks[i * blocks_per_row..];
        let mut sum = 0.0f32;

        // Temporary buffer for dequantized block values
        let mut temp_w = [0.0f32; QK_K];

        // Process each block: dequantize fully, then compute dot product
        for (block_idx, a_chunk) in a.chunks_exact(QK_K).enumerate() {
            let b = &row_blocks[block_idx];
            dequantize_q6_k_block(b, &mut temp_w);

            // Accumulate dot product
            for j in 0..QK_K {
                sum += a_chunk[j] * temp_w[j];
            }
        }

        *out_val = sum;
    }
}

/// Computes dot product of Q4_K weights and Q8_K quantized input.
///
/// This function computes the dot product directly in the quantized domain
/// using integer arithmetic, avoiding explicit dequantization. This is more
/// efficient than dequantizing to F32 first.
///
/// # Arguments
///
/// * `n` - Total number of elements (must be a multiple of 256).
/// * `w_blocks` - Q4_K weight blocks.
/// * `q_blocks` - Q8_K quantized input blocks (from [`crate::kernels::quantize::quantize_row_q8_k`]).
///
/// # Returns
///
/// The dot product as F32.
///
/// # Algorithm
///
/// Q4_K stores values as: `value = d * scale * q - dmin * min`
///
/// The dot product is computed as:
/// ```text
/// dot = Σ (d * scale_i * q_w_i * q_i_i * q.d) - (dmin * min_i * bsum_i * q.d)
/// ```
///
/// Where:
/// - `q_w_i` is the 4-bit weight quantized value
/// - `q_i_i` is the 8-bit input quantized value
/// - `scale_i`, `min_i` are per-sub-block scale/min factors
/// - `bsum_i` is the pre-computed sum of input values in the sub-block
///
/// # See Also
///
/// - [`matmul_vec_q4_k_scalar`] — Alternative that dequantizes weights to F32.
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

        // Block-level scale factors
        let d = w.d.to_f32();
        let dmin = w.dmin.to_f32();

        let mut sum_qs = 0i32;   // Accumulator for scaled quantized products
        let mut sum_mins = 0i32; // Accumulator for min corrections

        let mut is = 0;     // Sub-block scale index
        let mut q_idx = 0;  // Weight byte index

        // Process 4 groups of 2 sub-blocks each (8 sub-blocks total, 32 elements each)
        for _j in 0..4 {
            // Get scale and min for two consecutive sub-blocks
            let (sc1, m1) = get_scale_min_k4(is, &w.scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, &w.scales);

            // Input offset: is goes 0,2,4,6; each step = 32 elements
            let q_offset = is * 32;

            // --- Sub-block 1: lower 4 bits ---
            let mut sum_q1 = 0i32;
            for l in 0..32 {
                let w_q = (w.qs[q_idx + l] & 0xF) as i32;
                let i_q = q.qs[q_offset + l] as i32;
                sum_q1 += w_q * i_q;
            }
            sum_qs += sum_q1 * (sc1 as i32);

            // Use pre-computed bsums for min correction
            let isum1 = q.bsums[is * 2] as i32 + q.bsums[is * 2 + 1] as i32;
            sum_mins += isum1 * (m1 as i32);

            // --- Sub-block 2: upper 4 bits ---
            let mut sum_q2 = 0i32;
            for l in 0..32 {
                let w_q = (w.qs[q_idx + l] >> 4) as i32;
                let i_q = q.qs[q_offset + 32 + l] as i32;
                sum_q2 += w_q * i_q;
            }
            sum_qs += sum_q2 * (sc2 as i32);

            let isum2 = q.bsums[(is + 1) * 2] as i32 + q.bsums[(is + 1) * 2 + 1] as i32;
            sum_mins += isum2 * (m2 as i32);

            q_idx += 32;
            is += 2;
        }

        // Final accumulation: apply block scales
        sumf += q.d * d * (sum_qs as f32) - q.d * dmin * (sum_mins as f32);
    }

    sumf
}

/// Computes dot product of Q6_K weights and Q8_K quantized input.
///
/// This function computes the dot product directly in the quantized domain
/// using integer arithmetic. Q6_K stores 6-bit values split across `ql` (low 4 bits)
/// and `qh` (high 2 bits), which are reconstructed on-the-fly.
///
/// # Arguments
///
/// * `n` - Total number of elements (must be a multiple of 256).
/// * `w_blocks` - Q6_K weight blocks.
/// * `q_blocks` - Q8_K quantized input blocks.
///
/// # Returns
///
/// The dot product as F32.
///
/// # Algorithm
///
/// Q6_K stores values as: `value = d * scale * (q - 32)`
///
/// The 6-bit quantized value is reconstructed from:
/// - Low 4 bits from `ql`
/// - High 2 bits from `qh` (packed, 4 values share one byte)
///
/// The dot product computation:
/// ```text
/// dot = d * q.d * Σ scale_i * (q_w_i * q_i_i - 32 * q_i_i)
/// ```
///
/// The `-32` offset is the Q6_K zero-point.
///
/// # See Also
///
/// - [`matmul_vec_q6_k_scalar`] — Alternative that dequantizes weights to F32.
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

        let mut sum_qs = 0i32;

        // Process block in 2 halves (128 elements each)
        for j in 0..2 {
            // Pointers into the Q6_K block data
            let ql = &w.ql[j * 64..];   // Low 4 bits of quantized values
            let qh = &w.qh[j * 32..];   // High 2 bits (packed)
            let sc = &w.scales[j * 8..]; // Per-sub-block scales

            // Input offset for this half
            let q_offset = j * 128;

            // Process 32 positions, each producing 4 dot products
            for k in 0..32 {
                let is = k / 16; // Scale index offset (0 or 1 within this half)

                // High bits shared by 4 elements
                let qh_val = qh[k];

                // Get scales for 4 sub-blocks
                let sc0 = sc[is] as i32;
                let sc1 = sc[is + 2] as i32;
                let sc2 = sc[is + 4] as i32;
                let sc3 = sc[is + 6] as i32;

                // Element 0: low 4 bits from ql[k], high 2 bits from qh[0:1]
                let q0_w = ((ql[k] & 0xF) as i32) | (((qh_val & 0x03) as i32) << 4);
                let q0_i = q.qs[q_offset + k] as i32;
                sum_qs += sc0 * (q0_w * q0_i - 32 * q0_i);

                // Element 1: low 4 bits from ql[k+32], high 2 bits from qh[2:3]
                let q1_w = ((ql[k + 32] & 0xF) as i32) | (((qh_val & 0x0C) as i32) << 2);
                let q1_i = q.qs[q_offset + k + 32] as i32;
                sum_qs += sc1 * (q1_w * q1_i - 32 * q1_i);

                // Element 2: high 4 bits from ql[k], high 2 bits from qh[4:5]
                let q2_w = ((ql[k] >> 4) as i32) | ((qh_val & 0x30) as i32);
                let q2_i = q.qs[q_offset + k + 64] as i32;
                sum_qs += sc2 * (q2_w * q2_i - 32 * q2_i);

                // Element 3: high 4 bits from ql[k+32], high 2 bits from qh[6:7]
                let q3_w = ((ql[k + 32] >> 4) as i32) | (((qh_val & 0xC0) as i32) >> 2);
                let q3_i = q.qs[q_offset + k + 96] as i32;
                sum_qs += sc3 * (q3_w * q3_i - 32 * q3_i);
            }
        }

        // Apply block scales
        sumf += d * q.d * sum_qs as f32;
    }

    sumf
}