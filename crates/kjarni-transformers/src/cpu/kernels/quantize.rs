//! Quantization routines for converting F32 tensors to block-quantized formats.
//!
//! This module provides functions for quantizing floating-point weights and activations
//! into compressed integer formats. Quantization reduces memory footprint and enables
//! faster inference through integer arithmetic.
//!
//! # Overview
//!
//! Two quantization formats are supported:
//!
//! - **Q8_0**: 8-bit quantization with 32 elements per block. Each block has a single
//!   F16 scale factor, providing ~4x compression over F32.
//!
//! - **Q8_K**: 8-bit quantization with 256 elements per block (K-quants format). Each
//!   block has an F32 scale factor plus 16 sub-block sums for compatibility with
//!   K-quant dot product kernels.
//!
//! # Quantization Algorithm
//!
//! Both formats use symmetric affine quantization:
//!
//! 1. Find the absolute maximum value in the block: `max_abs = max(|x|)`
//! 2. Compute the scale: `scale = max_abs / 127`
//! 3. Quantize each value: `q = round(x / scale)` clamped to `[-128, 127]`
//!
//! Dequantization recovers approximate values: `x' = q * scale`
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::kernels::quantize::{quantize_matrix_q8_0, quantize_row_q8_k};
//! use ndarray::Array2;
//!
//! // Quantize a weight matrix to Q8_0
//! let weights = Array2::<f32>::zeros((4096, 2048));
//! let q8_blocks = quantize_matrix_q8_0(&weights)?;
//!
//! // Quantize an activation row to Q8_K (for K-quant dot products)
//! let activations = vec![0.0f32; 2048];
//! let q8k_blocks = quantize_row_q8_k(&activations);
//! ```
//!
//! # See Also
//!
//! - [`crate::kernels::q_common`] — Block structure definitions.
//! - [`crate::ops::matmul`] — Matmul functions using quantized weights.

use anyhow::Result;
use ndarray::{Array2, Axis};

use crate::cpu::kernels::q_common::{BlockQ8_0, BlockQ8_K, QK_K};

/// Block size for Q8_0 quantization (32 int8 values per block).
const Q8_0_BLOCK_SIZE: usize = 32;

/// Quantizes an F32 weight matrix to Q8_0 block format.
///
/// Converts a 2D weight matrix into a flat vector of quantized blocks. Each block
/// contains 32 int8 values with a shared F16 scale factor, providing approximately
/// 4x memory compression.
///
/// # Arguments
///
/// * `data` - Weight matrix of shape `[out_features, in_features]`.
///
/// # Returns
///
/// A flat vector of Q8_0 blocks in row-major order. The total number of blocks
/// is `out_features * (in_features / 32)`.
///
/// # Errors
///
/// Returns an error if `in_features` (columns) is not a multiple of 32.
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::kernels::quantize::quantize_matrix_q8_0;
/// use ndarray::Array2;
///
/// let weights = Array2::<f32>::from_elem((4096, 2048), 0.5);
/// let blocks = quantize_matrix_q8_0(&weights)?;
/// assert_eq!(blocks.len(), 4096 * (2048 / 32));
/// ```
///
/// # See Also
///
/// - [`quantize_row_q8_k`] — For activation quantization (Q8_K format).
pub fn quantize_matrix_q8_0(data: &Array2<f32>) -> Result<Vec<BlockQ8_0>> {
    let (rows, cols) = data.dim();

    // Validate that columns are aligned to block size
    if cols % Q8_0_BLOCK_SIZE != 0 {
        anyhow::bail!(
            "Matrix columns ({}) must be a multiple of {} for Q8_0 quantization",
            cols,
            Q8_0_BLOCK_SIZE
        );
    }

    let num_blocks_per_row = cols / Q8_0_BLOCK_SIZE;
    let mut out = Vec::with_capacity(rows * num_blocks_per_row);

    // Process each row independently
    for row in data.outer_iter() {
        // Split row into blocks of 32 elements
        for chunk in row.axis_chunks_iter(Axis(0), Q8_0_BLOCK_SIZE) {
            let mut block = BlockQ8_0 {
                d: half::f16::from_f32(0.0),
                qs: [0; 32],
            };

            // Step 1: Find absolute maximum for scale calculation
            let max_abs = chunk.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));

            if max_abs > 0.0 {
                // Step 2: Compute scale (maps max_abs to 127)
                let scale = max_abs / 127.0;
                let inv_scale = 1.0 / scale;
                block.d = half::f16::from_f32(scale);

                // Step 3: Quantize each value to int8
                for (i, &val) in chunk.iter().enumerate() {
                    let q = (val * inv_scale).round().clamp(-128.0, 127.0) as i8;
                    block.qs[i] = q;
                }
            }
            // Zero block: scale=0 and qs=0 is already correct

            out.push(block);
        }
    }

    Ok(out)
}

/// Quantizes an F32 activation row to Q8_K block format.
///
/// Converts a 1D activation vector into Q8_K blocks for use with K-quant dot product
/// kernels. Each block contains 256 int8 values, an F32 scale factor, and 16 sub-block
/// sums used by optimized dot product implementations.
///
/// This function is typically called once per inference step to quantize the input
/// activations before computing dot products against Q4_K or Q6_K weight matrices.
///
/// # Arguments
///
/// * `data` - Activation vector. Length must be a multiple of 256 (QK_K).
///
/// # Returns
///
/// A vector of Q8_K blocks. The number of blocks is `data.len() / 256`.
///
/// # Panics
///
/// Panics if `data.len()` is not a multiple of 256.
///
/// # Block Structure
///
/// Each Q8_K block contains:
/// - `d`: F32 scale factor for the entire block
/// - `qs`: 256 int8 quantized values
/// - `bsums`: 16 i16 sums, one per 16-element sub-block (used by K-quant kernels)
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::kernels::quantize::quantize_row_q8_k;
///
/// let activations = vec![0.5f32; 2048]; // Must be multiple of 256
/// let blocks = quantize_row_q8_k(&activations);
/// assert_eq!(blocks.len(), 2048 / 256);
/// ```
///
/// # See Also
///
/// - [`quantize_matrix_q8_0`] — For weight matrix quantization.
/// - [`crate::ops::matmul::matmul_2d_cpu_q6_k`] — Uses Q8_K for input quantization.
pub fn quantize_row_q8_k(data: &[f32]) -> Vec<BlockQ8_K> {
    assert_eq!(
        data.len() % QK_K,
        0,
        "Input length must be multiple of {} (QK_K)",
        QK_K
    );

    let num_blocks = data.len() / QK_K;
    let mut out = Vec::with_capacity(num_blocks);

    // Process each 256-element block
    for chunk in data.chunks_exact(QK_K) {
        let mut block = BlockQ8_K {
            d: 0.0,
            qs: [0; 256],
            bsums: [0; 16],
        };

        // Step 1: Find absolute maximum for scale calculation
        let mut max_abs = 0.0f32;
        for &x in chunk {
            let abs_x = x.abs();
            if abs_x > max_abs {
                max_abs = abs_x;
            }
        }

        // Zero block: skip quantization, push empty block
        if max_abs == 0.0 {
            out.push(block);
            continue;
        }

        // Step 2: Compute scale (maps max_abs to 127)
        let scale = max_abs / 127.0;
        let inv_scale = 1.0 / scale;
        block.d = scale;

        // Step 3: Quantize values and compute sub-block sums
        // The block is divided into 16 sub-blocks of 16 elements each.
        // bsums[j] stores the sum of quantized values in sub-block j.
        for j in 0..16 {
            let mut sum = 0i16;

            for i in 0..16 {
                let idx = j * 16 + i;
                let val = chunk[idx];

                // Quantize with clamping to int8 range
                let scaled = (val * inv_scale).round();
                let q = if scaled >= 127.0 {
                    127
                } else if scaled <= -128.0 {
                    -128
                } else {
                    scaled as i8
                };

                block.qs[idx] = q;
                sum += q as i16;
            }

            block.bsums[j] = sum;
        }

        out.push(block);
    }

    out
}
