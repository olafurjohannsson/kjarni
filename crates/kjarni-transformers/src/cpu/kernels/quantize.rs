//! Quantization routines for converting F32 tensors to block-quantized formats.

use anyhow::Result;
use ndarray::{Array2, Axis};

use crate::cpu::kernels::q_common::{BlockQ8_0, BlockQ8_K, QK_K};

/// Block size for Q8_0 quantization (32 int8 values per block).
const Q8_0_BLOCK_SIZE: usize = 32;

/// Quantizes an F32 weight matrix to Q8_0 block format.
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
        for chunk in row.axis_chunks_iter(Axis(0), Q8_0_BLOCK_SIZE) {
            let mut block = BlockQ8_0 {
                d: half::f16::from_f32(0.0),
                qs: [0; 32],
            };

            // Find absolute maximum for scale calculation
            let max_abs = chunk.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));

            if max_abs > 0.0 {
                //  Compute scale (maps max_abs to 127)
                let scale = max_abs / 127.0;
                let inv_scale = 1.0 / scale;
                block.d = half::f16::from_f32(scale);

                //  Quantize each value to int8
                for (i, &val) in chunk.iter().enumerate() {
                    let q = (val * inv_scale).round().clamp(-128.0, 127.0) as i8;
                    block.qs[i] = q;
                }
            }
            out.push(block);
        }
    }

    Ok(out)
}

/// Quantizes an F32 activation row to Q8_K block format.
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

        //Find absolute maximum for scale calculation
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

        //Compute scale (maps max_abs to 127)
        let scale = max_abs / 127.0;
        let inv_scale = 1.0 / scale;
        block.d = scale;

        //Quantize values and compute sub-block sums
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
