use ndarray::{Array2, Axis};
use anyhow::Result;
use crate::kernels::q_common::{BlockQ8_0, BlockQ8_K, QK_K};

/// Quantizes a full F32 weight matrix (rows = out_features, cols = in_features)
/// into a flat vector of Q8_0 blocks.
pub fn quantize_matrix_q8_0(data: &Array2<f32>) -> Result<Vec<BlockQ8_0>> {
    let (rows, cols) = data.dim();
    let k_per_block = 32; // Q8_0 block size
    if cols % k_per_block != 0 {
        anyhow::bail!("Matrix columns ({}) must be a multiple of {} for Q8_0 quantization", cols, k_per_block);
    }
    
    let num_blocks_per_row = cols / k_per_block;
    let mut out = Vec::with_capacity(rows * num_blocks_per_row);

    for row in data.outer_iter() {
        // Use axis_chunks_iter for ndarray views. This is the correct equivalent.
        for chunk in row.axis_chunks_iter(Axis(0), k_per_block) {
            let mut block = BlockQ8_0 { d: half::f16::from_f32(0.0), qs: [0; 32] };

            // Find the absolute maximum value in the chunk to determine the scale
            let max_abs = chunk.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));

            if max_abs > 0.0 {
                let scale = max_abs / 127.0;
                let iscale = 1.0 / scale;
                block.d = half::f16::from_f32(scale);
                // Quantize each value in the chunk
                for (i, &val) in chunk.iter().enumerate() {
                    let q = (val * iscale).round().clamp(-128.0, 127.0) as i8;
                    block.qs[i] = q;
                }
            }
            // If max_abs is 0, the block is already all zeros, which is correct.
            out.push(block);
        }
    }
    Ok(out)
}

/// Quantizes a row of F32 activations into Q8_K blocks.
/// This is done ONCE per inference step.
pub fn quantize_row_q8_k(data: &[f32]) -> Vec<BlockQ8_K> {
    assert_eq!(data.len() % QK_K, 0, "Input length must be multiple of 256");

    // #[cfg(target_arch = "x86_64")]
    // if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
    //     unsafe {
    //         return crate::kernels::x86::q4k_q8k::quantize_row_q8_k_avx2(data);
    //     }
    // }

    let num_blocks = data.len() / QK_K;
    let mut out = Vec::with_capacity(num_blocks);

    for chunk in data.chunks_exact(QK_K) {
        let mut block = BlockQ8_K {
            d: 0.0,
            qs: [0; 256],
            bsums: [0; 16],
        };

        // 1. Find absolute maximum
        let mut max_abs = 0.0f32;
        for &x in chunk {
            let abs_x = x.abs();
            if abs_x > max_abs {
                max_abs = abs_x;
            }
        }

        // 2. Calculate scale (map max_abs to 127)
        if max_abs == 0.0 {
            out.push(block); // All zeros
            continue;
        }

        let d = max_abs / 127.0;
        let id = 1.0 / d;
        block.d = d;

        // 3. Quantize and calculate block sums
        for j in 0..16 {
            // 16 sub-blocks of 16 elements
            let mut sum = 0i16;
            for i in 0..16 {
                let idx = j * 16 + i;
                let val = chunk[idx];

                // Round to nearest integer
                // let q = (val * id).round() as i8;
                let scaled = (val * id).round();
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
