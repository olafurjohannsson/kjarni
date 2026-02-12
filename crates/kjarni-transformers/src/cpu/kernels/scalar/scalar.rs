//! Scala kernel implementations for matrix operations.

use crate::cpu::kernels::{
    dequantize::{dequantize_q4_k_block, dequantize_q6_k_block, get_scale_min_k4},
    q_common::{BlockQ4_K, BlockQ6_K, BlockQ8_0, BlockQ8_K, QK_K},
};

/// Computes vector-matrix product for F32 input and BF16 weights.
pub fn matmul_vec_bf16_scalar(out_chunk: &mut [f32], a: &[f32], b_rows: &[u16], k: usize) {
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
pub fn matmul_vec_f32_scalar(out_chunk: &mut [f32], a: &[f32], b_rows: &[f32], k: usize) {
    for (i, out_val) in out_chunk.iter_mut().enumerate() {
        let b_row = &b_rows[i * k..(i + 1) * k];
        let sum: f32 = a.iter().zip(b_row.iter()).map(|(&x, &y)| x * y).sum();
        *out_val = sum;
    }
}

/// Computes vector-matrix product for F32 input and Q8_0 quantized weights.
pub fn matmul_vec_q8_0_scalar(out_chunk: &mut [f32], a: &[f32], b_blocks: &[BlockQ8_0], k: usize) {
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
pub fn vec_dot_q4k_q8k_scalar(n: usize, w_blocks: &[BlockQ4_K], q_blocks: &[BlockQ8_K]) -> f32 {
    let num_blocks = n / QK_K;
    let mut sumf = 0.0f32;

    for i in 0..num_blocks {
        let w = &w_blocks[i];
        let q = &q_blocks[i];

        // Block-level scale factors
        let d = w.d.to_f32();
        let dmin = w.dmin.to_f32();

        let mut sum_qs = 0i32; // Accumulator for scaled quantized products
        let mut sum_mins = 0i32; // Accumulator for min corrections

        let mut is = 0; // Sub-block scale index
        let mut q_idx = 0; // Weight byte index

        for _j in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, &w.scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, &w.scales);

            let q_offset = is * 32;

            let mut sum_q1 = 0i32;
            for l in 0..32 {
                let w_q = (w.qs[q_idx + l] & 0xF) as i32;
                let i_q = q.qs[q_offset + l] as i32;
                sum_q1 += w_q * i_q;
            }
            sum_qs += sum_q1 * (sc1 as i32);

            let isum1 = q.bsums[is * 2] as i32 + q.bsums[is * 2 + 1] as i32;
            sum_mins += isum1 * (m1 as i32);

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

        sumf += q.d * d * (sum_qs as f32) - q.d * dmin * (sum_mins as f32);
    }

    sumf
}

/// Computes dot product of Q6_K weights and Q8_K quantized input.
pub fn vec_dot_q6k_q8k_scalar(n: usize, w_blocks: &[BlockQ6_K], q_blocks: &[BlockQ8_K]) -> f32 {
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
            let ql = &w.ql[j * 64..]; // Low 4 bits of quantized values
            let qh = &w.qh[j * 32..]; // High 2 bits (packed)
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


#[cfg(test)]
mod matmul_scalar_tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use half::f16;
    use crate::cpu::kernels::quantize::quantize_row_q8_k;
    use crate::cpu::kernels::dequantize::dequantize_q8_0_block;

    const TEST_K: usize = 256; 
    const TEST_ROWS: usize = 4; 

    fn get_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn random_f32_vec(rng: &mut StdRng, len: usize) -> Vec<f32> {
        (0..len).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    fn f32_to_bf16_vec(data: &[f32]) -> Vec<u16> {
        data.iter().map(|&x| {
            let bits = x.to_bits();
            (bits >> 16) as u16
        }).collect()
    }
    fn random_q8_0_blocks(rng: &mut StdRng, count: usize) -> Vec<BlockQ8_0> {
        (0..count).map(|_| {
            let mut qs = [0i8; 32];
            for x in &mut qs { *x = rng.gen_range(-127..=127); }
            BlockQ8_0 {
                d: f16::from_f32(rng.gen_range(0.1..2.0)),
                qs
            }
        }).collect()
    }

    fn random_q4k_blocks(rng: &mut StdRng, count: usize) -> Vec<BlockQ4_K> {
        (0..count).map(|_| {
            let mut scales = [0u8; 12];
            let mut qs = [0u8; QK_K / 2];
            rng.fill(&mut scales);
            rng.fill(&mut qs);
            
            BlockQ4_K {
                d: f16::from_f32(rng.gen_range(0.1..1.0)),
                dmin: f16::from_f32(rng.gen_range(0.0..0.1)),
                scales,
                qs,
            }
        }).collect()
    }

    fn random_q6k_blocks(rng: &mut StdRng, count: usize) -> Vec<BlockQ6_K> {
        (0..count).map(|_| {
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut scales = [0i8; 16];
            rng.fill(&mut ql);
            rng.fill(&mut qh);
            rng.fill(&mut scales);

            BlockQ6_K {
                ql,
                qh,
                scales,
                d: f16::from_f32(rng.gen_range(0.1..1.0)),
            }
        }).collect()
    }

    #[test]
    fn test_parity_f32_vs_bf16() {
        let mut rng = get_rng();
        let k = TEST_K * 2;
        
        let input = random_f32_vec(&mut rng, k);
        let weights_f32 = random_f32_vec(&mut rng, k * TEST_ROWS);
        
        let mut out_f32 = vec![0.0; TEST_ROWS];
        let mut out_bf16 = vec![0.0; TEST_ROWS];

        matmul_vec_f32_scalar(&mut out_f32, &input, &weights_f32, k);

        let weights_bf16 = f32_to_bf16_vec(&weights_f32);
        matmul_vec_bf16_scalar(&mut out_bf16, &input, &weights_bf16, k);

        for i in 0..TEST_ROWS {
            let rel_diff = (out_f32[i] - out_bf16[i]).abs() / (out_f32[i].abs() + 1e-6);
            assert!(rel_diff < 0.02, "Row {}: F32 {} vs BF16 {}", i, out_f32[i], out_bf16[i]);
        }
    }

    #[test]
    fn test_parity_f32_vs_q8_0() {
        let mut rng = get_rng();
        let k = TEST_K; 
        
        let input = random_f32_vec(&mut rng, k);

        let blocks_q8 = random_q8_0_blocks(&mut rng, TEST_ROWS * (k / 32));
        let mut weights_f32_ref = vec![0.0f32; TEST_ROWS * k];
        for (i, block) in blocks_q8.iter().enumerate() {
            dequantize_q8_0_block(block, &mut weights_f32_ref[i*32..(i+1)*32]);
        }

        let mut out_f32 = vec![0.0; TEST_ROWS];
        let mut out_q8 = vec![0.0; TEST_ROWS];

        matmul_vec_f32_scalar(&mut out_f32, &input, &weights_f32_ref, k);

        matmul_vec_q8_0_scalar(&mut out_q8, &input, &blocks_q8, k);

        // 5. Compare
        for i in 0..TEST_ROWS {
            let diff = (out_f32[i] - out_q8[i]).abs();
            let magnitude = out_f32[i].abs().max(1.0);
            let rel_diff = diff / magnitude;
            
            assert!(rel_diff < 1e-5, 
                "Row {}: F32 {} vs Q8_0 {} (Diff: {}, Rel: {})", 
                i, out_f32[i], out_q8[i], diff, rel_diff);
        }
    }

    #[test]
    fn test_consistency_q4k_float_vs_int() {
        let mut rng = get_rng();
        let k = TEST_K; 
        
        let input = random_f32_vec(&mut rng, k);
        let weight_blocks = random_q4k_blocks(&mut rng, 1);
        let mut out_float = [0.0f32; 1];
        matmul_vec_q4_k_scalar(&mut out_float, &input, &weight_blocks, k);
        let input_q8k = quantize_row_q8_k(&input);
        
        let val_int = vec_dot_q4k_q8k_scalar(k, &weight_blocks, &input_q8k);

        let diff = (out_float[0] - val_int).abs();
        let magnitude = out_float[0].abs().max(1.0);
        let rel_err = diff / magnitude;
        
        assert!(rel_err < 0.02, "Q4K Float {} vs Int {} (Diff: {}, Rel: {})", out_float[0], val_int, diff, rel_err);
    }

    #[test]
    fn test_consistency_q6k_float_vs_int() {
        let mut rng = get_rng();
        let k = TEST_K; 
        
        let input = random_f32_vec(&mut rng, k);
        let weight_blocks = random_q6k_blocks(&mut rng, 1);
        let mut out_float = [0.0f32; 1];
        matmul_vec_q6_k_scalar(&mut out_float, &input, &weight_blocks, k);
        let input_q8k = quantize_row_q8_k(&input);

        let val_int = vec_dot_q6k_q8k_scalar(k, &weight_blocks, &input_q8k);
        let diff = (out_float[0] - val_int).abs();
        let magnitude = out_float[0].abs().max(1.0);
        let rel_err = diff / magnitude;
        
        assert!(rel_err < 0.02, "Q6K Float {} vs Int {} (Diff: {}, Rel: {})", out_float[0], val_int, diff, rel_err);
    }

    #[test]
    fn test_q8_0_empty_or_zero() {
        let k = 32;
        let a = vec![1.0; k];
        let mut b = BlockQ8_0 { d: f16::from_f32(1.0), qs: [0; 32] };
        let mut out = [0.0];
        matmul_vec_q8_0_scalar(&mut out, &a, &[b], k);
        assert_eq!(out[0], 0.0);
        b.qs.fill(1);
        matmul_vec_q8_0_scalar(&mut out, &a, &[b], k);
        assert_eq!(out[0], 32.0);
    
        b.d = f16::from_f32(0.5);
        matmul_vec_q8_0_scalar(&mut out, &a, &[b], k);
        assert_eq!(out[0], 16.0);
    }

    #[test]
    fn test_bf16_decoding_logic() {
        let a = vec![1.0, 2.0, 1.0];
        let b_rows = vec![
            0x3F80, 
            0x3F00, 
            0xC000, 
        ];
        
        let mut out = [0.0];
        matmul_vec_bf16_scalar(&mut out, &a, &b_rows, 3);
        
        assert_eq!(out[0], 0.0);
    }
}