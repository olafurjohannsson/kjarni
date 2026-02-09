use crate::cpu::kernels::q_common::BlockQ6_K;
use std::arch::x86_64::*;

// horizontally sum an AVX register
#[inline(always)]
unsafe fn hsum_ps_avx(v: __m256) -> f32 {
    unsafe {
        let vlow = _mm256_castps256_ps128(v);
        let vhigh = _mm256_extractf128_ps(v, 1);
        let vsum = _mm_add_ps(vlow, vhigh);
        let vsum = _mm_hadd_ps(vsum, vsum);
        let vsum = _mm_hadd_ps(vsum, vsum);
        _mm_cvtss_f32(vsum)
    }
}


#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn matmul_vec_q6_k_avx2(
    out_chunk: &mut [f32],
    a_ptr: *const f32,
    b_blocks: &[BlockQ6_K],
    k: usize,
) {
    let num_blocks_per_row = k / 256;
    unsafe {
        for (i, val) in out_chunk.iter_mut().enumerate() {
            let block_start_idx = i * num_blocks_per_row;
            let blocks = &b_blocks[block_start_idx..block_start_idx + num_blocks_per_row];

            let mut total_sum = 0.0;
            let mut a_ptr_block = a_ptr;

            for block in blocks {
                unsafe {
                    let d_val = block.d.to_f32();
                    let d_vec = _mm256_set1_ps(d_val);

                    let mut sum0 = _mm256_setzero_ps();
                    let mut sum1 = _mm256_setzero_ps();
                    let mut sum2 = _mm256_setzero_ps();
                    let mut sum3 = _mm256_setzero_ps();

                    let ql_ptr = block.ql.as_ptr();
                    let qh_ptr = block.qh.as_ptr();
                    let scales_ptr = block.scales.as_ptr();

                    let mask_low4 = _mm256_set1_epi8(0x0F);
                    let mask_low2 = _mm256_set1_epi8(0x03);
                    let m32 = _mm256_set1_epi8(32); 

                    // Iterate 2 halves (0..128 and 128..256)
                    for is in 0..2 {
                        // qh: 32 bytes per half (0..32 or 32..64)
                        let qh_base = is * 32;
                        let qh_vec = _mm256_loadu_si256(qh_ptr.add(qh_base) as *const __m256i);

                        // ql: 64 bytes per half (0..64 or 64..128)
                        let ql_base = is * 64;
                        let ql_vec_0 = _mm256_loadu_si256(ql_ptr.add(ql_base) as *const __m256i);
                        let ql_vec_1 = _mm256_loadu_si256(ql_ptr.add(ql_base + 32) as *const __m256i);

                        // Scales: 8 scales per half (0..8 or 8..16)
                        let s_base = is * 8;
                        let get_scale_vec = |offset: usize| {
                            let s_val = *scales_ptr.add(s_base + offset);
                            _mm256_mul_ps(_mm256_set1_ps(s_val as f32), d_vec)
                        };

                        // Stream 0: q0 (indices 0..32 in half) -> ql_vec_0 low, qh bits 0-1
                        let q0_lo = _mm256_and_si256(ql_vec_0, mask_low4);
                        let q0_hi = _mm256_and_si256(qh_vec, mask_low2);
                        let q0 = _mm256_or_si256(q0_lo, _mm256_slli_epi16(q0_hi, 4));
                        let q0_i8 = _mm256_sub_epi8(q0, m32);
                        let sc0 = get_scale_vec(0);
                        let sc1 = get_scale_vec(1);

                        // Stream 1: q2 (indices 64..96 in half) -> ql_vec_0 high, qh bits 4-5
                        let q2_lo = _mm256_and_si256(_mm256_srli_epi16(ql_vec_0, 4), mask_low4);
                        let q2_hi = _mm256_and_si256(_mm256_srli_epi16(qh_vec, 4), mask_low2);
                        let q2 = _mm256_or_si256(q2_lo, _mm256_slli_epi16(q2_hi, 4));
                        let q2_i8 = _mm256_sub_epi8(q2, m32);
                        let sc4 = get_scale_vec(4);
                        let sc5 = get_scale_vec(5);

                        // Stream 2: q1 (indices 32..64 in half) -> ql_vec_1 low, qh bits 2-3
                        let q1_lo = _mm256_and_si256(ql_vec_1, mask_low4);
                        let q1_hi = _mm256_and_si256(_mm256_srli_epi16(qh_vec, 2), mask_low2);
                        let q1 = _mm256_or_si256(q1_lo, _mm256_slli_epi16(q1_hi, 4));
                        let q1_i8 = _mm256_sub_epi8(q1, m32);
                        let sc2 = get_scale_vec(2);
                        let sc3 = get_scale_vec(3);

                        // Stream 3: q3 (indices 96..128 in half) -> ql_vec_1 high, qh bits 6-7
                        let q3_lo = _mm256_and_si256(_mm256_srli_epi16(ql_vec_1, 4), mask_low4);
                        let q3_hi = _mm256_and_si256(_mm256_srli_epi16(qh_vec, 6), mask_low2);
                        let q3 = _mm256_or_si256(q3_lo, _mm256_slli_epi16(q3_hi, 4));
                        let q3_i8 = _mm256_sub_epi8(q3, m32);
                        let sc6 = get_scale_vec(6);
                        let sc7 = get_scale_vec(7);

                        //  Dot Product
                        let dot_stream = |q_i8: __m256i,
                                              s_a: __m256,
                                              s_b: __m256,
                                              a_off: usize,
                                              sum: __m256|
                                              -> __m256 {
                            let q_low128 = _mm256_castsi256_si128(q_i8);
                            let q_high128 = _mm256_extracti128_si256(q_i8, 1);

                            let q_f32_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_low128));
                            let q_f32_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_low128, 8)));
                            let a0 = _mm256_loadu_ps(a_ptr_block.add(a_off));
                            let a1 = _mm256_loadu_ps(a_ptr_block.add(a_off + 8));

                            let mut acc = _mm256_fmadd_ps(_mm256_mul_ps(q_f32_0, s_a), a0, sum);
                            acc = _mm256_fmadd_ps(_mm256_mul_ps(q_f32_1, s_a), a1, acc);

                            let q_f32_2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_high128));
                            let q_f32_3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_high128, 8)));
                            let a2 = _mm256_loadu_ps(a_ptr_block.add(a_off + 16));
                            let a3 = _mm256_loadu_ps(a_ptr_block.add(a_off + 24));

                            acc = _mm256_fmadd_ps(_mm256_mul_ps(q_f32_2, s_b), a2, acc);
                            acc = _mm256_fmadd_ps(_mm256_mul_ps(q_f32_3, s_b), a3, acc);
                            acc
                        };

                        let input_base = is * 128;
                        sum0 = dot_stream(q0_i8, sc0, sc1, input_base + 0, sum0);   // q0
                        sum1 = dot_stream(q2_i8, sc4, sc5, input_base + 64, sum1);  // q2
                        sum2 = dot_stream(q1_i8, sc2, sc3, input_base + 32, sum2);  // q1
                        sum3 = dot_stream(q3_i8, sc6, sc7, input_base + 96, sum3);  // q3
                    }

                    a_ptr_block = a_ptr_block.add(256);
                    total_sum += hsum_ps_avx(_mm256_add_ps(
                        _mm256_add_ps(sum0, sum1),
                        _mm256_add_ps(sum2, sum3),
                    ));
                }
            }
            *val = total_sum;
        }
    }
}


#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod q6_k_avx2_test {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use half::f16;
    use crate::cpu::kernels::dequantize::dequantize_q6_k_block;

    fn ground_truth_matmul(a: &[f32], b_blocks: &[BlockQ6_K], k: usize) -> Vec<f32> {
        let blocks_per_row = k / 256;
        let rows = b_blocks.len() / blocks_per_row;
        let mut out = vec![0.0; rows];
        
        let mut weights_buf = [0.0f32; 256];

        for i in 0..rows {
            let mut row_sum = 0.0;
            for b in 0..blocks_per_row {
                let block_idx = i * blocks_per_row + b;
                
                dequantize_q6_k_block(&b_blocks[block_idx], &mut weights_buf);
                
                let input_chunk = &a[b * 256..(b + 1) * 256];
                row_sum += weights_buf.iter()
                    .zip(input_chunk.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f32>();
            }
            out[i] = row_sum;
        }
        out
    }

    fn get_rng() -> StdRng {
        StdRng::seed_from_u64(99)
    }

    fn random_f32_vec(rng: &mut StdRng, len: usize) -> Vec<f32> {
        (0..len).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    fn random_q6k_block(rng: &mut StdRng) -> BlockQ6_K {
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
            d: f16::from_f32(rng.gen_range(0.5..1.5)),
        }
    }

    #[test]
    fn test_avx2_q6k_correctness() {
        if !is_x86_feature_detected!("avx2") {
            println!("Skipping AVX2 Q6_K test");
            return;
        }

        let mut rng = get_rng();
        let k = 256 * 2; // 2 blocks width
        let rows = 2;

        let input = random_f32_vec(&mut rng, k);
        let blocks: Vec<BlockQ6_K> = (0..rows * 2).map(|_| random_q6k_block(&mut rng)).collect();
        
        let expected = ground_truth_matmul(&input, &blocks, k);

        let mut actual = vec![0.0f32; rows];
        unsafe {
            matmul_vec_q6_k_avx2(
                &mut actual, 
                input.as_ptr(), 
                &blocks, 
                k
            );
        }

        for i in 0..rows {
            let diff = (expected[i] - actual[i]).abs();
            let rel_err = diff / expected[i].abs().max(1.0);
            
            assert!(rel_err < 1e-4, 
                "Row {}: Ref {} vs AVX {} (RelErr: {})", i, expected[i], actual[i], rel_err);
        }
    }
}