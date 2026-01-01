use crate::kernels::q_common::BlockQ6_K;
use std::arch::x86_64::*;

/// Helper to horizontally sum an AVX register
#[inline(always)]
unsafe fn hsum_ps_avx(v: __m256) -> f32 {
    let vlow = _mm256_castps256_ps128(v);
    let vhigh = _mm256_extractf128_ps(v, 1);
    let vsum = _mm_add_ps(vlow, vhigh);
    let vsum = _mm_hadd_ps(vsum, vsum);
    let vsum = _mm_hadd_ps(vsum, vsum);
    _mm_cvtss_f32(vsum)
}

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn matmul_vec_q6_k_avx2(
    out_chunk: &mut [f32],
    a_ptr: *const f32,
    b_blocks: &[BlockQ6_K],
    k: usize,
) {
    let num_blocks_per_row = k / 256;

    for (i, val) in out_chunk.iter_mut().enumerate() {
        let block_start_idx = i * num_blocks_per_row;
        let blocks = &b_blocks[block_start_idx..block_start_idx + num_blocks_per_row];

        let mut total_sum = 0.0;
        let mut a_ptr_block = a_ptr;

        for block in blocks {
            // BlockQ6_K layout:
            // 256 weights total.
            // ql: 128 bytes (lower 4 bits of all 256 weights)
            // qh: 64 bytes (upper 2 bits of all 256 weights)
            // scales: 16 i8 values
            // d: f16 super-scale
            
            let d_val = block.d.to_f32();
            let d_vec = _mm256_set1_ps(d_val);
            
            // We accumulate into 4 registers corresponding to the 4 "streams" 
            // defined by the q6 packing layout (offsets 0, 64, 128, 192).
            let mut sum0 = _mm256_setzero_ps();
            let mut sum1 = _mm256_setzero_ps();
            let mut sum2 = _mm256_setzero_ps();
            let mut sum3 = _mm256_setzero_ps();

            let ql_ptr = block.ql.as_ptr();
            let qh_ptr = block.qh.as_ptr();
            let scales_ptr = block.scales.as_ptr();

            // We iterate 0..32 (processing 2 columns of 4 streams = 8 values per stream)
            // But to use AVX effectively, we step by 32 bytes of data logic?
            // Actually, best strategy for Q6 is processing 32 indices of the 'qh' array at a time.
            // But 'qh' is 64 bytes long. 
            // Let's iterate `is` from 0 to 2 (since 2 * 32 = 64).
            
            // Constant vectors for bit manipulation
            let mask_low4 = _mm256_set1_epi8(0x0F);
            let mask_low2 = _mm256_set1_epi8(0x03);
            let m32 = _mm256_set1_epi8(32); // Offset for q values

            for is in 0..2 { // 2 iterations covering the whole block
                // qh_ptr index base
                let idx_base = is * 32;
                
                // Load 32 bytes of High bits (qh)
                // These 32 bytes contain high bits for 4 * 32 = 128 weights.
                let qh_vec = _mm256_loadu_si256(qh_ptr.add(idx_base) as *const __m256i);

                // Load 32 bytes of Low bits (ql) for First Half (0..32 and 64..96)
                // ql layout: 0..127. 
                // Stream 0 (0..64) maps to ql[0..64]
                // Stream 1 (64..128) maps to ql[0..64] upper nibbles? 
                // NO: Standard GGUF Q6_K:
                // ql[i] = w[i].low4 | (w[i+64].low4 << 4)
                let ql_vec_0 = _mm256_loadu_si256(ql_ptr.add(idx_base) as *const __m256i);
                
                // Load 32 bytes of Low bits (ql) for Second Half (128..160 and 192..224)
                let ql_vec_1 = _mm256_loadu_si256(ql_ptr.add(idx_base + 64) as *const __m256i);

                // --- Extract Scales ---
                // scales[16] total. Each scale covers 16 weights.
                // We are processing 32 indices here.
                // Stream 0 (0..32): covers scales[0] and scales[1] (if is=0)
                // Stream 1 (64..96): covers scales[4] and scales[5]
                // Stream 2 (128..160): covers scales[8] and scales[9]
                // Stream 3 (192..224): covers scales[12] and scales[13]
                
                // This extraction is messy in a loop, so we assume `is` logic:
                // is=0 -> scales indices: 0,1, 4,5, 8,9, 12,13
                // is=1 -> scales indices: 2,3, 6,7, 10,11, 14,15
                
                // Helper to expand scales to f32 vectors
                let get_scale_vec = |offset: usize| {
                    let s_val = *scales_ptr.add(offset);
                    _mm256_mul_ps(_mm256_set1_ps(s_val as f32), d_vec)
                };

                let sc0 = get_scale_vec(is * 2 + 0);
                let sc1 = get_scale_vec(is * 2 + 1);
                let sc2 = get_scale_vec(is * 2 + 4);
                let sc3 = get_scale_vec(is * 2 + 5);
                let sc4 = get_scale_vec(is * 2 + 8);
                let sc5 = get_scale_vec(is * 2 + 9);
                let sc6 = get_scale_vec(is * 2 + 12);
                let sc7 = get_scale_vec(is * 2 + 13);

                // --- Reconstruct Weights ---

                // Stream 0: ql_vec_0 low nibbles | qh bits 0-1
                let q0_lo = _mm256_and_si256(ql_vec_0, mask_low4);
                let q0_hi = _mm256_and_si256(qh_vec, mask_low2);
                let q0 = _mm256_or_si256(q0_lo, _mm256_slli_epi16(q0_hi, 4));
                let q0_i8 = _mm256_sub_epi8(q0, m32); // sub 32

                // Stream 1: ql_vec_0 high nibbles | qh bits 2-3
                let q1_lo = _mm256_and_si256(_mm256_srli_epi16(ql_vec_0, 4), mask_low4);
                let q1_hi = _mm256_and_si256(_mm256_srli_epi16(qh_vec, 2), mask_low2);
                let q1 = _mm256_or_si256(q1_lo, _mm256_slli_epi16(q1_hi, 4));
                let q1_i8 = _mm256_sub_epi8(q1, m32);

                // Stream 2: ql_vec_1 low nibbles | qh bits 4-5
                let q2_lo = _mm256_and_si256(ql_vec_1, mask_low4);
                let q2_hi = _mm256_and_si256(_mm256_srli_epi16(qh_vec, 4), mask_low2);
                let q2 = _mm256_or_si256(q2_lo, _mm256_slli_epi16(q2_hi, 4));
                let q2_i8 = _mm256_sub_epi8(q2, m32);

                // Stream 3: ql_vec_1 high nibbles | qh bits 6-7
                let q3_lo = _mm256_and_si256(_mm256_srli_epi16(ql_vec_1, 4), mask_low4);
                let q3_hi = _mm256_and_si256(_mm256_srli_epi16(qh_vec, 6), mask_low2);
                let q3 = _mm256_or_si256(q3_lo, _mm256_slli_epi16(q3_hi, 4));
                let q3_i8 = _mm256_sub_epi8(q3, m32);

                // --- Dot Product (Splitting into low/high 128 lanes for float conversion) ---
                
                // Helper to process 32 bytes (one stream) against inputs
                // Takes 32-byte i8 vector, two scale vectors (first 16, second 16), and input ptr
                let mut dot_stream = |q_i8: __m256i, s_a: __m256, s_b: __m256, a_off: usize, sum: __m256| -> __m256 {
                    // Split i8 into low/high 128 bits
                    let q_low128 = _mm256_castsi256_si128(q_i8);
                    let q_high128 = _mm256_extracti128_si256(q_i8, 1);
                    
                    // Convert low 16 i8 -> 16 f32 (First Scale)
                    let q_f32_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_low128));
                    let q_f32_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_low128, 8)));
                    
                    let a0 = _mm256_loadu_ps(a_ptr_block.add(a_off));
                    let a1 = _mm256_loadu_ps(a_ptr_block.add(a_off + 8));
                    
                    let mut acc = _mm256_fmadd_ps(_mm256_mul_ps(q_f32_0, s_a), a0, sum);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(q_f32_1, s_a), a1, acc);

                    // Convert high 16 i8 -> 16 f32 (Second Scale)
                    let q_f32_2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_high128));
                    let q_f32_3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_high128, 8)));

                    let a2 = _mm256_loadu_ps(a_ptr_block.add(a_off + 16));
                    let a3 = _mm256_loadu_ps(a_ptr_block.add(a_off + 24));

                    acc = _mm256_fmadd_ps(_mm256_mul_ps(q_f32_2, s_b), a2, acc);
                    acc = _mm256_fmadd_ps(_mm256_mul_ps(q_f32_3, s_b), a3, acc);
                    
                    acc
                };

                // Apply to 4 streams
                let off = is * 32;
                sum0 = dot_stream(q0_i8, sc0, sc1, off + 0, sum0);
                sum1 = dot_stream(q1_i8, sc2, sc3, off + 64, sum1);
                sum2 = dot_stream(q2_i8, sc4, sc5, off + 128, sum2);
                sum3 = dot_stream(q3_i8, sc6, sc7, off + 192, sum3);
            }
            
            a_ptr_block = a_ptr_block.add(256);
            total_sum += hsum_ps_avx(_mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3)));
        }

        *val = total_sum;
    }
}