#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::cpu::kernels::{
    dequantize::get_scale_min_k4,
    q_common::{BlockQ4_K, BlockQ8_K, QK_K},
};

#[inline(always)]
unsafe fn hsum_i32_8(a: __m256i) -> i32 {
    unsafe {
        // Add upper and lower 128-bit halves
        let sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
        // Add upper and lower 64-bit halves
        let hi64 = _mm_unpackhi_epi64(sum128, sum128);
        let sum64 = _mm_add_epi32(hi64, sum128);
        let hi32 = _mm_shuffle_epi32(sum64, 0b10_11_00_01); // _MM_SHUFFLE(2,3,0,1)
        _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32))
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn vec_dot_q4k_q8k_avx2(
    n: usize,
    w_blocks: &[BlockQ4_K],
    q_blocks: &[BlockQ8_K],
) -> f32 {
    let num_blocks = n / QK_K;
    let mut acc = 0.0f32;
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);

        for i in 0..num_blocks {
            let w = &w_blocks[i];
            let q = &q_blocks[i];

            let d = w.d.to_f32();
            let dmin = w.dmin.to_f32();

            let mut sum_qs = 0;
            let mut sum_mins = 0;

            let mut is = 0;

            let w_ptr = w.qs.as_ptr();
            let q_ptr = q.qs.as_ptr();

            for j in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, &w.scales);
                let (sc2, m2) = get_scale_min_k4(is + 1, &w.scales);

                let q_offset = j * 64;
                let q_v1 = _mm256_loadu_si256(q_ptr.add(q_offset) as *const __m256i);
                let q_v2 = _mm256_loadu_si256(q_ptr.add(q_offset + 32) as *const __m256i);

                let w_offset = j * 32;
                let w_packed = _mm256_loadu_si256(w_ptr.add(w_offset) as *const __m256i);

                let w_low = _mm256_and_si256(w_packed, m4);
                let w_high_shifted = _mm256_srli_epi16(w_packed, 4);
                let w_high = _mm256_and_si256(w_high_shifted, m4);

                let dot1 = _mm256_maddubs_epi16(w_low, q_v1);
                let dot2 = _mm256_maddubs_epi16(w_high, q_v2);

                let ones = _mm256_set1_epi16(1);
                let sum1 = _mm256_madd_epi16(dot1, ones);
                let sum2 = _mm256_madd_epi16(dot2, ones);

                let s1 = hsum_i32_8(sum1);
                let s2 = hsum_i32_8(sum2);
                // --- Accumulate with Scales ---
                sum_qs += s1 * (sc1 as i32);
                sum_qs += s2 * (sc2 as i32);

                // --- Accumulate Mins ---
                let isum1 = q.bsums[is * 2] as i32 + q.bsums[is * 2 + 1] as i32;
                sum_mins += isum1 * (m1 as i32);

                let isum2 = q.bsums[(is + 1) * 2] as i32 + q.bsums[(is + 1) * 2 + 1] as i32;
                sum_mins += isum2 * (m2 as i32);

                is += 2;
            }

            acc += q.d * d * (sum_qs as f32) - q.d * dmin * (sum_mins as f32);
        }
    }
    acc
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod q4k_q8k_test {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use half::f16;
    use crate::cpu::kernels::dequantize::dequantize_q4_k_block;

    fn dequantize_q8k_to_f32(block: &BlockQ8_K) -> Vec<f32> {
        block.qs.iter()
            .map(|&q| q as f32 * block.d)
            .collect()
    }

    fn ground_truth_dot_product(w_blocks: &[BlockQ4_K], q_blocks: &[BlockQ8_K]) -> f32 {
        let mut total = 0.0;
        let mut w_f32 = [0.0f32; QK_K];

        for (w, q) in w_blocks.iter().zip(q_blocks.iter()) {
            dequantize_q4_k_block(w, &mut w_f32);
        
            let q_f32 = dequantize_q8k_to_f32(q);
            
            total += w_f32.iter()
                .zip(q_f32.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>();
        }
        total
    }


    fn get_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn random_q4k_block(rng: &mut StdRng) -> BlockQ4_K {
        let mut scales = [0u8; 12];
        let mut qs = [0u8; QK_K / 2];
        rng.fill(&mut scales);
        rng.fill(&mut qs);
        BlockQ4_K {
            d: f16::from_f32(rng.gen_range(0.5..1.5)),
            dmin: f16::from_f32(rng.gen_range(0.0..0.5)),
            scales,
            qs,
        }
    }

    fn random_q8k_block(rng: &mut StdRng) -> BlockQ8_K {
        let mut qs = [0i8; 256];
        let mut bsums = [0i16; 16];
        rng.fill(&mut qs);
        
        for i in 0..16 {
            let sum: i32 = qs[i*16..(i+1)*16].iter().map(|&x| x as i32).sum();
            bsums[i] = sum as i16;
        }

        BlockQ8_K {
            d: rng.gen_range(0.001..0.1),
            qs,
            bsums,
        }
    }

    #[test]
    fn test_avx2_q4k_q8k_correctness() {
        if !is_x86_feature_detected!("avx2") {
            println!("Skipping AVX2 test");
            return;
        }
        let mut rng = get_rng();
        let n = QK_K * 4; 

        let w_blocks: Vec<BlockQ4_K> = (0..4).map(|_| random_q4k_block(&mut rng)).collect();
        let q_blocks: Vec<BlockQ8_K> = (0..4).map(|_| random_q8k_block(&mut rng)).collect();

        let expected = ground_truth_dot_product(&w_blocks, &q_blocks);

        let actual = unsafe { vec_dot_q4k_q8k_avx2(n, &w_blocks, &q_blocks) };

        let diff = (expected - actual).abs();
        let rel_err = diff / expected.abs().max(1.0);

        assert!(rel_err < 5e-4, 
            "AVX2 mismatch! Expected: {}, Actual: {}, RelErr: {}", 
            expected, actual, rel_err);
    }
}