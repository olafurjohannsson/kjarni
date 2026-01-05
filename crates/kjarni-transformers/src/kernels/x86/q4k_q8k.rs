#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::kernels::dequantize::get_scale_min_k4;
use crate::kernels::q_common::{BlockQ4_K, BlockQ8_K, QK_K};

#[inline(always)]
unsafe fn hsum_i32_8(a: __m256i) -> i32 {
    unsafe {
        // Add upper and lower 128-bit halves
        let sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
        // Add upper and lower 64-bit halves
        let hi64 = _mm_unpackhi_epi64(sum128, sum128);
        let sum64 = _mm_add_epi32(hi64, sum128);
        // Add upper and lower 32-bit halves
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
        // Constants for masking and shuffling
        let m4 = _mm256_set1_epi8(0xF);

        for i in 0..num_blocks {
            let w = &w_blocks[i];
            let q = &q_blocks[i];

            let d = w.d.to_f32();
            let dmin = w.dmin.to_f32();

            let mut sum_qs = 0;
            let mut sum_mins = 0;

            let mut is = 0;

            // Pointers for vector loads
            let w_ptr = w.qs.as_ptr();
            let q_ptr = q.qs.as_ptr();

            // Process 4 pairs of sub-blocks (8 sub-blocks total)
            // Each iteration processes 64 weights (32 low nibbles + 32 high nibbles)
            for j in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, &w.scales);
                let (sc2, m2) = get_scale_min_k4(is + 1, &w.scales);

                // --- Load 32 bytes (64 weights) of Input Q8 ---
                // q_ptr offset: j * 64
                let q_offset = j * 64;
                let q_v1 = _mm256_loadu_si256(q_ptr.add(q_offset) as *const __m256i);
                let q_v2 = _mm256_loadu_si256(q_ptr.add(q_offset + 32) as *const __m256i);

                // --- Load 32 bytes (64 packed weights) of Weights Q4 ---
                // w_ptr offset: j * 32
                // Note: We load 32 bytes, but we only use them split into nibbles
                let w_offset = j * 32;
                let w_packed = _mm256_loadu_si256(w_ptr.add(w_offset) as *const __m256i);

                // --- Unpack Nibbles ---
                // Low nibbles (first 32 weights)
                let w_low = _mm256_and_si256(w_packed, m4);
                // High nibbles (next 32 weights) - shift right by 4
                // Note: AVX2 doesn't have _mm256_srli_epi8, so we use logic or 16-bit shifts
                // Standard trick: (x >> 4) & 0xF
                // But _mm256_srli_epi16 shifts 16-bit words.
                // 0xAB -> 0x0A. 0xCD -> 0x0C. Correct for bytes if we mask.
                let w_high_shifted = _mm256_srli_epi16(w_packed, 4);
                let w_high = _mm256_and_si256(w_high_shifted, m4);

                // --- Dot Product (maddubs) ---
                // Multiply unsigned w (u8) * signed q (i8) -> saturating i16
                let dot1 = _mm256_maddubs_epi16(w_low, q_v1);
                let dot2 = _mm256_maddubs_epi16(w_high, q_v2);

                // --- Horizontal Sum to i32 ---
                // madd_epi16 with 1s horizontally adds adjacent i16 pairs to i32
                let ones = _mm256_set1_epi16(1);
                let sum1 = _mm256_madd_epi16(dot1, ones);
                let sum2 = _mm256_madd_epi16(dot2, ones);

                // Extract sums from YMM to scalar
                // (This is slightly lazy, fully horizontal AVX sum is faster but more complex code)
                // let mut arr1 = [0i32; 8];
                // let mut arr2 = [0i32; 8];
                // _mm256_storeu_si256(arr1.as_mut_ptr() as *mut __m256i, sum1);
                // _mm256_storeu_si256(arr2.as_mut_ptr() as *mut __m256i, sum2);

                // let s1: i32 = arr1.iter().sum();
                // let s2: i32 = arr2.iter().sum();

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
