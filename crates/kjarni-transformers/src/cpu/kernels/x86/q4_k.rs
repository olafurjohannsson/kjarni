#![allow(unsafe_code)]
use crate::cpu::kernels::{
    dequantize::get_scale_min_k4, q_common::BlockQ4_K, x86::common::hsum_ps_avx,
};
use std::arch::x86_64::*;

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn matmul_vec_q4_k_avx2(
    out_chunk: &mut [f32],
    a_ptr: *const f32,
    b_blocks: &[BlockQ4_K],
    k: usize,
) {
    let num_blocks_per_row = k / 256;

    for (i, val) in out_chunk.iter_mut().enumerate() {
        let block_start_idx = i * num_blocks_per_row;
        let blocks = &b_blocks[block_start_idx..block_start_idx + num_blocks_per_row];

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let mut a_ptr_block = a_ptr;
        unsafe {
            for block in blocks {
                let d_vec = _mm256_set1_ps(block.d.to_f32());
                let dmin_vec = _mm256_set1_ps(block.dmin.to_f32());
                let qs_ptr = block.qs.as_ptr();

                // PART 0 (Bytes 0-31, Weights 0-63) ---
                {
                    let (sc1, m1) = get_scale_min_k4(0, &block.scales);
                    let (sc2, m2) = get_scale_min_k4(1, &block.scales);

                    let scale_vec1 = _mm256_mul_ps(_mm256_set1_ps(sc1 as f32), d_vec);
                    let min_vec1 = _mm256_mul_ps(_mm256_set1_ps(m1 as f32), dmin_vec);
                    let scale_vec2 = _mm256_mul_ps(_mm256_set1_ps(sc2 as f32), d_vec);
                    let min_vec2 = _mm256_mul_ps(_mm256_set1_ps(m2 as f32), dmin_vec);

                    let q_packed1 = _mm_loadu_si128(qs_ptr as *const __m128i);
                    let q_packed2 = _mm_loadu_si128(qs_ptr.add(16) as *const __m128i);

                    let a0 = _mm256_loadu_ps(a_ptr_block);
                    let a1 = _mm256_loadu_ps(a_ptr_block.add(8));
                    let a2 = _mm256_loadu_ps(a_ptr_block.add(16));
                    let a3 = _mm256_loadu_ps(a_ptr_block.add(24));

                    let q_low1 = _mm_and_si128(q_packed1, _mm_set1_epi8(0x0F));
                    let q_low2 = _mm_and_si128(q_packed2, _mm_set1_epi8(0x0F));

                    let q_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_low1));
                    let q_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_low1, 8)));
                    let q_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_low2));
                    let q_f4 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_low2, 8)));

                    acc0 = _mm256_fmadd_ps(
                        a0,
                        _mm256_sub_ps(_mm256_mul_ps(q_f1, scale_vec1), min_vec1),
                        acc0,
                    );
                    acc0 = _mm256_fmadd_ps(
                        a1,
                        _mm256_sub_ps(_mm256_mul_ps(q_f2, scale_vec1), min_vec1),
                        acc0,
                    );
                    acc0 = _mm256_fmadd_ps(
                        a2,
                        _mm256_sub_ps(_mm256_mul_ps(q_f3, scale_vec1), min_vec1),
                        acc0,
                    );
                    acc0 = _mm256_fmadd_ps(
                        a3,
                        _mm256_sub_ps(_mm256_mul_ps(q_f4, scale_vec1), min_vec1),
                        acc0,
                    );

                    // Processing high nibbles for Part 0
                    let a4 = _mm256_loadu_ps(a_ptr_block.add(32));
                    let a5 = _mm256_loadu_ps(a_ptr_block.add(40));
                    let a6 = _mm256_loadu_ps(a_ptr_block.add(48));
                    let a7 = _mm256_loadu_ps(a_ptr_block.add(56));

                    let q_hi1 = _mm_and_si128(_mm_srli_epi16(q_packed1, 4), _mm_set1_epi8(0x0F));
                    let q_hi2 = _mm_and_si128(_mm_srli_epi16(q_packed2, 4), _mm_set1_epi8(0x0F));

                    let q_f5 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi1));
                    let q_f6 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi1, 8)));
                    let q_f7 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi2));
                    let q_f8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi2, 8)));

                    acc0 = _mm256_fmadd_ps(
                        a4,
                        _mm256_sub_ps(_mm256_mul_ps(q_f5, scale_vec2), min_vec2),
                        acc0,
                    );
                    acc0 = _mm256_fmadd_ps(
                        a5,
                        _mm256_sub_ps(_mm256_mul_ps(q_f6, scale_vec2), min_vec2),
                        acc0,
                    );
                    acc0 = _mm256_fmadd_ps(
                        a6,
                        _mm256_sub_ps(_mm256_mul_ps(q_f7, scale_vec2), min_vec2),
                        acc0,
                    );
                    acc0 = _mm256_fmadd_ps(
                        a7,
                        _mm256_sub_ps(_mm256_mul_ps(q_f8, scale_vec2), min_vec2),
                        acc0,
                    );
                }

                // PART 1 (Bytes 32-63, Weights 64-127) ---
                {
                    let (sc1, m1) = get_scale_min_k4(2, &block.scales);
                    let (sc2, m2) = get_scale_min_k4(3, &block.scales);

                    let scale_vec1 = _mm256_mul_ps(_mm256_set1_ps(sc1 as f32), d_vec);
                    let min_vec1 = _mm256_mul_ps(_mm256_set1_ps(m1 as f32), dmin_vec);
                    let scale_vec2 = _mm256_mul_ps(_mm256_set1_ps(sc2 as f32), d_vec);
                    let min_vec2 = _mm256_mul_ps(_mm256_set1_ps(m2 as f32), dmin_vec);

                    let offset = 32;
                    let q_packed1 = _mm_loadu_si128(qs_ptr.add(offset) as *const __m128i);
                    let q_packed2 = _mm_loadu_si128(qs_ptr.add(offset + 16) as *const __m128i);

                    let a_base = a_ptr_block.add(64);

                    let a0 = _mm256_loadu_ps(a_base);
                    let a1 = _mm256_loadu_ps(a_base.add(8));
                    let a2 = _mm256_loadu_ps(a_base.add(16));
                    let a3 = _mm256_loadu_ps(a_base.add(24));

                    let q_low1 = _mm_and_si128(q_packed1, _mm_set1_epi8(0x0F));
                    let q_low2 = _mm_and_si128(q_packed2, _mm_set1_epi8(0x0F));

                    let q_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_low1));
                    let q_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_low1, 8)));
                    let q_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_low2));
                    let q_f4 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_low2, 8)));

                    acc1 = _mm256_fmadd_ps(
                        a0,
                        _mm256_sub_ps(_mm256_mul_ps(q_f1, scale_vec1), min_vec1),
                        acc1,
                    );
                    acc1 = _mm256_fmadd_ps(
                        a1,
                        _mm256_sub_ps(_mm256_mul_ps(q_f2, scale_vec1), min_vec1),
                        acc1,
                    );
                    acc1 = _mm256_fmadd_ps(
                        a2,
                        _mm256_sub_ps(_mm256_mul_ps(q_f3, scale_vec1), min_vec1),
                        acc1,
                    );
                    acc1 = _mm256_fmadd_ps(
                        a3,
                        _mm256_sub_ps(_mm256_mul_ps(q_f4, scale_vec1), min_vec1),
                        acc1,
                    );

                    let a4 = _mm256_loadu_ps(a_base.add(32));
                    let a5 = _mm256_loadu_ps(a_base.add(40));
                    let a6 = _mm256_loadu_ps(a_base.add(48));
                    let a7 = _mm256_loadu_ps(a_base.add(56));

                    let q_hi1 = _mm_and_si128(_mm_srli_epi16(q_packed1, 4), _mm_set1_epi8(0x0F));
                    let q_hi2 = _mm_and_si128(_mm_srli_epi16(q_packed2, 4), _mm_set1_epi8(0x0F));

                    let q_f5 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi1));
                    let q_f6 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi1, 8)));
                    let q_f7 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi2));
                    let q_f8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi2, 8)));

                    acc1 = _mm256_fmadd_ps(
                        a4,
                        _mm256_sub_ps(_mm256_mul_ps(q_f5, scale_vec2), min_vec2),
                        acc1,
                    );
                    acc1 = _mm256_fmadd_ps(
                        a5,
                        _mm256_sub_ps(_mm256_mul_ps(q_f6, scale_vec2), min_vec2),
                        acc1,
                    );
                    acc1 = _mm256_fmadd_ps(
                        a6,
                        _mm256_sub_ps(_mm256_mul_ps(q_f7, scale_vec2), min_vec2),
                        acc1,
                    );
                    acc1 = _mm256_fmadd_ps(
                        a7,
                        _mm256_sub_ps(_mm256_mul_ps(q_f8, scale_vec2), min_vec2),
                        acc1,
                    );
                }

                {
                    let (sc1, m1) = get_scale_min_k4(4, &block.scales);
                    let (sc2, m2) = get_scale_min_k4(5, &block.scales);

                    let scale_vec1 = _mm256_mul_ps(_mm256_set1_ps(sc1 as f32), d_vec);
                    let min_vec1 = _mm256_mul_ps(_mm256_set1_ps(m1 as f32), dmin_vec);
                    let scale_vec2 = _mm256_mul_ps(_mm256_set1_ps(sc2 as f32), d_vec);
                    let min_vec2 = _mm256_mul_ps(_mm256_set1_ps(m2 as f32), dmin_vec);

                    let offset = 64;
                    let q_packed1 = _mm_loadu_si128(qs_ptr.add(offset) as *const __m128i);
                    let q_packed2 = _mm_loadu_si128(qs_ptr.add(offset + 16) as *const __m128i);

                    let a_base = a_ptr_block.add(128);

                    let a0 = _mm256_loadu_ps(a_base);
                    let a1 = _mm256_loadu_ps(a_base.add(8));
                    let a2 = _mm256_loadu_ps(a_base.add(16));
                    let a3 = _mm256_loadu_ps(a_base.add(24));

                    let q_low1 = _mm_and_si128(q_packed1, _mm_set1_epi8(0x0F));
                    let q_low2 = _mm_and_si128(q_packed2, _mm_set1_epi8(0x0F));

                    let q_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_low1));
                    let q_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_low1, 8)));
                    let q_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_low2));
                    let q_f4 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_low2, 8)));

                    acc2 = _mm256_fmadd_ps(
                        a0,
                        _mm256_sub_ps(_mm256_mul_ps(q_f1, scale_vec1), min_vec1),
                        acc2,
                    );
                    acc2 = _mm256_fmadd_ps(
                        a1,
                        _mm256_sub_ps(_mm256_mul_ps(q_f2, scale_vec1), min_vec1),
                        acc2,
                    );
                    acc2 = _mm256_fmadd_ps(
                        a2,
                        _mm256_sub_ps(_mm256_mul_ps(q_f3, scale_vec1), min_vec1),
                        acc2,
                    );
                    acc2 = _mm256_fmadd_ps(
                        a3,
                        _mm256_sub_ps(_mm256_mul_ps(q_f4, scale_vec1), min_vec1),
                        acc2,
                    );

                    let a4 = _mm256_loadu_ps(a_base.add(32));
                    let a5 = _mm256_loadu_ps(a_base.add(40));
                    let a6 = _mm256_loadu_ps(a_base.add(48));
                    let a7 = _mm256_loadu_ps(a_base.add(56));

                    let q_hi1 = _mm_and_si128(_mm_srli_epi16(q_packed1, 4), _mm_set1_epi8(0x0F));
                    let q_hi2 = _mm_and_si128(_mm_srli_epi16(q_packed2, 4), _mm_set1_epi8(0x0F));

                    let q_f5 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi1));
                    let q_f6 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi1, 8)));
                    let q_f7 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi2));
                    let q_f8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi2, 8)));

                    acc2 = _mm256_fmadd_ps(
                        a4,
                        _mm256_sub_ps(_mm256_mul_ps(q_f5, scale_vec2), min_vec2),
                        acc2,
                    );
                    acc2 = _mm256_fmadd_ps(
                        a5,
                        _mm256_sub_ps(_mm256_mul_ps(q_f6, scale_vec2), min_vec2),
                        acc2,
                    );
                    acc2 = _mm256_fmadd_ps(
                        a6,
                        _mm256_sub_ps(_mm256_mul_ps(q_f7, scale_vec2), min_vec2),
                        acc2,
                    );
                    acc2 = _mm256_fmadd_ps(
                        a7,
                        _mm256_sub_ps(_mm256_mul_ps(q_f8, scale_vec2), min_vec2),
                        acc2,
                    );
                }

                {
                    let (sc1, m1) = get_scale_min_k4(6, &block.scales);
                    let (sc2, m2) = get_scale_min_k4(7, &block.scales);

                    let scale_vec1 = _mm256_mul_ps(_mm256_set1_ps(sc1 as f32), d_vec);
                    let min_vec1 = _mm256_mul_ps(_mm256_set1_ps(m1 as f32), dmin_vec);
                    let scale_vec2 = _mm256_mul_ps(_mm256_set1_ps(sc2 as f32), d_vec);
                    let min_vec2 = _mm256_mul_ps(_mm256_set1_ps(m2 as f32), dmin_vec);

                    let offset = 96;
                    let q_packed1 = _mm_loadu_si128(qs_ptr.add(offset) as *const __m128i);
                    let q_packed2 = _mm_loadu_si128(qs_ptr.add(offset + 16) as *const __m128i);

                    let a_base = a_ptr_block.add(192);

                    let a0 = _mm256_loadu_ps(a_base);
                    let a1 = _mm256_loadu_ps(a_base.add(8));
                    let a2 = _mm256_loadu_ps(a_base.add(16));
                    let a3 = _mm256_loadu_ps(a_base.add(24));

                    let q_low1 = _mm_and_si128(q_packed1, _mm_set1_epi8(0x0F));
                    let q_low2 = _mm_and_si128(q_packed2, _mm_set1_epi8(0x0F));

                    let q_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_low1));
                    let q_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_low1, 8)));
                    let q_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_low2));
                    let q_f4 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_low2, 8)));

                    acc3 = _mm256_fmadd_ps(
                        a0,
                        _mm256_sub_ps(_mm256_mul_ps(q_f1, scale_vec1), min_vec1),
                        acc3,
                    );
                    acc3 = _mm256_fmadd_ps(
                        a1,
                        _mm256_sub_ps(_mm256_mul_ps(q_f2, scale_vec1), min_vec1),
                        acc3,
                    );
                    acc3 = _mm256_fmadd_ps(
                        a2,
                        _mm256_sub_ps(_mm256_mul_ps(q_f3, scale_vec1), min_vec1),
                        acc3,
                    );
                    acc3 = _mm256_fmadd_ps(
                        a3,
                        _mm256_sub_ps(_mm256_mul_ps(q_f4, scale_vec1), min_vec1),
                        acc3,
                    );

                    let a4 = _mm256_loadu_ps(a_base.add(32));
                    let a5 = _mm256_loadu_ps(a_base.add(40));
                    let a6 = _mm256_loadu_ps(a_base.add(48));
                    let a7 = _mm256_loadu_ps(a_base.add(56));

                    let q_hi1 = _mm_and_si128(_mm_srli_epi16(q_packed1, 4), _mm_set1_epi8(0x0F));
                    let q_hi2 = _mm_and_si128(_mm_srli_epi16(q_packed2, 4), _mm_set1_epi8(0x0F));

                    let q_f5 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi1));
                    let q_f6 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi1, 8)));
                    let q_f7 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi2));
                    let q_f8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi2, 8)));

                    acc3 = _mm256_fmadd_ps(
                        a4,
                        _mm256_sub_ps(_mm256_mul_ps(q_f5, scale_vec2), min_vec2),
                        acc3,
                    );
                    acc3 = _mm256_fmadd_ps(
                        a5,
                        _mm256_sub_ps(_mm256_mul_ps(q_f6, scale_vec2), min_vec2),
                        acc3,
                    );
                    acc3 = _mm256_fmadd_ps(
                        a6,
                        _mm256_sub_ps(_mm256_mul_ps(q_f7, scale_vec2), min_vec2),
                        acc3,
                    );
                    acc3 = _mm256_fmadd_ps(
                        a7,
                        _mm256_sub_ps(_mm256_mul_ps(q_f8, scale_vec2), min_vec2),
                        acc3,
                    );
                }

                a_ptr_block = a_ptr_block.add(256);
            }

            let total_sum = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            *val = hsum_ps_avx(total_sum);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::kernels::q_common::BlockQ4_K;
    use crate::cpu::kernels::scalar::matmul_vec_q4_k_scalar;
    use half::f16;

    /// Create a deterministic Q4_K block with non-trivial values.
    /// This avoids degenerate cases (e.g. all zeros) while remaining predictable.
    fn create_test_block(seed: u8) -> BlockQ4_K {
        let mut qs = [0u8; 128];
        for (i, q) in qs.iter_mut().enumerate() {
            // Two 4-bit values per byte, range 0..15
            let lo = ((i as u8 + seed) & 0x0F) as u8;
            let hi = ((i as u8 + seed + 3) & 0x0F) as u8;
            *q = lo | (hi << 4);
        }

        BlockQ4_K {
            d: f16::from_f32(0.125),
            dmin: f16::from_f32(0.01),
            // Non-zero scales/mins to exercise all code paths
            scales: [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4],
            qs,
        }
    }

    /// The AVX2-enabled test body.
    /// This MUST live in a separate function with `#[target_feature]`.
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn run_q4_k_avx2_vs_scalar_test() {
        let k = 256;
        let m = 4; // number of output rows

        // Input vector
        let a: Vec<f32> = (0..k).map(|i| i as f32 * 0.01).collect();

        // Weight blocks: m rows × (k / 256) blocks per row
        let blocks: Vec<BlockQ4_K> = (0..m)
            .flat_map(|row| (0..(k / 256)).map(move |b| create_test_block((row * 7 + b) as u8)))
            .collect();

        let mut out_scalar = vec![0.0f32; m];
        let mut out_avx2 = vec![0.0f32; m];

        // Reference computation
        matmul_vec_q4_k_scalar(&mut out_scalar, &a, &blocks, k);

        // AVX2 computation
        unsafe { matmul_vec_q4_k_avx2(&mut out_avx2, a.as_ptr(), &blocks, k) };

        // Compare results
        for i in 0..m {
            let diff = (out_scalar[i] - out_avx2[i]).abs();
            assert!(
                diff < 1e-3,
                "Q4_K AVX2 mismatch at row {}: scalar={} avx2={} diff={}",
                i,
                out_scalar[i],
                out_avx2[i],
                diff
            );
        }
    }

    /// Public test entry point.
    /// This function is SAFE and can be called by the test harness.
    #[test]
    fn test_matmul_vec_q4_k_avx2_matches_scalar() {
        // Runtime feature detection is mandatory
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                run_q4_k_avx2_vs_scalar_test();
            }
        } else {
            // On non-AVX2 machines we skip silently
            eprintln!("skipping Q4_K AVX2 test: CPU does not support AVX2+FMA");
        }
    }
}
