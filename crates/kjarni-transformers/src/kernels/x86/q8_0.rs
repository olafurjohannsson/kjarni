#![allow(unsafe_code)]
use crate::kernels::{q_common::BlockQ8_0, x86::common::hsum_ps_avx};
use std::arch::x86_64::*;

/// Computes a vector-matrix multiplication for Q8_0 weights using AVX2.
///
/// This kernel performs on-the-fly dequantization and dot product accumulation.
/// It's designed for the "parallel-on-output" strategy where each thread computes
/// a chunk of the final output vector.
/// 1. Unrolled loop (4 blocks per iter) to hide FMA latency.
/// 2. Uses 4 separate accumulators to maximize ILP.
/// 3. Reduced pointer arithmetic overhead.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn matmul_vec_q8_0_avx2(
    out_chunk: &mut [f32],
    a_ptr: *const f32,
    b_blocks: &[BlockQ8_0],
    k: usize,
) {
    let num_blocks_per_row = k / 32;
    unsafe {
        for (i, val) in out_chunk.iter_mut().enumerate() {
            let block_start_idx = i * num_blocks_per_row;
            let row_blocks = &b_blocks[block_start_idx..block_start_idx + num_blocks_per_row];

            let mut a_ptr_local = a_ptr;

            // 4 separate accumulators to break dependency chains
            let mut sum0 = _mm256_setzero_ps();
            let mut sum1 = _mm256_setzero_ps();
            let mut sum2 = _mm256_setzero_ps();
            let mut sum3 = _mm256_setzero_ps();

            // Process blocks in groups of 4 (4 * 32 = 128 weights)
            let mut chunks = row_blocks.chunks_exact(4);
            while let Some(chunk) = chunks.next() {
                // --- Block 0 ---
                {
                    let b = &chunk[0];
                    let d_vec = _mm256_set1_ps(b.d.to_f32());
                    let q_ptr = b.qs.as_ptr() as *const __m128i;
                    let q_lo = _mm_loadu_si128(q_ptr);
                    let q_hi = _mm_loadu_si128(q_ptr.add(1));

                    let a0 = _mm256_loadu_ps(a_ptr_local);
                    let a1 = _mm256_loadu_ps(a_ptr_local.add(8));

                    let q0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo));
                    let q1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo, 8)));

                    sum0 = _mm256_fmadd_ps(_mm256_mul_ps(q0_f, d_vec), a0, sum0);
                    sum0 = _mm256_fmadd_ps(_mm256_mul_ps(q1_f, d_vec), a1, sum0);

                    let a2 = _mm256_loadu_ps(a_ptr_local.add(16));
                    let a3 = _mm256_loadu_ps(a_ptr_local.add(24));

                    let q2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi));
                    let q3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi, 8)));

                    sum0 = _mm256_fmadd_ps(_mm256_mul_ps(q2_f, d_vec), a2, sum0);
                    sum0 = _mm256_fmadd_ps(_mm256_mul_ps(q3_f, d_vec), a3, sum0);
                }

                // --- Block 1 ---
                {
                    let b = &chunk[1];
                    let d_vec = _mm256_set1_ps(b.d.to_f32());
                    let q_ptr = b.qs.as_ptr() as *const __m128i;
                    let q_lo = _mm_loadu_si128(q_ptr);
                    let q_hi = _mm_loadu_si128(q_ptr.add(1));

                    // Offset A by 32
                    let a0 = _mm256_loadu_ps(a_ptr_local.add(32));
                    let a1 = _mm256_loadu_ps(a_ptr_local.add(40));

                    let q0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo));
                    let q1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo, 8)));

                    sum1 = _mm256_fmadd_ps(_mm256_mul_ps(q0_f, d_vec), a0, sum1);
                    sum1 = _mm256_fmadd_ps(_mm256_mul_ps(q1_f, d_vec), a1, sum1);

                    let a2 = _mm256_loadu_ps(a_ptr_local.add(48));
                    let a3 = _mm256_loadu_ps(a_ptr_local.add(56));

                    let q2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi));
                    let q3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi, 8)));

                    sum1 = _mm256_fmadd_ps(_mm256_mul_ps(q2_f, d_vec), a2, sum1);
                    sum1 = _mm256_fmadd_ps(_mm256_mul_ps(q3_f, d_vec), a3, sum1);
                }

                // --- Block 2 ---
                {
                    let b = &chunk[2];
                    let d_vec = _mm256_set1_ps(b.d.to_f32());
                    let q_ptr = b.qs.as_ptr() as *const __m128i;
                    let q_lo = _mm_loadu_si128(q_ptr);
                    let q_hi = _mm_loadu_si128(q_ptr.add(1));

                    // Offset A by 64
                    let a0 = _mm256_loadu_ps(a_ptr_local.add(64));
                    let a1 = _mm256_loadu_ps(a_ptr_local.add(72));

                    let q0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo));
                    let q1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo, 8)));

                    sum2 = _mm256_fmadd_ps(_mm256_mul_ps(q0_f, d_vec), a0, sum2);
                    sum2 = _mm256_fmadd_ps(_mm256_mul_ps(q1_f, d_vec), a1, sum2);

                    let a2 = _mm256_loadu_ps(a_ptr_local.add(80));
                    let a3 = _mm256_loadu_ps(a_ptr_local.add(88));

                    let q2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi));
                    let q3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi, 8)));

                    sum2 = _mm256_fmadd_ps(_mm256_mul_ps(q2_f, d_vec), a2, sum2);
                    sum2 = _mm256_fmadd_ps(_mm256_mul_ps(q3_f, d_vec), a3, sum2);
                }

                // --- Block 3 ---
                {
                    let b = &chunk[3];
                    let d_vec = _mm256_set1_ps(b.d.to_f32());
                    let q_ptr = b.qs.as_ptr() as *const __m128i;
                    let q_lo = _mm_loadu_si128(q_ptr);
                    let q_hi = _mm_loadu_si128(q_ptr.add(1));

                    // Offset A by 96
                    let a0 = _mm256_loadu_ps(a_ptr_local.add(96));
                    let a1 = _mm256_loadu_ps(a_ptr_local.add(104));

                    let q0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo));
                    let q1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo, 8)));

                    sum3 = _mm256_fmadd_ps(_mm256_mul_ps(q0_f, d_vec), a0, sum3);
                    sum3 = _mm256_fmadd_ps(_mm256_mul_ps(q1_f, d_vec), a1, sum3);

                    let a2 = _mm256_loadu_ps(a_ptr_local.add(112));
                    let a3 = _mm256_loadu_ps(a_ptr_local.add(120));

                    let q2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi));
                    let q3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi, 8)));

                    sum3 = _mm256_fmadd_ps(_mm256_mul_ps(q2_f, d_vec), a2, sum3);
                    sum3 = _mm256_fmadd_ps(_mm256_mul_ps(q3_f, d_vec), a3, sum3);
                }

                a_ptr_local = a_ptr_local.add(128);
            }

            // Handle remainder blocks (less than 4)
            for b in chunks.remainder() {
                let d_vec = _mm256_set1_ps(b.d.to_f32());
                let q_ptr = b.qs.as_ptr() as *const __m128i;
                let q_lo = _mm_loadu_si128(q_ptr);
                let q_hi = _mm_loadu_si128(q_ptr.add(1));

                let a0 = _mm256_loadu_ps(a_ptr_local);
                let a1 = _mm256_loadu_ps(a_ptr_local.add(8));

                let q0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo));
                let q1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo, 8)));

                sum0 = _mm256_fmadd_ps(_mm256_mul_ps(q0_f, d_vec), a0, sum0);
                sum0 = _mm256_fmadd_ps(_mm256_mul_ps(q1_f, d_vec), a1, sum0);

                let a2 = _mm256_loadu_ps(a_ptr_local.add(16));
                let a3 = _mm256_loadu_ps(a_ptr_local.add(24));

                let q2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi));
                let q3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi, 8)));

                sum0 = _mm256_fmadd_ps(_mm256_mul_ps(q2_f, d_vec), a2, sum0);
                sum0 = _mm256_fmadd_ps(_mm256_mul_ps(q3_f, d_vec), a3, sum0);

                a_ptr_local = a_ptr_local.add(32);
            }

            let sum_total = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
            *val = hsum_ps_avx(sum_total);
        }
    }
}
