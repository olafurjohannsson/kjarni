//! AVX2/FMA accelerated kernels for BF16 weights.
//!
//! This module is part of the `unsafe` kernel zone. The functions within are
//! designed to be called from the safe dispatchers in the `ops` module.

#![allow(unsafe_code)]
use super::common::hsum_ps_avx;
use std::arch::x86_64::*;

/// Computes a vector-matrix multiplication (vec @ mat) for BF16 weights using AVX2/FMA.
///
/// This is the core routine for `LinearLayer` during single-token decoding.
/// It computes `out = a @ b.T` where `b` is stored row-major ([out_features, in_features]).
///
/// # Safety
///
/// This function is `unsafe` because it operates on raw pointers and relies on
/// AVX2/FMA CPU features being present, which must be checked by the caller.
/// The caller must also ensure that pointers are valid and slices have the correct dimensions.
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn matmul_vec_bf16(
    out_chunk: &mut [f32],
    a_ptr: *const f32,
    b_row_start: *const u16,
    k: usize,
) {
    let mut b_row_ptr = b_row_start;

    for val in out_chunk.iter_mut() {
        let mut a_chunk_ptr = a_ptr;
        let mut b_chunk_ptr = b_row_ptr;

        // Accumulators for the dot product, unrolled by 4 to hide FMA latency.
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        let mut n = k;
        while n >= 32 {
            // Prefetching can sometimes help, especially with large K.
            _mm_prefetch(b_chunk_ptr.add(128) as *const i8, _MM_HINT_T0);

            // Load 32 floats from the input vector 'a'
            let a0 = _mm256_loadu_ps(a_chunk_ptr);
            let a1 = _mm256_loadu_ps(a_chunk_ptr.add(8));
            let a2 = _mm256_loadu_ps(a_chunk_ptr.add(16));
            let a3 = _mm256_loadu_ps(a_chunk_ptr.add(24));

            // Load 32 bf16s (as u16) from the weight matrix 'b'
            let b0_u16 = _mm_loadu_si128(b_chunk_ptr as *const __m128i);
            let b1_u16 = _mm_loadu_si128(b_chunk_ptr.add(8) as *const __m128i);
            let b2_u16 = _mm_loadu_si128(b_chunk_ptr.add(16) as *const __m128i);
            let b3_u16 = _mm_loadu_si128(b_chunk_ptr.add(24) as *const __m128i);

            // Convert bf16 vectors to f32 vectors
            let b0_f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b0_u16), 16));
            let b1_f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b1_u16), 16));
            let b2_f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b2_u16), 16));
            let b3_f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b3_u16), 16));

            // Fused multiply-add
            sum0 = _mm256_fmadd_ps(a0, b0_f, sum0);
            sum1 = _mm256_fmadd_ps(a1, b1_f, sum1);
            sum2 = _mm256_fmadd_ps(a2, b2_f, sum2);
            sum3 = _mm256_fmadd_ps(a3, b3_f, sum3);

            a_chunk_ptr = a_chunk_ptr.add(32);
            b_chunk_ptr = b_chunk_ptr.add(32);
            n -= 32;
        }

        // Combine the four accumulators
        sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

        // Horizontally sum the final vector efficiently
        let mut sum = hsum_ps_avx(sum0);

        // Handle the remainder of the dot product
        while n > 0 {
            let val_a = *a_chunk_ptr;
            let val_b = f32::from_bits((*b_chunk_ptr as u32) << 16);
            sum += val_a * val_b;
            a_chunk_ptr = a_chunk_ptr.add(1);
            b_chunk_ptr = b_chunk_ptr.add(1);
            n -= 1;
        }

        *val = sum;
        b_row_ptr = b_row_ptr.add(k); // Move to the start of the next row in the weight matrix
    }
}

/// Computes the fused gate and up projections for a SwiGLU FFN with BF16 weights.
///
/// This kernel computes `(a @ gate_w.T)` and `(a @ up_w.T)` simultaneously,
/// maximizing cache reuse of the input vector `a`.
///
/// # Safety
///
/// This function is `unsafe` for the same reasons as `matmul_vec_bf16`.
/// The caller must ensure CPU features are present and all slices are valid.
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn swiglu_fused_gate_up_bf16(
    gate_out: &mut [f32],
    up_out: &mut [f32],
    a: &[f32],
    gate_w: &[u16],
    up_w: &[u16],
    k: usize,
    _n: usize,
) {
    // This kernel is parallelized on its output dimension by the caller (`ops` module).
    // Here we just implement the core computation for a single output element.
    let gate_w_row = &gate_w[..k];
    let up_w_row = &up_w[..k];

    let mut gate_sum_vec = _mm256_setzero_ps();
    let mut up_sum_vec = _mm256_setzero_ps();

    let mut j = 0;
    while j + 8 <= k {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(j));

        // Load and convert gate weights
        let gate_w_u16 = _mm_loadu_si128(gate_w_row.as_ptr().add(j) as *const __m128i);
        let gate_w_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(gate_w_u16), 16));

        // Load and convert up weights
        let up_w_u16 = _mm_loadu_si128(up_w_row.as_ptr().add(j) as *const __m128i);
        let up_w_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(up_w_u16), 16));

        // Fused multiply-add for both projections
        gate_sum_vec = _mm256_fmadd_ps(a_vec, gate_w_f32, gate_sum_vec);
        up_sum_vec = _mm256_fmadd_ps(a_vec, up_w_f32, up_sum_vec);

        j += 8;
    }

    let mut gate_sum = hsum_ps_avx(gate_sum_vec);
    let mut up_sum = hsum_ps_avx(up_sum_vec);

    // Remainder loop
    while j < k {
        let a_val = a[j];
        gate_sum += a_val * f32::from_bits((gate_w_row[j] as u32) << 16);
        up_sum += a_val * f32::from_bits((up_w_row[j] as u32) << 16);
        j += 1;
    }

    gate_out[0] = gate_sum;
    up_out[0] = up_sum;
}