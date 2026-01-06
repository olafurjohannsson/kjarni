//! AVX2/FMA accelerated kernels for F32 weights.
//!
//! This module is part of the `unsafe` kernel zone. The functions within are
//! designed to be called from the safe dispatchers in the `ops` module.

#![allow(unsafe_code)]
use super::common::hsum_ps_avx;
use std::arch::x86_64::*;

/// Computes a vector-matrix multiplication (vec @ mat) for F32 weights using AVX2/FMA.
///
/// This is the F32 equivalent of the BF16 matmul kernel.
///
/// # Safety
///
/// This function is `unsafe` for the same reasons as its BF16 counterpart.
/// The caller must ensure CPU features are present and all pointers/slices are valid.
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn matmul_vec_f32(
    out_chunk: &mut [f32],
    a_ptr: *const f32,
    b_row_start: *const f32,
    k: usize,
) {
    let mut b_row_ptr = b_row_start;
    unsafe {
        for val in out_chunk.iter_mut() {
            let mut a_chunk_ptr = a_ptr;
            let mut b_chunk_ptr = b_row_ptr;

            let mut sum0 = _mm256_setzero_ps();
            let mut sum1 = _mm256_setzero_ps();
            let mut sum2 = _mm256_setzero_ps();
            let mut sum3 = _mm256_setzero_ps();

            let mut n = k;
            while n >= 32 {
                let a0 = _mm256_loadu_ps(a_chunk_ptr);
                let a1 = _mm256_loadu_ps(a_chunk_ptr.add(8));
                let a2 = _mm256_loadu_ps(a_chunk_ptr.add(16));
                let a3 = _mm256_loadu_ps(a_chunk_ptr.add(24));

                // For F32, loading weights is a direct operation.
                let b0 = _mm256_loadu_ps(b_chunk_ptr);
                let b1 = _mm256_loadu_ps(b_chunk_ptr.add(8));
                let b2 = _mm256_loadu_ps(b_chunk_ptr.add(16));
                let b3 = _mm256_loadu_ps(b_chunk_ptr.add(24));

                sum0 = _mm256_fmadd_ps(a0, b0, sum0);
                sum1 = _mm256_fmadd_ps(a1, b1, sum1);
                sum2 = _mm256_fmadd_ps(a2, b2, sum2);
                sum3 = _mm256_fmadd_ps(a3, b3, sum3);

                a_chunk_ptr = a_chunk_ptr.add(32);
                b_chunk_ptr = b_chunk_ptr.add(32);
                n -= 32;
            }

            sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
            let mut sum = hsum_ps_avx(sum0);

            while n > 0 {
                sum += *a_chunk_ptr * *b_chunk_ptr;
                a_chunk_ptr = a_chunk_ptr.add(1);
                b_chunk_ptr = b_chunk_ptr.add(1);
                n -= 1;
            }

            *val = sum;
            b_row_ptr = b_row_ptr.add(k);
        }
    }
}
