//! Public, safe entry points for matrix multiplication operations.
//!
//! This module acts as a high-level dispatcher, selecting the most performant
//! CPU kernel available at runtime for a given operation and data type. It abstracts
//! away the `unsafe` kernel implementations, providing a robust and safe API to the
//! rest of the inference engine.

#[cfg(target_arch = "x86_64")]
use crate::kernels::q_common::BlockQ6_K;
use crate::kernels::{self, q_common::{BlockQ4_K, BlockQ8_0, QK_K}};
use half::bf16;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

/// Computes `C = A @ B.T` for F32 input `A` and BF16 weight matrix `B`.
pub fn matmul_2d_cpu_bf16(a: &ArrayView2<f32>, b_weights: &ArrayView2<bf16>) -> Array2<f32> {
    let (m, k) = a.dim();
    let (n, k2) = b_weights.dim();
    assert_eq!(k, k2, "Matmul dimension mismatch: A[k] != B[k]");

    let mut c = Array2::<f32>::zeros((m, n));
    let a_s = a.as_standard_layout();
    let b_s = b_weights.as_standard_layout();
    let a_slice = a_s.as_slice().expect("Input tensor 'a' must be contiguous");
    let b_slice = b_s.as_slice().expect("Weight tensor 'b' must be contiguous");

    if m == 1 {
        let out_slice = c.as_slice_mut().unwrap();
        let num_threads = rayon::current_num_threads();
        let chunk_size = (n + num_threads - 1) / num_threads;

        out_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let b_row_start_idx = chunk_idx * chunk_size;
                let b_chunk_ptr = unsafe { b_slice.as_ptr().add(b_row_start_idx * k) as *const u16 };

                unsafe {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        return kernels::x86::bf16::matmul_vec_bf16(out_chunk, a_slice.as_ptr(), b_chunk_ptr, k);
                    }
                    #[cfg(target_arch = "aarch64")]
                    if std::arch::is_aarch64_feature_detected!("neon") {
                        return kernels::aarch64::bf16::matmul_vec_bf16_neon(out_chunk, a_slice.as_ptr(), b_chunk_ptr, k);
                    }
                    let b_rows = &b_slice[b_row_start_idx * k..];
                    kernels::scalar::matmul_vec_bf16_scalar(out_chunk, a_slice, std::mem::transmute(b_rows), k);
                }
            });
    } else {
        // Batch parallelization for prefill
        c.outer_iter_mut().into_par_iter().zip(a.outer_iter()).for_each(|(mut c_row, a_row)| {
            let a_row_slice = a_row.as_slice().unwrap();
            let out_slice = c_row.as_slice_mut().unwrap();
            unsafe {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    return kernels::x86::bf16::matmul_vec_bf16(out_slice, a_row_slice.as_ptr(), b_slice.as_ptr() as *const u16, k);
                }
                #[cfg(target_arch = "aarch64")]
                if std::arch::is_aarch64_feature_detected!("neon") {
                    return kernels::aarch64::bf16::matmul_vec_bf16_neon(out_slice, a_row_slice.as_ptr(), b_slice.as_ptr() as *const u16, k);
                }
                kernels::scalar::matmul_vec_bf16_scalar(out_slice, a_row_slice, std::mem::transmute(b_slice), k);
            }
        });
    }
    c
}

/// Computes `C = A @ B.T` for F32 input `A` and F32 weight matrix `B`.
pub fn matmul_2d_cpu_f32(a: &ArrayView2<f32>, b_weights: &ArrayView2<f32>) -> Array2<f32> {
    // (Implementation is analogous to the BF16 version, calling the F32 kernels)
    let (m, k) = a.dim();
    let (n, k2) = b_weights.dim();
    assert_eq!(k, k2, "Matmul dimension mismatch: A[k] != B[k]");

    let mut c = Array2::<f32>::zeros((m, n));
    let a_s = a.as_standard_layout();
    let b_s = b_weights.as_standard_layout();
    let a_slice = a_s.as_slice().expect("Input tensor 'a' must be contiguous");
    let b_slice = b_s.as_slice().expect("Weight tensor 'b' must be contiguous");

    if m == 1 {
        let out_slice = c.as_slice_mut().unwrap();
        let num_threads = rayon::current_num_threads();
        let chunk_size = (n + num_threads - 1) / num_threads;

        out_slice.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, out_chunk)| {
            let b_row_start_idx = chunk_idx * chunk_size;
            let b_chunk_ptr = unsafe { b_slice.as_ptr().add(b_row_start_idx * k) };
            unsafe {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    return kernels::x86::f32::matmul_vec_f32(out_chunk, a_slice.as_ptr(), b_chunk_ptr, k);
                }
                #[cfg(target_arch = "aarch64")]
                if std::arch::is_aarch64_feature_detected!("neon") {
                    return kernels::aarch64::f32::matmul_vec_f32_neon(out_chunk, a_slice.as_ptr(), b_chunk_ptr, k);
                }
                let b_rows = &b_slice[b_row_start_idx * k..];
                kernels::scalar::matmul_vec_f32_scalar(out_chunk, a_slice, b_rows, k);
            }
        });
    } else {
        c.outer_iter_mut().into_par_iter().zip(a.outer_iter()).for_each(|(mut c_row, a_row)| {
            let a_row_slice = a_row.as_slice().unwrap();
            let out_slice = c_row.as_slice_mut().unwrap();
            unsafe {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    return kernels::x86::f32::matmul_vec_f32(out_slice, a_row_slice.as_ptr(), b_slice.as_ptr(), k);
                }
                #[cfg(target_arch = "aarch64")]
                if std::arch::is_aarch64_feature_detected!("neon") {
                    return kernels::aarch64::f32::matmul_vec_f32_neon(out_slice, a_row_slice.as_ptr(), b_slice.as_ptr(), k);
                }
                kernels::scalar::matmul_vec_f32_scalar(out_slice, a_row_slice, b_slice, k);
            }
        });
    }
    c
}

/// Computes `C = A @ B.T` for F32 input `A` and Q8_0 quantized weight matrix `B`.
pub fn matmul_2d_cpu_q8_0(a: &ArrayView2<f32>, b_weights: &[BlockQ8_0]) -> Array2<f32> {
    let (m, k) = a.dim();
    let k_per_block = std::mem::size_of_val(&b_weights[0].qs);
    let n = (b_weights.len() * k_per_block) / k;
    assert_eq!(k % k_per_block, 0, "Input features must be a multiple of the block size");

    let mut c = Array2::<f32>::zeros((m, n));
    let a_s = a.as_standard_layout();
    let a_slice = a_s.as_slice().expect("Input tensor 'a' must be contiguous");

    if m == 1 {
        let out_slice = c.as_slice_mut().unwrap();
        let num_threads = rayon::current_num_threads();
        let chunk_size = (n + num_threads - 1) / num_threads;

        out_slice.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, out_chunk)| {
            let b_block_start_idx = chunk_idx * chunk_size * (k / k_per_block);
            let b_blocks = &b_weights[b_block_start_idx..];
            
            // NOTE: No unsafe block needed here if the scalar function is safe
            kernels::scalar::matmul_vec_q8_0_scalar(out_chunk, a_slice, b_blocks, k);
        });
    } else {
        c.outer_iter_mut().into_par_iter().zip(a.outer_iter()).for_each(|(mut c_row, a_row)| {
            let a_row_slice = a_row.as_slice().unwrap();
            let out_slice = c_row.as_slice_mut().unwrap();
            kernels::scalar::matmul_vec_q8_0_scalar(out_slice, a_row_slice, b_weights, k);
        });
    }
    c
}

/// Computes `C = A @ B.T` for F32 input `A` and Q4_K quantized weight matrix `B`.
pub fn matmul_2d_cpu_q4_k(a: &ArrayView2<f32>, b_weights: &[BlockQ4_K]) -> Array2<f32> {
    let (m, k) = a.dim();
    let k_per_block = crate::kernels::q_common::QK_K;
    let n = (b_weights.len() * k_per_block) / k;
    assert_eq!(k % k_per_block, 0, "Input features must be a multiple of the QK_K block size");

    let mut c = Array2::<f32>::zeros((m, n));
    let a_s = a.as_standard_layout();
    let a_slice = a_s.as_slice().expect("Input tensor 'a' must be contiguous");

    if m == 1 {
        let out_slice = c.as_slice_mut().unwrap();
        let num_threads = rayon::current_num_threads();
        let chunk_size = (n + num_threads - 1) / num_threads;

        out_slice.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, out_chunk)| {
            let b_block_start_idx = chunk_idx * chunk_size * (k / k_per_block);
            let b_blocks = &b_weights[b_block_start_idx..];
            
            kernels::scalar::matmul_vec_q4_k_scalar(out_chunk, a_slice, b_blocks, k);
        });
    } else {
        c.outer_iter_mut().into_par_iter().zip(a.outer_iter()).for_each(|(mut c_row, a_row)| {
            let a_row_slice = a_row.as_slice().unwrap();
            let out_slice = c_row.as_slice_mut().unwrap();
            kernels::scalar::matmul_vec_q4_k_scalar(out_slice, a_row_slice, b_weights, k);
        });
    }
    c
}

/// Dispatcher for Q6_K 2D MatMul
pub fn matmul_2d_cpu_q6_k(
    input: &ArrayView2<f32>,
    weights: &[BlockQ6_K],
) -> Array2<f32> {
    let (batch_size, k) = input.dim();
    let num_blocks_per_row = k / QK_K;
    let out_features = weights.len() / num_blocks_per_row;

    let mut output = Array2::zeros((batch_size, out_features));

    // Parallelize over the batch and output features
    output.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(b_idx, mut out_row)| {
            let b = input.row(b_idx);
            let input_row = b.as_slice().unwrap();
            let out_slice = out_row.as_slice_mut().unwrap();

            unsafe {
                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        matmul_vec_q6_k_avx2(out_slice, input_row, weights, k);
                        return;
                    }
                }
                
                // Fallback to scalar
                matmul_vec_q6_k_scalar(out_slice, input_row, weights, k);
            }
        });

    output
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn matmul_vec_q6_k_avx2(
    out_chunk: &mut [f32],
    a: &[f32],
    b_blocks: &[BlockQ6_K],
    k: usize,
) {
    use std::arch::x86_64::*;

    use crate::kernels::q_common::QK_K;
    let blocks_per_row = k / QK_K;
    let mut temp_w = [0.0f32; QK_K]; // Stack buffer for dequantized weights

    for (i, out_val) in out_chunk.iter_mut().enumerate() {
        let row_blocks = &b_blocks[i * blocks_per_row..(i + 1) * blocks_per_row];
        let mut sum_vec = _mm256_setzero_ps();

        for (block_idx, a_chunk) in a.chunks_exact(QK_K).enumerate() {
            use crate::kernels::dequantize::dequantize_q6_k_block;

            let block = &row_blocks[block_idx];
            
            // 1. Dequantize Q6_K block to F32 temp buffer
            dequantize_q6_k_block(block, &mut temp_w);

            // 2. Perform AVX2 Dot Product
            let mut a_ptr = a_chunk.as_ptr();
            let mut w_ptr = temp_w.as_ptr();
            for _ in 0..32 { // 256 elements / 8 per YMM = 32 iterations
                let av = _mm256_loadu_ps(a_ptr);
                let wv = _mm256_loadu_ps(w_ptr);
                sum_vec = _mm256_fmadd_ps(av, wv, sum_vec);
                a_ptr = a_ptr.add(8);
                w_ptr = w_ptr.add(8);
            }
        }
        
        // 3. Horizontal sum
        let mut res = hsum_avx(sum_vec);
        *out_val = res;
    }
}

// Scalar Fallback
unsafe fn matmul_vec_q6_k_scalar(out: &mut [f32], a: &[f32], b: &[BlockQ6_K], k: usize) {
    let mut temp_w = [0.0f32; QK_K];
    let blocks_per_row = k / QK_K;
    for (i, out_val) in out.iter_mut().enumerate() {
        let row_blocks = &b[i * blocks_per_row..(i + 1) * blocks_per_row];
        let mut sum = 0.0f32;
        for (block_idx, a_chunk) in a.chunks_exact(QK_K).enumerate() {
            kernels::dequantize::dequantize_q6_k_block(&row_blocks[block_idx], &mut temp_w);
            for j in 0..QK_K {
                sum += a_chunk[j] * temp_w[j];
            }
        }
        *out_val = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let vlow = _mm256_castps256_ps128(v);
    let vhigh = _mm256_extractf128_ps(v, 1);
    let vsum = _mm_add_ps(vlow, vhigh);
    let vsum = _mm_hadd_ps(vsum, vsum);
    let vsum = _mm_hadd_ps(vsum, vsum);
    _mm_cvtss_f32(vsum)
}
