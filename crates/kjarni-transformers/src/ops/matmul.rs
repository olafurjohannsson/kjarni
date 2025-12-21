//! Public, safe entry points for matrix multiplication operations.
//!
//! This module acts as a high-level dispatcher, selecting the most performant
//! CPU kernel available at runtime for a given operation and data type. It abstracts
//! away the `unsafe` kernel implementations, providing a robust and safe API to the
//! rest of the inference engine.

use crate::kernels::{self, q_common::{BlockQ4_K, BlockQ8_0}};
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


// /// Computes `C = A @ B.T` for F32 input `A` and Q4_K quantized weight matrix `B`,
// /// using on-the-fly quantization of `A`.
// pub fn matmul_2d_cpu_f32_x_q4_k(a: &ArrayView2<f32>, b_weights: &[BlockQ4_K]) -> Array2<f32> {
//     let (m, k) = a.dim();
//     let k_per_block = crate::kernels::q_common::QK_K;
//     let n = (b_weights.len() * k_per_block) / k;
//     assert_eq!(k % k_per_block, 0, "Input features must be a multiple of the QK_K block size");

//     let mut c = Array2::<f32>::zeros((m, n));
//     let a_s = a.as_standard_layout();
//     let a_slice = a_s.as_slice().expect("Input tensor 'a' must be contiguous");

//     // 1. Quantize the input vector 'a' once.
//     let mut a_q8_blocks: Vec<BlockQ8_0> = vec![Default::default(); k / 32];
//     for (i, a_chunk) in a_slice.chunks_exact(32).enumerate() {
//         unsafe {
//             // This should ideally also have a SIMD version.
//             kernels::x86::q_x_q::quantize_f32_to_q8_0(a_chunk, &mut a_q8_blocks[i]);
//         }
//     }

//     // 2. Dispatch to the fused kernel.
//     if m == 1 {
//         let out_slice = c.as_slice_mut().unwrap();
//         // ... (parallelization logic) ...
//         // Inside the parallel loop:
//         unsafe {
//             #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
//             if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
//                 return kernels::x86::q_x_q::matmul_vec_q8_0_x_q4_k(out_chunk, &a_q8_blocks, b_chunk_blocks, k);
//             }
//             // ... scalar fallback ...
//         }
//     } else {
//         // ... batch prefill logic ...
//     }
//     c
// }