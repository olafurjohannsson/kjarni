//! Public, safe entry points for matrix multiplication operations.
//!
//! This module acts as a high-level dispatcher, selecting the most performant
//! CPU kernel available at runtime for a given operation and data type. It abstracts
//! away the `unsafe` kernel implementations, providing a robust and safe API to the
//! rest of the inference engine.

#[cfg(target_arch = "x86_64")]
use crate::kernels::q_common::BlockQ6_K;
use crate::{kernels::{self, q_common::{BlockQ4_K, BlockQ8_0, QK_K}, quantize::quantize_row_q8_k}, weights::gguf_block_group_for_row};

use crate::kernels::scalar::vec_dot_q4k_q8k_scalar;

#[cfg(target_arch = "x86_64")]
use crate::kernels::x86::q4k_q8k::vec_dot_q4k_q8k_avx2;

use half::bf16;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;



// TPS 13.5

pub fn matmul_2d_cpu_q8_0(a: &ArrayView2<f32>, b_weights: &[BlockQ8_0]) -> Array2<f32> {
    let (m, k) = a.dim();
    let k_per_block = 32; // Q8_0 has 32 values per block
    let num_blocks = b_weights.len();
    let n = (num_blocks * k_per_block) / k;
    assert_eq!(k % k_per_block, 0, "Input features must be a multiple of the block size");

    let mut c = Array2::<f32>::zeros((m, n));
    let a_s = a.as_standard_layout();

    if m == 1 { // Decode Path
        let a_slice = a_s.as_slice().unwrap();
        let out_slice = c.as_slice_mut().unwrap();
        let num_threads = rayon::current_num_threads();
        let chunk_size = (n + num_threads - 1) / num_threads;

        out_slice.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, out_chunk)| {
            let num_blocks_per_row = k / k_per_block;
            let b_block_start_idx = chunk_idx * chunk_size * num_blocks_per_row;
            let num_blocks_for_chunk = out_chunk.len() * num_blocks_per_row;
            let b_blocks_chunk = &b_weights[b_block_start_idx..b_block_start_idx + num_blocks_for_chunk];

            unsafe {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // Make sure you have created and imported this kernel
                    return kernels::x86::q8_0::matmul_vec_q8_0_avx2(
                        out_chunk,
                        a_slice.as_ptr(),
                        b_blocks_chunk,
                        k,
                    );
                }
                kernels::scalar::matmul_vec_q8_0_scalar(out_chunk, a_slice, b_blocks_chunk, k);
            }
        });
    } else { // Prefill Path
        c.outer_iter_mut().into_par_iter().zip(a.outer_iter()).for_each(|(mut c_row, a_row)| {
            let a_row_slice = a_row.as_slice().unwrap();
            let out_slice = c_row.as_slice_mut().unwrap();
            unsafe {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    return kernels::x86::q8_0::matmul_vec_q8_0_avx2(
                        out_slice,
                        a_row_slice.as_ptr(),
                        b_weights,
                        k,
                    );
                }
                kernels::scalar::matmul_vec_q8_0_scalar(out_slice, a_row_slice, b_weights, k);
            }
        });
    }
    c
}

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



pub fn matmul_2d_cpu_q4_k(a: &ArrayView2<f32>, b_weights: &[BlockQ4_K]) -> Array2<f32> {
    let (m, k) = a.dim();
    let k_per_block = QK_K;
    let n = (b_weights.len() * k_per_block) / k;
    
    let mut c = Array2::<f32>::zeros((m, n));
    let a_s = a.as_standard_layout();

    if m == 1 { // Decode Path
        let a_slice = a_s.as_slice().unwrap();
        let out_slice = c.as_slice_mut().unwrap();
        let num_threads = rayon::current_num_threads();
        let chunk_size = (n + num_threads - 1) / num_threads;

        out_slice.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, out_chunk)| {
            let num_blocks_per_row = k / k_per_block;
            let b_block_start_idx = chunk_idx * chunk_size * num_blocks_per_row;
            let num_blocks_for_chunk = out_chunk.len() * num_blocks_per_row;
            let b_blocks_chunk = &b_weights[b_block_start_idx..b_block_start_idx + num_blocks_for_chunk];

            unsafe {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // Make sure this path points to your new kernel
                    return kernels::x86::q4_k::matmul_vec_q4_k_avx2(
                        out_chunk,
                        a_slice.as_ptr(),
                        b_blocks_chunk,
                        k,
                    );
                }
                // Add a scalar fallback here if needed
            }
        });
    } else { // Prefill Path
        let b_slice = b_weights; // Use the whole slice
        c.outer_iter_mut().into_par_iter().zip(a.outer_iter()).for_each(|(mut c_row, a_row)| {
            let a_row_slice = a_row.as_slice().unwrap();
            let out_slice = c_row.as_slice_mut().unwrap();
            unsafe {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    return kernels::x86::q4_k::matmul_vec_q4_k_avx2(
                        out_slice,
                        a_row_slice.as_ptr(),
                        b_slice,
                        k,
                    );
                }
            }
        });
    }
    c
}

/// Dispatcher for Q6_K 2D MatMul
pub fn matmul_2d_cpu_q6_k2(
    input: &ArrayView2<f32>,
    weights: &[BlockQ6_K],
) -> Array2<f32> {
    let (m, k) = input.dim();
    let num_blocks_per_row = k / 256;
    let out_features = weights.len() / num_blocks_per_row;
    let mut output = Array2::zeros((m, out_features));

    // Notice: No quantize_row_q8_k calls anymore!
    
    if m == 1 { // Decode Path
        let r = input.row(0);
        let a_slice = r.as_slice().unwrap();
        let out_slice = output.as_slice_mut().unwrap();

        // 64 is a good chunk size for Q6 because it is computationally heavy
        out_slice.par_chunks_mut(64).enumerate().for_each(|(chunk_idx, out_chunk)| {
            unsafe {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // Offset calculation
                    let start_row = chunk_idx * 64;
                    let block_start = start_row * num_blocks_per_row;
                    let block_count = out_chunk.len() * num_blocks_per_row;
                    
                    return kernels::x86::q6_k::matmul_vec_q6_k_avx2(
                        out_chunk,
                        a_slice.as_ptr(),
                        &weights[block_start..block_start+block_count],
                        k
                    );
                }
                
                // Fallback (requires re-quantizing if you kept the old scalar kernel)
                // Or just implement a slow F32 scalar fallback here
            }
            
        });
    } else { 
        output.outer_iter_mut().into_par_iter().zip(input.outer_iter()).for_each(|(mut out_row, in_row)| {
            let a_slice = in_row.as_slice().unwrap();
            let a_q8 = quantize_row_q8_k(a_slice);
            
            let out_slice = out_row.as_slice_mut().unwrap();
            for (i, out_val) in out_slice.iter_mut().enumerate() {
                let start = i * num_blocks_per_row;
                let end = start + num_blocks_per_row;
                let w_row = &weights[start..end];
                *out_val = kernels::scalar::vec_dot_q6k_q8k_scalar(k, w_row, &a_q8);
            }
        });
    }

    output
}
pub fn matmul_2d_cpu_q6_k(
    input: &ArrayView2<f32>,
    weights: &[BlockQ6_K],
) -> Array2<f32> {
    let (m, k) = input.dim();
    let num_blocks_per_row = k / QK_K;
    let out_features = weights.len() / num_blocks_per_row;

    let mut output = Array2::zeros((m, out_features));

    if m == 1 { // Decode Path
        let r = input.row(0);
        let a_slice = r.as_slice().unwrap();
        let out_slice = output.as_slice_mut().unwrap();

        // Pre-quantize the input vector once.
        let a_q8 = quantize_row_q8_k(a_slice);

        let num_threads = rayon::current_num_threads();
        let chunk_size = (out_features + num_threads - 1) / num_threads;

        // Use the winning `par_chunks_mut` strategy.
        out_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                for (j, out_val) in out_chunk.iter_mut().enumerate() {
                    let i = chunk_idx * chunk_size + j;
                    let start = i * num_blocks_per_row;
                    let end = start + num_blocks_per_row;
                    let w_row = &weights[start..end];
                    
                    // Call the existing scalar kernel.
                    *out_val = kernels::scalar::vec_dot_q6k_q8k_scalar(k, w_row, &a_q8);
                }
            });
    } else { // Prefill Path
        output.outer_iter_mut().into_par_iter().zip(input.outer_iter()).for_each(|(mut out_row, in_row)| {
            let a_slice = in_row.as_slice().unwrap();
            let a_q8 = quantize_row_q8_k(a_slice);
            
            let out_slice = out_row.as_slice_mut().unwrap();
            for (i, out_val) in out_slice.iter_mut().enumerate() {
                let start = i * num_blocks_per_row;
                let end = start + num_blocks_per_row;
                let w_row = &weights[start..end];
                *out_val = kernels::scalar::vec_dot_q6k_q8k_scalar(k, w_row, &a_q8);
            }
        });
    }

    output
}

