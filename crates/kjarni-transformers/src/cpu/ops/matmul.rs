//! Public, safe entry points for matrix multiplication operations.
//!
//! This module acts as a high-level dispatcher, selecting the most performant
//! CPU kernel available at runtime for a given operation and data type. It abstracts
//! away the `unsafe` kernel implementations, providing a robust and safe API to the
//! rest of the inference engine.
//!
//! # Overview
//!
//! All functions in this module compute `C = A @ B^T` where:
//! - `A` is the input activation matrix of shape `[batch, in_features]` (always F32)
//! - `B` is the weight matrix of shape `[out_features, in_features]` (various dtypes)
//! - `C` is the output matrix of shape `[batch, out_features]` (always F32)
//!
//! The weight matrix `B` is stored in row-major order with each row representing
//! one output feature's weights. The transposition is implicit in the kernel
//! implementation.
//!
//! # Parallelization Strategy
//!
//! Each function uses one of two parallelization strategies based on the batch size:
//!
//! - **Decode path** (`batch == 1`): Parallelizes over output features. The single
//!   input vector is broadcast to all threads, and each thread computes a chunk
//!   of the output vector. This minimizes synchronization overhead for autoregressive
//!   token generation.
//!
//! - **Prefill path** (`batch > 1`): Parallelizes over batch rows. Each thread
//!   processes one input row independently, computing the full output row. This
//!   maximizes cache locality for prompt processing.
//!
//! # Kernel Selection
//!
//! At runtime, the dispatcher checks for CPU feature support and selects the
//! fastest available kernel:
//!
//! 1. **x86_64 with AVX2+FMA**: Uses hand-tuned SIMD kernels with 8-wide vectors.
//! 2. **aarch64 with NEON**: Uses ARM NEON intrinsics (for BF16 and F32).
//! 3. **Fallback**: Uses portable scalar kernels (slower but always available).
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::ops::matmul::matmul_2d_cpu_f32;
//! use ndarray::Array2;
//!
//! let input = Array2::<f32>::zeros((1, 2048));
//! let weights = Array2::<f32>::zeros((4096, 2048));
//! let output = matmul_2d_cpu_f32(&input.view(), &weights.view());
//! assert_eq!(output.shape(), &[1, 4096]);
//! ```
//!
//! # See Also
//!
//! - [`crate::kernels`] — Low-level SIMD kernel implementations.
//! - [`crate::linear_layer::LinearLayer`] — High-level wrapper using these functions.

#[cfg(target_arch = "x86_64")]
use crate::cpu::kernels::q_common::BlockQ6_K;
use crate::cpu::kernels::{
    self,
    q_common::{BlockQ4_K, BlockQ8_0, QK_K},
    quantize::quantize_row_q8_k,
};

use half::bf16;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;



/// Computes `C = A @ B^T` for F32 input `A` and Q8_0 quantized weight matrix `B`.
///
/// Q8_0 is an 8-bit block-quantized format where each block contains 32 int8 values
/// and a single F16 scale factor. This provides approximately 4x memory compression
/// compared to F32 with minimal quality loss.
///
/// # Arguments
///
/// * `a` - Input activation matrix of shape `[batch, in_features]`.
/// * `b_weights` - Quantized weight blocks. Total elements = `out_features * in_features / 32`.
///
/// # Returns
///
/// Output matrix of shape `[batch, out_features]`.
///
/// # Panics
///
/// Panics if `in_features` is not a multiple of 32 (the Q8_0 block size).
///
/// # Performance
///
/// Uses AVX2+FMA kernels on x86_64, falling back to scalar on other architectures.
pub fn matmul_2d_cpu_q8_0(a: &ArrayView2<f32>, b_weights: &[BlockQ8_0]) -> Array2<f32> {
    let (m, k) = a.dim();

    // Q8_0 stores 32 int8 values per block with one shared scale factor
    let k_per_block = 32;
    let num_blocks = b_weights.len();

    // Calculate output dimension: total quantized elements / input features
    let n = (num_blocks * k_per_block) / k;
    assert_eq!(
        k % k_per_block,
        0,
        "Input features must be a multiple of the block size"
    );

    let mut c = Array2::<f32>::zeros((m, n));

    // Ensure input is in standard row-major layout for contiguous memory access
    let a_s = a.as_standard_layout();

    if m == 1 {
        // === DECODE PATH ===
        // Single input vector: parallelize over output features.
        // Each thread computes a chunk of the output vector using the shared input.

        let a_slice = a_s.as_slice().unwrap();
        let out_slice = c.as_slice_mut().unwrap();

        // Divide output evenly among available threads
        let num_threads = rayon::current_num_threads();
        let chunk_size = (n + num_threads - 1) / num_threads;

        out_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                // Calculate which weight blocks this thread needs
                let num_blocks_per_row = k / k_per_block;
                let b_block_start_idx = chunk_idx * chunk_size * num_blocks_per_row;
                let num_blocks_for_chunk = out_chunk.len() * num_blocks_per_row;
                let b_blocks_chunk =
                    &b_weights[b_block_start_idx..b_block_start_idx + num_blocks_for_chunk];

                // Dispatch to fastest available kernel
                unsafe {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        return kernels::x86::q8_0::matmul_vec_q8_0_avx2(
                            out_chunk,
                            a_slice.as_ptr(),
                            b_blocks_chunk,
                            k,
                        );
                    }
                    // Fallback to portable scalar implementation
                    kernels::scalar::matmul_vec_q8_0_scalar(
                        out_chunk,
                        a_slice,
                        b_blocks_chunk,
                        k,
                    );
                }
            });
    } else {
        // === PREFILL PATH ===
        // Multiple input rows: parallelize over batch dimension.
        // Each thread processes one complete row independently.

        c.outer_iter_mut()
            .into_par_iter()
            .zip(a.outer_iter())
            .for_each(|(mut c_row, a_row)| {
                let a_row_slice = a_row.as_slice().unwrap();
                let out_slice = c_row.as_slice_mut().unwrap();

                // Each row uses all weight blocks
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

/// Computes `C = A @ B^T` for F32 input `A` and BF16 weight matrix `B`.
///
/// BF16 (brain floating-point 16) uses the same exponent range as F32 but with
/// a reduced mantissa, providing 2x memory savings with good numerical stability
/// for neural network inference.
///
/// # Arguments
///
/// * `a` - Input activation matrix of shape `[batch, in_features]` in F32.
/// * `b_weights` - Weight matrix of shape `[out_features, in_features]` in BF16.
///
/// # Returns
///
/// Output matrix of shape `[batch, out_features]` in F32.
///
/// # Panics
///
/// Panics if the inner dimensions don't match (`a.shape()[1] != b_weights.shape()[1]`).
///
/// # Performance
///
/// Uses AVX2+FMA on x86_64, NEON on aarch64, with scalar fallback.
/// The kernel converts BF16 to F32 on-the-fly using bit manipulation.
pub fn matmul_2d_cpu_bf16(a: &ArrayView2<f32>, b_weights: &ArrayView2<bf16>) -> Array2<f32> {
    let (m, k) = a.dim(); // m = batch size, k = input features
    let (n, k2) = b_weights.dim(); // n = output features
    assert_eq!(k, k2, "Matmul dimension mismatch: A[k] != B[k]");

    let mut c = Array2::<f32>::zeros((m, n));

    // Ensure both tensors are in contiguous row-major layout
    let a_s = a.as_standard_layout();
    let b_s = b_weights.as_standard_layout();
    let a_slice = a_s.as_slice().expect("Input tensor 'a' must be contiguous");
    let b_slice = b_s
        .as_slice()
        .expect("Weight tensor 'b' must be contiguous");

    if m == 1 {
        // === DECODE PATH ===
        // Parallelize over output features for single-token inference

        let out_slice = c.as_slice_mut().unwrap();
        let num_threads = rayon::current_num_threads();
        let chunk_size = (n + num_threads - 1) / num_threads;

        out_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                // Calculate pointer to this thread's weight rows
                let b_row_start_idx = chunk_idx * chunk_size;
                let b_chunk_ptr =
                    unsafe { b_slice.as_ptr().add(b_row_start_idx * k) as *const u16 };

                // Dispatch to architecture-specific kernel
                unsafe {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        return kernels::x86::bf16::matmul_vec_bf16(
                            out_chunk,
                            a_slice.as_ptr(),
                            b_chunk_ptr,
                            k,
                        );
                    }
                    #[cfg(target_arch = "aarch64")]
                    if std::arch::is_aarch64_feature_detected!("neon") {
                        return kernels::aarch64::bf16::matmul_vec_bf16_neon(
                            out_chunk,
                            a_slice.as_ptr(),
                            b_chunk_ptr,
                            k,
                        );
                    }
                    // Scalar fallback for other architectures
                    let b_rows = &b_slice[b_row_start_idx * k..];
                    kernels::scalar::matmul_vec_bf16_scalar(
                        out_chunk,
                        a_slice,
                        std::mem::transmute(b_rows),
                        k,
                    );
                }
            });
    } else {
        // === PREFILL PATH ===
        // Parallelize over batch rows for prompt processing

        c.outer_iter_mut()
            .into_par_iter()
            .zip(a.outer_iter())
            .for_each(|(mut c_row, a_row)| {
                let a_row_slice = a_row.as_slice().unwrap();
                let out_slice = c_row.as_slice_mut().unwrap();

                // Each row computes against all weights
                unsafe {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        return kernels::x86::bf16::matmul_vec_bf16(
                            out_slice,
                            a_row_slice.as_ptr(),
                            b_slice.as_ptr() as *const u16,
                            k,
                        );
                    }
                    #[cfg(target_arch = "aarch64")]
                    if std::arch::is_aarch64_feature_detected!("neon") {
                        return kernels::aarch64::bf16::matmul_vec_bf16_neon(
                            out_slice,
                            a_row_slice.as_ptr(),
                            b_slice.as_ptr() as *const u16,
                            k,
                        );
                    }
                    kernels::scalar::matmul_vec_bf16_scalar(
                        out_slice,
                        a_row_slice,
                        std::mem::transmute(b_slice),
                        k,
                    );
                }
            });
    }
    c
}

use faer::{MatRef, MatMut, Parallelism};

pub fn matmul_2d_cpu_f32_faer(a: &ArrayView2<f32>, b_weights: &ArrayView2<f32>) -> Array2<f32> {
    let (m, k) = a.dim();
    let (n, k2) = b_weights.dim();
    assert_eq!(k, k2, "Matmul dimension mismatch");

    let mut c = Array2::<f32>::zeros((m, n));

    let a_slice = a.as_slice().expect("A must be contiguous");
    let b_slice = b_weights.as_slice().expect("B must be contiguous");
    let c_slice = c.as_slice_mut().expect("C must be contiguous");

    // Remove ::<f32>. Rust infers the types automatically.
    let mat_a = faer::mat::from_row_major_slice(a_slice, m, k);
    let mat_b = faer::mat::from_row_major_slice(b_slice, n, k);
    let mut mat_c = faer::mat::from_row_major_slice_mut(c_slice, m, n);

    faer::linalg::matmul::matmul(
        mat_c,
        mat_a,
        mat_b.transpose(), // This results in Contiguous x Contiguous dot products
        None,
        1.0,
        Parallelism::Rayon(0),
    );

    c
}

/// No-alloc version of matmul_2d_cpu_f32
/// 
/// Computes `C = A @ B^T + bias` for F32 input `A` and F32 weight matrix `B`,
/// writing directly to a pre-allocated output buffer.
///
/// Uses the vec kernel - optimized for decode (m=1) and small batches.
///
/// # Arguments
///
/// * `a` - Input activation matrix of shape `[batch, in_features]`.
/// * `b_weights` - Weight matrix of shape `[out_features, in_features]`.
/// * `bias` - Optional bias vector of shape `[out_features]`.
/// * `output` - Pre-allocated output matrix of shape `[batch, out_features]`.
///
/// # Panics
///
/// Panics if dimensions don't match.
pub fn matmul_2d_f32_noalloc(
    a: &ArrayView2<f32>,
    b_weights: &ArrayView2<f32>,
    bias: Option<&[f32]>,
    output: &mut Array2<f32>,
) {
    let (m, k) = a.dim();
    let (n, k2) = b_weights.dim();
    
    debug_assert_eq!(k, k2, "Matmul dimension mismatch: A[k]={} != B[k]={}", k, k2);
    debug_assert_eq!(output.dim(), (m, n), "Output shape mismatch: expected ({}, {}), got {:?}", m, n, output.dim());
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), n, "Bias length {} != out_features {}", b.len(), n);
    }

    // Ensure inputs are contiguous
    let a_s = a.as_standard_layout();
    let b_s = b_weights.as_standard_layout();
    let a_slice = a_s.as_slice().expect("Input tensor 'a' must be contiguous");
    let b_slice = b_s.as_slice().expect("Weight tensor 'b' must be contiguous");

    if m == 1 {
        // === DECODE PATH ===
        // Parallelize over output features for single-token inference
        let out_slice = output.as_slice_mut().unwrap();
        let num_threads = rayon::current_num_threads();
        let chunk_size = (n + num_threads - 1) / num_threads;

        out_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let b_row_start_idx = chunk_idx * chunk_size;
                let b_chunk_ptr = unsafe { b_slice.as_ptr().add(b_row_start_idx * k) };

                unsafe {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        kernels::x86::f32::matmul_vec_f32(
                            out_chunk,
                            a_slice.as_ptr(),
                            b_chunk_ptr,
                            k,
                        );
                        
                        // Fuse bias addition
                        if let Some(bias) = bias {
                            let bias_start = b_row_start_idx;
                            for (i, val) in out_chunk.iter_mut().enumerate() {
                                *val += bias[bias_start + i];
                            }
                        }
                        return;
                    }
                    
                    #[cfg(target_arch = "aarch64")]
                    if std::arch::is_aarch64_feature_detected!("neon") {
                        kernels::aarch64::f32::matmul_vec_f32_neon(
                            out_chunk,
                            a_slice.as_ptr(),
                            b_chunk_ptr,
                            k,
                        );
                        
                        if let Some(bias) = bias {
                            let bias_start = b_row_start_idx;
                            for (i, val) in out_chunk.iter_mut().enumerate() {
                                *val += bias[bias_start + i];
                            }
                        }
                        return;
                    }
                    
                    // Scalar fallback
                    let b_rows = &b_slice[b_row_start_idx * k..];
                    kernels::scalar::matmul_vec_f32_scalar(out_chunk, a_slice, b_rows, k);
                    
                    if let Some(bias) = bias {
                        let bias_start = b_row_start_idx;
                        for (i, val) in out_chunk.iter_mut().enumerate() {
                            *val += bias[bias_start + i];
                        }
                    }
                }
            });
    } else {
        // === PREFILL PATH ===
        // Parallelize over batch rows
        let bias_slice = bias;
        
        output
            .outer_iter_mut()
            .into_par_iter()
            .zip(a.outer_iter())
            .for_each(|(mut c_row, a_row)| {
                let a_row_slice = a_row.as_slice().unwrap();
                let out_slice = c_row.as_slice_mut().unwrap();

                unsafe {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        kernels::x86::f32::matmul_vec_f32(
                            out_slice,
                            a_row_slice.as_ptr(),
                            b_slice.as_ptr(),
                            k,
                        );
                        
                        if let Some(bias) = bias_slice {
                            for (val, &b) in out_slice.iter_mut().zip(bias.iter()) {
                                *val += b;
                            }
                        }
                        return;
                    }
                    
                    #[cfg(target_arch = "aarch64")]
                    if std::arch::is_aarch64_feature_detected!("neon") {
                        kernels::aarch64::f32::matmul_vec_f32_neon(
                            out_slice,
                            a_row_slice.as_ptr(),
                            b_slice.as_ptr(),
                            k,
                        );
                        
                        if let Some(bias) = bias_slice {
                            for (val, &b) in out_slice.iter_mut().zip(bias.iter()) {
                                *val += b;
                            }
                        }
                        return;
                    }
                    
                    kernels::scalar::matmul_vec_f32_scalar(out_slice, a_row_slice, b_slice, k);
                    
                    if let Some(bias) = bias_slice {
                        for (val, &b) in out_slice.iter_mut().zip(bias.iter()) {
                            *val += b;
                        }
                    }
                }
            });
    }
}

/// No-alloc version of matmul_2d_cpu_f32_batched
///
/// Computes `C = A @ B^T + bias` for F32 input `A` and F32 weight matrix `B`,
/// writing directly to a pre-allocated output buffer.
///
/// Uses the 4x3 block kernel - optimized for large batches (m >= ~1000).
/// Falls back to `matmul_2d_f32_noalloc` for single-token inputs.
///
/// # Arguments
///
/// * `a` - Input activation matrix of shape `[batch, in_features]`.
/// * `b_weights` - Weight matrix of shape `[out_features, in_features]`.
/// * `bias` - Optional bias vector of shape `[out_features]`.
/// * `output` - Pre-allocated output matrix of shape `[batch, out_features]`.
///
/// # Panics
///
/// Panics if dimensions don't match.
pub fn matmul_2d_f32_batched_noalloc(
    a: &ArrayView2<f32>,
    b_weights: &ArrayView2<f32>,
    bias: Option<&[f32]>,
    output: &mut Array2<f32>,
) {
    let (m, k) = a.dim();
    let (n, k2) = b_weights.dim();
    
    debug_assert_eq!(k, k2, "Matmul dimension mismatch: A[k]={} != B[k]={}", k, k2);
    debug_assert_eq!(output.dim(), (m, n), "Output shape mismatch: expected ({}, {}), got {:?}", m, n, output.dim());
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), n, "Bias length {} != out_features {}", b.len(), n);
    }

    // For single token, use the vec-kernel optimized path
    if m == 1 {
        return matmul_2d_f32_noalloc(a, b_weights, bias, output);
    }

    let a_s = a.as_standard_layout();
    let b_s = b_weights.as_standard_layout();
    let a_slice = a_s.as_slice().expect("Input must be contiguous");
    let b_slice = b_s.as_slice().expect("Weights must be contiguous");
    let c_slice = output.as_slice_mut().expect("Output must be contiguous");

    const BLOCK_SIZE: usize = 64;

    c_slice
        .par_chunks_mut(BLOCK_SIZE * n)
        .zip(a_slice.par_chunks(BLOCK_SIZE * k))
        .for_each(|(out_block, in_block)| {
            let weights_ptr = b_slice.as_ptr();
            let bias_ptr = bias.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());

            let num_tokens = in_block.len() / k;
            let mut t = 0;

            // === MAIN LOOP: Process 4 tokens × 3 outputs at a time ===
            while t + 4 <= num_tokens {
                let in_ptr = unsafe { in_block.as_ptr().add(t * k) };
                let mut j = 0;

                // 4x3 kernel loop
                while j + 3 <= n {
                    let w_ptr = unsafe { weights_ptr.add(j * k) };
                    let out_ptr = unsafe { out_block.as_mut_ptr().add(t * n + j) };
                    let b_ptr = if !bias_ptr.is_null() {
                        unsafe { bias_ptr.add(j) }
                    } else {
                        std::ptr::null()
                    };

                    unsafe {
                        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                            kernels::x86::f32::matmul_block_4x3_f32(
                                out_ptr, n, in_ptr, w_ptr, k, b_ptr,
                            );
                        }
                    }
                    j += 3;
                }

                // Remaining output features (n % 3 != 0)
                while j < n {
                    let w_ptr = unsafe { weights_ptr.add(j * k) };
                    let b_val = if !bias_ptr.is_null() {
                        unsafe { *bias_ptr.add(j) }
                    } else {
                        0.0
                    };

                    for row in 0..4 {
                        let a_row = unsafe { in_ptr.add(row * k) };
                        let dst = unsafe { out_block.as_mut_ptr().add((t + row) * n + j) };

                        let mut sum = 0.0f32;
                        for i in 0..k {
                            sum += unsafe { *a_row.add(i) * *w_ptr.add(i) };
                        }
                        unsafe { *dst = sum + b_val };
                    }
                    j += 1;
                }
                t += 4;
            }

            // === CLEANUP: Remaining tokens (num_tokens % 4 != 0) ===
            while t < num_tokens {
                let in_ptr = unsafe { in_block.as_ptr().add(t * k) };

                for j in 0..n {
                    let w_ptr = unsafe { weights_ptr.add(j * k) };
                    let b_val = if !bias_ptr.is_null() {
                        unsafe { *bias_ptr.add(j) }
                    } else {
                        0.0
                    };

                    let mut sum = 0.0f32;
                    for i in 0..k {
                        sum += unsafe { *in_ptr.add(i) * *w_ptr.add(i) };
                    }
                    out_block[t * n + j] = sum + b_val;
                }
                t += 1;
            }
        });
}

/// Computes `C = A @ B^T` for F32 input `A` and F32 weight matrix `B`.
///
/// This is the highest-precision matmul variant, used when accuracy is paramount
/// or when model weights are already in F32 format.
///
/// # Arguments
///
/// * `a` - Input activation matrix of shape `[batch, in_features]`.
/// * `b_weights` - Weight matrix of shape `[out_features, in_features]`.
///
/// # Returns
///
/// Output matrix of shape `[batch, out_features]`.
///
/// # Panics
///
/// Panics if the inner dimensions don't match (`a.shape()[1] != b_weights.shape()[1]`).
///
/// # Performance
///
/// Uses AVX2+FMA on x86_64, NEON on aarch64, with scalar fallback.
/// via [`crate::utils::tensor_ops::matmul_2d_faer`] instead.
pub fn matmul_2d_cpu_f32(a: &ArrayView2<f32>, b_weights: &ArrayView2<f32>) -> Array2<f32> {
    let (m, k) = a.dim(); // m = batch size, k = input features
    let (n, k2) = b_weights.dim(); // n = output features
    assert_eq!(k, k2, "Matmul dimension mismatch: A[k] != B[k]");

    let mut c = Array2::<f32>::zeros((m, n));

    // Ensure both tensors are in contiguous row-major layout
    let a_s = a.as_standard_layout();
    let b_s = b_weights.as_standard_layout();
    let a_slice = a_s.as_slice().expect("Input tensor 'a' must be contiguous");
    let b_slice = b_s
        .as_slice()
        .expect("Weight tensor 'b' must be contiguous");

    if m == 1 {
        // === DECODE PATH ===
        // Parallelize over output features for single-token inference

        let out_slice = c.as_slice_mut().unwrap();
        let num_threads = rayon::current_num_threads();
        let chunk_size = (n + num_threads - 1) / num_threads;

        out_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                // Calculate pointer to this thread's weight rows
                let b_row_start_idx = chunk_idx * chunk_size;
                let b_chunk_ptr = unsafe { b_slice.as_ptr().add(b_row_start_idx * k) };

                // Dispatch to architecture-specific kernel
                unsafe {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        return kernels::x86::f32::matmul_vec_f32(
                            out_chunk,
                            a_slice.as_ptr(),
                            b_chunk_ptr,
                            k,
                        );
                    }
                    #[cfg(target_arch = "aarch64")]
                    if std::arch::is_aarch64_feature_detected!("neon") {
                        return kernels::aarch64::f32::matmul_vec_f32_neon(
                            out_chunk,
                            a_slice.as_ptr(),
                            b_chunk_ptr,
                            k,
                        );
                    }
                    // Scalar fallback for other architectures
                    let b_rows = &b_slice[b_row_start_idx * k..];
                    kernels::scalar::matmul_vec_f32_scalar(out_chunk, a_slice, b_rows, k);
                }
            });
    } else {
        // === PREFILL PATH ===
        // Parallelize over batch rows for prompt processing

        c.outer_iter_mut()
            .into_par_iter()
            .zip(a.outer_iter())
            .for_each(|(mut c_row, a_row)| {
                let a_row_slice = a_row.as_slice().unwrap();
                let out_slice = c_row.as_slice_mut().unwrap();

                // Each row computes against all weights
                unsafe {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        return kernels::x86::f32::matmul_vec_f32(
                            out_slice,
                            a_row_slice.as_ptr(),
                            b_slice.as_ptr(),
                            k,
                        );
                    }
                    #[cfg(target_arch = "aarch64")]
                    if std::arch::is_aarch64_feature_detected!("neon") {
                        return kernels::aarch64::f32::matmul_vec_f32_neon(
                            out_slice,
                            a_row_slice.as_ptr(),
                            b_slice.as_ptr(),
                            k,
                        );
                    }
                    kernels::scalar::matmul_vec_f32_scalar(out_slice, a_row_slice, b_slice, k);
                }
            });
    }
    c
}

/// Computes `C = A @ B^T` for F32 input `A` and F32 weight matrix `B`.
/// 
/// Batched version using the 4x3 block kernel for improved throughput on
/// multi-token inputs (encoding/prefill). Falls back to `matmul_2d_cpu_f32`
/// for single-token (decode) inputs.
///
/// # Arguments
///
/// * `a` - Input activation matrix of shape `[batch, in_features]`.
/// * `b_weights` - Weight matrix of shape `[out_features, in_features]`.
/// * `bias` - Optional bias vector of shape `[out_features]`.
///
/// # Returns
///
/// Output matrix of shape `[batch, out_features]`.
pub fn matmul_2d_cpu_f32_batched(
    a: &ArrayView2<f32>, 
    b_weights: &ArrayView2<f32>,
    bias: Option<&[f32]>,
) -> Array2<f32> {
    let (m, k) = a.dim();
    let (n, k2) = b_weights.dim();
    assert_eq!(k, k2, "Matmul dimension mismatch: A[k]={} != B[k]={}", k, k2);
    
    if let Some(b) = bias {
        assert_eq!(b.len(), n, "Bias length {} != out_features {}", b.len(), n);
    }

    // For single token, use the decode-optimized path
    if m == 1 {
        let mut result = matmul_2d_cpu_f32(a, b_weights);
        if let Some(b) = bias {
            let out_slice = result.as_slice_mut().unwrap();
            for (i, val) in out_slice.iter_mut().enumerate() {
                *val += b[i];
            }
        }
        return result;
    }

    let mut c = Array2::<f32>::zeros((m, n));
    
    let a_s = a.as_standard_layout();
    let b_s = b_weights.as_standard_layout();
    let a_slice = a_s.as_slice().expect("Input must be contiguous");
    let b_slice = b_s.as_slice().expect("Weights must be contiguous");
    let c_slice = c.as_slice_mut().expect("Output must be contiguous");

    const BLOCK_SIZE: usize = 64;

    c_slice
        .par_chunks_mut(BLOCK_SIZE * n)
        .zip(a_slice.par_chunks(BLOCK_SIZE * k))
        .for_each(|(out_block, in_block)| {
            let weights_ptr = b_slice.as_ptr();
            let bias_ptr = bias.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());

            let num_tokens = in_block.len() / k;
            let mut t = 0;

            // === MAIN LOOP: Process 4 tokens × 3 outputs at a time ===
            while t + 4 <= num_tokens {
                let in_ptr = unsafe { in_block.as_ptr().add(t * k) };
                let mut j = 0;

                // 4x3 kernel loop
                while j + 3 <= n {
                    let w_ptr = unsafe { weights_ptr.add(j * k) };
                    let out_ptr = unsafe { out_block.as_mut_ptr().add(t * n + j) };
                    let b_ptr = if !bias_ptr.is_null() {
                        unsafe { bias_ptr.add(j) }
                    } else {
                        std::ptr::null()
                    };

                    unsafe {
                        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                            kernels::x86::f32::matmul_block_4x3_f32(
                                out_ptr, n, in_ptr, w_ptr, k, b_ptr,
                            );
                        }
                    }
                    j += 3;
                }

                // Remaining output features (n % 3 != 0)
                while j < n {
                    let w_ptr = unsafe { weights_ptr.add(j * k) };
                    let b_val = if !bias_ptr.is_null() {
                        unsafe { *bias_ptr.add(j) }
                    } else {
                        0.0
                    };

                    for row in 0..4 {
                        let a_row = unsafe { in_ptr.add(row * k) };
                        let dst = unsafe { out_block.as_mut_ptr().add((t + row) * n + j) };

                        let mut sum = 0.0f32;
                        for i in 0..k {
                            sum += unsafe { *a_row.add(i) * *w_ptr.add(i) };
                        }
                        unsafe { *dst = sum + b_val };
                    }
                    j += 1;
                }
                t += 4;
            }

            // === CLEANUP: Remaining tokens (num_tokens % 4 != 0) ===
            while t < num_tokens {
                let in_ptr = unsafe { in_block.as_ptr().add(t * k) };

                for j in 0..n {
                    let w_ptr = unsafe { weights_ptr.add(j * k) };
                    let b_val = if !bias_ptr.is_null() {
                        unsafe { *bias_ptr.add(j) }
                    } else {
                        0.0
                    };

                    let mut sum = 0.0f32;
                    for i in 0..k {
                        sum += unsafe { *in_ptr.add(i) * *w_ptr.add(i) };
                    }
                    out_block[t * n + j] = sum + b_val;
                }
                t += 1;
            }
        });

    c
}

/// Computes `C = A @ B^T` for F32 input `A` and Q4_K quantized weight matrix `B`.
///
/// Q4_K is a 4-bit block-quantized format using "K-quants" with 256 elements per block.
/// Each block contains multiple sub-scales for improved accuracy, providing approximately
/// 8x memory compression compared to F32 while maintaining reasonable quality.
///
/// # Arguments
///
/// * `a` - Input activation matrix of shape `[batch, in_features]`.
/// * `b_weights` - Quantized weight blocks. Total elements = `out_features * in_features / 256`.
///
/// # Returns
///
/// Output matrix of shape `[batch, out_features]`.
///
/// # Performance
///
/// Uses AVX2+FMA kernels on x86_64. Currently no scalar fallback is implemented,
/// so this will silently produce zeros on non-x86 platforms.
///
/// # See Also
///
/// - [`matmul_2d_cpu_q6_k`] — Higher precision 6-bit variant.
/// - [`matmul_2d_cpu_q8_0`] — 8-bit variant with simpler block structure.
pub fn matmul_2d_cpu_q4_k(a: &ArrayView2<f32>, b_weights: &[BlockQ4_K]) -> Array2<f32> {
    let (m, k) = a.dim();

    // Q4_K uses 256 elements per block (QK_K constant from GGML)
    let k_per_block = QK_K;

    // Calculate output dimension from total blocks
    let n = (b_weights.len() * k_per_block) / k;

    let mut c = Array2::<f32>::zeros((m, n));
    let a_s = a.as_standard_layout();

    if m == 1 {
        // === DECODE PATH ===
        // Parallelize over output features for single-token inference

        let a_slice = a_s.as_slice().unwrap();
        let out_slice = c.as_slice_mut().unwrap();
        let num_threads = rayon::current_num_threads();
        let chunk_size = (n + num_threads - 1) / num_threads;

        out_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                // Calculate which weight blocks this thread needs
                let num_blocks_per_row = k / k_per_block;
                let b_block_start_idx = chunk_idx * chunk_size * num_blocks_per_row;
                let num_blocks_for_chunk = out_chunk.len() * num_blocks_per_row;
                let b_blocks_chunk =
                    &b_weights[b_block_start_idx..b_block_start_idx + num_blocks_for_chunk];

                // Dispatch to AVX2 kernel (no scalar fallback currently)
                unsafe {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        return kernels::x86::q4_k::matmul_vec_q4_k_avx2(
                            out_chunk,
                            a_slice.as_ptr(),
                            b_blocks_chunk,
                            k,
                        );
                    }
                    // TODO: Add scalar fallback for non-x86 platforms
                }
            });
    } else {
        // === PREFILL PATH ===
        // Parallelize over batch rows for prompt processing

        c.outer_iter_mut()
            .into_par_iter()
            .zip(a.outer_iter())
            .for_each(|(mut c_row, a_row)| {
                let a_row_slice = a_row.as_slice().unwrap();
                let out_slice = c_row.as_slice_mut().unwrap();

                // Each row uses all weight blocks
                unsafe {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        return kernels::x86::q4_k::matmul_vec_q4_k_avx2(
                            out_slice,
                            a_row_slice.as_ptr(),
                            b_weights,
                            k,
                        );
                    }
                }
            });
    }
    c
}

/// Computes `C = A @ B^T` for F32 input `A` and Q6_K quantized weight matrix `B`.
///
/// **Experimental AVX2 variant** — Uses direct F32 input without pre-quantization
/// on the decode path for potentially better performance. Falls back to scalar
/// with Q8_K quantization on the prefill path.
///
/// # Arguments
///
/// * `input` - Input activation matrix of shape `[batch, in_features]`.
/// * `weights` - Quantized weight blocks.
///
/// # Returns
///
/// Output matrix of shape `[batch, out_features]`.
///
/// # Performance
///
/// Decode path uses AVX2 with fixed chunk size of 64 (tuned for Q6_K's
/// computational intensity). Prefill path uses scalar kernels with on-the-fly
/// Q8_K quantization of the input.
#[cfg(target_arch = "x86_64")]
pub fn matmul_2d_cpu_q6_k2(input: &ArrayView2<f32>, weights: &[BlockQ6_K]) -> Array2<f32> {
    let (m, k) = input.dim();

    // Q6_K uses 256 elements per block
    let num_blocks_per_row = k / 256;
    let out_features = weights.len() / num_blocks_per_row;
    let mut output = Array2::zeros((m, out_features));

    if m == 1 {
        // === DECODE PATH ===
        // Uses direct F32 input with AVX2 kernel (no input quantization needed)

        let r = input.row(0);
        let a_slice = r.as_slice().unwrap();
        let out_slice = output.as_slice_mut().unwrap();

        // Fixed chunk size of 64 is optimal for Q6_K's computational intensity
        out_slice
            .par_chunks_mut(64)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                unsafe {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        // Calculate block range for this chunk
                        let start_row = chunk_idx * 64;
                        let block_start = start_row * num_blocks_per_row;
                        let block_count = out_chunk.len() * num_blocks_per_row;

                        return kernels::x86::q6_k::matmul_vec_q6_k_avx2(
                            out_chunk,
                            a_slice.as_ptr(),
                            &weights[block_start..block_start + block_count],
                            k,
                        );
                    }
                    // No scalar fallback for AVX2 path
                }
            });
    } else {
        // === PREFILL PATH ===
        // Uses scalar kernel with Q8_K quantization of input

        output
            .outer_iter_mut()
            .into_par_iter()
            .zip(input.outer_iter())
            .for_each(|(mut out_row, in_row)| {
                let a_slice = in_row.as_slice().unwrap();

                // Quantize input row to Q8_K format for dot product
                let a_q8 = quantize_row_q8_k(a_slice);

                let out_slice = out_row.as_slice_mut().unwrap();
                for (i, out_val) in out_slice.iter_mut().enumerate() {
                    // Extract weight blocks for this output feature
                    let start = i * num_blocks_per_row;
                    let end = start + num_blocks_per_row;
                    let w_row = &weights[start..end];

                    // Compute dot product using scalar kernel
                    *out_val = kernels::scalar::vec_dot_q6k_q8k_scalar(k, w_row, &a_q8);
                }
            });
    }

    output
}
/// Computes `C = A @ B^T` for F32 input `A` and Q6_K quantized weight matrix `B`.
///
/// Q6_K is a 6-bit block-quantized format using "K-quants" with 256 elements per block.
/// It provides approximately 5x memory compression with higher precision than Q4_K,
/// making it suitable for models where quality is prioritized over memory savings.
///
/// # Arguments
///
/// * `input` - Input activation matrix of shape `[batch, in_features]`.
/// * `weights` - Quantized weight blocks. Total elements = `out_features * in_features / 256`.
///
/// # Returns
///
/// Output matrix of shape `[batch, out_features]`.
///
/// # Implementation Details
///
/// This function quantizes the F32 input to Q8_K format before computing the dot
/// product. This is necessary because the Q6_K dot product kernel expects a
/// quantized input for efficient computation.
///
/// # Performance
///
/// Uses scalar kernels with Q8_K input quantization. The decode path pre-quantizes
/// the input once and shares it across all output computations. Consider using
/// [`matmul_2d_cpu_q6_k2`] for potentially faster AVX2 decode path.
///
/// # See Also
///
/// - [`matmul_2d_cpu_q6_k2`] — Experimental AVX2 variant.
/// - [`matmul_2d_cpu_q4_k`] — Lower precision 4-bit variant.
#[cfg(target_arch = "x86_64")]
pub fn matmul_2d_cpu_q6_k(input: &ArrayView2<f32>, weights: &[BlockQ6_K]) -> Array2<f32> {
    let (m, k) = input.dim();

    // Q6_K uses 256 elements per block (QK_K constant)
    let num_blocks_per_row = k / QK_K;
    let out_features = weights.len() / num_blocks_per_row;

    let mut output = Array2::zeros((m, out_features));

    if m == 1 {
        // === DECODE PATH ===
        // Pre-quantize input once, then parallelize over output features

        let r = input.row(0);
        let a_slice = r.as_slice().unwrap();
        let out_slice = output.as_slice_mut().unwrap();

        // Quantize input to Q8_K format once (shared across all threads)
        let a_q8 = quantize_row_q8_k(a_slice);

        // Divide output evenly among threads
        let num_threads = rayon::current_num_threads();
        let chunk_size = (out_features + num_threads - 1) / num_threads;

        out_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                for (j, out_val) in out_chunk.iter_mut().enumerate() {
                    // Map local index to global output index
                    let i = chunk_idx * chunk_size + j;

                    // Extract weight blocks for this output feature
                    let start = i * num_blocks_per_row;
                    let end = start + num_blocks_per_row;
                    let w_row = &weights[start..end];

                    // Compute Q6_K x Q8_K dot product
                    *out_val = kernels::scalar::vec_dot_q6k_q8k_scalar(k, w_row, &a_q8);
                }
            });
    } else {
        // === PREFILL PATH ===
        // Each row quantizes its input independently

        output
            .outer_iter_mut()
            .into_par_iter()
            .zip(input.outer_iter())
            .for_each(|(mut out_row, in_row)| {
                let a_slice = in_row.as_slice().unwrap();

                // Quantize this row's input to Q8_K format
                let a_q8 = quantize_row_q8_k(a_slice);

                let out_slice = out_row.as_slice_mut().unwrap();
                for (i, out_val) in out_slice.iter_mut().enumerate() {
                    // Extract weight blocks for this output feature
                    let start = i * num_blocks_per_row;
                    let end = start + num_blocks_per_row;
                    let w_row = &weights[start..end];

                    // Compute Q6_K x Q8_K dot product
                    *out_val = kernels::scalar::vec_dot_q6k_q8k_scalar(k, w_row, &a_q8);
                }
            });
    }

    output
}



// =========================================================================
// TESTS
// =========================================================================

#[cfg(test)]
mod matmul_tests {
    use super::*;
    use ndarray::Array2;
    use std::time::Instant;

    // =========================================================================
    // Test Utilities
    // =========================================================================

    fn make_input(rows: usize, cols: usize, seed: usize) -> Array2<f32> {
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i + seed) % 1000) as f32 * 0.001 - 0.5)
            .collect();
        Array2::from_shape_vec((rows, cols), data).unwrap()
    }

    fn make_bias(n: usize, base: f32) -> Vec<f32> {
        (0..n).map(|i| base + i as f32 * 0.0001).collect()
    }

    fn max_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn mean_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        sum / a.len() as f32
    }

    /// Reference implementation using the known-working matmul_2d_cpu_f32
    fn reference_matmul_with_bias(
        a: &ArrayView2<f32>,
        b: &ArrayView2<f32>,
        bias: Option<&[f32]>,
    ) -> Array2<f32> {
        let mut result = matmul_2d_cpu_f32(a, b);
        if let Some(bias) = bias {
            for mut row in result.rows_mut() {
                for (val, &b) in row.iter_mut().zip(bias.iter()) {
                    *val += b;
                }
            }
        }
        result
    }

    // =========================================================================
    // F32 Standard Path Tests (matmul_2d_cpu_f32)
    // =========================================================================

    #[test]
    fn test_f32_decode_tiny() {
        let (m, k, n) = (1, 4, 4);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let result = matmul_2d_cpu_f32(&a.view(), &b.view());

        assert_eq!(result.dim(), (m, n));
        println!("\n=== F32 Decode Tiny (m={}, k={}, n={}) ===", m, k, n);
        println!("Output shape: {:?}", result.dim());
    }

    #[test]
    fn test_f32_decode_medium() {
        let (m, k, n) = (1, 384, 1536);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let result = matmul_2d_cpu_f32(&a.view(), &b.view());

        assert_eq!(result.dim(), (m, n));
        println!("\n=== F32 Decode Medium (m={}, k={}, n={}) ===", m, k, n);
    }

    #[test]
    fn test_f32_prefill_small() {
        let (m, k, n) = (8, 64, 128);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let result = matmul_2d_cpu_f32(&a.view(), &b.view());

        assert_eq!(result.dim(), (m, n));
        println!("\n=== F32 Prefill Small (m={}, k={}, n={}) ===", m, k, n);
    }

    // =========================================================================
    // F32 Batched Path Tests (matmul_2d_cpu_f32_batched)
    // =========================================================================

    #[test]
    fn test_batched_parity_tiny_no_bias() {
        let (m, k, n) = (4, 4, 3);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), None);
        let actual = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), None);

        let diff = max_diff(&expected, &actual);
        println!("\n=== Batched Parity Tiny No Bias (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-6, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_parity_tiny_with_bias() {
        let (m, k, n) = (4, 4, 3);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let actual = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), Some(&bias));

        let diff = max_diff(&expected, &actual);
        println!("\n=== Batched Parity Tiny With Bias (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-6, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_parity_small_no_bias() {
        // Tests main 4x3 loop with exact multiple of 3 outputs
        let (m, k, n) = (8, 32, 12); // n=12 is divisible by 3
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), None);
        let actual = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), None);

        let diff = max_diff(&expected, &actual);
        println!("\n=== Batched Parity Small No Bias (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_parity_small_remainder_outputs() {
        // Tests output remainder loop (n % 3 != 0)
        let (m, k, n) = (8, 32, 14); // n=14 leaves remainder of 2
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), None);
        let actual = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), None);

        let diff = max_diff(&expected, &actual);
        println!("\n=== Batched Parity Small Remainder Outputs (m={}, k={}, n={}, n%3={}) ===", m, k, n, n % 3);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_parity_small_remainder_tokens() {
        // Tests token cleanup loop (m % 4 != 0)
        let (m, k, n) = (10, 32, 12); // m=10 leaves remainder of 2 tokens
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), None);
        let actual = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), None);

        let diff = max_diff(&expected, &actual);
        println!("\n=== Batched Parity Small Remainder Tokens (m={}, k={}, m%4={}) ===", m, k, m % 4);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_parity_small_both_remainders() {
        // Tests both remainder loops
        let (m, k, n) = (10, 32, 14); // m%4=2, n%3=2
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let actual = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), Some(&bias));

        let diff = max_diff(&expected, &actual);
        println!("\n=== Batched Parity Both Remainders (m={}, k={}, n={}, m%4={}, n%3={}) ===", 
                 m, k, n, m % 4, n % 3);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_parity_medium_no_bias() {
        // Medium test - this is where the previous bug showed up
        let (m, k, n) = (64, 128, 256);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), None);
        let actual = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), None);

        let diff = max_diff(&expected, &actual);
        let mean = mean_diff(&expected, &actual);
        println!("\n=== Batched Parity Medium No Bias (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff:  {:.2e}", diff);
        println!("Mean diff: {:.2e}", mean);
        assert!(diff < 1e-4, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_parity_medium_with_bias() {
        let (m, k, n) = (64, 128, 256);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let actual = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), Some(&bias));

        let diff = max_diff(&expected, &actual);
        let mean = mean_diff(&expected, &actual);
        println!("\n=== Batched Parity Medium With Bias (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff:  {:.2e}", diff);
        println!("Mean diff: {:.2e}", mean);
        assert!(diff < 1e-4, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_parity_large_no_bias() {
        // MiniLM-like dimensions
        let (m, k, n) = (120 * 24, 384, 384); // batch=120, seq=24
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), None);
        let actual = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), None);

        let diff = max_diff(&expected, &actual);
        let mean = mean_diff(&expected, &actual);
        println!("\n=== Batched Parity Large (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff:  {:.2e}", diff);
        println!("Mean diff: {:.2e}", mean);
        assert!(diff < 1e-3, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_parity_minilm_ffn() {
        // MiniLM FFN layer dimensions
        let (m, k, n) = (120 * 24, 384, 1536);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let actual = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), Some(&bias));

        let diff = max_diff(&expected, &actual);
        let mean = mean_diff(&expected, &actual);
        println!("\n=== Batched Parity MiniLM FFN (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff:  {:.2e}", diff);
        println!("Mean diff: {:.2e}", mean);
        assert!(diff < 1e-3, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_parity_non_aligned_k() {
        // k not divisible by 8 (tests SIMD remainder)
        let (m, k, n) = (8, 37, 12);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), None);
        let actual = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), None);

        let diff = max_diff(&expected, &actual);
        println!("\n=== Batched Parity Non-Aligned K (k={}, k%8={}) ===", k, k % 8);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_single_token_fallback() {
        // m=1 should fall back to decode path
        let (m, k, n) = (1, 384, 1536);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let actual = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), Some(&bias));

        let diff = max_diff(&expected, &actual);
        println!("\n=== Batched Single Token Fallback ===");
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Max diff {} exceeds tolerance", diff);
    }

    // =========================================================================
    // BF16 Tests
    // =========================================================================

    #[test]
    fn test_bf16_decode() {
        use half::bf16;
        
        let (m, k, n) = (1, 384, 1536);
        let a = make_input(m, k, 0);
        let b_f32 = make_input(n, k, 100);
        let b_bf16: Array2<bf16> = b_f32.mapv(bf16::from_f32);

        let result = matmul_2d_cpu_bf16(&a.view(), &b_bf16.view());

        // Compare against F32 result (allowing for BF16 precision loss)
        let expected = matmul_2d_cpu_f32(&a.view(), &b_f32.view());
        let diff = max_diff(&expected, &result);

        println!("\n=== BF16 Decode (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff vs F32: {:.2e}", diff);
        // BF16 has ~3 decimal digits of precision, so expect ~1e-3 error
        assert!(diff < 1e-2, "BF16 diff {} too large", diff);
    }

    #[test]
    fn test_bf16_prefill() {
        use half::bf16;
        
        let (m, k, n) = (32, 384, 1536);
        let a = make_input(m, k, 0);
        let b_f32 = make_input(n, k, 100);
        let b_bf16: Array2<bf16> = b_f32.mapv(bf16::from_f32);

        let result = matmul_2d_cpu_bf16(&a.view(), &b_bf16.view());
        let expected = matmul_2d_cpu_f32(&a.view(), &b_f32.view());
        let diff = max_diff(&expected, &result);

        println!("\n=== BF16 Prefill (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff vs F32: {:.2e}", diff);
        assert!(diff < 1e-2, "BF16 diff {} too large", diff);
    }

    // =========================================================================
    // Performance Benchmarks
    // =========================================================================

    #[test]
    fn perf_f32_standard_vs_batched() {
        let (m, k, n) = (120 * 24, 384, 1536); // MiniLM FFN
        let iterations = 10;

        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        // Warmup
        let _ = matmul_2d_cpu_f32(&a.view(), &b.view());
        let _ = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), Some(&bias));

        // Benchmark standard (no bias - bias added separately)
        let start = Instant::now();
        for _ in 0..iterations {
            let mut result = matmul_2d_cpu_f32(&a.view(), &b.view());
            // Add bias like LinearLayer::matmul does
            for mut row in result.rows_mut() {
                for (val, &b) in row.iter_mut().zip(bias.iter()) {
                    *val += b;
                }
            }
            std::hint::black_box(result);
        }
        let std_time = start.elapsed();

        // Benchmark batched (bias inline)
        let start = Instant::now();
        for _ in 0..iterations {
            let result = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), Some(&bias));
            std::hint::black_box(result);
        }
        let batched_time = start.elapsed();

        let ops = 2 * m * k * n * iterations;
        let std_gflops = ops as f64 / std_time.as_secs_f64() / 1e9;
        let batched_gflops = ops as f64 / batched_time.as_secs_f64() / 1e9;

        println!("\n=== PERF: Standard vs Batched F32 Matmul ===");
        println!("Dimensions: m={}, k={}, n={}", m, k, n);
        println!("Standard (row-parallel): {:?} ({:.2} GFLOPS)", std_time / iterations as u32, std_gflops);
        println!("Batched (4x3 blocks):    {:?} ({:.2} GFLOPS)", batched_time / iterations as u32, batched_gflops);
        println!("Speedup: {:.2}x", std_time.as_secs_f64() / batched_time.as_secs_f64());
    }

    #[test]
    fn perf_bf16_decode_vs_prefill() {
        use half::bf16;
        
        let k = 384;
        let n = 1536;
        let iterations = 100;

        let a_decode = make_input(1, k, 0);
        let a_prefill = make_input(32, k, 0);
        let b_f32 = make_input(n, k, 100);
        let b_bf16: Array2<bf16> = b_f32.mapv(bf16::from_f32);

        // Warmup
        let _ = matmul_2d_cpu_bf16(&a_decode.view(), &b_bf16.view());
        let _ = matmul_2d_cpu_bf16(&a_prefill.view(), &b_bf16.view());

        let start = Instant::now();
        for _ in 0..iterations {
            let result = matmul_2d_cpu_bf16(&a_decode.view(), &b_bf16.view());
            std::hint::black_box(result);
        }
        let decode_time = start.elapsed();

        let start = Instant::now();
        for _ in 0..iterations {
            let result = matmul_2d_cpu_bf16(&a_prefill.view(), &b_bf16.view());
            std::hint::black_box(result);
        }
        let prefill_time = start.elapsed();

        println!("\n=== PERF: BF16 Decode vs Prefill ===");
        println!("Decode (m=1):   {:?} per call", decode_time / iterations as u32);
        println!("Prefill (m=32): {:?} per call", prefill_time / iterations as u32);
        println!("Tokens/sec decode:  {:.0}", iterations as f64 / decode_time.as_secs_f64());
        println!("Tokens/sec prefill: {:.0}", 32.0 * iterations as f64 / prefill_time.as_secs_f64());
    }

    // =========================================================================
    // No-Alloc Tests (matmul_2d_f32_noalloc)
    // =========================================================================

    #[test]
    fn test_noalloc_decode_no_bias() {
        let (m, k, n) = (1, 384, 384);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), None);
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_noalloc(&a.view(), &b.view(), None, &mut output);

        let diff = max_diff(&expected, &output);
        println!("\n=== No-Alloc Decode No Bias (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_noalloc_decode_with_bias() {
        let (m, k, n) = (1, 384, 384);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);

        let diff = max_diff(&expected, &output);
        println!("\n=== No-Alloc Decode With Bias (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_noalloc_decode_ffn_dims() {
        // FFN up-projection: 384 -> 1536
        let (m, k, n) = (1, 384, 1536);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);

        let diff = max_diff(&expected, &output);
        println!("\n=== No-Alloc Decode FFN Dims (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_noalloc_small_batch_no_bias() {
        let (m, k, n) = (16, 384, 384);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), None);
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_noalloc(&a.view(), &b.view(), None, &mut output);

        let diff = max_diff(&expected, &output);
        println!("\n=== No-Alloc Small Batch No Bias (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-4, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_noalloc_small_batch_with_bias() {
        let (m, k, n) = (16, 384, 384);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);

        let diff = max_diff(&expected, &output);
        println!("\n=== No-Alloc Small Batch With Bias (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-4, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_noalloc_medium_batch() {
        let (m, k, n) = (256, 384, 384);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);

        let diff = max_diff(&expected, &output);
        println!("\n=== No-Alloc Medium Batch (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-4, "Max diff {} exceeds tolerance", diff);
    }

    // =========================================================================
    // No-Alloc Batched Tests (matmul_2d_f32_batched_noalloc)
    // =========================================================================

    #[test]
    fn test_batched_noalloc_single_token_fallback() {
        // Should fall back to vec kernel
        let (m, k, n) = (1, 384, 1536);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_batched_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);

        let diff = max_diff(&expected, &output);
        println!("\n=== Batched No-Alloc Single Token Fallback ===");
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_noalloc_tiny_no_bias() {
        let (m, k, n) = (4, 4, 3);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), None);
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_batched_noalloc(&a.view(), &b.view(), None, &mut output);

        let diff = max_diff(&expected, &output);
        println!("\n=== Batched No-Alloc Tiny No Bias (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-6, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_noalloc_tiny_with_bias() {
        let (m, k, n) = (4, 4, 3);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_batched_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);

        let diff = max_diff(&expected, &output);
        println!("\n=== Batched No-Alloc Tiny With Bias (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-6, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_noalloc_remainder_outputs() {
        // n % 3 != 0
        let (m, k, n) = (8, 32, 14);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_batched_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);

        let diff = max_diff(&expected, &output);
        println!("\n=== Batched No-Alloc Remainder Outputs (n={}, n%3={}) ===", n, n % 3);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_noalloc_remainder_tokens() {
        // m % 4 != 0
        let (m, k, n) = (10, 32, 12);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_batched_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);

        let diff = max_diff(&expected, &output);
        println!("\n=== Batched No-Alloc Remainder Tokens (m={}, m%4={}) ===", m, m % 4);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_noalloc_both_remainders() {
        // m % 4 != 0 AND n % 3 != 0
        let (m, k, n) = (10, 32, 14);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_batched_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);

        let diff = max_diff(&expected, &output);
        println!("\n=== Batched No-Alloc Both Remainders (m%4={}, n%3={}) ===", m % 4, n % 3);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_noalloc_medium() {
        let (m, k, n) = (64, 128, 256);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_batched_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);

        let diff = max_diff(&expected, &output);
        let mean = mean_diff(&expected, &output);
        println!("\n=== Batched No-Alloc Medium (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff:  {:.2e}", diff);
        println!("Mean diff: {:.2e}", mean);
        assert!(diff < 1e-4, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_noalloc_large_minilm() {
        // MiniLM-like dimensions
        let (m, k, n) = (120 * 24, 384, 384);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_batched_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);

        let diff = max_diff(&expected, &output);
        let mean = mean_diff(&expected, &output);
        println!("\n=== Batched No-Alloc Large MiniLM (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff:  {:.2e}", diff);
        println!("Mean diff: {:.2e}", mean);
        assert!(diff < 1e-3, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_noalloc_minilm_ffn() {
        // MiniLM FFN dimensions
        let (m, k, n) = (120 * 24, 384, 1536);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_batched_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);

        let diff = max_diff(&expected, &output);
        let mean = mean_diff(&expected, &output);
        println!("\n=== Batched No-Alloc MiniLM FFN (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff:  {:.2e}", diff);
        println!("Mean diff: {:.2e}", mean);
        assert!(diff < 1e-3, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_noalloc_non_aligned_k() {
        // k % 8 != 0 (tests SIMD remainder in 4x3 kernel)
        let (m, k, n) = (8, 37, 12);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), None);
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_batched_noalloc(&a.view(), &b.view(), None, &mut output);

        let diff = max_diff(&expected, &output);
        println!("\n=== Batched No-Alloc Non-Aligned K (k={}, k%8={}) ===", k, k % 8);
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Max diff {} exceeds tolerance", diff);
    }

    // =========================================================================
    // Cross-Validation: Allocating vs No-Alloc
    // =========================================================================

    #[test]
    fn test_noalloc_matches_allocating_decode() {
        let (m, k, n) = (1, 384, 1536);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        // Allocating version
        let mut expected = matmul_2d_cpu_f32(&a.view(), &b.view());
        for (val, &b) in expected.as_slice_mut().unwrap().iter_mut().zip(bias.iter()) {
            *val += b;
        }

        // No-alloc version
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);

        let diff = max_diff(&expected, &output);
        println!("\n=== No-Alloc Matches Allocating (Decode) ===");
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-6, "No-alloc should match allocating exactly");
    }

    #[test]
    fn test_noalloc_matches_allocating_batched() {
        let (m, k, n) = (256, 384, 384);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        // Allocating version
        let expected = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), Some(&bias));

        // No-alloc version
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_batched_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);

        let diff = max_diff(&expected, &output);
        println!("\n=== No-Alloc Matches Allocating (Batched) ===");
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-6, "No-alloc should match allocating exactly");
    }

    // =========================================================================
    // Performance Comparison
    // =========================================================================

    #[test]
    fn bench_noalloc_vs_allocating() {
        use std::time::Instant;

        println!("\n");
        println!("╔══════════════════════════════════════════════════════════════════════╗");
        println!("║           NO-ALLOC VS ALLOCATING BENCHMARK                           ║");
        println!("╚══════════════════════════════════════════════════════════════════════╝");

        let configs = [
            ("Decode (m=1)", 1, 384, 1536, 1000, 100),
            ("Small batch (m=16)", 16, 384, 384, 500, 50),
            ("Medium batch (m=256)", 256, 384, 384, 100, 10),
            ("Large batch (m=2880)", 2880, 384, 384, 50, 5),
        ];

        for (name, m, k, n, iterations, warmup) in configs {
            let a = make_input(m, k, 0);
            let b = make_input(n, k, 100);
            let bias = make_bias(n, 0.01);
            let mut output = Array2::zeros((m, n));

            // Warmup
            for _ in 0..warmup {
                if m == 1 {
                    let mut r = matmul_2d_cpu_f32(&a.view(), &b.view());
                    for (val, &b) in r.as_slice_mut().unwrap().iter_mut().zip(bias.iter()) {
                        *val += b;
                    }
                } else {
                    let _ = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), Some(&bias));
                }
                matmul_2d_f32_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);
            }

            // Benchmark allocating
            let start = Instant::now();
            for _ in 0..iterations {
                if m == 1 {
                    let mut r = matmul_2d_cpu_f32(&a.view(), &b.view());
                    for (val, &b) in r.as_slice_mut().unwrap().iter_mut().zip(bias.iter()) {
                        *val += b;
                    }
                    std::hint::black_box(r);
                } else {
                    let r = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), Some(&bias));
                    std::hint::black_box(r);
                }
            }
            let alloc_time = start.elapsed();

            // Benchmark no-alloc
            let start = Instant::now();
            for _ in 0..iterations {
                matmul_2d_f32_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);
                std::hint::black_box(&output);
            }
            let noalloc_time = start.elapsed();

            let alloc_per = alloc_time / iterations as u32;
            let noalloc_per = noalloc_time / iterations as u32;
            let speedup = alloc_time.as_secs_f64() / noalloc_time.as_secs_f64();

            println!("\n=== {} ===", name);
            println!("Allocating: {:>10.2?}", alloc_per);
            println!("No-alloc:   {:>10.2?}", noalloc_per);
            println!("Speedup:    {:.2}x", speedup);
        }
    }

    #[test]
    fn bench_batched_noalloc_vs_allocating() {
        use std::time::Instant;

        println!("\n");
        println!("╔══════════════════════════════════════════════════════════════════════╗");
        println!("║           BATCHED NO-ALLOC VS ALLOCATING BENCHMARK                   ║");
        println!("╚══════════════════════════════════════════════════════════════════════╝");

        let configs = [
            ("MiniLM QKV (m=2880, n=384)", 2880, 384, 384, 50, 5),
            ("MiniLM FFN up (m=2880, n=1536)", 2880, 384, 1536, 30, 3),
            ("MiniLM FFN down (m=2880, n=384)", 2880, 1536, 384, 30, 3),
            ("BERT QKV (m=2048, n=768)", 2048, 768, 768, 20, 3),
        ];

        for (name, m, k, n, iterations, warmup) in configs {
            let a = make_input(m, k, 0);
            let b = make_input(n, k, 100);
            let bias = make_bias(n, 0.01);
            let mut output = Array2::zeros((m, n));

            // Warmup
            for _ in 0..warmup {
                let _ = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), Some(&bias));
                matmul_2d_f32_batched_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);
            }

            // Benchmark allocating
            let start = Instant::now();
            for _ in 0..iterations {
                let r = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), Some(&bias));
                std::hint::black_box(r);
            }
            let alloc_time = start.elapsed();

            // Benchmark no-alloc
            let start = Instant::now();
            for _ in 0..iterations {
                matmul_2d_f32_batched_noalloc(&a.view(), &b.view(), Some(&bias), &mut output);
                std::hint::black_box(&output);
            }
            let noalloc_time = start.elapsed();

            let alloc_per = alloc_time / iterations as u32;
            let noalloc_per = noalloc_time / iterations as u32;
            let speedup = alloc_time.as_secs_f64() / noalloc_time.as_secs_f64();

            println!("\n=== {} ===", name);
            println!("Allocating: {:>10.2?}", alloc_per);
            println!("No-alloc:   {:>10.2?}", noalloc_per);
            println!("Speedup:    {:.2}x", speedup);
        }
    }
}