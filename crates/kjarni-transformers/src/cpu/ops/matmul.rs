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
/// For maximum performance on F32 weights, consider using the `faer` library
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
