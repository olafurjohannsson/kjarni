use crate::cpu::kernels::q_common::BlockQ6_K;
use crate::cpu::kernels::{
    self,
    q_common::{BlockQ4_K, BlockQ8_0, BlockQ8_K, QK_K},
    quantize::quantize_row_q8_k,
};

use half::bf16;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

/// Computes `C = A @ B^T` for F32 input `A` and Q8_0 quantized weight matrix `B`.
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
        // Single input vector: parallelize over output features.

        let a_slice = a_s.as_slice().unwrap();
        let out_slice = c.as_slice_mut().unwrap();

        // Divide output evenly among available threads
        let num_threads = rayon::current_num_threads();
        let chunk_size = n.div_ceil(num_threads);

        out_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let num_blocks_per_row = k / k_per_block;
                let b_block_start_idx = chunk_idx * chunk_size * num_blocks_per_row;
                let num_blocks_for_chunk = out_chunk.len() * num_blocks_per_row;
                let b_blocks_chunk =
                    &b_weights[b_block_start_idx..b_block_start_idx + num_blocks_for_chunk];

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
                    kernels::scalar::matmul_vec_q8_0_scalar(out_chunk, a_slice, b_blocks_chunk, k);
                }
            });
    } else {
        c.outer_iter_mut()
            .into_par_iter()
            .zip(a.outer_iter())
            .for_each(|(mut c_row, a_row)| {
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

/// Computes `C = A @ B^T` for F32 input `A` and BF16 weight matrix `B`.
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
        let out_slice = c.as_slice_mut().unwrap();
        let num_threads = rayon::current_num_threads();
        let chunk_size = n.div_ceil(num_threads);

        out_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let b_row_start_idx = chunk_idx * chunk_size;
                let b_chunk_ptr =
                    unsafe { b_slice.as_ptr().add(b_row_start_idx * k) as *const u16 };

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
                        std::mem::transmute::<&[half::bf16], &[u16]>(b_rows),
                        k,
                    );
                }
            });
    } else {
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
                        std::mem::transmute::<&[half::bf16], &[u16]>(b_slice),
                        k,
                    );
                }
            });
    }
    c
}

use faer::Parallelism;

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
    let mat_c = faer::mat::from_row_major_slice_mut(c_slice, m, n);

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
pub fn matmul_2d_f32_noalloc(
    a: &ArrayView2<f32>,
    b_weights: &ArrayView2<f32>,
    bias: Option<&[f32]>,
    output: &mut Array2<f32>,
) {
    let (m, k) = a.dim();
    let (n, k2) = b_weights.dim();

    debug_assert_eq!(
        k, k2,
        "Matmul dimension mismatch: A[k]={} != B[k]={}",
        k, k2
    );
    debug_assert_eq!(
        output.dim(),
        (m, n),
        "Output shape mismatch: expected ({}, {}), got {:?}",
        m,
        n,
        output.dim()
    );
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), n, "Bias length {} != out_features {}", b.len(), n);
    }

    // Ensure inputs are contiguous
    let a_s = a.as_standard_layout();
    let b_s = b_weights.as_standard_layout();
    let a_slice = a_s.as_slice().expect("Input tensor 'a' must be contiguous");
    let b_slice = b_s
        .as_slice()
        .expect("Weight tensor 'b' must be contiguous");

    if m == 1 {
        let out_slice = output.as_slice_mut().unwrap();
        let num_threads = rayon::current_num_threads();
        let chunk_size = n.div_ceil(num_threads);

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
pub fn matmul_2d_f32_batched_noalloc(
    a: &ArrayView2<f32>,
    b_weights: &ArrayView2<f32>,
    bias: Option<&[f32]>,
    output: &mut Array2<f32>,
) {
    let (m, k) = a.dim();
    let (n, k2) = b_weights.dim();

    debug_assert_eq!(
        k, k2,
        "Matmul dimension mismatch: A[k]={} != B[k]={}",
        k, k2
    );
    debug_assert_eq!(
        output.dim(),
        (m, n),
        "Output shape mismatch: expected ({}, {}), got {:?}",
        m,
        n,
        output.dim()
    );
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

            // 4 tokens × 3 outputs at a time
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

            // remaining tokens (num_tokens % 4 != 0)
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
        let out_slice = c.as_slice_mut().unwrap();
        let num_threads = rayon::current_num_threads();
        let chunk_size = n.div_ceil(num_threads);

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
pub fn matmul_2d_cpu_f32_batched(
    a: &ArrayView2<f32>,
    b_weights: &ArrayView2<f32>,
    bias: Option<&[f32]>,
) -> Array2<f32> {
    let (m, k) = a.dim();
    let (n, k2) = b_weights.dim();
    assert_eq!(
        k, k2,
        "Matmul dimension mismatch: A[k]={} != B[k]={}",
        k, k2
    );

    if let Some(b) = bias {
        assert_eq!(b.len(), n, "Bias length {} != out_features {}", b.len(), n);
    }

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
/// The activations are quantised to Q8_K once per input row and the dot products run
/// in the integer domain, which is what llama.cpp does. Measured on a real 8192x3072
/// `ffn_gate` on an i7-13700, both paths across the rayon pool, that is 105 GB/s
/// against 54 for expanding every 4-bit weight to f32 first: 1.95x.
///
/// The cost is that the activations pass through int8. On real weights that is about
/// 0.2% relative error, rising to roughly 2% when one large activation stretches the
/// absmax scale of its 256-wide block. `matmul_2d_cpu_q6_k` has always worked this
/// way, so this makes the two quantised CPU paths consistent rather than introducing
/// a new compromise. `x86::q4_k::matmul_vec_q4_k_avx2` still holds the exact f32 path
/// for anything that needs it.
pub fn matmul_2d_cpu_q4_k(a: &ArrayView2<f32>, b_weights: &[BlockQ4_K]) -> Array2<f32> {
    let (m, k) = a.dim();
    let blocks_per_row = k / QK_K;
    let n = (b_weights.len() * QK_K) / k;

    let mut c = Array2::<f32>::zeros((m, n));
    let a_s = a.as_standard_layout();

    // Feature detection once, not once per output row.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let has_avx2 = false;

    // One output row: the weight row against an already-quantised input row.
    let dot = |w_row: &[BlockQ4_K], a_q8: &[BlockQ8_K]| -> f32 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if has_avx2 {
            return unsafe { kernels::x86::q4k_q8k::vec_dot_q4k_q8k_avx2(k, w_row, a_q8) };
        }
        kernels::scalar::vec_dot_q4k_q8k_scalar(k, w_row, a_q8)
    };

    let rows_into = |out: &mut [f32], first_row: usize, a_q8: &[BlockQ8_K]| {
        for (j, o) in out.iter_mut().enumerate() {
            let start = (first_row + j) * blocks_per_row;
            *o = dot(&b_weights[start..start + blocks_per_row], a_q8);
        }
    };

    if m == 1 {
        let a_slice = a_s.as_slice().unwrap();
        // Quantised once and shared by every thread: the cost is amortised over all
        // `n` output rows, which is why the integer path wins despite the extra pass.
        let a_q8 = quantize_row_q8_k(a_slice);

        let out_slice = c.as_slice_mut().unwrap();
        let chunk_size = n.div_ceil(rayon::current_num_threads());
        out_slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                rows_into(out_chunk, chunk_idx * chunk_size, &a_q8);
            });
    } else {
        c.outer_iter_mut()
            .into_par_iter()
            .zip(a.outer_iter())
            .for_each(|(mut c_row, a_row)| {
                let a_q8 = quantize_row_q8_k(a_row.as_slice().unwrap());
                rows_into(c_row.as_slice_mut().unwrap(), 0, &a_q8);
            });
    }
    c
}

/// Computes `C = A @ B^T` for F32 input `A` and Q6_K quantized weight matrix `B`.
pub fn matmul_2d_cpu_q6_k2(input: &ArrayView2<f32>, weights: &[BlockQ6_K]) -> Array2<f32> {
    let (m, k) = input.dim();

    // Q6_K uses 256 elements per block
    let num_blocks_per_row = k / 256;
    let out_features = weights.len() / num_blocks_per_row;
    let mut output = Array2::zeros((m, out_features));

    if m == 1 {
        let r = input.row(0);
        let a_slice = r.as_slice().unwrap();
        let out_slice = output.as_slice_mut().unwrap();

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

                        kernels::x86::q6_k::matmul_vec_q6_k_avx2(
                            out_chunk,
                            a_slice.as_ptr(),
                            &weights[block_start..block_start + block_count],
                            k,
                        )
                    }
                    // scalar fallback ?
                }
            });
    } else {
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
pub fn matmul_2d_cpu_q6_k(input: &ArrayView2<f32>, weights: &[BlockQ6_K]) -> Array2<f32> {
    let (m, k) = input.dim();

    // Q6_K uses 256 elements per block (QK_K constant)
    let num_blocks_per_row = k / QK_K;
    let out_features = weights.len() / num_blocks_per_row;

    let mut output = Array2::zeros((m, out_features));

    if m == 1 {
        let r = input.row(0);
        let a_slice = r.as_slice().unwrap();
        let out_slice = output.as_slice_mut().unwrap();

        // Quantize input to Q8_K format once (shared across all threads)
        let a_q8 = quantize_row_q8_k(a_slice);

        // Divide output evenly among threads
        let num_threads = rayon::current_num_threads();
        let chunk_size = out_features.div_ceil(num_threads);

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
        output
            .outer_iter_mut()
            .into_par_iter()
            .zip(input.outer_iter())
            .for_each(|(mut out_row, in_row)| {
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

/// C[..m, ..n] = A[m, k] @ W[n, k]^T + bias, writing into a caller-owned buffer.
///
/// faer has no bias argument and needs exact dimensions, so the output is viewed
/// as its first `m * n` elements. Nothing here allocates: every matrix is a
/// borrowed view over memory the caller already owns.
pub fn matmul_2d_f32_faer_noalloc(
    a: &ArrayView2<f32>,
    b_weights: &ArrayView2<f32>,
    bias: Option<&[f32]>,
    output: &mut Array2<f32>,
) {
    use rayon::prelude::*;

    let (m, k) = a.dim();
    let (n, _) = b_weights.dim();

    let a_slice = a.to_slice().expect("input must be contiguous");
    let w_slice = b_weights.to_slice().expect("weights must be contiguous");
    let out_slice = &mut output.as_slice_mut().expect("output must be contiguous")[..m * n];

    {
        let a_f = faer::mat::from_row_major_slice(a_slice, m, k);
        let b_f = faer::mat::from_row_major_slice(w_slice, n, k);
        let c_f = faer::mat::from_row_major_slice_mut(out_slice, m, n);
        faer::linalg::matmul::matmul(
            c_f,
            a_f,
            b_f.transpose(),
            None,
            1.0f32,
            faer::Parallelism::Rayon(0),
        );
    }

    if let Some(b) = bias {
        out_slice.par_chunks_mut(n).for_each(|row| {
            for (v, &bv) in row.iter_mut().zip(b.iter()) {
                *v += bv;
            }
        });
    }
}

/// Vector-kernel matmul parallelised over output columns instead of rows.
///
/// The row-parallel path gives every task a whole row of A and streams the entire
/// weight matrix past it, so at m rows the weights are pulled through cache m
/// times, and when m is below the thread count most threads get nothing. A
/// MiniLM encode of 18 tokens runs about 70 MFLOP per layer in a millisecond,
/// which is 2.9 GFLOP/s per thread against a per-core peak near 160, and
/// profiling puts 15% of that encode in crossbeam and rayon bookkeeping.
///
/// Splitting by column instead gives each thread its own band of weights, read
/// once and reused across every row, and all threads get equal work regardless
/// of m. Bands are multiples of the 16 floats in a cache line so that two
/// threads never share a line of the output.
///
/// Falls back to the row-parallel path where the AVX2 kernel is unavailable.
pub fn matmul_2d_f32_noalloc_par_n(
    a: &ArrayView2<f32>,
    b_weights: &ArrayView2<f32>,
    bias: Option<&[f32]>,
    output: &mut Array2<f32>,
) {
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    return matmul_2d_f32_noalloc(a, b_weights, bias, output);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return matmul_2d_f32_noalloc(a, b_weights, bias, output);
        }

        let (m, k) = a.dim();
        let (n, k2) = b_weights.dim();
        debug_assert_eq!(k, k2);
        debug_assert_eq!(output.dim(), (m, n));

        let a_s = a.as_standard_layout();
        let b_s = b_weights.as_standard_layout();
        let a_slice = a_s.as_slice().expect("input must be contiguous");
        let b_slice = b_s.as_slice().expect("weights must be contiguous");

        struct OutPtr(*mut f32);
        // SAFETY: each task writes only its own band of columns, and the bands are
        // disjoint, so no two threads touch the same element.
        unsafe impl Send for OutPtr {}
        unsafe impl Sync for OutPtr {}
        let out = OutPtr(output.as_mut_ptr());

        const LINE: usize = 16;
        let threads = rayon::current_num_threads();
        let band = n.div_ceil(threads).div_ceil(LINE).max(1) * LINE;

        (0..n.div_ceil(band)).into_par_iter().for_each(|bi| {
            let out = &out;
            let j0 = bi * band;
            let j_end = (j0 + band).min(n);
            let width = j_end - j0;

            unsafe {
                let b_band = b_slice.as_ptr().add(j0 * k);
                for t in 0..m {
                    let dst = std::slice::from_raw_parts_mut(out.0.add(t * n + j0), width);
                    kernels::x86::f32::matmul_vec_f32(dst, a_slice.as_ptr().add(t * k), b_band, k);
                    if let Some(bs) = bias {
                        for (i, v) in dst.iter_mut().enumerate() {
                            *v += bs[j0 + i];
                        }
                    }
                }
            }
        });
    }
}

/// Column-parallel matmul using the 4x3 register tile.
///
/// Same band structure as [`matmul_2d_f32_noalloc_par_n`], but each band is
/// worked in 4 row by 3 column tiles instead of one output at a time. The tile
/// does 12 FMAs per 7 loads where the vector kernel does one per two, and since
/// the inner loop was restructured to hold three weight vectors and stream the
/// input rows it fits the sixteen AVX2 registers exactly, with no stack traffic.
///
/// Rows past a multiple of four, and columns past a multiple of three inside a
/// band, fall back to the vector kernel.
pub fn matmul_2d_f32_tile43_par_n(
    a: &ArrayView2<f32>,
    b_weights: &ArrayView2<f32>,
    bias: Option<&[f32]>,
    output: &mut Array2<f32>,
) {
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    return matmul_2d_f32_noalloc(a, b_weights, bias, output);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return matmul_2d_f32_noalloc(a, b_weights, bias, output);
        }

        let (m, k) = a.dim();
        let (n, k2) = b_weights.dim();
        debug_assert_eq!(k, k2);
        debug_assert_eq!(output.dim(), (m, n));

        let a_s = a.as_standard_layout();
        let b_s = b_weights.as_standard_layout();
        let a_slice = a_s.as_slice().expect("input must be contiguous");
        let b_slice = b_s.as_slice().expect("weights must be contiguous");

        struct OutPtr(*mut f32);
        // SAFETY: each task writes only its own band of columns, and the bands
        // are disjoint, so no two threads touch the same element.
        unsafe impl Send for OutPtr {}
        unsafe impl Sync for OutPtr {}
        let out = OutPtr(output.as_mut_ptr());

        const LINE: usize = 16;
        let threads = rayon::current_num_threads();
        let band = n.div_ceil(threads).div_ceil(LINE).max(1) * LINE;

        (0..n.div_ceil(band)).into_par_iter().for_each(|bi| {
            let out = &out;
            let j0 = bi * band;
            let j_end = (j0 + band).min(n);
            let bias_ptr = bias.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());

            unsafe {
                let base = out.0;
                let mut t = 0;

                while t + 4 <= m {
                    let in_ptr = a_slice.as_ptr().add(t * k);
                    let mut j = j0;

                    while j + 3 <= j_end {
                        kernels::x86::f32::matmul_block_4x3_f32(
                            base.add(t * n + j),
                            n,
                            in_ptr,
                            b_slice.as_ptr().add(j * k),
                            k,
                            if bias_ptr.is_null() {
                                std::ptr::null()
                            } else {
                                bias_ptr.add(j)
                            },
                        );
                        j += 3;
                    }

                    if j < j_end {
                        for row in 0..4 {
                            let dst = std::slice::from_raw_parts_mut(
                                base.add((t + row) * n + j),
                                j_end - j,
                            );
                            kernels::x86::f32::matmul_vec_f32(
                                dst,
                                in_ptr.add(row * k),
                                b_slice.as_ptr().add(j * k),
                                k,
                            );
                            if !bias_ptr.is_null() {
                                for (i, v) in dst.iter_mut().enumerate() {
                                    *v += *bias_ptr.add(j + i);
                                }
                            }
                        }
                    }
                    t += 4;
                }

                while t < m {
                    let dst = std::slice::from_raw_parts_mut(base.add(t * n + j0), j_end - j0);
                    kernels::x86::f32::matmul_vec_f32(
                        dst,
                        a_slice.as_ptr().add(t * k),
                        b_slice.as_ptr().add(j0 * k),
                        k,
                    );
                    if !bias_ptr.is_null() {
                        for (i, v) in dst.iter_mut().enumerate() {
                            *v += *bias_ptr.add(j0 + i);
                        }
                    }
                    t += 1;
                }
            }
        });
    }
}

#[cfg(test)]
mod matmul_tests {
    use super::*;
    use ndarray::Array2;

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

    #[test]
    fn test_batched_parity_tiny_no_bias() {
        let (m, k, n) = (4, 4, 3);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), None);
        let actual = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), None);

        let diff = max_diff(&expected, &actual);
        println!(
            "\n=== Batched Parity Tiny No Bias (m={}, k={}, n={}) ===",
            m, k, n
        );
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
        println!(
            "\n=== Batched Parity Tiny With Bias (m={}, k={}, n={}) ===",
            m, k, n
        );
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-6, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_parity_small_no_bias() {
        let (m, k, n) = (8, 32, 12); // n=12 is divisible by 3
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), None);
        let actual = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), None);

        let diff = max_diff(&expected, &actual);
        println!(
            "\n=== Batched Parity Small No Bias (m={}, k={}, n={}) ===",
            m, k, n
        );
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
        println!(
            "\n=== Batched Parity Small Remainder Outputs (m={}, k={}, n={}, n%3={}) ===",
            m,
            k,
            n,
            n % 3
        );
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
        println!(
            "\n=== Batched Parity Small Remainder Tokens (m={}, k={}, m%4={}) ===",
            m,
            k,
            m % 4
        );
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
        println!(
            "\n=== Batched Parity Both Remainders (m={}, k={}, n={}, m%4={}, n%3={}) ===",
            m,
            k,
            n,
            m % 4,
            n % 3
        );
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_parity_medium_no_bias() {
        let (m, k, n) = (64, 128, 256);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), None);
        let actual = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), None);

        let diff = max_diff(&expected, &actual);
        let mean = mean_diff(&expected, &actual);
        println!(
            "\n=== Batched Parity Medium No Bias (m={}, k={}, n={}) ===",
            m, k, n
        );
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
        println!(
            "\n=== Batched Parity Medium With Bias (m={}, k={}, n={}) ===",
            m, k, n
        );
        println!("Max diff:  {:.2e}", diff);
        println!("Mean diff: {:.2e}", mean);
        assert!(diff < 1e-4, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_parity_large_no_bias() {
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
        let (m, k, n) = (120 * 24, 384, 1536);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);
        let bias = make_bias(n, 0.01);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), Some(&bias));
        let actual = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), Some(&bias));

        let diff = max_diff(&expected, &actual);
        let mean = mean_diff(&expected, &actual);
        println!(
            "\n=== Batched Parity MiniLM FFN (m={}, k={}, n={}) ===",
            m, k, n
        );
        println!("Max diff:  {:.2e}", diff);
        println!("Mean diff: {:.2e}", mean);
        assert!(diff < 1e-3, "Max diff {} exceeds tolerance", diff);
    }

    #[test]
    fn test_batched_parity_non_aligned_k() {
        let (m, k, n) = (8, 37, 12);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), None);
        let actual = matmul_2d_cpu_f32_batched(&a.view(), &b.view(), None);

        let diff = max_diff(&expected, &actual);
        println!(
            "\n=== Batched Parity Non-Aligned K (k={}, k%8={}) ===",
            k,
            k % 8
        );
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

    #[test]
    fn test_bf16_decode() {
        use half::bf16;

        let (m, k, n) = (1, 384, 1536);
        let a = make_input(m, k, 0);
        let b_f32 = make_input(n, k, 100);
        let b_bf16: Array2<bf16> = b_f32.mapv(bf16::from_f32);

        let result = matmul_2d_cpu_bf16(&a.view(), &b_bf16.view());
        let expected = matmul_2d_cpu_f32(&a.view(), &b_f32.view());
        let diff = max_diff(&expected, &result);

        println!("\n=== BF16 Decode (m={}, k={}, n={}) ===", m, k, n);
        println!("Max diff vs F32: {:.2e}", diff);
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

    #[test]
    fn test_noalloc_decode_no_bias() {
        let (m, k, n) = (1, 384, 384);
        let a = make_input(m, k, 0);
        let b = make_input(n, k, 100);

        let expected = reference_matmul_with_bias(&a.view(), &b.view(), None);
        let mut output = Array2::zeros((m, n));
        matmul_2d_f32_noalloc(&a.view(), &b.view(), None, &mut output);

        let diff = max_diff(&expected, &output);
        println!(
            "\n=== No-Alloc Decode No Bias (m={}, k={}, n={}) ===",
            m, k, n
        );
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
        println!(
            "\n=== No-Alloc Decode With Bias (m={}, k={}, n={}) ===",
            m, k, n
        );
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
        println!(
            "\n=== No-Alloc Decode FFN Dims (m={}, k={}, n={}) ===",
            m, k, n
        );
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
        println!(
            "\n=== No-Alloc Small Batch No Bias (m={}, k={}, n={}) ===",
            m, k, n
        );
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
        println!(
            "\n=== No-Alloc Small Batch With Bias (m={}, k={}, n={}) ===",
            m, k, n
        );
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
        println!(
            "\n=== No-Alloc Medium Batch (m={}, k={}, n={}) ===",
            m, k, n
        );
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-4, "Max diff {} exceeds tolerance", diff);
    }

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
        println!(
            "\n=== Batched No-Alloc Tiny No Bias (m={}, k={}, n={}) ===",
            m, k, n
        );
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
        println!(
            "\n=== Batched No-Alloc Tiny With Bias (m={}, k={}, n={}) ===",
            m, k, n
        );
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
        println!(
            "\n=== Batched No-Alloc Remainder Outputs (n={}, n%3={}) ===",
            n,
            n % 3
        );
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
        println!(
            "\n=== Batched No-Alloc Remainder Tokens (m={}, m%4={}) ===",
            m,
            m % 4
        );
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
        println!(
            "\n=== Batched No-Alloc Both Remainders (m%4={}, n%3={}) ===",
            m % 4,
            n % 3
        );
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
        println!(
            "\n=== Batched No-Alloc Medium (m={}, k={}, n={}) ===",
            m, k, n
        );
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
        println!(
            "\n=== Batched No-Alloc Large MiniLM (m={}, k={}, n={}) ===",
            m, k, n
        );
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
        println!(
            "\n=== Batched No-Alloc MiniLM FFN (m={}, k={}, n={}) ===",
            m, k, n
        );
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
        println!(
            "\n=== Batched No-Alloc Non-Aligned K (k={}, k%8={}) ===",
            k,
            k % 8
        );
        println!("Max diff: {:.2e}", diff);
        assert!(diff < 1e-5, "Max diff {} exceeds tolerance", diff);
    }

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
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod q6k_path_tests {
    use super::*;
    use crate::weights::ModelWeights;

    /// What the two Q6_K decode paths actually cost, measured on real weights.
    ///
    /// A Q4_K_M file is 42% Q6_K by bytes -- `token_embd`, `output`, and half of
    /// `ffn_down` and `attn_v` -- so whichever path runs here decides a large share of
    /// decode, not a rounding error.
    ///
    /// `matmul_2d_cpu_q6_k` (wired into `LinearLayer::forward`) squeezes the activations
    /// to int8 with `quantize_row_q8_k` and calls `vec_dot_q6k_q8k_scalar`. That name is
    /// misleading: LLVM auto-vectorises it into packed SSE2 integer code (`pmullw`,
    /// `pmaddwd`, `punpcklbw`), which is why it is not the disaster it looks like.
    ///
    /// `matmul_2d_cpu_q6_k2` keeps the activations in f32 and calls the hand-written
    /// AVX2 kernel. It is the more accurate of the two by four orders of magnitude, and
    /// measured interleaved on this machine it is also marginally faster.
    ///
    /// The scalar kernel is not buggy: given the same Q8_K blocks it reproduces exact
    /// arithmetic to ~2e-4. The entire gap is the activation quantisation in front of it,
    /// which is why the error grows with the size of the largest activation.
    #[test]
    fn q6k_int8_activation_cost_against_the_f32_path() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("Skipping: needs AVX2+FMA");
            return;
        }
        let path = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join(".cache/kjarni/llama-3.2-3b-instruct-q4_k_m/Llama-3.2-3B-Instruct-Q4_K_M.gguf");
        if !path.exists() {
            println!("Skipping: model not present");
            return;
        }
        let w = ModelWeights::new(&path).unwrap();
        assert_eq!(
            w.tensor_dtype("output.weight").unwrap(),
            crate::tensor::DType::Q6_K
        );

        let k = 3072usize;
        let rows = 256usize;
        let blocks_per_row = k / 256;
        let all: Vec<BlockQ6_K> = w
            .with_raw_tensor("output.weight", |v| {
                Ok(
                    bytemuck::cast_slice::<u8, BlockQ6_K>(&v.bytes)[..rows * blocks_per_row]
                        .to_vec(),
                )
            })
            .unwrap();

        let mut base = vec![0.0f32; k];
        let mut state = 0x2545F4914F6CDD1Du64;
        for x in base.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // 24 bits mapped onto [-1, 1], the scale a post-RMSNorm hidden state has.
            *x = ((state >> 40) as f32 / 8_388_608.0) - 1.0;
        }

        // Absmax int8 spends its whole range on the largest element of the block, so a
        // single outlier coarsens every other value sharing it. Sweeping the outlier is
        // sweeping the only axis that matters.
        for outlier in [0.0f32, 3.0, 10.0, 30.0] {
            let mut a = base.clone();
            if outlier > 0.0 {
                a[11] = outlier;
            }
            let input = ndarray::Array2::from_shape_vec((1, k), a.clone()).unwrap();

            let mut exact = vec![0.0f64; rows];
            let mut wf = [0.0f32; 256];
            for (r, e) in exact.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for b in 0..blocks_per_row {
                    crate::cpu::kernels::dequantize::dequantize_q6_k_block(
                        &all[r * blocks_per_row + b],
                        &mut wf,
                    );
                    for j in 0..256 {
                        acc += wf[j] as f64 * a[b * 256 + j] as f64;
                    }
                }
                *e = acc;
            }

            let wired = matmul_2d_cpu_q6_k(&input.view(), &all);
            let avx2 = matmul_2d_cpu_q6_k2(&input.view(), &all);
            let err = |got: &ndarray::Array2<f32>| {
                (0..rows).fold(0.0f64, |worst, r| {
                    worst.max((got[[0, r]] as f64 - exact[r]).abs() / exact[r].abs().max(1.0))
                })
            };
            let (e_wired, e_avx2) = (err(&wired), err(&avx2));
            println!("  outlier {outlier:>5.1}: wired int8 {e_wired:.3e}   avx2 f32 {e_avx2:.3e}");

            // The f32 path carries no activation quantisation, so it must stay near exact.
            assert!(e_avx2 < 1e-4, "AVX2 path drifted from exact: {e_avx2:.3e}");
            // And it must never be the worse of the two; if that flips, it has a bug.
            assert!(
                e_avx2 <= e_wired,
                "AVX2 ({e_avx2:.3e}) worse than int8 ({e_wired:.3e})"
            );
        }
    }
}

#[cfg(test)]
mod par_n_tests {
    use super::*;
    use ndarray::Array2;

    /// Column-parallel must agree with the row-parallel path at every shape,
    /// including n that is not a multiple of the cache-line band.
    #[test]
    fn agrees_with_row_parallel() {
        for (m, k, n) in [
            (2, 384, 1152),
            (5, 384, 384),
            (14, 384, 1536),
            (20, 768, 3072),
            (31, 1024, 4096),
            (64, 384, 384),
            (7, 33, 11),
            (3, 17, 5),
        ] {
            let a = Array2::from_shape_fn((m, k), |(i, j)| {
                ((i * 31 + j * 17) % 97) as f32 / 97.0 - 0.5
            });
            let w =
                Array2::from_shape_fn((n, k), |(i, j)| ((i * 13 + j * 7) % 89) as f32 / 89.0 - 0.5);
            let bias: Vec<f32> = (0..n).map(|i| (i % 11) as f32 / 11.0).collect();

            for b in [None, Some(bias.as_slice())] {
                let mut want = Array2::zeros((m, n));
                let mut got = Array2::zeros((m, n));
                matmul_2d_f32_noalloc(&a.view(), &w.view(), b, &mut want);
                matmul_2d_f32_noalloc_par_n(&a.view(), &w.view(), b, &mut got);
                let d = want
                    .iter()
                    .zip(got.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                assert!(d < 1e-5, "m={m} k={k} n={n} bias={} diff {d}", b.is_some());
            }
        }
    }
}
