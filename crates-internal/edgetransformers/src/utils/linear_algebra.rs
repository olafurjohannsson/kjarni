//! Linear algebra operations for transformers
//! Universal implementation.
//! Uses "Kernel Hoisting" (Merging math into the chunk loop) to avoid inlining barriers.

use faer::Parallelism;
use ndarray::{Array2, Array3, Array4, ArrayView2, ArrayView4, Axis, Zip};
use rayon::prelude::*;
const MASK_VALUE: f32 = -1e9;

// =========================================================================
//  SECTION 1: HARDWARE SPECIFIC KERNELS
// =========================================================================

/// 1. AVX2 CHUNK KERNEL (x86 Only)
/// Handles a batch of outputs using AVX2.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn compute_chunk_avx2(
    out_chunk: &mut [f32],
    a_ptr_base: *const f32,
    mut b_ptr: *const u16,
    k: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // Outer Loop: Iterate over the rows in this chunk
    for val in out_chunk.iter_mut() {
        let mut a_ptr = a_ptr_base;

        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        let mut n = k;

        // Inner Loop: The Dot Product
        while n >= 32 {
            _mm_prefetch(b_ptr.add(128) as *const i8, _MM_HINT_T0);
            _mm_prefetch(a_ptr.add(128) as *const i8, _MM_HINT_T0);

            let a0 = _mm256_loadu_ps(a_ptr);
            let a1 = _mm256_loadu_ps(a_ptr.add(8));
            let a2 = _mm256_loadu_ps(a_ptr.add(16));
            let a3 = _mm256_loadu_ps(a_ptr.add(24));

            let b0_u16 = _mm_loadu_si128(b_ptr as *const __m128i);
            let b1_u16 = _mm_loadu_si128(b_ptr.add(8) as *const __m128i);
            let b2_u16 = _mm_loadu_si128(b_ptr.add(16) as *const __m128i);
            let b3_u16 = _mm_loadu_si128(b_ptr.add(24) as *const __m128i);

            // BF16 expansion
            let b0_u32 = _mm256_cvtepu16_epi32(b0_u16);
            let b1_u32 = _mm256_cvtepu16_epi32(b1_u16);
            let b2_u32 = _mm256_cvtepu16_epi32(b2_u16);
            let b3_u32 = _mm256_cvtepu16_epi32(b3_u16);

            let b0_f = _mm256_castsi256_ps(_mm256_slli_epi32(b0_u32, 16));
            let b1_f = _mm256_castsi256_ps(_mm256_slli_epi32(b1_u32, 16));
            let b2_f = _mm256_castsi256_ps(_mm256_slli_epi32(b2_u32, 16));
            let b3_f = _mm256_castsi256_ps(_mm256_slli_epi32(b3_u32, 16));

            sum0 = _mm256_fmadd_ps(a0, b0_f, sum0);
            sum1 = _mm256_fmadd_ps(a1, b1_f, sum1);
            sum2 = _mm256_fmadd_ps(a2, b2_f, sum2);
            sum3 = _mm256_fmadd_ps(a3, b3_f, sum3);

            a_ptr = a_ptr.add(32);
            b_ptr = b_ptr.add(32);
            n -= 32;
        }

        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), sum0);
        let mut sum = temp.iter().sum::<f32>();

        // Remainder
        while n > 0 {
            let val_a = *a_ptr;
            let val_b = f32::from_bits((*b_ptr as u32) << 16);
            sum += val_a * val_b;
            a_ptr = a_ptr.add(1);
            b_ptr = b_ptr.add(1);
            n -= 1;
        }

        *val = sum;
        // Do NOT advance a_ptr_base (we reuse input vector)
        // Advance b_ptr to the next row of weights
        b_ptr = b_ptr.add(k); // Note: b_ptr was incremented inside inner loop? No.
        // Wait: The inner loop increments local b_ptr copy.
        // The outer b_ptr needs to jump K elements *after* the inner loop finishes?
        // No, in the logic above `b_ptr.add(32)` modifies the loop variable.
        // We need to ensure we point to the START of the next row.
        // Since we modified b_ptr inside the `while n >= 32`, it is currently at end of row.
        // So actually, we don't need to add K again if we consumed the whole row.
        // BUT: if n > 0 (remainder), we consumed it.
        // Logic check: `b_ptr` is local to the function args.
        // We need a separate cursor for the inner loop.
    }
}

// Redefine to fix cursor logic clearly
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn compute_chunk_avx2_clean(
    out_chunk: &mut [f32],
    a_ptr_base: *const f32,
    mut b_row_start: *const u16,
    k: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    for val in out_chunk.iter_mut() {
        let mut a_ptr = a_ptr_base;
        let mut b_ptr = b_row_start;

        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        let mut n = k;

        while n >= 32 {
            _mm_prefetch(b_ptr.add(128) as *const i8, _MM_HINT_T0);
            _mm_prefetch(a_ptr.add(128) as *const i8, _MM_HINT_T0);

            let a0 = _mm256_loadu_ps(a_ptr);
            let a1 = _mm256_loadu_ps(a_ptr.add(8));
            let a2 = _mm256_loadu_ps(a_ptr.add(16));
            let a3 = _mm256_loadu_ps(a_ptr.add(24));

            let b0_u16 = _mm_loadu_si128(b_ptr as *const __m128i);
            let b1_u16 = _mm_loadu_si128(b_ptr.add(8) as *const __m128i);
            let b2_u16 = _mm_loadu_si128(b_ptr.add(16) as *const __m128i);
            let b3_u16 = _mm_loadu_si128(b_ptr.add(24) as *const __m128i);

            let b0_u32 = _mm256_cvtepu16_epi32(b0_u16);
            let b1_u32 = _mm256_cvtepu16_epi32(b1_u16);
            let b2_u32 = _mm256_cvtepu16_epi32(b2_u16);
            let b3_u32 = _mm256_cvtepu16_epi32(b3_u16);

            let b0_f = _mm256_castsi256_ps(_mm256_slli_epi32(b0_u32, 16));
            let b1_f = _mm256_castsi256_ps(_mm256_slli_epi32(b1_u32, 16));
            let b2_f = _mm256_castsi256_ps(_mm256_slli_epi32(b2_u32, 16));
            let b3_f = _mm256_castsi256_ps(_mm256_slli_epi32(b3_u32, 16));

            sum0 = _mm256_fmadd_ps(a0, b0_f, sum0);
            sum1 = _mm256_fmadd_ps(a1, b1_f, sum1);
            sum2 = _mm256_fmadd_ps(a2, b2_f, sum2);
            sum3 = _mm256_fmadd_ps(a3, b3_f, sum3);

            a_ptr = a_ptr.add(32);
            b_ptr = b_ptr.add(32);
            n -= 32;
        }

        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), sum0);
        let mut sum = temp.iter().sum::<f32>();

        while n > 0 {
            let val_a = *a_ptr;
            let val_b = f32::from_bits((*b_ptr as u32) << 16);
            sum += val_a * val_b;
            a_ptr = a_ptr.add(1);
            b_ptr = b_ptr.add(1);
            n -= 1;
        }

        *val = sum;
        // Move to start of next row
        b_row_start = b_row_start.add(k);
    }
}

/// 2. ARM NEON CHUNK KERNEL
#[cfg(target_arch = "aarch64")]
unsafe fn compute_chunk_neon(
    out_chunk: &mut [f32],
    a_ptr_base: *const f32,
    mut b_row_start: *const u16,
    k: usize,
) {
    use std::arch::aarch64::*;

    for val in out_chunk.iter_mut() {
        let mut a_ptr = a_ptr_base;
        let mut b_ptr = b_row_start;
        let mut sum_v = vdupq_n_f32(0.0);
        let mut n = k;

        while n >= 16 {
            _mm_prefetch(a_ptr.add(64) as *const i8, _MM_HINT_T0);

            let a0 = vld1q_f32(a_ptr);
            let a1 = vld1q_f32(a_ptr.add(4));
            let a2 = vld1q_f32(a_ptr.add(8));
            let a3 = vld1q_f32(a_ptr.add(12));

            let b_raw_0 = vld1q_u16(b_ptr);
            let b_raw_1 = vld1q_u16(b_ptr.add(8));

            let b0_u32 = vmovl_u16(vget_low_u16(b_raw_0));
            let b1_u32 = vmovl_high_u16(b_raw_0);
            let b2_u32 = vmovl_u16(vget_low_u16(b_raw_1));
            let b3_u32 = vmovl_high_u16(b_raw_1);

            let shift = vdupq_n_s32(16);
            let b0_f = vreinterpretq_f32_u32(vshlq_u32(b0_u32, shift));
            let b1_f = vreinterpretq_f32_u32(vshlq_u32(b1_u32, shift));
            let b2_f = vreinterpretq_f32_u32(vshlq_u32(b2_u32, shift));
            let b3_f = vreinterpretq_f32_u32(vshlq_u32(b3_u32, shift));

            sum_v = vfmaq_f32(sum_v, a0, b0_f);
            sum_v = vfmaq_f32(sum_v, a1, b1_f);
            sum_v = vfmaq_f32(sum_v, a2, b2_f);
            sum_v = vfmaq_f32(sum_v, a3, b3_f);

            a_ptr = a_ptr.add(16);
            b_ptr = b_ptr.add(16);
            n -= 16;
        }

        let sum = vaddvq_f32(sum_v);
        let mut s_scalar = 0.0;
        while n > 0 {
            let val_a = *a_ptr;
            let val_b = f32::from_bits((*b_ptr as u32) << 16);
            s_scalar += val_a * val_b;
            a_ptr = a_ptr.add(1);
            b_ptr = b_ptr.add(1);
            n -= 1;
        }
        *val = sum + s_scalar;
        b_row_start = b_row_start.add(k);
    }
}

/// 3. FALLBACK CHUNK KERNEL
unsafe fn compute_chunk_fallback(
    out_chunk: &mut [f32],
    a_ptr_base: *const f32,
    mut b_row_start: *const u16,
    k: usize,
) {
    for val in out_chunk.iter_mut() {
        let mut sum = 0.0;
        let mut a_ptr = a_ptr_base;
        let mut b_ptr = b_row_start;
        for _ in 0..k {
            let val_a = *a_ptr;
            let val_b = f32::from_bits((*b_ptr as u32) << 16);
            sum += val_a * val_b;
            a_ptr = a_ptr.add(1);
            b_ptr = b_ptr.add(1);
        }
        *val = sum;
        b_row_start = b_row_start.add(k);
    }
}

// =========================================================================
//  SECTION 2: DISPATCHER
// =========================================================================

pub fn matmul_2d_mixed_bf16(a: &ArrayView2<f32>, b_weights: &ArrayView2<u16>) -> Array2<f32> {
    let (m, k) = a.dim();
    let (n, k2) = b_weights.dim();
    assert_eq!(k, k2, "Dim mismatch");

    let mut c = Array2::<f32>::zeros((m, n));

    let b_s = b_weights.as_standard_layout();
    let b_addr = b_s.as_slice().expect("Weights not contiguous").as_ptr() as usize;

    let a_s = a.as_standard_layout();
    let a_addr = a_s.as_slice().expect("Input not contiguous").as_ptr() as usize;

    let c_ptr_head = c.as_mut_ptr();

    // Check Hardware ONCE
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");

    if m == 1 {
        let output_slice = unsafe { std::slice::from_raw_parts_mut(c_ptr_head, n) };
        let num_threads = rayon::current_num_threads();

        // FIX: Threshold on total work, not just n
        let total_work = n * k;
        let min_work_per_thread = 32 * 1024; // ~32K ops minimum per thread
        let min_parallel_work = num_threads * min_work_per_thread;

        if total_work >= min_parallel_work && n >= num_threads {
            // Parallel path
            let chunk_size = (n + num_threads - 1) / num_threads;
            output_slice
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let a_ptr = a_addr as *const f32;
                    let b_base_ptr = b_addr as *const u16;
                    let start_n = chunk_idx * chunk_size;
                    let b_curr = unsafe { b_base_ptr.add(start_n * k) };

                    unsafe {
                        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                        if use_avx2 {
                            compute_chunk_avx2_clean(out_chunk, a_ptr, b_curr, k);
                            return;
                        }

                        #[cfg(target_arch = "aarch64")]
                        {
                            compute_chunk_neon(out_chunk, a_ptr, b_curr, k);
                            return;
                        }

                        compute_chunk_fallback(out_chunk, a_ptr, b_curr, k);
                    }
                });
        } else {
            // Serial
            let a_ptr = a_addr as *const f32;
            let b_curr = b_addr as *const u16;
            unsafe {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if use_avx2 {
                    compute_chunk_avx2_clean(output_slice, a_ptr, b_curr, k);
                    return c;
                }

                #[cfg(target_arch = "aarch64")]
                {
                    compute_chunk_neon(output_slice, a_ptr, b_curr, k);
                    return c;
                }

                compute_chunk_fallback(output_slice, a_ptr, b_curr, k);
            }
        }
    } else {
        // --- BATCH > 1 ---
        let c_addr = c_ptr_head as usize;
        (0..m).into_par_iter().for_each(|i| unsafe {
            let a_ptr = a_addr as *const f32;
            let b_base_ptr = b_addr as *const u16;
            let c_ptr = c_addr as *mut f32;

            let a_row = a_ptr.add(i * k);
            let c_row = std::slice::from_raw_parts_mut(c_ptr.add(i * n), n);
            let b_curr = b_base_ptr;

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if use_avx2 {
                compute_chunk_avx2_clean(c_row, a_row, b_curr, k);
                return;
            }

            #[cfg(target_arch = "aarch64")]
            {
                compute_chunk_neon(c_row, a_row, b_curr, k);
                return;
            }

            compute_chunk_fallback(c_row, a_row, b_curr, k);
        });
    }
    c
}

/// F32 matmul - weights stored [out, in], same as BF16
/// Computes: input @ weights.T  (but weights are already transposed in storage)
pub fn matmul_2d_f32_notranspose(a: &ArrayView2<f32>, b_weights: &ArrayView2<f32>) -> Array2<f32> {
    let (m, k) = a.dim();
    let (n, k2) = b_weights.dim(); // [out, in] = [n, k]
    assert_eq!(k, k2, "Dim mismatch");

    let mut c = Array2::<f32>::zeros((m, n));

    let b_s = b_weights.as_standard_layout();
    let b_addr = b_s.as_slice().expect("Weights not contiguous").as_ptr() as usize;

    let a_s = a.as_standard_layout();
    let a_addr = a_s.as_slice().expect("Input not contiguous").as_ptr() as usize;

    let c_ptr_head = c.as_mut_ptr();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");

    if m == 1 {
        let output_slice = unsafe { std::slice::from_raw_parts_mut(c_ptr_head, n) };
        let num_threads = rayon::current_num_threads();

        let total_work = n * k;
        let min_work_per_thread = 32 * 1024;
        let min_parallel_work = num_threads * min_work_per_thread;

        if total_work >= min_parallel_work && n >= num_threads {
            let chunk_size = (n + num_threads - 1) / num_threads;
            output_slice
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let a_ptr = a_addr as *const f32;
                    let b_base_ptr = b_addr as *const f32;
                    let start_n = chunk_idx * chunk_size;
                    let b_curr = unsafe { b_base_ptr.add(start_n * k) };

                    unsafe {
                        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                        if use_avx2 {
                            compute_chunk_avx2_f32(out_chunk, a_ptr, b_curr, k);
                            return;
                        }

                        #[cfg(target_arch = "aarch64")]
                        {
                            compute_chunk_neon_f32(out_chunk, a_ptr, b_curr, k);
                            return;
                        }

                        compute_chunk_fallback_f32(out_chunk, a_ptr, b_curr, k);
                    }
                });
        } else {
            let a_ptr = a_addr as *const f32;
            let b_curr = b_addr as *const f32;
            unsafe {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if use_avx2 {
                    compute_chunk_avx2_f32(output_slice, a_ptr, b_curr, k);
                    return c;
                }

                #[cfg(target_arch = "aarch64")]
                {
                    compute_chunk_neon_f32(output_slice, a_ptr, b_curr, k);
                    return c;
                }

                compute_chunk_fallback_f32(output_slice, a_ptr, b_curr, k);
            }
        }
    } else {
        let c_addr = c_ptr_head as usize;
        (0..m).into_par_iter().for_each(|i| unsafe {
            let a_ptr = (a_addr as *const f32).add(i * k);
            let b_base_ptr = b_addr as *const f32;
            let c_ptr = c_addr as *mut f32;
            let c_row = std::slice::from_raw_parts_mut(c_ptr.add(i * n), n);

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if use_avx2 {
                compute_chunk_avx2_f32(c_row, a_ptr, b_base_ptr, k);
                return;
            }

            #[cfg(target_arch = "aarch64")]
            {
                compute_chunk_neon_f32(c_row, a_ptr, b_base_ptr, k);
                return;
            }

            compute_chunk_fallback_f32(c_row, a_ptr, b_base_ptr, k);
        });
    }
    c
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn compute_chunk_avx2_f32(
    out_chunk: &mut [f32],
    a_ptr_base: *const f32,
    mut b_row_start: *const f32,
    k: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    for val in out_chunk.iter_mut() {
        let mut a_ptr = a_ptr_base;
        let mut b_ptr = b_row_start;

        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        let mut n = k;

        while n >= 32 {
            let a0 = _mm256_loadu_ps(a_ptr);
            let a1 = _mm256_loadu_ps(a_ptr.add(8));
            let a2 = _mm256_loadu_ps(a_ptr.add(16));
            let a3 = _mm256_loadu_ps(a_ptr.add(24));

            let b0 = _mm256_loadu_ps(b_ptr);
            let b1 = _mm256_loadu_ps(b_ptr.add(8));
            let b2 = _mm256_loadu_ps(b_ptr.add(16));
            let b3 = _mm256_loadu_ps(b_ptr.add(24));

            sum0 = _mm256_fmadd_ps(a0, b0, sum0);
            sum1 = _mm256_fmadd_ps(a1, b1, sum1);
            sum2 = _mm256_fmadd_ps(a2, b2, sum2);
            sum3 = _mm256_fmadd_ps(a3, b3, sum3);

            a_ptr = a_ptr.add(32);
            b_ptr = b_ptr.add(32);
            n -= 32;
        }

        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), sum0);
        let mut sum: f32 = temp.iter().sum();

        while n > 0 {
            sum += *a_ptr * *b_ptr;
            a_ptr = a_ptr.add(1);
            b_ptr = b_ptr.add(1);
            n -= 1;
        }

        *val = sum;
        b_row_start = b_row_start.add(k);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn compute_chunk_neon_f32(
    out_chunk: &mut [f32],
    a_ptr_base: *const f32,
    mut b_row_start: *const f32,
    k: usize,
) {
    use std::arch::aarch64::*;

    for val in out_chunk.iter_mut() {
        let mut a_ptr = a_ptr_base;
        let mut b_ptr = b_row_start;
        let mut sum_v = vdupq_n_f32(0.0);
        let mut n = k;

        while n >= 16 {
            let a0 = vld1q_f32(a_ptr);
            let a1 = vld1q_f32(a_ptr.add(4));
            let a2 = vld1q_f32(a_ptr.add(8));
            let a3 = vld1q_f32(a_ptr.add(12));

            let b0 = vld1q_f32(b_ptr);
            let b1 = vld1q_f32(b_ptr.add(4));
            let b2 = vld1q_f32(b_ptr.add(8));
            let b3 = vld1q_f32(b_ptr.add(12));

            sum_v = vfmaq_f32(sum_v, a0, b0);
            sum_v = vfmaq_f32(sum_v, a1, b1);
            sum_v = vfmaq_f32(sum_v, a2, b2);
            sum_v = vfmaq_f32(sum_v, a3, b3);

            a_ptr = a_ptr.add(16);
            b_ptr = b_ptr.add(16);
            n -= 16;
        }

        let mut sum = vaddvq_f32(sum_v);

        while n > 0 {
            sum += *a_ptr * *b_ptr;
            a_ptr = a_ptr.add(1);
            b_ptr = b_ptr.add(1);
            n -= 1;
        }

        *val = sum;
        b_row_start = b_row_start.add(k);
    }
}

unsafe fn compute_chunk_fallback_f32(
    out_chunk: &mut [f32],
    a_ptr_base: *const f32,
    mut b_row_start: *const f32,
    k: usize,
) {
    for val in out_chunk.iter_mut() {
        let mut sum = 0.0;
        let mut a_ptr = a_ptr_base;
        let mut b_ptr = b_row_start;
        for _ in 0..k {
            sum += *a_ptr * *b_ptr;
            a_ptr = a_ptr.add(1);
            b_ptr = b_ptr.add(1);
        }
        *val = sum;
        b_row_start = b_row_start.add(k);
    }
}

// =========================================================================
//  SECTION 3: STANDARD HELPERS (Unchanged)
// =========================================================================

#[inline]
pub fn matmul_2d(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> Array2<f32> {
    let (m, k) = a.dim();
    let (k2, n) = b.dim();
    assert_eq!(k, k2, "Dim mismatch");

    let mut c = Array2::<f32>::zeros((m, n));
    let a_s = a.as_standard_layout();
    let a_sl = a_s.as_slice().unwrap();
    let b_s = b.as_standard_layout();
    let b_sl = b_s.as_slice().unwrap();
    let c_sl = c.as_slice_mut().unwrap();

    faer::linalg::matmul::matmul(
        faer::mat::from_row_major_slice_mut(c_sl, m, n),
        faer::mat::from_row_major_slice(a_sl, m, k),
        faer::mat::from_row_major_slice(b_sl, k, n),
        None,
        1.0,
        Parallelism::Rayon(0),
    );
    c
}

#[inline]
pub fn matmul_2d_transposed(a: &ArrayView2<f32>, b_transposed: &ArrayView2<f32>) -> Array2<f32> {
    let (m, k) = a.dim();
    let (n, k2) = b_transposed.dim();
    assert_eq!(k, k2, "Dim mismatch");

    let mut c = Array2::<f32>::zeros((m, n));
    let a_s = a.as_standard_layout();
    let a_sl = a_s.as_slice().unwrap();
    let b_s = b_transposed.as_standard_layout();
    let b_sl = b_s.as_slice().unwrap();
    let c_sl = c.as_slice_mut().unwrap();

    faer::linalg::matmul::matmul(
        faer::mat::from_row_major_slice_mut(c_sl, m, n),
        faer::mat::from_row_major_slice(a_sl, m, k),
        faer::mat::from_row_major_slice(b_sl, n, k).transpose(),
        None,
        1.0,
        Parallelism::Rayon(0),
    );
    c
}

#[inline]
pub fn matmul_3d_2d(a: &Array3<f32>, b: &Array2<f32>) -> Array3<f32> {
    let (batch, m, k) = a.dim();
    let (k2, n) = b.dim();
    assert_eq!(k, k2);
    let a_flat = a.view().into_shape_with_order((batch * m, k)).unwrap();
    let b_view = b.view();
    let c_flat = matmul_2d(&a_flat, &b_view);
    c_flat.into_shape_with_order((batch, m, n)).unwrap()
}

/// Performs matmul for a 3D input and a 2D weight matrix in [Out, In] layout.
///
/// This is the CPU equivalent of the optimized GPU kernels.
/// `a` has shape [Batch, Seq, In], `b_transposed` has shape [Out, In].
#[inline]
pub fn matmul_3d_2d_transposed(a: &Array3<f32>, b_transposed: &Array2<f32>) -> Array3<f32> {
    let (batch, m, k) = a.dim();
    let (n, k2) = b_transposed.dim(); // Shape is [Out, In]
    assert_eq!(k, k2, "Matmul inner dimensions do not match");

    let a_flat = a.view().into_shape_with_order((batch * m, k)).unwrap();
    
    // We use the existing matmul_2d_transposed which correctly handles the layout
    let c_flat = matmul_2d_transposed(&a_flat.view(), &b_transposed.view());
    
    c_flat.into_shape_with_order((batch, m, n)).unwrap()
}

#[inline]
pub fn matmul_4d(a: &Array4<f32>, b: &Array4<f32>) -> Array4<f32> {
    let (batch, heads, seq1, dim) = a.dim();
    let seq2 = b.shape()[3];
    let mut output = Array4::<f32>::zeros((batch, heads, seq1, seq2));

    Zip::from(output.outer_iter_mut())
        .and(a.outer_iter())
        .and(b.outer_iter())
        .par_for_each(|mut out_b, a_b, b_b| {
            Zip::from(out_b.outer_iter_mut())
                .and(a_b.outer_iter())
                .and(b_b.outer_iter())
                .for_each(|mut out_h, a_h, b_h| {
                    let a_s = a_h.as_standard_layout();
                    let b_s = b_h.as_standard_layout();
                    let o_sl = out_h.as_slice_mut().unwrap();

                    faer::linalg::matmul::matmul(
                        faer::mat::from_row_major_slice_mut(o_sl, seq1, seq2),
                        faer::mat::from_row_major_slice(a_s.as_slice().unwrap(), seq1, dim),
                        faer::mat::from_row_major_slice(b_s.as_slice().unwrap(), dim, seq2),
                        None,
                        1.0,
                        Parallelism::Rayon(0),
                    );
                });
        });
    output
}

pub fn matmul_4d_decode_gqa(
    q: &Array4<f32>,
    k_transposed: &ArrayView4<f32>,
    n_rep: usize,
) -> Array4<f32> {
    let (batch, heads, _, dim) = q.dim();
    let cache_len = k_transposed.shape()[3];
    let mut out = Array4::<f32>::zeros((batch, heads, 1, cache_len));

    out.outer_iter_mut()
        .zip(q.outer_iter())
        .enumerate()
        .par_bridge()
        .for_each(|(b_idx, (mut out_b, q_b))| {
            let k_batch = k_transposed.index_axis(Axis(0), b_idx);
            out_b
                .outer_iter_mut()
                .zip(q_b.outer_iter())
                .enumerate()
                .for_each(|(h_idx, (mut out_h, q_h))| {
                    let kv_head_idx = h_idx / n_rep;
                    let k_head = k_batch.index_axis(Axis(0), kv_head_idx);
                    let q_ptr = q_h.as_ptr();
                    let out_s = out_h.as_slice_mut().unwrap();

                    for t in 0..cache_len {
                        let mut sum = 0.0;
                        for d in 0..dim {
                            unsafe {
                                sum += *q_ptr.add(d) * *k_head.uget((d, t));
                            }
                        }
                        out_s[t] = sum;
                    }
                });
        });
    out
}

pub fn matmul_4d_context_gqa(
    scores: &Array4<f32>,
    v: &ArrayView4<f32>,
    n_rep: usize,
) -> Array4<f32> {
    let (batch, heads, _, cache_len) = scores.dim();
    let dim = v.shape()[3];
    let mut out = Array4::<f32>::zeros((batch, heads, 1, dim));

    out.outer_iter_mut()
        .zip(scores.outer_iter())
        .enumerate()
        .par_bridge()
        .for_each(|(b_idx, (mut out_b, s_b))| {
            let v_batch = v.index_axis(Axis(0), b_idx);
            out_b
                .outer_iter_mut()
                .zip(s_b.outer_iter())
                .enumerate()
                .for_each(|(h_idx, (mut out_h, s_h))| {
                    let kv_head_idx = h_idx / n_rep;
                    let v_head = v_batch.index_axis(Axis(0), kv_head_idx);
                    let s_ptr = s_h.as_ptr();
                    let out_s = out_h.as_slice_mut().unwrap();

                    for d in 0..dim {
                        let mut sum = 0.0;
                        for t in 0..cache_len {
                            unsafe {
                                sum += *s_ptr.add(t) * *v_head.uget((t, d));
                            }
                        }
                        out_s[d] = sum;
                    }
                });
        });
    out
}

/// Optimized Decode Score (Non-GQA): Q=[B, H, 1, D], K=[B, H, D, S]
pub fn matmul_4d_decode(q: &Array4<f32>, k_transposed: &Array4<f32>) -> Array4<f32> {
    let (batch, heads, _, dim) = q.dim();
    let cache_len = k_transposed.shape()[3];
    let mut out = Array4::<f32>::zeros((batch, heads, 1, cache_len));

    out.outer_iter_mut()
        .zip(q.outer_iter())
        .zip(k_transposed.outer_iter())
        .par_bridge()
        .for_each(|((mut out_b, q_b), k_b)| {
            out_b
                .outer_iter_mut()
                .zip(q_b.outer_iter())
                .zip(k_b.outer_iter())
                .for_each(|((mut out_h, q_h), k_h)| {
                    let q_ptr = q_h.as_ptr();
                    let out_s = out_h.as_slice_mut().unwrap();
                    // Simple dot product loop (k_h is transposed so rows are D, cols are S)
                    // If k_h is from standard layout [D, S], then accessing (d,t) strides 1.
                    for t in 0..cache_len {
                        let mut sum = 0.0;
                        for d in 0..dim {
                            unsafe {
                                sum += *q_ptr.add(d) * k_h[[d, t]];
                            }
                        }
                        out_s[t] = sum;
                    }
                });
        });
    out
}

/// Optimized Decode Context (Non-GQA): Scores=[B, H, 1, S], V=[B, H, S, D]
pub fn matmul_4d_context(scores: &Array4<f32>, v: &Array4<f32>) -> Array4<f32> {
    let (batch, heads, _, cache_len) = scores.dim();
    let dim = v.shape()[3];
    let mut out = Array4::<f32>::zeros((batch, heads, 1, dim));

    out.outer_iter_mut()
        .zip(scores.outer_iter())
        .zip(v.outer_iter())
        .par_bridge()
        .for_each(|((mut out_b, s_b), v_b)| {
            out_b
                .outer_iter_mut()
                .zip(s_b.outer_iter())
                .zip(v_b.outer_iter())
                .for_each(|((mut out_h, s_h), v_h)| {
                    let s_ptr = s_h.as_ptr();
                    let out_s = out_h.as_slice_mut().unwrap();
                    for d in 0..dim {
                        let mut sum = 0.0;
                        for t in 0..cache_len {
                            unsafe {
                                sum += *s_ptr.add(t) * v_h[[t, d]];
                            }
                        }
                        out_s[d] = sum;
                    }
                });
        });
    out
}

// =========================================================================
//  SECTION 4: FIXED MASKING UTILITY
// =========================================================================

/// Safe implementation of masking using NDArray Zip.
/// This handles Broadcasting (Batch=1 mask -> Batch=4 scores)
/// and Memory Strides correctly.
pub fn apply_attention_mask(mut scores: Array4<f32>, mask: &Array2<f32>) -> Array4<f32> {
    let (batch, heads, seq_q, seq_k) = scores.dim();

    // Safety check: Encoder mask length must match Key length
    if mask.shape()[1] != seq_k {
        return scores;
    }

    // Expand mask: [MaskBatch, SeqK] → [MaskBatch, 1, 1, SeqK]
    let mask_expanded = mask.view().insert_axis(Axis(1)).insert_axis(Axis(1));

    // Broadcast and apply
    if let Some(broadcast_mask) = mask_expanded.broadcast((batch, heads, seq_q, seq_k)) {
        Zip::from(&mut scores)
            .and(&broadcast_mask)
            .par_for_each(|s, &m| {
                // If mask is 0.0, set score to very low value
                if m == 0.0 {
                    *s = MASK_VALUE;
                }
            });
    }

    scores
}
