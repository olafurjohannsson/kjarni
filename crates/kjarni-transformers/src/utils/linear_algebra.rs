//! Linear algebra operations for transformers

use faer::Parallelism;
use ndarray::{Array2, Array3, Array4, ArrayView2, ArrayView4, Axis, Zip};
use rayon::prelude::*;
const MASK_VALUE: f32 = -1e9;


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
        unsafe {
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
}

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
            std::arch::aarch64::__prefetch(b_ptr.add(128) as *const i8, 0, 3);

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

unsafe fn compute_chunk_fallback(
    out_chunk: &mut [f32],
    a_ptr_base: *const f32,
    mut b_row_start: *const u16,
    k: usize,
) {
    for val in out_chunk.iter_mut() {
        unsafe {
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
}



pub fn matmul_dequant_q4_k(_a: &ArrayView2<f32>, _b_weights: &Array3<u8>) -> Array2<f32> {
    unimplemented!("Q4_K matmul not implemented yet");
}

pub fn matmul_2d_mixed_bf16_new(
    a: &ArrayView2<f32>,
    b_weights: &ArrayView2<half::bf16>,
) -> Array2<f32> {
    let (m, k) = a.dim();
    let (n, k2) = b_weights.dim();
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
        let min_work_per_thread = 32 * 1024; // 32K ops minimum per thread
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
    } else { // batch > 1
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
        // batch 
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
            unsafe {
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
        }

        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        let mut temp = [0.0f32; 8];
        unsafe {
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
        unsafe {
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
}


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
#[inline]
pub fn matmul_3d_2d_transposed(a: &Array3<f32>, b_transposed: &Array2<f32>) -> Array3<f32> {
    let (batch, m, k) = a.dim();
    let (n, k2) = b_transposed.dim(); // Shape is [Out, In]
    assert_eq!(k, k2, "Matmul inner dimensions do not match");

    let a_flat = a.view().into_shape_with_order((batch * m, k)).unwrap();
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
                    let o_s = out_h.as_slice_mut().expect("Output buffer must be contiguous");

                    faer::linalg::matmul::matmul(
                        faer::mat::from_row_major_slice_mut(o_s, seq1, seq2),
                        faer::mat::from_row_major_slice(a_s.as_slice().unwrap(), seq1, dim),
                        faer::mat::from_row_major_slice(b_s.as_slice().unwrap(), dim, seq2),
                        None,
                        1.0,
                        Parallelism::None, // No internal threads; we are already parallel
                    );
                });
        });
        
    output
}


#[inline]
pub fn matmul_4d_noalloc(
    a: &Array4<f32>,
    b: &Array4<f32>,
    out: &mut Array4<f32>,
) {
    let (batch, heads, seq1, dim) = a.dim();
    let seq2 = b.shape()[3];

    debug_assert_eq!(out.dim(), (batch, heads, seq1, seq2), "Output shape mismatch");

    Zip::from(out.outer_iter_mut())
        .and(a.outer_iter())
        .and(b.outer_iter())
        .par_for_each(|mut out_b, a_b, b_b| {
            Zip::from(out_b.outer_iter_mut())
                .and(a_b.outer_iter())
                .and(b_b.outer_iter())
                .for_each(|mut out_h, a_h, b_h| {
                    let a_s = a_h.as_standard_layout();
                    let b_s = b_h.as_standard_layout();
                    let o_s = out_h.as_slice_mut().expect("Output buffer must be contiguous");

                    faer::linalg::matmul::matmul(
                        faer::mat::from_row_major_slice_mut(o_s, seq1, seq2),
                        faer::mat::from_row_major_slice(a_s.as_slice().unwrap(), seq1, dim),
                        faer::mat::from_row_major_slice(b_s.as_slice().unwrap(), dim, seq2),
                        None,
                        1.0,
                        Parallelism::None,
                    );
                });
        });
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

///  Scores=[B, H, 1, S], V=[B, H, S, D]
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

pub fn apply_attention_mask(mut scores: Array4<f32>, mask: &Array2<f32>) -> Array4<f32> {
    let (batch, heads, seq_q, seq_k) = scores.dim();

    if mask.shape()[1] != seq_k {
        return scores;
    }

    let mask_expanded = mask.view().insert_axis(Axis(1)).insert_axis(Axis(1));

    // Broadcast and apply
    if let Some(broadcast_mask) = mask_expanded.broadcast((batch, heads, seq_q, seq_k)) {
        Zip::from(&mut scores)
            .and(&broadcast_mask)
            .par_for_each(|s, &m| {
                if m == 0.0 {
                    *s = MASK_VALUE;
                }
            });
    }

    scores
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3, Array4};

    fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "{}: length mismatch", msg);
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            assert!(
                diff <= tol,
                "{}: mismatch at {}: {} vs {} (diff: {})",
                msg, i, x, y, diff
            );
        }
    }

    fn reference_matmul_2d(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k, k2);
        let mut c = Array2::<f32>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[[i, l]] * b[[l, j]];
                }
                c[[i, j]] = sum;
            }
        }
        c
    }

    fn reference_matmul_2d_transposed(a: &Array2<f32>, b_t: &Array2<f32>) -> Array2<f32> {
        let (m, k) = a.dim();
        let (n, k2) = b_t.dim();
        assert_eq!(k, k2);
        let mut c = Array2::<f32>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[[i, l]] * b_t[[j, l]];
                }
                c[[i, j]] = sum;
            }
        }
        c
    }
    #[test]
    fn test_matmul_2d_simple() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = matmul_2d(&a.view(), &b.view());
        let expected = reference_matmul_2d(&a, &b);

        assert_close(
            result.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            1e-5,
            "matmul_2d simple",
        );
    }

    #[test]
    fn test_matmul_2d_single_row() {
        let a = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array2::from_shape_vec((4, 3), (0..12).map(|x| x as f32).collect()).unwrap();

        let result = matmul_2d(&a.view(), &b.view());
        let expected = reference_matmul_2d(&a, &b);

        assert_close(
            result.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            1e-5,
            "matmul_2d single row",
        );
    }

    #[test]
    fn test_matmul_2d_single_column() {
        let a = Array2::from_shape_vec((4, 3), (0..12).map(|x| x as f32).collect()).unwrap();
        let b = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();

        let result = matmul_2d(&a.view(), &b.view());
        let expected = reference_matmul_2d(&a, &b);

        assert_close(
            result.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            1e-5,
            "matmul_2d single column",
        );
    }

    #[test]
    fn test_matmul_2d_square() {
        let a = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32);
        let b = Array2::from_shape_fn((4, 4), |(i, j)| (i + j) as f32);

        let result = matmul_2d(&a.view(), &b.view());
        let expected = reference_matmul_2d(&a, &b);

        assert_close(
            result.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            1e-5,
            "matmul_2d square",
        );
    }

    #[test]
    fn test_matmul_2d_large() {
        let a = Array2::from_shape_fn((64, 128), |(i, j)| ((i + j) % 10) as f32 * 0.1);
        let b = Array2::from_shape_fn((128, 32), |(i, j)| ((i * j) % 7) as f32 * 0.1);

        let result = matmul_2d(&a.view(), &b.view());
        let expected = reference_matmul_2d(&a, &b);

        assert_close(
            result.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            1e-4,
            "matmul_2d large",
        );
    }

    #[test]
    fn test_matmul_2d_transposed_simple() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b_t = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = matmul_2d_transposed(&a.view(), &b_t.view());
        let expected = reference_matmul_2d_transposed(&a, &b_t);

        assert_close(
            result.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            1e-5,
            "matmul_2d_transposed simple",
        );
    }

    #[test]
    fn test_matmul_2d_transposed_single_row() {
        let a = Array2::from_shape_vec((1, 64), (0..64).map(|x| x as f32 * 0.1).collect()).unwrap();
        let b_t = Array2::from_shape_fn((32, 64), |(i, j)| ((i + j) % 5) as f32);

        let result = matmul_2d_transposed(&a.view(), &b_t.view());
        let expected = reference_matmul_2d_transposed(&a, &b_t);

        assert_close(
            result.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            1e-4,
            "matmul_2d_transposed single row",
        );
    }

    #[test]
    fn test_matmul_2d_transposed_large() {
        let a = Array2::from_shape_fn((16, 256), |(i, j)| ((i + j) % 10) as f32 * 0.1);
        let b_t = Array2::from_shape_fn((64, 256), |(i, j)| ((i * j) % 7) as f32 * 0.1);

        let result = matmul_2d_transposed(&a.view(), &b_t.view());
        let expected = reference_matmul_2d_transposed(&a, &b_t);

        assert_close(
            result.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            1e-3,
            "matmul_2d_transposed large",
        );
    }

    #[test]
    fn test_matmul_2d_f32_notranspose_simple() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b_t = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = matmul_2d_f32_notranspose(&a.view(), &b_t.view());
        let expected = reference_matmul_2d_transposed(&a, &b_t);

        assert_close(
            result.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            1e-5,
            "matmul_2d_f32_notranspose simple",
        );
    }

    #[test]
    fn test_matmul_2d_f32_notranspose_single_row() {
        let a = Array2::from_shape_vec((1, 64), (0..64).map(|x| x as f32 * 0.1).collect()).unwrap();
        let b_t = Array2::from_shape_fn((32, 64), |(i, j)| ((i + j) % 5) as f32);

        let result = matmul_2d_f32_notranspose(&a.view(), &b_t.view());
        let expected = reference_matmul_2d_transposed(&a, &b_t);

        assert_close(
            result.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            1e-4,
            "matmul_2d_f32_notranspose single row",
        );
    }

    #[test]
    fn test_matmul_2d_f32_notranspose_batch() {
        let a = Array2::from_shape_fn((8, 128), |(i, j)| (i * 128 + j) as f32 * 0.01);
        let b_t = Array2::from_shape_fn((64, 128), |(i, j)| ((i + j) % 10) as f32 * 0.1);

        let result = matmul_2d_f32_notranspose(&a.view(), &b_t.view());
        let expected = reference_matmul_2d_transposed(&a, &b_t);

        assert_close(
            result.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            1e-3,
            "matmul_2d_f32_notranspose batch",
        );
    }

    #[test]
    fn test_matmul_2d_mixed_bf16_simple() {
        let a = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b_f32 = Array2::from_shape_vec((2, 4), vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let b_u16: Array2<u16> = b_f32.mapv(|x| half::bf16::from_f32(x).to_bits());

        let result = matmul_2d_mixed_bf16(&a.view(), &b_u16.view());

        assert!((result[[0, 0]] - 1.0).abs() < 0.01);
        assert!((result[[0, 1]] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_matmul_2d_mixed_bf16_identity() {
        let a = Array2::from_shape_fn((1, 8), |(_, j)| (j + 1) as f32);
        let mut b_f32 = Array2::<f32>::zeros((8, 8));
        for i in 0..8 {
            b_f32[[i, i]] = 1.0;
        }
        let b_u16: Array2<u16> = b_f32.mapv(|x| half::bf16::from_f32(x).to_bits());

        let result = matmul_2d_mixed_bf16(&a.view(), &b_u16.view());

        for j in 0..8 {
            assert!(
                (result[[0, j]] - (j + 1) as f32).abs() < 0.1,
                "identity mismatch at {}",
                j
            );
        }
    }

    #[test]
    fn test_matmul_2d_mixed_bf16_batch() {
        let a = Array2::from_shape_fn((4, 32), |(i, j)| ((i + j) % 5) as f32 * 0.5);
        let b_f32 = Array2::from_shape_fn((16, 32), |(i, j)| ((i * j) % 7) as f32 * 0.3);
        let b_u16: Array2<u16> = b_f32.mapv(|x| half::bf16::from_f32(x).to_bits());

        let result = matmul_2d_mixed_bf16(&a.view(), &b_u16.view());
        let expected = reference_matmul_2d_transposed(&a, &b_f32);

        assert_close(
            result.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            0.5,
            "matmul_2d_mixed_bf16 batch",
        );
    }

    #[test]
    fn test_matmul_2d_mixed_bf16_new_simple() {
        let a = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b_f32 = Array2::from_shape_vec((2, 4), vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let b_bf16: Array2<half::bf16> = b_f32.mapv(half::bf16::from_f32);

        let result = matmul_2d_mixed_bf16_new(&a.view(), &b_bf16.view());

        assert!((result[[0, 0]] - 1.0).abs() < 0.01);
        assert!((result[[0, 1]] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_matmul_2d_mixed_bf16_new_batch() {
        let a = Array2::from_shape_fn((4, 32), |(i, j)| ((i + j) % 5) as f32 * 0.5);
        let b_f32 = Array2::from_shape_fn((16, 32), |(i, j)| ((i * j) % 7) as f32 * 0.3);
        let b_bf16: Array2<half::bf16> = b_f32.mapv(half::bf16::from_f32);

        let result = matmul_2d_mixed_bf16_new(&a.view(), &b_bf16.view());
        let expected = reference_matmul_2d_transposed(&a, &b_f32);

        assert_close(
            result.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            0.5,
            "matmul_2d_mixed_bf16_new batch",
        );
    }

    #[test]
    fn test_matmul_3d_2d_simple() {
        let a = Array3::from_shape_fn((2, 3, 4), |(b, i, j)| (b * 12 + i * 4 + j) as f32);
        let b = Array2::from_shape_fn((4, 5), |(i, j)| (i + j) as f32);

        let result = matmul_3d_2d(&a, &b);

        assert_eq!(result.dim(), (2, 3, 5));

        for batch in 0..2 {
            let a_slice = a.slice(ndarray::s![batch, .., ..]).to_owned();
            let expected = reference_matmul_2d(&a_slice, &b);
            let result_slice = result.slice(ndarray::s![batch, .., ..]);
            assert_close(
                result_slice.as_slice().unwrap(),
                expected.as_slice().unwrap(),
                1e-4,
                &format!("matmul_3d_2d batch {}", batch),
            );
        }
    }

    #[test]
    fn test_matmul_3d_2d_transformer_shape() {
        let a = Array3::from_shape_fn((2, 16, 64), |(b, s, h)| ((b + s + h) % 10) as f32 * 0.1);
        let b = Array2::from_shape_fn((64, 128), |(i, j)| ((i * j) % 7) as f32 * 0.1);

        let result = matmul_3d_2d(&a, &b);

        assert_eq!(result.dim(), (2, 16, 128));
    }

    #[test]
    fn test_matmul_3d_2d_transposed_simple() {
        let a = Array3::from_shape_fn((2, 3, 4), |(b, i, j)| (b * 12 + i * 4 + j) as f32);
        let b_t = Array2::from_shape_fn((5, 4), |(i, j)| (i + j) as f32);

        let result = matmul_3d_2d_transposed(&a, &b_t);

        assert_eq!(result.dim(), (2, 3, 5));

        for batch in 0..2 {
            let a_slice = a.slice(ndarray::s![batch, .., ..]).to_owned();
            let expected = reference_matmul_2d_transposed(&a_slice, &b_t);
            let result_slice = result.slice(ndarray::s![batch, .., ..]);
            assert_close(
                result_slice.as_slice().unwrap(),
                expected.as_slice().unwrap(),
                1e-4,
                &format!("matmul_3d_2d_transposed batch {}", batch),
            );
        }
    }

    #[test]
    fn test_matmul_3d_2d_transposed_transformer_shape() {
        let a = Array3::from_shape_fn((2, 16, 768), |(b, s, h)| ((b + s + h) % 10) as f32 * 0.1);
        let b_t = Array2::from_shape_fn((3072, 768), |(i, j)| ((i + j) % 7) as f32 * 0.01);

        let result = matmul_3d_2d_transposed(&a, &b_t);

        assert_eq!(result.dim(), (2, 16, 3072));
    }

    #[test]
    fn test_matmul_4d_simple() {
        let a = Array4::from_shape_fn((1, 2, 3, 4), |(b, h, i, j)| (b + h + i + j) as f32);
        let b = Array4::from_shape_fn((1, 2, 4, 5), |(b, h, i, j)| (b * h + i + j) as f32);

        let result = matmul_4d(&a, &b);

        assert_eq!(result.dim(), (1, 2, 3, 5));
    }

    #[test]
    fn test_matmul_4d_attention_shape() {
        let batch = 2;
        let heads = 8;
        let seq = 16;
        let head_dim = 64;

        let q = Array4::from_shape_fn((batch, heads, seq, head_dim), |(b, h, s, d)| {
            ((b + h + s + d) % 10) as f32 * 0.1
        });
        let k_t = Array4::from_shape_fn((batch, heads, head_dim, seq), |(b, h, d, s)| {
            ((b * h + d + s) % 7) as f32 * 0.1
        });

        let scores = matmul_4d(&q, &k_t);

        assert_eq!(scores.dim(), (batch, heads, seq, seq));
    }

    #[test]
    fn test_matmul_4d_noalloc_simple() {
        let a = Array4::from_shape_fn((1, 2, 3, 4), |(b, h, i, j)| (b + h + i + j) as f32);
        let b = Array4::from_shape_fn((1, 2, 4, 5), |(b, h, i, j)| (b * h + i + j) as f32);
        let mut out = Array4::<f32>::zeros((1, 2, 3, 5));

        matmul_4d_noalloc(&a, &b, &mut out);

        let expected = matmul_4d(&a, &b);
        assert_close(
            out.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            1e-5,
            "matmul_4d_noalloc",
        );
    }

    #[test]
    fn test_matmul_4d_noalloc_reuse_buffer() {
        let a1 = Array4::from_shape_fn((2, 4, 8, 16), |(b, h, i, j)| (b + h + i + j) as f32);
        let b1 = Array4::from_shape_fn((2, 4, 16, 8), |(b, h, i, j)| (b * h + i + j) as f32);

        let a2 = Array4::from_shape_fn((2, 4, 8, 16), |(b, h, i, j)| (b * 2 + h + i + j) as f32);
        let b2 = Array4::from_shape_fn((2, 4, 16, 8), |(b, h, i, j)| (b + h * 2 + i + j) as f32);

        let mut out = Array4::<f32>::zeros((2, 4, 8, 8));

        matmul_4d_noalloc(&a1, &b1, &mut out);
        let expected1 = matmul_4d(&a1, &b1);
        assert_close(
            out.as_slice().unwrap(),
            expected1.as_slice().unwrap(),
            1e-5,
            "matmul_4d_noalloc first",
        );

        matmul_4d_noalloc(&a2, &b2, &mut out);
        let expected2 = matmul_4d(&a2, &b2);
        assert_close(
            out.as_slice().unwrap(),
            expected2.as_slice().unwrap(),
            1e-5,
            "matmul_4d_noalloc second",
        );
    }

    #[test]
    fn test_matmul_4d_decode_simple() {
        let q = Array4::from_shape_fn((1, 4, 1, 64), |(_, h, _, d)| (h + d) as f32 * 0.1);
        let k_t = Array4::from_shape_fn((1, 4, 64, 8), |(_, h, d, s)| (h * d + s) as f32 * 0.01);

        let result = matmul_4d_decode(&q, &k_t);

        assert_eq!(result.dim(), (1, 4, 1, 8));
    }

    #[test]
    fn test_matmul_4d_decode_batched() {
        let batch = 4;
        let heads = 8;
        let head_dim = 64;
        let cache_len = 128;

        let q = Array4::from_shape_fn((batch, heads, 1, head_dim), |(b, h, _, d)| {
            ((b + h + d) % 10) as f32 * 0.1
        });
        let k_t = Array4::from_shape_fn((batch, heads, head_dim, cache_len), |(b, h, d, s)| {
            ((b * h + d + s) % 7) as f32 * 0.1
        });

        let result = matmul_4d_decode(&q, &k_t);

        assert_eq!(result.dim(), (batch, heads, 1, cache_len));
    }

    #[test]
    fn test_matmul_4d_context_simple() {
        let scores = Array4::from_shape_fn((1, 4, 1, 8), |(_, h, _, s)| (h + s) as f32 * 0.1);
        let v = Array4::from_shape_fn((1, 4, 8, 64), |(_, h, s, d)| (h * s + d) as f32 * 0.01);

        let result = matmul_4d_context(&scores, &v);

        assert_eq!(result.dim(), (1, 4, 1, 64));
    }

    #[test]
    fn test_matmul_4d_context_batched() {
        let batch = 4;
        let heads = 8;
        let cache_len = 128;
        let head_dim = 64;

        let scores = Array4::from_shape_fn((batch, heads, 1, cache_len), |(b, h, _, s)| {
            ((b + h + s) % 10) as f32 * 0.01
        });
        let v = Array4::from_shape_fn((batch, heads, cache_len, head_dim), |(b, h, s, d)| {
            ((b * h + s + d) % 7) as f32 * 0.1
        });

        let result = matmul_4d_context(&scores, &v);

        assert_eq!(result.dim(), (batch, heads, 1, head_dim));
    }

    #[test]
    fn test_matmul_4d_decode_gqa_simple() {
        let batch = 1;
        let heads = 8;
        let kv_heads = 2;
        let n_rep = heads / kv_heads;
        let head_dim = 64;
        let cache_len = 16;

        let q = Array4::from_shape_fn((batch, heads, 1, head_dim), |(b, h, _, d)| {
            (b + h + d) as f32 * 0.1
        });
        let k_t = Array4::from_shape_fn((batch, kv_heads, head_dim, cache_len), |(b, h, d, s)| {
            (b * h + d + s) as f32 * 0.01
        });

        let result = matmul_4d_decode_gqa(&q, &k_t.view(), n_rep);

        assert_eq!(result.dim(), (batch, heads, 1, cache_len));
    }

    #[test]
    fn test_matmul_4d_decode_gqa_batched() {
        let batch = 2;
        let heads = 32;
        let kv_heads = 8;
        let n_rep = heads / kv_heads;
        let head_dim = 128;
        let cache_len = 64;

        let q = Array4::from_shape_fn((batch, heads, 1, head_dim), |(b, h, _, d)| {
            ((b + h + d) % 10) as f32 * 0.1
        });
        let k_t = Array4::from_shape_fn((batch, kv_heads, head_dim, cache_len), |(b, h, d, s)| {
            ((b * h + d + s) % 7) as f32 * 0.1
        });

        let result = matmul_4d_decode_gqa(&q, &k_t.view(), n_rep);

        assert_eq!(result.dim(), (batch, heads, 1, cache_len));
    }

    #[test]
    fn test_matmul_4d_context_gqa_simple() {
        let batch = 1;
        let heads = 8;
        let kv_heads = 2;
        let n_rep = heads / kv_heads;
        let cache_len = 16;
        let head_dim = 64;

        let scores = Array4::from_shape_fn((batch, heads, 1, cache_len), |(b, h, _, s)| {
            (b + h + s) as f32 * 0.01
        });
        let v = Array4::from_shape_fn((batch, kv_heads, cache_len, head_dim), |(b, h, s, d)| {
            (b * h + s + d) as f32 * 0.1
        });

        let result = matmul_4d_context_gqa(&scores, &v.view(), n_rep);

        assert_eq!(result.dim(), (batch, heads, 1, head_dim));
    }

    #[test]
    fn test_matmul_4d_context_gqa_batched() {
        let batch = 2;
        let heads = 32;
        let kv_heads = 8;
        let n_rep = heads / kv_heads;
        let cache_len = 64;
        let head_dim = 128;

        let scores = Array4::from_shape_fn((batch, heads, 1, cache_len), |(b, h, _, s)| {
            ((b + h + s) % 10) as f32 * 0.01
        });
        let v = Array4::from_shape_fn((batch, kv_heads, cache_len, head_dim), |(b, h, s, d)| {
            ((b * h + s + d) % 7) as f32 * 0.1
        });

        let result = matmul_4d_context_gqa(&scores, &v.view(), n_rep);

        assert_eq!(result.dim(), (batch, heads, 1, head_dim));
    }

    #[test]
    fn test_apply_attention_mask_simple() {
        let scores = Array4::from_shape_fn((1, 2, 4, 4), |_| 1.0f32);
        let mask = Array2::from_shape_vec((1, 4), vec![1.0, 1.0, 0.0, 0.0]).unwrap();

        let result = apply_attention_mask(scores, &mask);

        for h in 0..2 {
            for q in 0..4 {
                assert_eq!(result[[0, h, q, 0]], 1.0);
                assert_eq!(result[[0, h, q, 1]], 1.0);
                assert_eq!(result[[0, h, q, 2]], MASK_VALUE);
                assert_eq!(result[[0, h, q, 3]], MASK_VALUE);
            }
        }
    }

    #[test]
    fn test_apply_attention_mask_broadcast() {
        let scores = Array4::from_shape_fn((4, 8, 16, 32), |_| 0.5f32);
        let mask = Array2::from_shape_fn((1, 32), |(_, j)| if j < 16 { 1.0 } else { 0.0 });

        let result = apply_attention_mask(scores, &mask);

        for b in 0..4 {
            for h in 0..8 {
                for q in 0..16 {
                    for k in 0..32 {
                        if k < 16 {
                            assert_eq!(result[[b, h, q, k]], 0.5);
                        } else {
                            assert_eq!(result[[b, h, q, k]], MASK_VALUE);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_apply_attention_mask_all_ones() {
        let scores = Array4::from_shape_fn((2, 4, 8, 8), |(b, h, q, k)| (b + h + q + k) as f32);
        let mask = Array2::ones((2, 8));

        let result = apply_attention_mask(scores.clone(), &mask);

        assert_close(
            result.as_slice().unwrap(),
            scores.as_slice().unwrap(),
            1e-6,
            "apply_attention_mask all ones",
        );
    }

    #[test]
    fn test_apply_attention_mask_all_zeros() {
        let scores = Array4::from_shape_fn((2, 4, 8, 8), |_| 1.0f32);
        let mask = Array2::zeros((2, 8));

        let result = apply_attention_mask(scores, &mask);

        for val in result.iter() {
            assert_eq!(*val, MASK_VALUE);
        }
    }

    #[test]
    fn test_apply_attention_mask_length_mismatch() {
        let scores = Array4::from_shape_fn((1, 2, 4, 8), |_| 1.0f32);
        let mask = Array2::zeros((1, 16));

        let result = apply_attention_mask(scores.clone(), &mask);

        assert_close(
            result.as_slice().unwrap(),
            scores.as_slice().unwrap(),
            1e-6,
            "apply_attention_mask length mismatch returns unchanged",
        );
    }

    #[test]
    fn test_apply_attention_mask_per_batch() {
        let scores = Array4::from_shape_fn((2, 1, 2, 4), |_| 1.0f32);
        let mask = Array2::from_shape_vec((2, 4), vec![
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 0.0, 0.0,
        ]).unwrap();

        let result = apply_attention_mask(scores, &mask);

        assert_eq!(result[[0, 0, 0, 3]], 1.0);
        assert_eq!(result[[1, 0, 0, 2]], MASK_VALUE);
        assert_eq!(result[[1, 0, 0, 3]], MASK_VALUE);
        assert_eq!(result[[1, 0, 0, 0]], 1.0);
    }

    #[test]
    fn test_matmul_2d_zeros() {
        let a = Array2::<f32>::zeros((4, 8));
        let b = Array2::from_shape_fn((8, 4), |(i, j)| (i + j) as f32);

        let result = matmul_2d(&a.view(), &b.view());

        for val in result.iter() {
            assert_eq!(*val, 0.0);
        }
    }

    #[test]
    fn test_matmul_2d_identity() {
        let a = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32);
        let mut identity = Array2::<f32>::zeros((4, 4));
        for i in 0..4 {
            identity[[i, i]] = 1.0;
        }

        let result = matmul_2d(&a.view(), &identity.view());

        assert_close(
            result.as_slice().unwrap(),
            a.as_slice().unwrap(),
            1e-6,
            "matmul_2d identity",
        );
    }

    #[test]
    fn test_matmul_negative_values() {
        let a = Array2::from_shape_fn((3, 4), |(i, j)| -((i + j) as f32));
        let b = Array2::from_shape_fn((4, 3), |(i, j)| (i as f32) - (j as f32));

        let result = matmul_2d(&a.view(), &b.view());
        let expected = reference_matmul_2d(&a, &b);

        assert_close(
            result.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            1e-5,
            "matmul negative values",
        );
    }

    #[test]
    fn test_matmul_small_values() {
        let a = Array2::from_shape_fn((4, 8), |(i, j)| ((i + j) as f32) * 1e-6);
        let b = Array2::from_shape_fn((8, 4), |(i, j)| ((i * j) as f32) * 1e-6);

        let result = matmul_2d(&a.view(), &b.view());
        let expected = reference_matmul_2d(&a, &b);

        assert_close(
            result.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            1e-10,
            "matmul small values",
        );
    }
}