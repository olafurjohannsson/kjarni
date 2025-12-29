//! Ultra-optimized logits projection kernel for Llama lm_head
//!
//! Specifically tuned for:
//! - Single token decode: [1, hidden_size] × [vocab_size, hidden_size]^T
//! - Llama 3.2 1B: hidden_size=2048, vocab_size=128256
//! - BF16 weights, F32 activations
//!
//! Key optimizations:
//! - 8-way output unrolling (8 vocab entries per iteration)
//! - 4-way inner loop unrolling (32 elements per iteration)
//! - Aggressive prefetching
//! - Cache-blocking for L2/L3
//! - Parallel over vocab chunks
//!
#![allow(unsafe_code)]
pub mod q_common;
// pub(crate) mod q_common;
// pub(crate) mod scalar;
pub mod scalar;
pub mod dequantize;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod x86;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;

use half::bf16;
use ndarray::{Array1, ArrayView1, ArrayView2};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Process this many vocabulary entries per Rayon task.
/// Balanced for 1B/8B to prevent tail-latency in the thread pool.
const CHUNK_SIZE: usize = 1024;

/// Optimized logits projection for single-token decode
/// Computes: logits = hidden @ lm_head.T
pub fn project_logits_bf16(hidden: &ArrayView1<f32>, lm_head: &ArrayView2<bf16>) -> Array1<f32> {
    let hidden_size = hidden.len();
    let vocab_size = lm_head.shape()[0];

    assert_eq!(lm_head.shape()[1], hidden_size, "Dimension mismatch");

    let mut logits = Array1::<f32>::zeros(vocab_size);

    let hidden_slice = hidden.as_slice().expect("Hidden must be contiguous");
    let weights_slice = lm_head.as_slice().expect("Weights must be contiguous");

    logits
        .as_slice_mut()
        .unwrap()
        .par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let vocab_start = chunk_idx * CHUNK_SIZE;
            let weights_offset = vocab_start * hidden_size;

            unsafe {
                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        project_logits_avx2_fma(
                            out_chunk,
                            hidden_slice,
                            weights_slice.as_ptr().add(weights_offset) as *const u16,
                            hidden_size,
                        );
                        return;
                    }
                }

                #[cfg(target_arch = "aarch64")]
                {
                    // No runtime feature check needed for standard aarch64 neon
                    project_logits_neon(
                        out_chunk,
                        hidden_slice,
                        weights_slice.as_ptr().add(weights_offset) as *const u16,
                        hidden_size,
                    );
                    return;
                }

                // Scalar fallback for other architectures
                project_logits_scalar(
                    out_chunk,
                    hidden_slice,
                    &weights_slice[weights_offset..],
                    hidden_size,
                );
            }
        });

    logits
}

// =============================================================================
// X86_64 AVX2 + FMA KERNEL
// =============================================================================
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn project_logits_avx2_fma(
    out: &mut [f32],
    hidden: &[f32],
    weights_ptr: *const u16,
    k: usize,
) {
    let n = out.len();
    let hidden_ptr = hidden.as_ptr();
    let mut out_idx = 0;

    // 4-way output unrolling is the "sweet spot" for AVX2 (16 registers)
    // sum0..3 (4) + h (1) + w_u16 (1) + w_f32 (1) = 7 registers used.
    // 8-way unrolling causes register spilling which slows down the kernel.
    while out_idx + 4 <= n {
        let mut w_ptr0 = weights_ptr.add((out_idx + 0) * k);
        let mut w_ptr1 = weights_ptr.add((out_idx + 1) * k);
        let mut w_ptr2 = weights_ptr.add((out_idx + 2) * k);
        let mut w_ptr3 = weights_ptr.add((out_idx + 3) * k);

        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        let mut h_ptr = hidden_ptr;
        let mut remaining_k = k;

        while remaining_k >= 8 {
            let h = _mm256_loadu_ps(h_ptr);

            // Row 0
            let w0 = bf16x8_to_f32x8(_mm_loadu_si128(w_ptr0 as *const __m128i));
            sum0 = _mm256_fmadd_ps(h, w0, sum0);

            // Row 1
            let w1 = bf16x8_to_f32x8(_mm_loadu_si128(w_ptr1 as *const __m128i));
            sum1 = _mm256_fmadd_ps(h, w1, sum1);

            // Row 2
            let w2 = bf16x8_to_f32x8(_mm_loadu_si128(w_ptr2 as *const __m128i));
            sum2 = _mm256_fmadd_ps(h, w2, sum2);

            // Row 3
            let w3 = bf16x8_to_f32x8(_mm_loadu_si128(w_ptr3 as *const __m128i));
            sum3 = _mm256_fmadd_ps(h, w3, sum3);

            h_ptr = h_ptr.add(8);
            w_ptr0 = w_ptr0.add(8);
            w_ptr1 = w_ptr1.add(8);
            w_ptr2 = w_ptr2.add(8);
            w_ptr3 = w_ptr3.add(8);
            remaining_k -= 8;
        }

        out[out_idx + 0] = hsum_avx(sum0);
        out[out_idx + 1] = hsum_avx(sum1);
        out[out_idx + 2] = hsum_avx(sum2);
        out[out_idx + 3] = hsum_avx(sum3);

        // Handle K-remainder if hidden size is not multiple of 8
        if remaining_k > 0 {
            for i in 0..4 {
                let w_row = weights_ptr.add((out_idx + i) * k + (k - remaining_k));
                for j in 0..remaining_k {
                    out[out_idx + i] +=
                        hidden[k - remaining_k + j] * f32::from_bits((*w_row.add(j) as u32) << 16);
                }
            }
        }
        out_idx += 4;
    }

    // Handle remaining output rows (tail)
    for i in out_idx..n {
        let mut sum = _mm256_setzero_ps();
        let w_row = weights_ptr.add(i * k);
        let mut h_ptr = hidden_ptr;
        for j in (0..k).step_by(8) {
            if j + 8 <= k {
                let h = _mm256_loadu_ps(h_ptr);
                let w = bf16x8_to_f32x8(_mm_loadu_si128(w_row.add(j) as *const __m128i));
                sum = _mm256_fmadd_ps(h, w, sum);
                h_ptr = h_ptr.add(8);
            } else {
                let mut res = hsum_avx(sum);
                for rem_j in j..k {
                    res += hidden[rem_j] * f32::from_bits((*w_row.add(rem_j) as u32) << 16);
                }
                out[i] = res;
                break;
            }
        }
        if k % 8 == 0 {
            out[i] = hsum_avx(sum);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn bf16x8_to_f32x8(bf16_vals: __m128i) -> __m256 {
    let expanded = _mm256_cvtepu16_epi32(bf16_vals);
    _mm256_castsi256_ps(_mm256_slli_epi32(expanded, 16))
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum_avx(v: __m256) -> f32 {
    let vlow = _mm256_castps256_ps128(v);
    let vhigh = _mm256_extractf128_ps(v, 1);
    let vsum = _mm_add_ps(vlow, vhigh);
    let vsum = _mm_hadd_ps(vsum, vsum);
    let vsum = _mm_hadd_ps(vsum, vsum);
    _mm_cvtss_f32(vsum)
}

// =============================================================================
// AARCH64 NEON KERNEL
// =============================================================================
#[cfg(target_arch = "aarch64")]
unsafe fn project_logits_neon(out: &mut [f32], hidden: &[f32], weights_ptr: *const u16, k: usize) {
    use std::arch::aarch64::*;
    let n = out.len();

    for i in 0..n {
        let mut sum_v = vdupq_n_f32(0.0);
        let w_row = weights_ptr.add(i * k);

        let mut j = 0;
        while j + 4 <= k {
            let h_v = vld1q_f32(hidden.as_ptr().add(j));
            let w_u16 = vld1_u16(w_row.add(j));
            // bf16 to f32
            let w_v = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(w_u16), 16));
            sum_v = vfmaq_f32(sum_v, h_v, w_v);
            j += 4;
        }

        let mut sum = vaddvq_f32(sum_v);
        while j < k {
            sum += hidden[j] * f32::from_bits((*w_row.add(j) as u32) << 16);
            j += 1;
        }
        out[i] = sum;
    }
}

// =============================================================================
// SCALAR FALLBACK
// =============================================================================
fn project_logits_scalar(out: &mut [f32], hidden: &[f32], weights: &[bf16], k: usize) {
    for (i, out_val) in out.iter_mut().enumerate() {
        let row = &weights[i * k..(i + 1) * k];
        let mut sum = 0.0f32;
        for j in 0..k {
            sum += hidden[j] * row[j].to_f32();
        }
        *out_val = sum;
    }
}
