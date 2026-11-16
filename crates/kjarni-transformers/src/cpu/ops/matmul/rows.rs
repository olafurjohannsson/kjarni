//! One input row against a run of weight rows, on the calling thread.
//!
//! The matmuls in this module's siblings split a single input row's output
//! rows over every thread. For a caller multiplying many small matrices at
//! once, such as a mixture of experts decoding one token through eight
//! 512-row experts, that is a dispatch per matrix with a few dozen rows per
//! thread, and the dispatches cost more than the arithmetic. These compute a
//! stretch of rows where they are called, so the caller can hand every
//! matrix's rows to the threads in one go.

use half::bf16;
use ndarray::ArrayView2;

use crate::cpu::kernels::{
    self,
    q_common::{BlockQ4_K, BlockQ5_K, BlockQ6_K, BlockQ8_0, BlockQ8_K},
};

/// `out[i] = w.row(i) · a` for f32 weights holding `out.len()` rows.
pub fn f32_rows_into(out: &mut [f32], a: &[f32], w: ArrayView2<f32>) {
    for (o, row) in out.iter_mut().zip(w.outer_iter()) {
        *o = match row.as_slice() {
            Some(r) => r.iter().zip(a).map(|(x, y)| x * y).sum(),
            None => row.iter().zip(a).map(|(x, y)| x * y).sum(),
        };
    }
}

/// As [`f32_rows_into`], for bf16 weights.
pub fn bf16_rows_into(out: &mut [f32], a: &[f32], w: ArrayView2<bf16>) {
    for (o, row) in out.iter_mut().zip(w.outer_iter()) {
        *o = row.iter().zip(a).map(|(x, y)| x.to_f32() * y).sum();
    }
}

/// As [`f32_rows_into`], for Q8_0 weights of `k` columns, `out.len()` rows
/// of blocks in `w`.
pub fn q8_0_rows_into(out: &mut [f32], a: &[f32], w: &[BlockQ8_0], k: usize) {
    // SAFETY: the features are checked first; `w` holds `out.len()` rows of
    // `k / 32` blocks and `a` holds `k` values.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { kernels::x86::q8_0::matmul_vec_q8_0_avx2(out, a.as_ptr(), w, k) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { kernels::aarch64::q8_0::matmul_vec_q8_0_neon(out, a.as_ptr(), w, k) };
    }
    kernels::scalar::matmul_vec_q8_0_scalar(out, a, w, k)
}

/// `out[i]` is weight row `i` of `w` against `a_q8`, the input already
/// quantised to Q8_K, for Q4_K weights of `k` columns.
pub fn q4_k_rows_into(out: &mut [f32], a_q8: &[BlockQ8_K], w: &[BlockQ4_K], k: usize) {
    let per_row = k / kernels::q_common::QK_K;
    for (o, row) in out.iter_mut().zip(w.chunks_exact(per_row)) {
        // SAFETY: the features are checked first; each row holds k / 256 blocks.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            *o = unsafe { kernels::x86::q4k_q8k::vec_dot_q4k_q8k_avx2(k, row, a_q8) };
            continue;
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            *o = unsafe { kernels::aarch64::q4k_q8k::vec_dot_q4k_q8k_neon(k, row, a_q8) };
            continue;
        }
        *o = kernels::scalar::vec_dot_q4k_q8k_scalar(k, row, a_q8);
    }
}

/// As [`q4_k_rows_into`], for Q5_K weights.
pub fn q5_k_rows_into(out: &mut [f32], a_q8: &[BlockQ8_K], w: &[BlockQ5_K], k: usize) {
    let per_row = k / kernels::q_common::QK_K;
    for (o, row) in out.iter_mut().zip(w.chunks_exact(per_row)) {
        // SAFETY: AVX2 is checked first; each row holds k / 256 blocks.
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            *o = unsafe { kernels::x86::k_quants::vec_dot_q5k_q8k_avx2(k, row, a_q8) };
            continue;
        }
        *o = kernels::scalar::vec_dot_q5k_q8k_scalar(k, row, a_q8);
    }
}

/// As [`q4_k_rows_into`], for Q6_K weights.
pub fn q6_k_rows_into(out: &mut [f32], a_q8: &[BlockQ8_K], w: &[BlockQ6_K], k: usize) {
    let per_row = k / kernels::q_common::QK_K;
    for (o, row) in out.iter_mut().zip(w.chunks_exact(per_row)) {
        // SAFETY: every aarch64 target Kjarni builds for has NEON enabled at
        // compile time.
        #[cfg(target_arch = "aarch64")]
        {
            *o = unsafe { kernels::aarch64::q6k_q8k::vec_dot_q6k_q8k_neon(k, row, a_q8) };
            continue;
        }
        // SAFETY: AVX2 is checked first; each row holds k / 256 blocks.
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            *o = unsafe { kernels::x86::k_quants::vec_dot_q6k_q8k_avx2(k, row, a_q8) };
            continue;
        }
        #[allow(unreachable_code)]
        {
            *o = kernels::scalar::vec_dot_q6k_q8k_scalar(k, row, a_q8);
        }
    }
}
