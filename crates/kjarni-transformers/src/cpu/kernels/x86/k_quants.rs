//! AVX2 dot products of Q5_K and Q6_K weights with Q8_K activations.
//!
//! Every step before the final scale is integer arithmetic, so each kernel
//! returns exactly what its scalar counterpart in `scalar.rs` returns.

#![allow(unsafe_code)]

use std::arch::x86_64::*;

use crate::cpu::kernels::dequantize::get_scale_min_k4;
use crate::cpu::kernels::q_common::{BlockQ5_K, BlockQ6_K, BlockQ8_K, QK_K};

/// The sum of the eight i32 lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn hsum_i32(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
    _mm_cvtsi128_si32(s)
}

/// Q6_K weights against Q8_K activations.
///
/// A Q6_K block holds 16 sub-blocks of 16 weights, each with its own scale, and
/// stores weights offset by 32. Sub-block `s` covers the same 16 activations as
/// the Q8_K block's `bsums[s]`, so the offset is taken off once per sub-block:
/// `sum(sc * (w - 32) * a) = sum(sc * w * a) - 32 * sum(sc * bsums)`.
///
/// # Safety
/// The CPU must support AVX2, and both slices must hold `n / 256` blocks.
#[target_feature(enable = "avx2")]
pub unsafe fn vec_dot_q6k_q8k_avx2(
    n: usize,
    w_blocks: &[BlockQ6_K],
    q_blocks: &[BlockQ8_K],
) -> f32 {
    let low4 = _mm256_set1_epi8(0x0F);
    let low2 = _mm256_set1_epi8(0x03);
    let mut acc = 0.0f32;
    for (w, q) in w_blocks.iter().zip(q_blocks).take(n / QK_K) {
        let mut sums = _mm256_setzero_si256();
        for half in 0..2 {
            // SAFETY: `half < 2`, so every 32-byte load stays inside its array:
            // ql[half*64 + 32 + 32] <= 128, qh[half*32 + 32] <= 64,
            // qs[half*128 + 96 + 32] <= 256.
            let (ql0, ql1, qh, a) = unsafe {
                let ql = w.ql.as_ptr().add(half * 64);
                let qs = q.qs.as_ptr().add(half * 128);
                (
                    _mm256_loadu_si256(ql as *const __m256i),
                    _mm256_loadu_si256(ql.add(32) as *const __m256i),
                    _mm256_loadu_si256(w.qh.as_ptr().add(half * 32) as *const __m256i),
                    [
                        _mm256_loadu_si256(qs as *const __m256i),
                        _mm256_loadu_si256(qs.add(32) as *const __m256i),
                        _mm256_loadu_si256(qs.add(64) as *const __m256i),
                        _mm256_loadu_si256(qs.add(96) as *const __m256i),
                    ],
                )
            };
            // Each qh byte holds the top two bits of four weights, one per stream.
            // Shifting 16-bit lanes is safe here: every result is masked to the
            // bits that came from its own byte.
            let hi = |shift: i32| -> __m256i {
                let bits = match shift {
                    0 => qh,
                    2 => _mm256_srli_epi16::<2>(qh),
                    4 => _mm256_srli_epi16::<4>(qh),
                    _ => _mm256_srli_epi16::<6>(qh),
                };
                _mm256_slli_epi16::<4>(_mm256_and_si256(bits, low2))
            };
            let weights = [
                _mm256_or_si256(_mm256_and_si256(ql0, low4), hi(0)),
                _mm256_or_si256(_mm256_and_si256(ql1, low4), hi(2)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(ql0), low4), hi(4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(ql1), low4), hi(6)),
            ];
            for (stream, (wv, av)) in weights.iter().zip(a.iter()).enumerate() {
                // Unsigned 6-bit weights times signed activations, in pairs:
                // at most 2 * 63 * 128, which fits an i16.
                let products = _mm256_maddubs_epi16(*wv, *av);
                // The first 16 positions belong to one sub-block, the next 16 to
                // the following one.
                let s = half * 8 + stream * 2;
                let scale = _mm256_set_m128i(
                    _mm_set1_epi16(w.scales[s + 1] as i16),
                    _mm_set1_epi16(w.scales[s] as i16),
                );
                sums = _mm256_add_epi32(sums, _mm256_madd_epi16(products, scale));
            }
        }
        let offset: i32 = w
            .scales
            .iter()
            .zip(q.bsums.iter())
            .map(|(&sc, &bs)| sc as i32 * bs as i32)
            .sum();
        let sum_qs = hsum_i32(sums) - 32 * offset;
        acc += w.d.to_f32() * q.d * sum_qs as f32;
    }
    acc
}

/// Q5_K weights against Q8_K activations.
///
/// Eight sub-blocks of 32 weights, each with a 6-bit scale and minimum. Bit
/// `s` of `qh[l]` is the fifth bit of weight `l` in sub-block `s`.
///
/// # Safety
/// The CPU must support AVX2, and both slices must hold `n / 256` blocks.
#[target_feature(enable = "avx2")]
pub unsafe fn vec_dot_q5k_q8k_avx2(
    n: usize,
    w_blocks: &[BlockQ5_K],
    q_blocks: &[BlockQ8_K],
) -> f32 {
    let low4 = _mm256_set1_epi8(0x0F);
    let sixteen = _mm256_set1_epi8(16);
    let mut acc = 0.0f32;
    for (w, q) in w_blocks.iter().zip(q_blocks).take(n / QK_K) {
        // SAFETY: qh is 32 bytes.
        let qh = unsafe { _mm256_loadu_si256(w.qh.as_ptr() as *const __m256i) };
        let mut sums = _mm256_setzero_si256();
        let mut sum_mins = 0i32;
        for j in 0..4 {
            // SAFETY: j < 4: qs[j*32 + 32] <= 128 and q.qs[j*64 + 64] <= 256.
            let (packed, a_low, a_high) = unsafe {
                (
                    _mm256_loadu_si256(w.qs.as_ptr().add(j * 32) as *const __m256i),
                    _mm256_loadu_si256(q.qs.as_ptr().add(j * 64) as *const __m256i),
                    _mm256_loadu_si256(q.qs.as_ptr().add(j * 64 + 32) as *const __m256i),
                )
            };
            for (half, a) in [(0usize, a_low), (1, a_high)] {
                let sub = 2 * j + half;
                let nibbles = if half == 0 {
                    _mm256_and_si256(packed, low4)
                } else {
                    _mm256_and_si256(_mm256_srli_epi16::<4>(packed), low4)
                };
                let bit = _mm256_set1_epi8((1u8 << sub) as i8);
                let has_bit = _mm256_cmpeq_epi8(_mm256_and_si256(qh, bit), bit);
                let weights = _mm256_or_si256(nibbles, _mm256_and_si256(has_bit, sixteen));
                // At most 2 * 31 * 128 per pair, which fits an i16.
                let products = _mm256_maddubs_epi16(weights, a);
                let (sc, m) = get_scale_min_k4(sub, &w.scales);
                sums = _mm256_add_epi32(
                    sums,
                    _mm256_madd_epi16(products, _mm256_set1_epi16(sc as i16)),
                );
                sum_mins += (q.bsums[sub * 2] as i32 + q.bsums[sub * 2 + 1] as i32) * m as i32;
            }
        }
        let sum_qs = hsum_i32(sums);
        acc += q.d * w.d.to_f32() * sum_qs as f32 - q.d * w.dmin.to_f32() * sum_mins as f32;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::kernels::quantize::quantize_row_q8_k;
    use crate::cpu::kernels::scalar::{vec_dot_q5k_q8k_scalar, vec_dot_q6k_q8k_scalar};
    use half::f16;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn activations(rng: &mut StdRng, k: usize) -> Vec<BlockQ8_K> {
        let mut a: Vec<f32> = (0..k).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
        // Outliers push other values towards zero and the extremes to ±127.
        a[3] = 25.0;
        a[k - 1] = -31.0;
        quantize_row_q8_k(&a)
    }

    #[test]
    fn q6_k_matches_the_scalar_kernel_exactly() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = StdRng::seed_from_u64(7);
        let blocks = 6;
        let w: Vec<BlockQ6_K> = (0..blocks)
            .map(|_| {
                let mut b = BlockQ6_K {
                    ql: [0; 128],
                    qh: [0; 64],
                    scales: [0; 16],
                    d: f16::from_f32(rng.gen_range(0.001..0.05)),
                };
                rng.fill(&mut b.ql[..]);
                rng.fill(&mut b.qh[..]);
                for s in b.scales.iter_mut() {
                    *s = rng.gen_range(-128i16..=127) as i8;
                }
                b
            })
            .collect();
        let a = activations(&mut rng, blocks * QK_K);
        let fast = unsafe { vec_dot_q6k_q8k_avx2(blocks * QK_K, &w, &a) };
        let slow = vec_dot_q6k_q8k_scalar(blocks * QK_K, &w, &a);
        assert_eq!(fast.to_bits(), slow.to_bits(), "{fast} vs {slow}");
    }

    #[test]
    fn q4_k_matches_the_scalar_kernel_exactly() {
        use crate::cpu::kernels::q_common::BlockQ4_K;
        use crate::cpu::kernels::scalar::vec_dot_q4k_q8k_scalar;
        use crate::cpu::kernels::x86::q4k_q8k::vec_dot_q4k_q8k_avx2;
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let mut rng = StdRng::seed_from_u64(5);
        let blocks = 6;
        let w: Vec<BlockQ4_K> = (0..blocks)
            .map(|_| {
                let mut b = BlockQ4_K {
                    d: f16::from_f32(rng.gen_range(0.001..0.05)),
                    dmin: f16::from_f32(rng.gen_range(0.0..0.02)),
                    scales: [0; 12],
                    qs: [0; 128],
                };
                rng.fill(&mut b.scales[..]);
                rng.fill(&mut b.qs[..]);
                b
            })
            .collect();
        let a = activations(&mut rng, blocks * QK_K);
        let fast = unsafe { vec_dot_q4k_q8k_avx2(blocks * QK_K, &w, &a) };
        let slow = vec_dot_q4k_q8k_scalar(blocks * QK_K, &w, &a);
        assert_eq!(fast.to_bits(), slow.to_bits(), "{fast} vs {slow}");
    }

    #[test]
    fn q5_k_matches_the_scalar_kernel_exactly() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = StdRng::seed_from_u64(11);
        let blocks = 6;
        let w: Vec<BlockQ5_K> = (0..blocks)
            .map(|_| {
                let mut b = BlockQ5_K {
                    d: f16::from_f32(rng.gen_range(0.001..0.05)),
                    dmin: f16::from_f32(rng.gen_range(0.0..0.02)),
                    scales: [0; 12],
                    qh: [0; 32],
                    qs: [0; 128],
                };
                rng.fill(&mut b.scales[..]);
                rng.fill(&mut b.qh[..]);
                rng.fill(&mut b.qs[..]);
                b
            })
            .collect();
        let a = activations(&mut rng, blocks * QK_K);
        let fast = unsafe { vec_dot_q5k_q8k_avx2(blocks * QK_K, &w, &a) };
        let slow = vec_dot_q5k_q8k_scalar(blocks * QK_K, &w, &a);
        assert_eq!(fast.to_bits(), slow.to_bits(), "{fast} vs {slow}");
    }

    /// Decode-time cost of a Qwen 3.5 LM head (248,320 x 2,560) in Q6_K, both
    /// kernels, and the bandwidth it implies. Run with --ignored --nocapture.
    #[test]
    #[ignore = "a timing, not a check; allocates 520 MB"]
    fn q6_k_lm_head_timing() {
        use rayon::prelude::*;
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let (rows, k) = (248_320usize, 2_560usize);
        let per_row = k / QK_K;
        let block = BlockQ6_K {
            ql: [0x5A; 128],
            qh: [0xA5; 64],
            scales: [3; 16],
            d: f16::from_f32(0.01),
        };
        let w = vec![block; rows * per_row];
        let mut rng = StdRng::seed_from_u64(1);
        let a = activations(&mut rng, k);
        let mut out = vec![0f32; rows];
        let bytes = (w.len() * std::mem::size_of::<BlockQ6_K>()) as f64;
        for (label, fast) in [("scalar", false), ("avx2", true), ("avx2", true)] {
            let t = std::time::Instant::now();
            out.par_chunks_mut(1024).enumerate().for_each(|(c, o)| {
                for (j, v) in o.iter_mut().enumerate() {
                    let r = c * 1024 + j;
                    let row = &w[r * per_row..(r + 1) * per_row];
                    *v = if fast {
                        unsafe { vec_dot_q6k_q8k_avx2(k, row, &a) }
                    } else {
                        vec_dot_q6k_q8k_scalar(k, row, &a)
                    };
                }
            });
            let s = t.elapsed().as_secs_f64();
            println!("{label}: {:.1} ms, {:.1} GB/s", s * 1e3, bytes / s / 1e9);
        }
        // What the memory delivers to a plain parallel read of the same bytes.
        let raw: &[u64] = bytemuck::cast_slice(&w[..w.len() / 8 * 8]);
        for _ in 0..2 {
            let t = std::time::Instant::now();
            let total: u64 = raw
                .par_chunks(1 << 16)
                .map(|c| c.iter().fold(0u64, |a, &b| a ^ b))
                .reduce(|| 0, |a, b| a ^ b);
            let s = t.elapsed().as_secs_f64();
            println!(
                "plain read: {:.1} ms, {:.1} GB/s ({total})",
                s * 1e3,
                (raw.len() * 8) as f64 / s / 1e9
            );
        }
    }
}
