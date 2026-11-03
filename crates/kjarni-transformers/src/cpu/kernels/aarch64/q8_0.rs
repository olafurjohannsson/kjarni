#![allow(unsafe_code)]
use crate::cpu::kernels::q_common::BlockQ8_0;
use std::arch::aarch64::*;

/// Widens 16 int8 weights to four f32x4 vectors.
#[inline]
#[target_feature(enable = "neon")]
fn widen_i8_to_f32(q: int8x16_t) -> [float32x4_t; 4] {
    let lo = vmovl_s8(vget_low_s8(q));
    let hi = vmovl_high_s8(q);
    [
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),
        vcvtq_f32_s32(vmovl_high_s16(lo)),
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
        vcvtq_f32_s32(vmovl_high_s16(hi)),
    ]
}

/// Computes a vector-matrix multiplication for Q8_0 weights using NEON.
///
/// Each block's 32 products go into four separate sums, so the multiply-adds can run
/// in parallel on the core's FP units, and the block scale is applied once to their
/// total rather than to every weight.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn matmul_vec_q8_0_neon(
    out_chunk: &mut [f32],
    a_ptr: *const f32,
    b_blocks: &[BlockQ8_0],
    k: usize,
) {
    let blocks_per_row = k / 32;

    for (i, val) in out_chunk.iter_mut().enumerate() {
        let start = i * blocks_per_row;
        let row_blocks = &b_blocks[start..start + blocks_per_row];
        let mut total = vdupq_n_f32(0.0);

        for (block_idx, block) in row_blocks.iter().enumerate() {
            // SAFETY: `qs` holds 32 bytes, and the caller guarantees `a_ptr` points to
            // `k` floats, which covers all `k / 32` blocks of this row.
            let (q_lo, q_hi, a) = unsafe {
                let q = block.qs.as_ptr();
                let a = a_ptr.add(block_idx * 32);
                (
                    vld1q_s8(q),
                    vld1q_s8(q.add(16)),
                    [
                        vld1q_f32(a),
                        vld1q_f32(a.add(4)),
                        vld1q_f32(a.add(8)),
                        vld1q_f32(a.add(12)),
                        vld1q_f32(a.add(16)),
                        vld1q_f32(a.add(20)),
                        vld1q_f32(a.add(24)),
                        vld1q_f32(a.add(28)),
                    ],
                )
            };

            let w_lo = widen_i8_to_f32(q_lo);
            let w_hi = widen_i8_to_f32(q_hi);

            let s0 = vfmaq_f32(vmulq_f32(w_lo[0], a[0]), w_hi[0], a[4]);
            let s1 = vfmaq_f32(vmulq_f32(w_lo[1], a[1]), w_hi[1], a[5]);
            let s2 = vfmaq_f32(vmulq_f32(w_lo[2], a[2]), w_hi[2], a[6]);
            let s3 = vfmaq_f32(vmulq_f32(w_lo[3], a[3]), w_hi[3], a[7]);

            let block_sum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
            total = vfmaq_f32(total, block_sum, vdupq_n_f32(block.d.to_f32()));
        }

        *val = vaddvq_f32(total);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::kernels::scalar::matmul_vec_q8_0_scalar;
    use half::f16;

    fn run(a: &[f32], blocks: &[BlockQ8_0], rows: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows];
        unsafe { matmul_vec_q8_0_neon(&mut out, a.as_ptr(), blocks, a.len()) };
        out
    }

    #[test]
    fn matches_hand_computed_value() {
        // Every weight is 2 * 0.5 = 1, so the row is the sum of 0..32.
        let a: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let blocks = [BlockQ8_0 {
            d: f16::from_f32(0.5),
            qs: [2; 32],
        }];
        assert_eq!(run(&a, &blocks, 1), [496.0]);
    }

    #[test]
    fn keeps_each_weight_in_its_own_lane() {
        // Only the weight at `pos` is set, so the result must be that one activation.
        let a: Vec<f32> = (0..32).map(|i| (i + 1) as f32).collect();
        for pos in 0..32 {
            let mut qs = [0i8; 32];
            qs[pos] = -3;
            let blocks = [BlockQ8_0 {
                d: f16::from_f32(1.0),
                qs,
            }];
            assert_eq!(run(&a, &blocks, 1), [-3.0 * (pos + 1) as f32], "pos {pos}");
        }
    }

    #[test]
    fn matches_scalar_across_rows_and_extreme_weights() {
        let k = 32 * 5;
        let rows = 7;
        let a: Vec<f32> = (0..k)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01)
            .collect();
        let blocks: Vec<BlockQ8_0> = (0..rows * 5)
            .map(|b| {
                let mut qs = [0i8; 32];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = match i % 5 {
                        0 => i8::MIN,
                        1 => i8::MAX,
                        _ => ((b * 31 + i * 17) % 255) as u8 as i8,
                    };
                }
                BlockQ8_0 {
                    d: f16::from_f32(0.01 + b as f32 * 0.003),
                    qs,
                }
            })
            .collect();

        let mut expected = vec![0.0f32; rows];
        matmul_vec_q8_0_scalar(&mut expected, &a, &blocks, k);
        let actual = run(&a, &blocks, rows);

        for (row, (e, g)) in expected.iter().zip(&actual).enumerate() {
            assert!(
                (e - g).abs() <= e.abs().max(1.0) * 1e-5,
                "row {row}: {e} vs {g}"
            );
        }
    }
}
