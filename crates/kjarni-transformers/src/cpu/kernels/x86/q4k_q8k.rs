#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::cpu::kernels::{
    dequantize::get_scale_min_k4,
    q_common::{BlockQ4_K, BlockQ8_K, QK_K},
};

#[allow(
    dead_code,
    reason = "SIMD kernel for the Q4_K path, which linear_algebra.rs still leaves unimplemented!()"
)]
#[inline(always)]
unsafe fn hsum_i32_8(a: __m256i) -> i32 {
    unsafe {
        // Add upper and lower 128-bit halves
        let sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
        // Add upper and lower 64-bit halves
        let hi64 = _mm_unpackhi_epi64(sum128, sum128);
        let sum64 = _mm_add_epi32(hi64, sum128);
        let hi32 = _mm_shuffle_epi32(sum64, 0b10_11_00_01); // _MM_SHUFFLE(2,3,0,1)
        _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32))
    }
}

#[allow(
    dead_code,
    reason = "SIMD kernel for the Q4_K path, which linear_algebra.rs still leaves unimplemented!()"
)]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn vec_dot_q4k_q8k_avx2(
    n: usize,
    w_blocks: &[BlockQ4_K],
    q_blocks: &[BlockQ8_K],
) -> f32 {
    let num_blocks = n / QK_K;
    let mut acc = 0.0f32;
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);

        for i in 0..num_blocks {
            let w = &w_blocks[i];
            let q = &q_blocks[i];

            let d = w.d.to_f32();
            let dmin = w.dmin.to_f32();

            let mut sum_qs = 0;
            let mut sum_mins = 0;

            let mut is = 0;

            let w_ptr = w.qs.as_ptr();
            let q_ptr = q.qs.as_ptr();

            for j in 0..4 {
                let (sc1, m1) = get_scale_min_k4(is, &w.scales);
                let (sc2, m2) = get_scale_min_k4(is + 1, &w.scales);

                let q_offset = j * 64;
                let q_v1 = _mm256_loadu_si256(q_ptr.add(q_offset) as *const __m256i);
                let q_v2 = _mm256_loadu_si256(q_ptr.add(q_offset + 32) as *const __m256i);

                let w_offset = j * 32;
                let w_packed = _mm256_loadu_si256(w_ptr.add(w_offset) as *const __m256i);

                let w_low = _mm256_and_si256(w_packed, m4);
                let w_high_shifted = _mm256_srli_epi16(w_packed, 4);
                let w_high = _mm256_and_si256(w_high_shifted, m4);

                let dot1 = _mm256_maddubs_epi16(w_low, q_v1);
                let dot2 = _mm256_maddubs_epi16(w_high, q_v2);

                let ones = _mm256_set1_epi16(1);
                let sum1 = _mm256_madd_epi16(dot1, ones);
                let sum2 = _mm256_madd_epi16(dot2, ones);

                let s1 = hsum_i32_8(sum1);
                let s2 = hsum_i32_8(sum2);
                sum_qs += s1 * (sc1 as i32);
                sum_qs += s2 * (sc2 as i32);

                let isum1 = q.bsums[is * 2] as i32 + q.bsums[is * 2 + 1] as i32;
                sum_mins += isum1 * (m1 as i32);

                let isum2 = q.bsums[(is + 1) * 2] as i32 + q.bsums[(is + 1) * 2 + 1] as i32;
                sum_mins += isum2 * (m2 as i32);

                is += 2;
            }

            acc += q.d * d * (sum_qs as f32) - q.d * dmin * (sum_mins as f32);
        }
    }
    acc
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod q4k_q8k_test {
    use super::*;
    use crate::cpu::kernels::dequantize::dequantize_q4_k_block;
    use half::f16;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn dequantize_q8k_to_f32(block: &BlockQ8_K) -> Vec<f32> {
        block.qs.iter().map(|&q| q as f32 * block.d).collect()
    }

    fn ground_truth_dot_product(w_blocks: &[BlockQ4_K], q_blocks: &[BlockQ8_K]) -> f32 {
        let mut total = 0.0;
        let mut w_f32 = [0.0f32; QK_K];

        for (w, q) in w_blocks.iter().zip(q_blocks.iter()) {
            dequantize_q4_k_block(w, &mut w_f32);

            let q_f32 = dequantize_q8k_to_f32(q);

            total += w_f32
                .iter()
                .zip(q_f32.iter())
                .map(|(a, b)| a * b)
                .sum::<f32>();
        }
        total
    }

    fn get_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn random_q4k_block(rng: &mut StdRng) -> BlockQ4_K {
        let mut scales = [0u8; 12];
        let mut qs = [0u8; QK_K / 2];
        rng.fill(&mut scales);
        rng.fill(&mut qs);
        BlockQ4_K {
            d: f16::from_f32(rng.gen_range(0.5..1.5)),
            dmin: f16::from_f32(rng.gen_range(0.0..0.5)),
            scales,
            qs,
        }
    }

    fn random_q8k_block(rng: &mut StdRng) -> BlockQ8_K {
        let mut qs = [0i8; 256];
        let mut bsums = [0i16; 16];
        rng.fill(&mut qs);

        for i in 0..16 {
            let sum: i32 = qs[i * 16..(i + 1) * 16].iter().map(|&x| x as i32).sum();
            bsums[i] = sum as i16;
        }

        BlockQ8_K {
            d: rng.gen_range(0.001..0.1),
            qs,
            bsums,
        }
    }

    /// Exact reference: dequantise the weights and dot them against the *unquantised*
    /// f32 activations, accumulated in f64 so the reference is not itself the error.
    fn reference_f32_activations(w_blocks: &[BlockQ4_K], a: &[f32]) -> f64 {
        let mut total = 0.0f64;
        let mut w_f32 = [0.0f32; QK_K];
        for (bi, w) in w_blocks.iter().enumerate() {
            dequantize_q4_k_block(w, &mut w_f32);
            for j in 0..QK_K {
                total += w_f32[j] as f64 * a[bi * QK_K + j] as f64;
            }
        }
        total
    }

    /// The two Q4_K paths do not compute the same thing, and this pins down by how much.
    ///
    /// The live kernel (`matmul_vec_q4_k_avx2`) expands each 4-bit weight to f32 and
    /// multiplies against the activations at full f32 precision. The integer kernel
    /// (`vec_dot_q4k_q8k_avx2`) first squeezes the activations into int8 via absmax,
    /// then multiplies in the integer domain. Swapping the second in for the first is
    /// therefore not a refactor: it trades activation precision for speed, and this
    /// test exists to say what that trade actually costs before anyone takes it.
    ///
    /// Activations are tested twice. Gaussian is the flattering case. Real decoder
    /// activations carry outliers, and absmax quantisation spends its whole int8 range
    /// on the largest value, so one big element coarsens every other element in the
    /// same 256-wide block. The outlier case is the one that matters.
    #[test]
    fn integer_path_error_against_live_path() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("Skipping: needs AVX2+FMA");
            return;
        }
        use crate::cpu::kernels::quantize::quantize_row_q8_k;
        use crate::cpu::kernels::x86::q4_k::matmul_vec_q4_k_avx2;

        let mut rng = get_rng();
        let blocks_per_row = 8;
        let k = QK_K * blocks_per_row;

        let w_blocks: Vec<BlockQ4_K> =
            (0..blocks_per_row).map(|_| random_q4k_block(&mut rng)).collect();

        for (label, a) in [
            ("gaussian", {
                let v: Vec<f32> = (0..k).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
                v
            }),
            ("with outliers", {
                let mut v: Vec<f32> = (0..k).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
                // One large value per 256-block, which is what absmax has to accommodate.
                for b in 0..blocks_per_row {
                    v[b * QK_K + 7] = 40.0;
                }
                v
            }),
        ] {
            let exact = reference_f32_activations(&w_blocks, &a);

            let mut live_out = [0.0f32; 1];
            unsafe { matmul_vec_q4_k_avx2(&mut live_out, a.as_ptr(), &w_blocks, k) };

            let q_blocks = quantize_row_q8_k(&a);
            let int_out = unsafe { vec_dot_q4k_q8k_avx2(k, &w_blocks, &q_blocks) };

            let denom = exact.abs().max(1.0);
            let live_err = (live_out[0] as f64 - exact).abs() / denom;
            let int_err = (int_out as f64 - exact).abs() / denom;

            // How much cancellation is in this dot product? If the sum of magnitudes
            // dwarfs the result, relative-to-result error is measuring cancellation
            // rather than kernel quality, and the test data is the thing at fault.
            let mut abs_sum = 0.0f64;
            let mut w_f32 = [0.0f32; QK_K];
            for (bi, w) in w_blocks.iter().enumerate() {
                dequantize_q4_k_block(w, &mut w_f32);
                for j in 0..QK_K {
                    abs_sum += (w_f32[j] as f64 * a[bi * QK_K + j] as f64).abs();
                }
            }
            println!(
                "{label:>14}: exact {exact:>12.4}  live {:>12.4} (rel {live_err:.2e})  \
                 int8 {:>12.4} (rel {int_err:.2e})  sum|wa| {abs_sum:>12.1}  \
                 cancellation {:.0}x",
                live_out[0], int_out, abs_sum / exact.abs().max(1e-9)
            );

            // The live path keeps the activations in f32, so it should track the exact
            // result to roughly f32 accumulation error and nothing worse.
            assert!(
                live_err < 1e-5,
                "{label}: live kernel drifted from exact: rel {live_err:.3e}"
            );

            // The integer path is allowed to be worse, but it has to stay in the range
            // where a quantised model is still the model. This bound is the finding,
            // not a target: if it tightens or loosens, the trade has changed.
            assert!(
                int_err < 1e-1,
                "{label}: int8 activation path lost more than expected: rel {int_err:.3e}"
            );
        }
    }

    /// The same comparison against weights that a real model actually contains.
    ///
    /// The synthetic test above builds Q4_K blocks from random `scales` bytes, which
    /// produces sub-block scales and mins no quantiser would ever emit. The resulting
    /// dot products cancel to roughly a thousandth of their own magnitude, and any
    /// error measured relative to that residue looks enormous whatever the kernel does.
    /// Real weights cancel far less, so this is the test that says whether swapping in
    /// the integer kernel is safe.
    ///
    /// Skips when the GGUF is not on this machine, so it stays a normal `cargo test`.
    /// Prints what dtypes the GGUF actually stores, per tensor kind. Diagnostic only.
    ///
    /// Q4_K_M is a mixture, not a uniform 4-bit file: llama.cpp keeps some tensors at
    /// Q6_K and leaves norms in F32. Knowing which is which is what makes a decode
    /// profile readable, because each dtype lands in a different kernel.
    #[test]
    #[ignore = "diagnostic"]
    fn dump_gguf_dtypes() {
        use crate::weights::ModelWeights;
        let path = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(
            ".cache/kjarni/llama-3.2-3b-instruct-q4_k_m/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        );
        if !path.exists() {
            println!("Skipping: model not present");
            return;
        }
        let w = ModelWeights::new(&path).unwrap();

        let per_layer = [
            "attn_q", "attn_k", "attn_v", "attn_output",
            "ffn_gate", "ffn_up", "ffn_down",
            "attn_norm", "ffn_norm",
        ];
        let mut totals: std::collections::BTreeMap<String, (usize, usize)> = Default::default();

        for kind in per_layer {
            let mut seen: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
            for layer in 0..28 {
                let n = format!("blk.{layer}.{kind}.weight");
                if let (Ok(dt), Ok(sz)) = (w.tensor_dtype(&n), w.tensor_size_bytes(&n)) {
                    let e = seen.entry(format!("{dt:?}")).or_insert((0, 0));
                    e.0 += 1;
                    e.1 += sz;
                    let t = totals.entry(format!("{dt:?}")).or_insert((0, 0));
                    t.0 += 1;
                    t.1 += sz;
                }
            }
            for (dt, (count, bytes)) in seen {
                println!("  {kind:<14} {dt:<6} x{count:<3} {:>8.1} MB", bytes as f64 / 1e6);
            }
        }
        for n in ["token_embd.weight", "output.weight", "output_norm.weight"] {
            if let (Ok(dt), Ok(sz)) = (w.tensor_dtype(n), w.tensor_size_bytes(n)) {
                println!("  {n:<20} {:<6}     {:>8.1} MB", format!("{dt:?}"), sz as f64 / 1e6);
                let t = totals.entry(format!("{dt:?}")).or_insert((0, 0));
                t.0 += 1;
                t.1 += sz;
            }
        }
        println!("  ---- totals by dtype ----");
        for (dt, (count, bytes)) in totals {
            println!("  {dt:<6} x{count:<4} {:>9.1} MB", bytes as f64 / 1e6);
        }
    }

    /// The same comparison against weights that a real model actually contains.
    ///
    /// The synthetic test above builds Q4_K blocks from random `scales` bytes, which
    /// produces sub-block scales and mins no quantiser would ever emit. The resulting
    /// dot products cancel to roughly a thousandth of their own magnitude, and any
    /// error measured relative to that residue looks enormous whatever the kernel does.
    /// Real weights cancel far less, so this is the test that says whether swapping in
    /// the integer kernel is safe.
    ///
    /// Skips when the GGUF is not on this machine, so it stays a normal `cargo test`.
    #[test]
    fn integer_path_error_on_real_q4k_weights() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("Skipping: needs AVX2+FMA");
            return;
        }
        use crate::cpu::kernels::quantize::quantize_row_q8_k;
        use crate::cpu::kernels::x86::q4_k::matmul_vec_q4_k_avx2;
        use crate::weights::ModelWeights;

        let path = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(
            ".cache/kjarni/llama-3.2-3b-instruct-q4_k_m/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        );
        if !path.exists() {
            println!("Skipping: {} not present", path.display());
            return;
        }
        let weights = ModelWeights::new(&path).expect("load gguf");

        // Whichever naming the loader exposes.
        let name = ["blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"]
            .into_iter()
            .find(|n| weights.tensor_dtype(n).is_ok())
            .expect("no q_proj tensor found under either naming");
        let dt = weights.tensor_dtype(name).unwrap();
        assert_eq!(dt, crate::tensor::DType::Q4_K, "{name} is {dt:?}, expected Q4_K");

        let blocks: Vec<BlockQ4_K> = weights
            .with_raw_tensor(name, |v| Ok(bytemuck::cast_slice::<u8, BlockQ4_K>(&v.bytes).to_vec()))
            .expect("read blocks");

        let mut rng = get_rng();
        let blocks_per_row = 8;
        let k = QK_K * blocks_per_row;
        let w_blocks = &blocks[..blocks_per_row];

        println!("  real weights from {name}");
        // Sweep the outlier magnitude. Absmax int8 spends its whole range on the
        // largest element of each 256-wide block, so one big activation coarsens
        // every other activation sharing that block. This is the axis that decides
        // whether the integer kernel is usable.
        for outlier in [0.0f32, 5.0, 20.0, 50.0] {
            let mut a: Vec<f32> = (0..k).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            if outlier > 0.0 {
                for b in 0..blocks_per_row {
                    a[b * QK_K + 7] = outlier;
                }
            }

            let exact = reference_f32_activations(w_blocks, &a);
            let mut live_out = [0.0f32; 1];
            unsafe { matmul_vec_q4_k_avx2(&mut live_out, a.as_ptr(), w_blocks, k) };
            let q_blocks = quantize_row_q8_k(&a);
            let int_out = unsafe { vec_dot_q4k_q8k_avx2(k, w_blocks, &q_blocks) };

            let mut abs_sum = 0.0f64;
            let mut w_f32 = [0.0f32; QK_K];
            for (bi, w) in w_blocks.iter().enumerate() {
                dequantize_q4_k_block(w, &mut w_f32);
                for j in 0..QK_K {
                    abs_sum += (w_f32[j] as f64 * a[bi * QK_K + j] as f64).abs();
                }
            }

            let denom = exact.abs().max(1e-9);
            println!(
                "  outlier {outlier:>5.1}: exact {exact:>11.4}  live {:>11.4} (rel {:.2e})  \
                 int8 {:>11.4} (rel {:.2e})  cancellation {:.0}x",
                live_out[0],
                (live_out[0] as f64 - exact).abs() / denom,
                int_out,
                (int_out as f64 - exact).abs() / denom,
                abs_sum / denom
            );
        }
    }

    /// Throughput of the two Q4_K CPU paths on a real weight matrix.
    ///
    /// The live path (`matmul_2d_cpu_q4_k`) expands each 4-bit weight to f32 and does
    /// f32 FMA against f32 activations. The unused path quantises the activations to
    /// Q8_K once per row and multiplies in the integer domain, which is what llama.cpp
    /// does. The integer kernel is already written and tested; the only reason it is
    /// not wired in is that nothing calls it. This says whether wiring it would pay.
    #[test]
    #[ignore = "benchmark"]
    fn bench_q4k_live_vs_integer() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        use crate::cpu::kernels::quantize::quantize_row_q8_k;
        use crate::weights::ModelWeights;
        use std::time::Instant;

        let path = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(
            ".cache/kjarni/llama-3.2-3b-instruct-q4_k_m/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        );
        if !path.exists() {
            println!("Skipping: model not present");
            return;
        }
        let w = ModelWeights::new(&path).unwrap();
        let name = "blk.0.ffn_gate.weight";
        let blocks: Vec<BlockQ4_K> = match w.get_typed_tensor(name).unwrap() {
            crate::tensor::CpuTensor::Q4_K(m) => m.blocks.clone(),
            _ => return,
        };
        let k = 3072usize;
        let bpr = k / QK_K;
        let n = blocks.len() / bpr;

        let mut rng = get_rng();
        let a: Vec<f32> = (0..k).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let input = ndarray::Array2::from_shape_vec((1, k), a.clone()).unwrap();

        let best = |mut f: Box<dyn FnMut()>| {
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };

        let v = input.view();
        let live = best(Box::new(|| {
            std::hint::black_box(crate::cpu::ops::matmul::matmul_2d_cpu_q4_k(&v, &blocks));
        }));

        // Same shape of work and the same parallelism as the live path: quantise the
        // row once, then split the output rows across the rayon pool. Comparing a
        // single-threaded kernel against a parallel one measures the thread count.
        use rayon::prelude::*;
        let integer = best(Box::new(|| {
            let q8 = quantize_row_q8_k(&a);
            let mut out = vec![0.0f32; n];
            let chunk = n.div_ceil(rayon::current_num_threads());
            out.par_chunks_mut(chunk).enumerate().for_each(|(ci, c)| {
                for (j, o) in c.iter_mut().enumerate() {
                    let i = ci * chunk + j;
                    let row = &blocks[i * bpr..(i + 1) * bpr];
                    *o = unsafe { vec_dot_q4k_q8k_avx2(k, row, &q8) };
                }
            });
            std::hint::black_box(out);
        }));

        let bytes = blocks.len() * std::mem::size_of::<BlockQ4_K>();
        println!("  {name} [{n}x{k}], {:.1} MB of Q4_K", bytes as f64 / 1e6);
        println!(
            "    live f32-dequant  : {:>7.3} ms  {:>6.1} GB/s",
            live * 1e3,
            bytes as f64 / live / 1e9
        );
        println!(
            "    integer Q4_K x Q8_K: {:>7.3} ms  {:>6.1} GB/s   ({:.2}x)",
            integer * 1e3,
            bytes as f64 / integer / 1e9,
            live / integer
        );
    }

    #[test]
    fn test_avx2_q4k_q8k_correctness() {
        if !is_x86_feature_detected!("avx2") {
            println!("Skipping AVX2 test");
            return;
        }
        let mut rng = get_rng();
        let n = QK_K * 4;

        let w_blocks: Vec<BlockQ4_K> = (0..4).map(|_| random_q4k_block(&mut rng)).collect();
        let q_blocks: Vec<BlockQ8_K> = (0..4).map(|_| random_q8k_block(&mut rng)).collect();

        let expected = ground_truth_dot_product(&w_blocks, &q_blocks);

        let actual = unsafe { vec_dot_q4k_q8k_avx2(n, &w_blocks, &q_blocks) };

        let diff = (expected - actual).abs();
        let rel_err = diff / expected.abs().max(1.0);

        assert!(
            rel_err < 5e-4,
            "AVX2 mismatch! Expected: {}, Actual: {}, RelErr: {}",
            expected,
            actual,
            rel_err
        );
    }
}
