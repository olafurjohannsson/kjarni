//! Where each matmul kernel wins in isolation, at the shapes these encoders use.
//!
//! Measures every (m, k, n) the five supported encoders generate against the
//! vector, blocked and faer kernels.
//!
//! Read the output with care. It says the vector kernel wins 52 of 132
//! shape/size combinations, but changing the dispatch to match cost time end to
//! end: every candidate threshold landed within 2.4% of the shipped one on the
//! real encoders. These loops are cache resident and single shaped, so they do
//! not reproduce what the same kernel does inside a forward pass. Treat this as
//! a description of the kernels, never as a reason to change the dispatch. Only
//! `encoder_benchmark` can justify that.

use std::time::Instant;

use kjarni_transformers::cpu::ops::matmul::{
    matmul_2d_f32_batched_noalloc, matmul_2d_f32_faer_noalloc, matmul_2d_f32_noalloc,
};
use ndarray::Array2;

/// (label, k = in features, n = out features)
const SHAPES: &[(&str, usize, usize)] = &[
    ("minilm qkv", 384, 1152),
    ("minilm out", 384, 384),
    ("minilm fc1", 384, 1536),
    ("minilm fc2", 1536, 384),
    ("base   qkv", 768, 2304),
    ("base   out", 768, 768),
    ("base   fc1", 768, 3072),
    ("base   fc2", 3072, 768),
    ("bge    qkv", 1024, 3072),
    ("bge    out", 1024, 1024),
    ("bge    fc1", 1024, 4096),
    ("bge    fc2", 4096, 1024),
];

#[tokio::test]
#[cfg_attr(debug_assertions, ignore = "timings are meaningless unoptimised")]
async fn kernel_crossover_by_shape() {
    eprintln!("\n  best of 5, microseconds, winner per row\n");
    eprintln!(
        "  {:<12}{:>7}{:>11}{:>11}{:>11}   winner",
        "shape", "m", "vector", "blocked", "faer"
    );

    for (label, k, n) in SHAPES.iter().copied() {
        for m in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let a = Array2::<f32>::from_elem((m, k), 0.01);
            let w = Array2::<f32>::from_elem((n, k), 0.02);
            let bias = vec![0.0f32; n];
            let mut out = Array2::<f32>::zeros((m, n));
            let (av, wv) = (a.view(), w.view());

            let mut best = [f64::MAX; 3];
            for _ in 0..5 {
                let t = Instant::now();
                matmul_2d_f32_noalloc(&av, &wv, Some(&bias), &mut out);
                best[0] = best[0].min(t.elapsed().as_secs_f64() * 1e6);

                let t = Instant::now();
                matmul_2d_f32_batched_noalloc(&av, &wv, Some(&bias), &mut out);
                best[1] = best[1].min(t.elapsed().as_secs_f64() * 1e6);

                let t = Instant::now();
                matmul_2d_f32_faer_noalloc(&av, &wv, Some(&bias), &mut out);
                best[2] = best[2].min(t.elapsed().as_secs_f64() * 1e6);
            }

            let names = ["vector", "blocked", "faer"];
            let win = (0..3)
                .min_by(|&i, &j| best[i].partial_cmp(&best[j]).expect("finite"))
                .expect("three");
            let second = (0..3)
                .filter(|&i| i != win)
                .map(|i| best[i])
                .fold(f64::MAX, f64::min);
            eprintln!(
                "  {label:<12}{m:>7}{:>11.1}{:>11.1}{:>11.1}   {:<8}{:.2}x",
                best[0],
                best[1],
                best[2],
                names[win],
                second / best[win]
            );
        }
        eprintln!();
    }
}
