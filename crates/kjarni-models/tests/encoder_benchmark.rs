//! Encoder timings across models, sizes and both entry points.
//!
//! Mirrors `bench/encoder_bench.py` row for row so the two tables can be compared
//! directly. Not an assertion of anything: it prints numbers.
//!
//!   cargo test --release --test encoder_benchmark -- --nocapture

use std::time::Instant;

use kjarni_models::SentenceEncoder;
use kjarni_transformers::models::get_default_cache_dir;
use kjarni_transformers::traits::Device;
use kjarni_transformers::utils::alloc_stats;

/// Warmup iterations discarded before timing, then the number of timed
/// iterations whose minimum is reported.
const WARMUP: usize = 3;
const ENCODE_RUNS: usize = 9;
const BATCH_RUNS: usize = 7;

const MODELS: &[(&str, &str, usize)] = &[
    ("sentence-transformers_all-MiniLM-L6-v2", "minilm 22M", 64),
    (
        "distilbert_distilbert-base-uncased-finetuned-sst-2-english",
        "distilbert 66M",
        64,
    ),
    ("sentence-transformers_all-mpnet-base-v2", "mpnet 110M", 64),
    ("nomic-ai_nomic-embed-text-v1.5", "nomic 137M", 64),
    ("BAAI_bge-m3", "bge-m3 567M", 64),
];

fn docs(n: usize, sentences: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            (0..sentences)
                .map(|j| format!("Document number {i} sentence {j} covers refunds, delivery windows and account settings in some detail. "))
                .collect()
        })
        .collect()
}

#[tokio::test]
#[cfg_attr(debug_assertions, ignore = "timings are meaningless unoptimised")]
async fn encoder_timings() {
    eprintln!(
        "\n  {:<16}{:>7}{:>6}{:>5}{:>11}{:>13}",
        "model", "call", "docs", "sent", "ms", "allocations"
    );

    for (dir_name, label, max_docs) in MODELS.iter().copied() {
        let dir = get_default_cache_dir().join(dir_name);
        if !dir.join("model.safetensors").exists() {
            eprintln!("  {label:<16} not cached");
            continue;
        }
        let encoder = match SentenceEncoder::from_pretrained(&dir, Device::Cpu, None, None, None) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("  {label:<16} load failed: {e}");
                continue;
            }
        };

        // Single documents from one sentence up to roughly 1800 tokens. bge-m3
        // has an 8192 token window; everything else truncates, which is itself
        // worth seeing.
        for sentences in [1usize, 12, 48, 100] {
            let text = docs(1, sentences).remove(0);
            // Report the minimum, not the mean. Three identical runs of the
            // mean-based harness showed a 10% median and 27% worst-case spread
            // per row, which buried the effects being measured. The minimum
            // discards scheduler hiccups and thread migrations, which only ever
            // add time, and repeats to within about 2%.
            for _ in 0..WARMUP {
                let _ = encoder.encode(&text).await.expect("warmup");
            }
            alloc_stats::reset_alloc_count();
            let mut best = f64::MAX;
            for _ in 0..ENCODE_RUNS {
                let t = Instant::now();
                let _ = encoder.encode(&text).await.expect("encode");
                best = best.min(t.elapsed().as_secs_f64() * 1000.0);
            }
            eprintln!(
                "  {label:<16}{:>7}{:>6}{sentences:>5}{:>11.2}{:>13}",
                "encode",
                1,
                best,
                alloc_stats::alloc_count() / ENCODE_RUNS
            );
        }

        for sentences in [1usize, 12] {
            let mut n = 1;
            while n <= max_docs {
                let corpus = docs(n, sentences);
                let refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
                let runs = if n * sentences > 100 { 4 } else { BATCH_RUNS };
                for _ in 0..WARMUP {
                    let _ = encoder.encode_batch(&refs).await.expect("warmup");
                }
                alloc_stats::reset_alloc_count();
                let mut best = f64::MAX;
                for _ in 0..runs {
                    let t = Instant::now();
                    let _ = encoder.encode_batch(&refs).await.expect("encode_batch");
                    best = best.min(t.elapsed().as_secs_f64() * 1000.0);
                }
                eprintln!(
                    "  {label:<16}{:>7}{n:>6}{sentences:>5}{:>11.2}{:>13}",
                    "batch",
                    best,
                    alloc_stats::alloc_count() / runs
                );
                n *= 4;
            }
        }
    }
    eprintln!();
}
