//! Emotion detection examples.
//!
//! Run with: cargo run --example emotion

use kjarni::classifier::{self, Classifier, ClassificationMode};

use std::time::{Duration, Instant};
use kjarni::embedder::Embedder;

#[tokio::main]
async fn main() {
    let e = Embedder::builder("minilm-l6-v2")
        .quiet(true)
        .build()
        .await
        .expect("failed to load embedder");

    let text = "The quick brown fox jumps over the lazy dog";

    // Warmup
    e.embed(text).await.unwrap();

    let n = 100;
    let mut times = Vec::with_capacity(n);

    for _ in 0..n {
        let start = Instant::now();
        e.embed(text).await.unwrap();
        times.push(start.elapsed());
    }

    let total: Duration = times.iter().sum();
    let min = times.iter().min().unwrap();
    let max = times.iter().max().unwrap();

    println!("Encode ({n} runs)");
    println!("  avg: {:?}", total / n as u32);
    println!("  min: {:?}", min);
    println!("  max: {:?}", max);
}