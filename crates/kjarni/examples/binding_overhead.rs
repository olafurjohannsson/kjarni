//! The Rust side of bench/bindings: one text through the public API, as JSON.

use std::time::Instant;

use kjarni::embedder::Embedder;

const WARMUP: usize = 200;
const CALLS: usize = 3000;
const TEXT: &str = "how do I get a refund for a damaged order?";

#[tokio::main]
async fn main() {
    let embedder = Embedder::builder("minilm-l6-v2")
        .quiet(true)
        .build()
        .await
        .unwrap();
    let mut times = Vec::with_capacity(CALLS);
    for i in 0..WARMUP + CALLS {
        let start = Instant::now();
        embedder.embed(TEXT).await.unwrap();
        if i >= WARMUP {
            times.push(start.elapsed().as_secs_f64() * 1e6);
        }
    }
    times.sort_by(f64::total_cmp);
    println!(
        "{{\"lang\": \"Rust\", \"median_us\": {:.1}, \"min_us\": {:.1}}}",
        times[CALLS / 2],
        times[0]
    );
}
