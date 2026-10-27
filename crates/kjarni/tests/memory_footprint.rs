//! A loaded model holds about one copy of its weights.
//!
//! Three separate duplications once held Whisper-small at 1.9 times its weight
//! file and distilbart at 1.85: the global mmap cache kept every weight file
//! resident for the life of the process, `StdFeedForward` stored its matrices
//! twice, and the encoder, the decoder and the pipeline each loaded the token
//! embedding table on their own. None of that is visible in any output, only in
//! memory, so this checks the memory directly.
//!
//! One test in its own file, so it has the process to itself: cargo runs the
//! tests within a file in parallel, and another test's allocations would land in
//! the same counters. Linux only, since it reads `/proc/self/status`.

#![cfg(target_os = "linux")]

use std::path::Path;

/// Resident memory of the process in bytes: heap plus mapped files.
fn rss_bytes() -> u64 {
    status_bytes("VmRSS:")
}

/// The highest resident memory the process has reached.
fn peak_bytes() -> u64 {
    status_bytes("VmHWM:")
}

fn status_bytes(key: &str) -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    let kb: u64 = status
        .lines()
        .find(|l| l.starts_with(key))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .expect("VmRSS in /proc/self/status");
    kb * 1024
}

fn file_bytes(dir: &Path) -> Option<u64> {
    std::fs::metadata(dir.join("model.safetensors"))
        .ok()
        .map(|m| m.len())
}

/// Within this of the weight file: room for buffers, the tokenizer and
/// allocator overhead, well short of the second copy the bugs above added.
const MAX_RATIO: f64 = 1.3;

/// Peak while loading. The file is read as the weights are copied out, and each
/// tensor's pages are released once copied, so the peak stays near one copy; it
/// was twice the file when the whole mapping stayed resident during the load.
const MAX_PEAK_RATIO: f64 = 1.5;

#[tokio::test]
async fn models_hold_one_copy_of_their_weights() {
    let cache = kjarni_transformers::models::get_default_cache_dir();
    let whisper_dir = cache.join("openai_whisper-small");
    let bart_dir = cache.join("olafuraron_distilbart-cnn-12-6");
    let (Some(whisper_file), Some(bart_file)) = (file_bytes(&whisper_dir), file_bytes(&bart_dir))
    else {
        eprintln!("skipping: needs whisper-small and distilbart-cnn downloaded");
        return;
    };

    let before = rss_bytes();
    let peak_before = peak_bytes();
    let transcriber = kjarni::transcriber::Transcriber::builder("whisper-small")
        .quiet()
        .build()
        .await
        .expect("load whisper-small");
    let whisper = rss_bytes() - before;
    let ratio = whisper as f64 / whisper_file as f64;
    eprintln!(
        "whisper-small: {:.0} MB for a {:.0} MB file ({ratio:.2}x)",
        whisper as f64 / 1e6,
        whisper_file as f64 / 1e6
    );
    assert!(
        ratio < MAX_RATIO,
        "whisper-small holds {ratio:.2}x its weight file"
    );
    // Only meaningful for the first model: the peak is a high-water mark.
    let peak = (peak_bytes() - peak_before.max(before)) as f64 / whisper_file as f64;
    eprintln!("whisper-small peak while loading: {peak:.2}x its file");
    assert!(
        peak < MAX_PEAK_RATIO,
        "whisper-small peaked at {peak:.2}x its weight file"
    );
    drop(transcriber);

    let before = rss_bytes();
    let summarizer = kjarni::Summarizer::builder("distilbart-cnn")
        .quiet()
        .build()
        .await
        .expect("load distilbart-cnn");
    let bart = rss_bytes().saturating_sub(before);
    let ratio = bart as f64 / bart_file as f64;
    eprintln!(
        "distilbart-cnn: {:.0} MB for a {:.0} MB file ({ratio:.2}x)",
        bart as f64 / 1e6,
        bart_file as f64 / 1e6
    );
    assert!(
        ratio < MAX_RATIO,
        "distilbart-cnn holds {ratio:.2}x its weight file"
    );
    drop(summarizer);
}
