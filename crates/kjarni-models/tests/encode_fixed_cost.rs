//! Where does an 18 token encode spend 5 milliseconds?
//!
//! That input is about 0.02 GFLOP, which is microseconds of arithmetic, so almost
//! all of it is fixed per-call cost. This splits a single encode into tokenize,
//! embed, buffer creation and forward, so the fixed part can be named rather than
//! guessed at.

use std::time::Instant;

use kjarni_models::SentenceEncoder;
use kjarni_transformers::cpu::encoder::traits::EncoderLanguageModel;
use kjarni_transformers::models::get_default_cache_dir;
use kjarni_transformers::traits::Device;

#[tokio::test]
#[cfg_attr(debug_assertions, ignore = "timings are meaningless unoptimised")]
async fn where_a_small_encode_goes() {
    let dir = get_default_cache_dir().join("sentence-transformers_all-MiniLM-L6-v2");
    if !dir.join("model.safetensors").exists() {
        return;
    }
    let encoder =
        SentenceEncoder::from_pretrained(&dir, Device::Cpu, None, None, None).expect("load");
    let ops = encoder.encoder_cpu_ops().expect("cpu ops");
    let cpu = ops.encoder();

    for (label, sentences) in [("18 tokens", 1usize), ("216 tokens", 12)] {
        let text: String = (0..sentences)
            .map(|j| format!("Document number 0 sentence {j} covers refunds, delivery windows and account settings in some detail. "))
            .collect();
        let refs = [text.as_str()];

        const RUNS: usize = 20;
        let _ = encoder.encode(&text).await.expect("warmup");

        // Whole call, for reference.
        let t = Instant::now();
        for _ in 0..RUNS {
            let _ = encoder.encode(&text).await.expect("encode");
        }
        let whole = t.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;

        // Tokenize.
        let t = Instant::now();
        for _ in 0..RUNS {
            let _ = encoder.encode_batch_texts(&refs).expect("tokenize");
        }
        let tokenize = t.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;

        let encoded = encoder.encode_batch_texts(&refs).expect("tokenize");
        let seq = encoded[0].get_ids().len();
        let mut ids = ndarray::Array2::<u32>::zeros((1, seq));
        let mut mask = ndarray::Array2::<f32>::zeros((1, seq));
        for (i, &id) in encoded[0].get_ids().iter().enumerate() {
            ids[[0, i]] = id;
            mask[[0, i]] = encoded[0].get_attention_mask()[i] as f32;
        }

        // Embedding lookup plus its norm.
        let t = Instant::now();
        for _ in 0..RUNS {
            let h = ops.embed_tokens(&ids, None, 0).expect("embed");
            let _ = cpu.embed_norm(&h).expect("embed_norm");
        }
        let embed = t.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;

        // Buffer creation, which happens on every single call.
        let t = Instant::now();
        for _ in 0..RUNS {
            let _ = cpu.create_buffers(1, seq);
        }
        let buffers = t.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;

        // The forward pass itself, buffers already in hand.
        let hidden = ops.embed_tokens(&ids, None, 0).expect("embed");
        let normalized = cpu.embed_norm(&hidden).expect("embed_norm");
        let mut bufs = cpu.create_buffers(1, seq);
        let t = Instant::now();
        for _ in 0..RUNS {
            let _ = cpu
                .forward_with_buffers(&normalized, &mask, &mut bufs)
                .expect("forward");
        }
        let forward = t.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;

        eprintln!("\n  {label} ({seq} padded)");
        eprintln!("    whole encode()      {whole:>8.3} ms");
        eprintln!("    tokenize            {tokenize:>8.3} ms");
        eprintln!("    embed + norm        {embed:>8.3} ms");
        eprintln!("    create_buffers      {buffers:>8.3} ms");
        eprintln!("    forward_with_buffers{forward:>8.3} ms");
        eprintln!(
            "    unaccounted         {:>8.3} ms",
            whole - tokenize - embed - buffers - forward
        );
    }
    eprintln!();
}
