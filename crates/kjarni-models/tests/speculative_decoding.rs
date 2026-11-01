//! Does the speculative decoding path work?
//!
//! Nothing in the public API sets `speculation: Some(..)`, so these tests are what
//! exercises `run_speculative_generation_loop`. Qwen2.5-0.5B and 1.5B share a
//! vocabulary (151,936), which is what a draft model needs.

use kjarni_models::models::llama::LlamaModel;
use kjarni_models::models::qwen::QwenModel;
use kjarni_transformers::common::stream::TokenType;
use kjarni_transformers::common::{DecodingStrategy, GenerationConfig, SpeculationParams};
use kjarni_transformers::cpu::decoder::{
    CpuDecoderBackend, SpeculationStats, run_prompt_lookup_generation_loop,
    run_speculative_generation_loop,
};
use kjarni_transformers::decoder::backend::AnyDecoderBackend;
use kjarni_transformers::decoder::generator::run_generation_loop;
use kjarni_transformers::decoder::traits::DecoderLanguageModel;
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::models::registry::ModelType;
use kjarni_transformers::{Device, LanguageModel};
use std::sync::Arc;
use std::time::Instant;

type DynModel = Arc<dyn DecoderLanguageModel + Send + Sync>;

async fn load(mt: ModelType) -> Arc<QwenModel> {
    Arc::new(
        QwenModel::from_registry(mt, None, Device::Cpu, None, None)
            .await
            .expect("load"),
    )
}

fn greedy(n: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: Some(n),
        strategy: DecodingStrategy::Greedy,
        ..Default::default()
    }
}

/// Greedy with the repetition penalty the CLI actually defaults to.
///
/// Speculation has to apply `repetition_penalty` too, or `--draft` produces
/// different, more repetitive text than the same command without it. The default
/// config has penalty 1.0, which would hide that.
fn greedy_penalised(n: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: Some(n),
        strategy: DecodingStrategy::Greedy,
        repetition_penalty: 1.1,
        ..Default::default()
    }
}

async fn collect(
    mut rx: tokio::sync::mpsc::Receiver<anyhow::Result<kjarni_transformers::common::StreamedToken>>,
) -> String {
    let mut out = String::new();
    while let Some(Ok(t)) = rx.recv().await {
        if t.token_type != TokenType::Prompt {
            out.push_str(&t.text);
        }
    }
    out
}

#[tokio::test]
#[ignore = "loads two real models from the local cache"]
async fn speculative_matches_plain_greedy() {
    let target = load(ModelType::Qwen2_5_1_5B_Instruct).await;
    let draft = load(ModelType::Qwen2_5_0_5B_Instruct).await;

    assert_eq!(
        target.vocab_size(),
        draft.vocab_size(),
        "a draft model must share the target's vocabulary"
    );

    let prompt = "The capital of France is Paris. The capital of Germany is";
    let ids: Vec<u32> = target
        .tokenizer()
        .encode(prompt, false)
        .unwrap()
        .get_ids()
        .to_vec();

    // plain decoding, for both the answer and the time to beat
    let (tx, rx) = tokio::sync::mpsc::channel(256);
    let t0 = Instant::now();
    let m: Arc<dyn DecoderLanguageModel + Send + Sync> = target.clone();
    let run = run_generation_loop(
        m,
        AnyDecoderBackend::cpu(),
        ids.clone(),
        greedy(24),
        tx,
        None,
    );
    let (r, plain) = tokio::join!(run, collect(rx));
    r.expect("plain generation");
    let t_plain = t0.elapsed().as_secs_f64();

    // speculative, greedy acceptance so it must reproduce the same text exactly
    let (tx2, rx2) = tokio::sync::mpsc::channel(256);
    let spec = SpeculationParams {
        num_tokens: 4,
        prompt_lookup: None,
    };
    let t1 = Instant::now();
    let tm: Arc<dyn DecoderLanguageModel + Send + Sync> = target.clone();
    let dm: Arc<dyn DecoderLanguageModel + Send + Sync> = draft.clone();
    let run2 = run_speculative_generation_loop(
        tm,
        AnyDecoderBackend::Cpu(CpuDecoderBackend::new()),
        dm,
        AnyDecoderBackend::Cpu(CpuDecoderBackend::new()),
        ids,
        greedy(24),
        &spec,
        tx2,
        None,
    );
    let (r2, spec_text) = tokio::join!(run2, collect(rx2));
    r2.expect("speculative generation");
    let t_spec = t1.elapsed().as_secs_f64();

    println!("\n  plain        {t_plain:>6.2}s  {plain:?}");
    println!("  speculative  {t_spec:>6.2}s  {spec_text:?}");
    println!("  speedup      {:.2}x", t_plain / t_spec);

    // A speculative round accepts a batch at once, so it can overshoot
    // max_new_tokens by up to num_tokens. Compare the shared prefix.
    let n = plain.len().min(spec_text.len());
    assert_eq!(
        &plain[..n],
        &spec_text[..n],
        "greedy speculative decoding must reproduce plain greedy output"
    );
    assert!(n > 0, "produced nothing");
}

#[tokio::test]
#[ignore = "loads two real models from the local cache"]
async fn speculation_depth_sweep() {
    let target = load(ModelType::Qwen2_5_1_5B_Instruct).await;
    let draft = load(ModelType::Qwen2_5_0_5B_Instruct).await;

    let prompt = "Write a short list of European capitals. The capital of France is";
    let ids: Vec<u32> = target
        .tokenizer()
        .encode(prompt, false)
        .unwrap()
        .get_ids()
        .to_vec();

    let (tx, rx) = tokio::sync::mpsc::channel(512);
    let t0 = Instant::now();
    let m: Arc<dyn DecoderLanguageModel + Send + Sync> = target.clone();
    let run = run_generation_loop(
        m,
        AnyDecoderBackend::cpu(),
        ids.clone(),
        greedy(48),
        tx,
        None,
    );
    let (r, _) = tokio::join!(run, collect(rx));
    r.expect("plain");
    let t_plain = t0.elapsed().as_secs_f64();
    println!(
        "\n  plain 1.5B         {t_plain:>6.2}s   {:.1} tok/s",
        48.0 / t_plain
    );
    println!(
        "\n  {:>6}  {:>9}  {:>10}  {:>8}",
        "draft k", "time", "tok/s", "speedup"
    );

    for k in [2usize, 4, 6, 8] {
        let (tx2, rx2) = tokio::sync::mpsc::channel(512);
        let spec = SpeculationParams {
            num_tokens: k,
            prompt_lookup: None,
        };
        let t1 = Instant::now();
        let tm: Arc<dyn DecoderLanguageModel + Send + Sync> = target.clone();
        let dm: Arc<dyn DecoderLanguageModel + Send + Sync> = draft.clone();
        let run2 = run_speculative_generation_loop(
            tm,
            AnyDecoderBackend::Cpu(CpuDecoderBackend::new()),
            dm,
            AnyDecoderBackend::Cpu(CpuDecoderBackend::new()),
            ids.clone(),
            greedy(48),
            &spec,
            tx2,
            None,
        );
        let (r2, _text) = tokio::join!(run2, collect(rx2));
        r2.expect("speculative");
        let t = t1.elapsed().as_secs_f64();
        println!(
            "  {:>6}  {:>8.2}s  {:>9.1}  {:>7.2}x",
            k,
            t,
            48.0 / t,
            t_plain / t
        );
    }
}

#[tokio::test]
#[ignore = "loads two real models from the local cache"]
async fn speculation_respects_repetition_penalty() {
    let target = load(ModelType::Qwen2_5_1_5B_Instruct).await;
    let draft = load(ModelType::Qwen2_5_0_5B_Instruct).await;

    let prompt = "The capital of France is Paris. The capital of Germany is";
    let ids: Vec<u32> = target
        .tokenizer()
        .encode(prompt, false)
        .unwrap()
        .get_ids()
        .to_vec();

    let (tx, rx) = tokio::sync::mpsc::channel(256);
    let m: Arc<dyn DecoderLanguageModel + Send + Sync> = target.clone();
    let run = run_generation_loop(
        m,
        AnyDecoderBackend::cpu(),
        ids.clone(),
        greedy_penalised(24),
        tx,
        None,
    );
    let (r, plain) = tokio::join!(run, collect(rx));
    r.expect("plain");

    let (tx2, rx2) = tokio::sync::mpsc::channel(256);
    let spec = SpeculationParams {
        num_tokens: 4,
        prompt_lookup: None,
    };
    let tm: Arc<dyn DecoderLanguageModel + Send + Sync> = target.clone();
    let dm: Arc<dyn DecoderLanguageModel + Send + Sync> = draft.clone();
    let run2 = run_speculative_generation_loop(
        tm,
        AnyDecoderBackend::Cpu(CpuDecoderBackend::new()),
        dm,
        AnyDecoderBackend::Cpu(CpuDecoderBackend::new()),
        ids,
        greedy_penalised(24),
        &spec,
        tx2,
        None,
    );
    let (r2, spec_text) = tokio::join!(run2, collect(rx2));
    r2.expect("speculative");

    println!("\n  plain        {plain:?}");
    println!("  speculative  {spec_text:?}");

    let n = plain.len().min(spec_text.len());
    assert_eq!(
        &plain[..n],
        &spec_text[..n],
        "speculation must apply the same logit penalties as plain decoding"
    );
}

#[tokio::test]
#[ignore = "loads two real models from the local cache"]
async fn speculation_does_not_emit_stop_tokens() {
    let target = load(ModelType::Qwen2_5_1_5B_Instruct).await;
    let draft = load(ModelType::Qwen2_5_0_5B_Instruct).await;

    // A prompt short enough that the model finishes and hits its stop token
    // well inside the budget, so the stop path is actually exercised.
    let prompt = "<|im_start|>user\nWhat is the capital of Spain? Answer in one \
                  sentence.<|im_end|>\n<|im_start|>assistant\n";
    let ids: Vec<u32> = target
        .tokenizer()
        .encode(prompt, false)
        .unwrap()
        .get_ids()
        .to_vec();

    let (tx, rx) = tokio::sync::mpsc::channel(256);
    let spec = SpeculationParams {
        num_tokens: 6,
        prompt_lookup: None,
    };
    let tm: Arc<dyn DecoderLanguageModel + Send + Sync> = target.clone();
    let dm: Arc<dyn DecoderLanguageModel + Send + Sync> = draft.clone();
    let run = run_speculative_generation_loop(
        tm,
        AnyDecoderBackend::Cpu(CpuDecoderBackend::new()),
        dm,
        AnyDecoderBackend::Cpu(CpuDecoderBackend::new()),
        ids,
        greedy(40),
        &spec,
        tx,
        None,
    );
    let (r, text) = tokio::join!(run, collect(rx));
    r.expect("speculative");

    println!("\n  {text:?}");
    // The stop check ran after the send, so every speculative reply ended with
    // the terminator as literal text.
    for marker in ["<|im_end|>", "<|endoftext|>"] {
        assert!(
            !text.contains(marker),
            "a stop token reached the stream as text: {text:?}"
        );
    }
    assert!(!text.is_empty(), "generated nothing");
}

/// What a run produced and how long it took.
struct Run {
    text: String,
    secs: f64,
    stats: Option<SpeculationStats>,
}

fn print_run(name: &str, run: &Run, plain_secs: f64, tokens: usize) {
    let (per_pass, accepted) = run.stats.map_or((1.0, String::from("-")), |s| {
        (
            s.tokens_per_pass(),
            format!("{}/{}", s.accepted, s.proposed),
        )
    });
    println!(
        "  {name:<22} {:>7.2}s {:>7.1} tok/s {:>6.2}x {per_pass:>6.2} tok/pass  accepted {accepted}",
        run.secs,
        tokens as f64 / run.secs,
        plain_secs / run.secs,
    );
}

async fn run_plain(model: &DynModel, ids: &[u32], config: GenerationConfig) -> Run {
    let (tx, rx) = tokio::sync::mpsc::channel(4096);
    let start = Instant::now();
    let run = run_generation_loop(
        model.clone(),
        AnyDecoderBackend::for_model(model.as_ref()).unwrap(),
        ids.to_vec(),
        config,
        tx,
        None,
    );
    let (result, text) = tokio::join!(run, collect(rx));
    result.expect("plain generation");
    Run {
        text,
        secs: start.elapsed().as_secs_f64(),
        stats: None,
    }
}

async fn run_draft(
    model: &DynModel,
    draft: &DynModel,
    ids: &[u32],
    config: GenerationConfig,
    depth: usize,
) -> Run {
    let (tx, rx) = tokio::sync::mpsc::channel(4096);
    let start = Instant::now();
    let spec = SpeculationParams {
        num_tokens: depth,
        prompt_lookup: None,
    };
    let run = run_speculative_generation_loop(
        model.clone(),
        AnyDecoderBackend::for_model(model.as_ref()).unwrap(),
        draft.clone(),
        AnyDecoderBackend::for_model(draft.as_ref()).unwrap(),
        ids.to_vec(),
        config,
        &spec,
        tx,
        None,
    );
    let (result, text) = tokio::join!(run, collect(rx));
    Run {
        text,
        secs: start.elapsed().as_secs_f64(),
        stats: Some(result.expect("speculative generation")),
    }
}

async fn run_lookup(model: &DynModel, ids: &[u32], config: GenerationConfig, depth: usize) -> Run {
    let (tx, rx) = tokio::sync::mpsc::channel(4096);
    let start = Instant::now();
    let spec = SpeculationParams {
        num_tokens: depth,
        prompt_lookup: Some(3),
    };
    let run = run_prompt_lookup_generation_loop(
        model.clone(),
        AnyDecoderBackend::for_model(model.as_ref()).unwrap(),
        ids.to_vec(),
        config,
        &spec,
        3,
        tx,
        None,
    );
    let (result, text) = tokio::join!(run, collect(rx));
    Run {
        text,
        secs: start.elapsed().as_secs_f64(),
        stats: Some(result.expect("prompt lookup generation")),
    }
}

async fn load_qwen(mt: ModelType, device: Device) -> DynModel {
    Arc::new(
        QwenModel::from_registry(mt, None, device, None, None)
            .await
            .expect("load"),
    )
}

fn encode(model: &DynModel, prompt: &str) -> Vec<u32> {
    model
        .tokenizer()
        .encode(prompt, false)
        .unwrap()
        .get_ids()
        .to_vec()
}

const FREE_PROMPT: &str = "The capital of France is Paris. The capital of Germany is";
const COPY_PROMPT: &str = "Fix the typo in this sentence and repeat the rest exactly.\n\
    Input: The maintenance team replaced the worn seal on the second pump, tested the \
    pressure valve, and wrote the rezults in the log before the night shift arrived.\n\
    Output: The maintenance team replaced";

/// Plain greedy text is the reference: a speculative run must reproduce it exactly, at
/// every depth, on the device under test. The penalised config covers the penalties that
/// differ by row within one verifying pass.
async fn assert_speculation_is_exact(device: Device) {
    let target = load_qwen(ModelType::Qwen2_5_1_5B_Instruct, device).await;
    let draft = load_qwen(ModelType::Qwen2_5_0_5B_Instruct, device).await;

    for prompt in [FREE_PROMPT, COPY_PROMPT] {
        let ids = encode(&target, prompt);
        for config in [greedy(40), greedy_penalised(40)] {
            let plain = run_plain(&target, &ids, config.clone()).await;
            for depth in [1, 4, 8] {
                let drafted = run_draft(&target, &draft, &ids, config.clone(), depth).await;
                assert_eq!(drafted.text, plain.text, "draft, depth {depth}");
                let looked_up = run_lookup(&target, &ids, config.clone(), depth).await;
                assert_eq!(looked_up.text, plain.text, "lookup, depth {depth}");
            }
        }
    }
}

#[tokio::test]
#[ignore = "loads two real models from the local cache"]
async fn speculation_is_exact_on_the_cpu() {
    assert_speculation_is_exact(Device::Cpu).await;
}

#[tokio::test]
#[ignore = "needs a GPU and two real models in the local cache"]
async fn speculation_is_exact_on_the_gpu() {
    assert_speculation_is_exact(Device::Wgpu).await;
}

/// A draft on the CPU can serve a target on the GPU, which leaves GPU memory to the target.
#[tokio::test]
#[ignore = "needs a GPU and two real models in the local cache"]
async fn a_cpu_draft_can_serve_a_gpu_target() {
    let target = load_qwen(ModelType::Qwen2_5_1_5B_Instruct, Device::Wgpu).await;
    let draft = load_qwen(ModelType::Qwen2_5_0_5B_Instruct, Device::Cpu).await;
    let ids = encode(&target, FREE_PROMPT);
    let plain = run_plain(&target, &ids, greedy(40)).await;
    let drafted = run_draft(&target, &draft, &ids, greedy(40), 4).await;
    assert_eq!(drafted.text, plain.text);
}

/// A run stops at `max_new_tokens` even when a round would have accepted more.
#[tokio::test]
#[ignore = "loads a real model from the local cache"]
async fn speculation_stops_at_the_token_limit() {
    let target = load_qwen(ModelType::Qwen2_5_0_5B_Instruct, Device::Cpu).await;
    let ids = encode(&target, COPY_PROMPT);
    for limit in [1, 2, 7, 10] {
        let run = run_lookup(&target, &ids, greedy(limit), 8).await;
        assert_eq!(run.stats.unwrap().generated, limit, "limit {limit}");
    }
}

/// Prints what one pass costs by batch size, with the cache at a fixed length.
#[tokio::test]
#[ignore = "profile over a real model"]
async fn speculation_pass_cost() {
    use kjarni_transformers::decoder::traits::DecoderGenerationBackend;

    let device = bench_device();
    let target = load_bench_model(&bench_target(), device).await;
    let backend = AnyDecoderBackend::for_model(target.as_ref()).unwrap();
    let ids = encode(&target, COPY_PROMPT);
    let mut cache = target.new_cache(1, 400, 1).unwrap();
    let array = ndarray::Array2::from_shape_vec((1, ids.len()), ids.clone()).unwrap();
    backend
        .prefill(target.as_ref(), &array, cache.as_mut())
        .await
        .unwrap();
    let len = ids.len();
    let mut token = backend.new_decode_token().unwrap();
    backend.update_decode_token(&mut token, ids[0]).unwrap();

    for _ in 0..2 {
        let start = Instant::now();
        for _ in 0..5 {
            cache.set_seq_length(len);
            backend
                .decode_one(target.as_ref(), &token, len + 1, cache.as_mut())
                .await
                .unwrap();
        }
        let one = start.elapsed().as_secs_f64() * 200.0;
        println!("{device:?} decode_one        {one:>8.1} ms");
        for n in [1usize, 2, 5, 9] {
            let start = Instant::now();
            for _ in 0..5 {
                cache.set_seq_length(len);
                backend
                    .forward_batch(target.as_ref(), &ids[..n], cache.as_mut())
                    .await
                    .unwrap();
            }
            let ms = start.elapsed().as_secs_f64() * 200.0;
            println!(
                "{device:?} forward_batch {n}   {ms:>8.1} ms, {:>5.1} ms per row",
                ms / n as f64
            );
        }
    }
}

fn bench_device() -> Device {
    match std::env::var("KJARNI_BENCH_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu,
        _ => Device::Wgpu,
    }
}

fn bench_target() -> String {
    std::env::var("KJARNI_BENCH_TARGET").unwrap_or_else(|_| "qwen2.5-1.5b-instruct".into())
}

fn bench_draft() -> String {
    std::env::var("KJARNI_BENCH_DRAFT").unwrap_or_else(|_| "qwen2.5-0.5b-instruct".into())
}

async fn load_bench_model(name: &str, device: Device) -> DynModel {
    let model_type = ModelType::from_cli_name(name).expect("a model name from `kjarni model list`");
    let config = std::env::var("KJARNI_BENCH_GGUF")
        .is_ok()
        .then(|| ModelLoadConfig {
            use_gguf: true,
            ..Default::default()
        });
    if name.starts_with("qwen") {
        Arc::new(
            QwenModel::from_registry(model_type, None, device, None, config)
                .await
                .expect("load"),
        )
    } else {
        Arc::new(
            LlamaModel::from_registry(model_type, None, device, None, config)
                .await
                .expect("load"),
        )
    }
}

/// Prints the table: `cargo test --profile ci -p kjarni-models --test speculative_decoding
/// speculation_benchmark -- --ignored --nocapture`. `KJARNI_BENCH_DEVICE=cpu` picks the CPU,
/// `KJARNI_BENCH_TARGET` and `KJARNI_BENCH_DRAFT` the models, `KJARNI_BENCH_TOKENS` the length.
/// `KJARNI_BENCH_GGUF=1` loads the quantized weights.
#[tokio::test]
#[ignore = "benchmark over real models"]
async fn speculation_benchmark() {
    let device = bench_device();
    let tokens = std::env::var("KJARNI_BENCH_TOKENS").map_or(60, |t| t.parse().unwrap());
    let target = load_bench_model(&bench_target(), device).await;
    // "none" measures prompt lookup alone, for a target with no draft of its family.
    let draft_name = bench_draft();
    let draft = if draft_name == "none" {
        None
    } else {
        Some(load_bench_model(&draft_name, device).await)
    };

    for (name, prompt) in [("free text", FREE_PROMPT), ("copying", COPY_PROMPT)] {
        println!("\n{device:?}, {name}, {tokens} tokens");
        let ids = encode(&target, prompt);

        // The first runs warm up shaders and the page cache, and are not reported.
        run_plain(&target, &ids, greedy(4)).await;

        let plain = run_plain(&target, &ids, greedy(tokens)).await;
        print_run("plain", &plain, plain.secs, tokens);
        if let Some(draft) = &draft {
            run_draft(&target, draft, &ids, greedy(4), 4).await;
            for depth in [2, 4, 8] {
                let run = run_draft(&target, draft, &ids, greedy(tokens), depth).await;
                print_run(&format!("draft, depth {depth}"), &run, plain.secs, tokens);
                assert_eq!(run.text, plain.text);
            }
        }
        for depth in [4, 8] {
            let run = run_lookup(&target, &ids, greedy(tokens), depth).await;
            print_run(&format!("lookup, depth {depth}"), &run, plain.secs, tokens);
            assert_eq!(run.text, plain.text);
        }
    }
}
