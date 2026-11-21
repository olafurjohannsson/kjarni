use crate::models::llama::config::LlamaConfig;
use crate::models::llama::model::LlamaModel;
use anyhow::Result;
use kjarni_transformers::common::{DecodingStrategy, GenerationConfig};
use kjarni_transformers::decoder::prelude::*;
use kjarni_transformers::models::LanguageModel;
use kjarni_transformers::models::registry::model_cache_dir;
use kjarni_transformers::prelude::*;
use std::sync::Arc;

const LLAMA_32_1B: &str = "meta-llama_Llama-3.2-1B";
const LLAMA_32_8B: &str = "meta-llama_Llama-3.2-8B-Instruct";

/// Helper function to load the Llama model for testing.
async fn load_llama_for_test() -> Result<LlamaModel> {
    let p = model_cache_dir(LLAMA_32_1B);
    let path = p.as_path();
    LlamaModel::from_pretrained(path, Device::Cpu, None, None, None)
}

async fn load_llama_8b_for_test() -> Result<LlamaModel> {
    let p = model_cache_dir(LLAMA_32_8B);
    let path = p.as_path();
    LlamaModel::from_pretrained(path, Device::Cpu, None, None, None)
}

#[ignore = "Requires large model"]
#[tokio::test]
async fn test_llama3_8b_architectural_properties() -> Result<()> {
    if !std::path::Path::new("meta-llama_Llama-3.2-8B-Instruct").exists() {
        log::warn!("Skipping Llama-3.2-8B test since model files not found in cache.");
        return Ok(());
    }
    let model = load_llama_8b_for_test().await?;
    let config = model.config();

    assert_eq!(config.vocab_size, 128256);
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_hidden_layers, 32);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 8); // GQA
    assert_eq!(config.intermediate_size, 14336);
    assert_eq!(config.max_position_embeddings, 8192);
    assert_eq!(config.rope_theta, 500000.0);
    assert_eq!(config.rms_norm_eps, 1e-5);

    assert_eq!(
        model.vocab_size(),
        config.vocab_size,
        "vocab_size mismatch - check for hardcoded value"
    );
    assert_eq!(
        model.hidden_size(),
        config.hidden_size,
        "hidden_size mismatch - check for hardcoded value"
    );
    assert_eq!(
        model.num_layers(),
        config.num_hidden_layers,
        "num_layers mismatch - check for hardcoded value"
    );
    assert_eq!(
        model.num_heads(),
        config.num_attention_heads,
        "num_heads mismatch - check for hardcoded value"
    );

    // Check token IDs match config.json
    assert_eq!(model.bos_token_id(), Some(128000));
    assert_eq!(model.eos_token_id(), Some(128009));

    Ok(())
}

#[ignore = "Requires large model"]
#[tokio::test]
async fn test_llama_trait_config_consistency() -> Result<()> {
    let model_1b = load_llama_for_test().await?;
    assert_trait_matches_config(&model_1b, "Llama-3.2-1B");

    if let Ok(model_8b) = load_llama_8b_for_test().await {
        assert_trait_matches_config(&model_8b, "Llama-3-8B");
    }

    Ok(())
}

fn assert_trait_matches_config(model: &LlamaModel, name: &str) {
    let config = model.config();

    assert_eq!(
        model.vocab_size(),
        config.vocab_size,
        "{}: vocab_size() should come from config",
        name
    );
    assert_eq!(
        model.hidden_size(),
        config.hidden_size,
        "{}: hidden_size() should come from config",
        name
    );
    assert_eq!(
        model.num_layers(),
        config.num_hidden_layers,
        "{}: num_layers() should come from config",
        name
    );
    assert_eq!(
        model.num_heads(),
        config.num_attention_heads,
        "{}: num_heads() should come from config",
        name
    );
}

#[test]
fn test_llama_config_parsing_8b() {
    let json = r#"{
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "vocab_size": 128256,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-05,
        "rope_theta": 500000.0,
        "bos_token_id": 128000,
        "eos_token_id": 128009,
        "tie_word_embeddings": false
    }"#;

    let config = LlamaConfig::from_json(json).unwrap();

    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_hidden_layers, 32);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.intermediate_size, 14336);
    assert_eq!(config.get_head_dim(), 128); // 4096 / 32
    assert_eq!(config.get_kv_dim(), 1024); // 8 * 128
    // assert!(config.uses_gqa());
}

#[test]
fn test_llama_config_parsing_1b() {
    let json = r#"{
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 8192,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-05,
        "rope_theta": 500000.0,
        "bos_token_id": 128000,
        "eos_token_id": 128001
    }"#;

    let config = LlamaConfig::from_json(json).unwrap();

    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.num_hidden_layers, 16);
    assert_eq!(config.get_head_dim(), 64); // 2048 / 32
    assert_eq!(config.get_kv_dim(), 512); // 8 * 64
}

#[ignore = "Requires large model"]
#[tokio::test]
async fn test_llama3_2_1b_architectural_properties() -> Result<()> {
    {
        let model = load_llama_for_test().await?;
        let config = model.config();

        assert_eq!(config.vocab_size, 128256);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 16);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8); // GQA
        assert_eq!(config.max_position_embeddings, 131072);
        assert_eq!(config.rope_theta, 500000.0);

        assert_eq!(model.vocab_size(), 128256);
        assert_eq!(model.hidden_size(), 2048);
        assert_eq!(model.num_layers(), 16);
        assert_eq!(model.num_heads(), 32);
        assert_eq!(model.max_length(), 131072);

        assert_eq!(model.bos_token_id(), Some(128000));
        assert_eq!(model.eos_token_id(), Some(128001));
        assert_eq!(model.pad_token_id(), None);
    }
    kjarni_transformers::weights::clear_mmap_cache();
    Ok(())
}

#[ignore = "GPU required"]
#[tokio::test]
async fn test_llama3_2_1b_generation_parity() -> Result<()> {
    let prompt = "The field of Artificial Intelligence has seen a lot of progress";
    let expected_output =
        "The field of Artificial Intelligence has seen a lot of progress in the last few years.";

    let config = GenerationConfig {
        max_new_tokens: Some(6),
        strategy: DecodingStrategy::Greedy,
        repetition_penalty: 1.0,
        add_bos_token: true,
        ..Default::default()
    };
    let p = model_cache_dir(LLAMA_32_1B);
    let path = p.as_path();
    {
        let llama_model = LlamaModel::from_pretrained(path, Device::Cpu, None, None, None)?;
        let _pp = model_cache_dir(LLAMA_32_1B);
        let path22 = p.as_path();
        let llama_gpu = LlamaModel::from_pretrained(
            path22,
            Device::Wgpu,
            Some(WgpuContext::new().await?),
            None,
            None,
        )?;
        let generator_gpu = DecoderGenerator::new(Arc::new(llama_gpu))?;
        let generated_text_gpu = generator_gpu.generate(prompt, &config, None).await?;
        let concat_prompt_gpu = prompt.to_string() + "" + &generated_text_gpu;
        assert_eq!(concat_prompt_gpu.trim(), expected_output.trim());

        let generator = DecoderGenerator::new(Arc::new(llama_model))?;
        let generated_text = generator.generate(prompt, &config, None).await?;

        let concat_prompt = prompt.to_string() + "" + &generated_text;
        assert_eq!(concat_prompt.trim(), expected_output.trim());
    }
    kjarni_transformers::weights::clear_mmap_cache();
    Ok(())
}

/// Times every GPU kernel of a 512-token prefill and of four decode steps of
/// Llama 3.2 1B from its GGUF, totalled per kernel. A tool rather than a check:
/// run it with `--features kjarni-transformers/profiling --release -- --nocapture`.
#[tokio::test]
#[ignore = "a profile: needs a GPU, the 1B Instruct GGUF and the profiling feature"]
async fn profile_the_1b_on_the_gpu() -> Result<()> {
    use kjarni_transformers::cache::Cache;
    use ndarray::Array2;

    let path = model_cache_dir("meta-llama_Llama-3.2-1B-Instruct").join("model.gguf");
    let ctx = WgpuContext::new().await?;
    let m = LlamaModel::from_pretrained(&path, Device::Wgpu, Some(ctx.clone()), None, None)?;
    let backend = GpuDecoderBackend::new(ctx.clone())?;
    let context: usize = 512;
    let ids: Vec<u32> = (0..context as u32)
        .map(|i| 1000 + (i * 37) % 20000)
        .collect();
    let prompt = Array2::from_shape_vec((1, context), ids)?;
    let argmax = |l: &ndarray::Array1<f32>| {
        l.iter()
            .enumerate()
            .fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
            .0 as u32
    };
    let report = |name: &str, totals: Vec<(String, f64, usize)>, wall: f64| {
        let sum: f64 = totals.iter().map(|t| t.1).sum();
        println!("== {name}: wall {wall:.1} ms, timed GPU {sum:.1} ms");
        for (l, ms, n) in totals.iter().take(18) {
            println!("{:>8.1} ms {:>5.1}% {:>5}x  {l}", ms, 100.0 * ms / sum, n);
        }
    };
    let flush = |c: &Arc<WgpuContext>| {
        let mut enc = c.device.create_command_encoder(&Default::default());
        c.profiler.process_results(&mut enc);
        c.queue.submit(Some(enc.finish()));
    };
    // A warm-up, whose timings are dropped.
    let mut cache: Box<dyn Cache> = m.new_cache(1, context + 8, 1)?;
    backend.prefill(&m, &prompt, cache.as_mut()).await?;
    flush(&ctx);
    ctx.profiler.totals(&ctx).await;

    let mut cache = m.new_cache(1, context + 8, 1)?;
    let t = std::time::Instant::now();
    let l = backend.prefill(&m, &prompt, cache.as_mut()).await?;
    let wall = t.elapsed().as_secs_f64() * 1e3;
    flush(&ctx);
    report("prefill 512", ctx.profiler.totals(&ctx).await, wall);

    let mut next = argmax(&l);
    let mut tok = backend.new_decode_token()?;
    let t = std::time::Instant::now();
    for i in 0..4 {
        backend.update_decode_token(&mut tok, next)?;
        let l = backend
            .decode_one(&m, &tok, context + 1 + i, cache.as_mut())
            .await?;
        next = argmax(&l);
    }
    let wall = t.elapsed().as_secs_f64() * 1e3;
    flush(&ctx);
    report("decode 4 tokens", ctx.profiler.totals(&ctx).await, wall);
    Ok(())
}
