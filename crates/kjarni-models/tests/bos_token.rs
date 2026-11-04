//! Whether a prompt starts with BOS is the model's decision, not the config's.
//!
//! `GenerationConfig::default()` has `add_bos_token: true`, and the CLI and the
//! browser build their configs from that default. Qwen's config.json names a BOS
//! id (151643, `<|endoftext|>`) that its tokenizer never adds, so before the model
//! decided, every Qwen prompt got one extra token in front. Llama's tokenizer does
//! add `<|begin_of_text|>`, so it must keep getting it.
//!
//! Token ids are from the HuggingFace tokenizers for the same prompt.
//!
//!     cargo test --release -p kjarni-models --test bos_token

use std::path::PathBuf;
use std::sync::Arc;

use kjarni_models::models::llama::LlamaModel;
use kjarni_models::models::qwen::QwenModel;
use kjarni_transformers::common::GenerationConfig;
use kjarni_transformers::decoder::generator::DecoderGenerator;
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::pipeline::DecoderLoader;
use kjarni_transformers::tensor::DType;
use kjarni_transformers::traits::Device;

const PROMPT: &str = "The capital of Iceland is";

fn cached(name: &str) -> Option<PathBuf> {
    let d = kjarni_transformers::models::get_default_cache_dir().join(name);
    d.join("model.safetensors").exists().then_some(d)
}

fn f32_config() -> ModelLoadConfig {
    ModelLoadConfig {
        target_dtype: Some(DType::F32),
        ..Default::default()
    }
}

#[test]
fn qwen_prompt_gets_no_bos_under_the_default_config() {
    let Some(dir) = cached("Qwen_Qwen2.5-0.5B-Instruct") else {
        eprintln!("skipping: qwen2.5-0.5b not cached");
        return;
    };
    let model: QwenModel =
        DecoderLoader::load_from_pretrained(&dir, Device::Cpu, None, Some(f32_config()), None)
            .expect("load qwen");
    let generator = DecoderGenerator::new(Arc::new(model)).expect("generator");

    let tokens = generator
        .encode(PROMPT, &GenerationConfig::default())
        .expect("encode");

    assert_eq!(tokens, vec![785, 6722, 315, 38396, 374]);
}

#[test]
fn llama_prompt_starts_with_begin_of_text() {
    let Some(dir) = cached("meta-llama_Llama-3.2-1B-Instruct") else {
        eprintln!("skipping: llama-3.2-1b-instruct not cached");
        return;
    };
    let model: LlamaModel =
        DecoderLoader::load_from_pretrained(&dir, Device::Cpu, None, Some(f32_config()), None)
            .expect("load llama");
    let generator = DecoderGenerator::new(Arc::new(model)).expect("generator");

    let with_bos = generator
        .encode(PROMPT, &GenerationConfig::default())
        .expect("encode");
    assert_eq!(with_bos, vec![128000, 791, 6864, 315, 39496, 374]);

    // The config can still turn it off.
    let without = GenerationConfig {
        add_bos_token: false,
        ..Default::default()
    };
    let tokens = generator.encode(PROMPT, &without).expect("encode");
    assert_eq!(tokens, vec![791, 6864, 315, 39496, 374]);
}
