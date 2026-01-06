//! LLaMA-style decoder-only language model.
//!
//! This module provides the `LlamaModel`, a model container responsible for loading
//! weights and configuration for Llama and its variants.
//!
//! The actual text generation is handled by the generic `Generator` struct.

use crate::models::llama::config::LlamaConfig;
use crate::models::llama::model::LlamaModel;
use anyhow::Result;
use kjarni_transformers::common::{BeamSearchParams, DecodingStrategy, GenerationConfig};
use kjarni_transformers::decoder::prelude::*;
use kjarni_transformers::models::{LanguageModel, ModelType};
use kjarni_transformers::prelude::*;
use std::path::{self, Path};
use std::sync::Arc;

/// Helper function to load the Llama model for testing.
async fn load_llama_for_test() -> Result<LlamaModel> {
    LlamaModel::from_pretrained(
        Path::new("/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B"),
        Device::Cpu,
        None,
        None,
        None,
    )
}

async fn load_llama_8b_for_test() -> Result<LlamaModel> {
    LlamaModel::from_pretrained(
        Path::new("/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-8B-Instruct"),
        Device::Cpu,
        None,
        None,
        None,
    )
}

//
#[tokio::test]
async fn test_llama3_8b_architectural_properties() -> Result<()> {
    if std::path::Path::new("/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-8B-Instruct").exists() == false {
        log::warn!("Skipping Llama-3.2-8B test since model files not found in cache.");
        return Ok(());
    }
    // 1. Arrange: Load the model.
    let model = load_llama_8b_for_test().await?;
    let config = model.config();

    // 2. Assert: Check architectural values directly from the config struct.
    assert_eq!(config.vocab_size, 128256);
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_hidden_layers, 32);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 8); // GQA
    assert_eq!(config.intermediate_size, 14336);
    assert_eq!(config.max_position_embeddings, 8192);
    assert_eq!(config.rope_theta, 500000.0);
    assert_eq!(config.rms_norm_eps, 1e-5);

    // 3. Assert: Check that the trait implementations correctly expose these values.
    // This catches hardcoded values like the bug we just found!
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
    assert_eq!(model.eos_token_id(), Some(128009)); // Note: 8B uses 128009, not 128001

    Ok(())
}

/// Generic test to ensure trait values match config values.
/// Run this for ANY model to catch hardcoded values.
#[tokio::test]
async fn test_llama_trait_config_consistency() -> Result<()> {
    // Test with 1B
    let model_1b = load_llama_for_test().await?;
    assert_trait_matches_config(&model_1b, "Llama-3.2-1B");

    // Test with 8B (if available)
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

#[tokio::test]
async fn test_llama3_2_1b_architectural_properties() -> Result<()> {
    // 1. Arrange: Load the model.
    let model = load_llama_for_test().await?;
    let config = model.config();

    // 2. Assert: Check architectural values directly from the config struct.
    assert_eq!(config.vocab_size, 128256);
    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.num_hidden_layers, 16);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 8); // GQA
    assert_eq!(config.max_position_embeddings, 131072);
    assert_eq!(config.rope_theta, 500000.0);

    // 3. Assert: Check that the trait implementations correctly expose these values.
    assert_eq!(model.vocab_size(), 128256);
    assert_eq!(model.hidden_size(), 2048);
    assert_eq!(model.num_layers(), 16);
    assert_eq!(model.num_heads(), 32);
    assert_eq!(model.max_length(), 131072);

    // Check token IDs.
    assert_eq!(model.bos_token_id(), Some(128000));
    assert_eq!(model.eos_token_id(), Some(128001));
    // Llama often doesn't define a pad token, so this should be None.
    assert_eq!(model.pad_token_id(), None);

    Ok(())
}

#[tokio::test]
async fn test_llama3_2_1b_generation_parity() -> Result<()> {
    // This test verifies that our Llama implementation produces a known, correct output
    // for a deterministic (greedy) generation task.

    // 1. Setup: Define the model, prompt, and expected output.
    // let model_type = ModelType::Llama3_2_1B;
    let prompt = "The field of Artificial Intelligence has seen a lot of progress";

    // The "golden" output string for generating 5 new tokens, based on previous correct runs.
    let expected_output =
        "The field of Artificial Intelligence has seen a lot of progress in the last few years.";

    // Create a config for deterministic, greedy decoding.
    let config = GenerationConfig {
        max_new_tokens: Some(6),
        strategy: DecodingStrategy::Greedy,
        repetition_penalty: 1.0, // No penalty.
        add_bos_token: true,     // CRITICAL for Llama models.
        ..Default::default()
    };
    // decoder_layer::tests::test_decoder_layer_with_rope_and_gqa

    // 2. Load model and create the generator.
    let llama_model = LlamaModel::from_pretrained(
        Path::new("/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B"),
        Device::Cpu,
        None,
        None,
        None,
    )?;

    let llama_gpu = LlamaModel::from_pretrained(
        Path::new("/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B"),
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

    Ok(())
}
