//! GPT-2 style decoder-only language model.
//!
//! This module provides the `Gpt2Model`, a model container responsible for loading
//! weights and configuration for models like GPT-2, DistilGPT2, etc.
//!
//! The actual text generation is handled by the generic `Generator` struct,
//! which can operate on any model that implements the `DecoderLanguageModel` trait.

use anyhow::Result;
use kjarni_transformers::common::{DecodingStrategy, GenerationConfig};
use kjarni_transformers::decoder::prelude::*;
use kjarni_transformers::models::{LanguageModel, ModelType};
use kjarni_transformers::prelude::*;
use std::sync::Arc;

use crate::models::gpt2::model::Gpt2Model;

/// Helper function to load the DistilGPT2 model for testing.
async fn load_distilgpt2_for_test() -> Result<Gpt2Model> {
    Gpt2Model::from_registry(ModelType::DistilGpt2, None, Device::Cpu, None, None).await
}

#[tokio::test]
async fn test_distilgpt2_generation_parity() -> Result<()> {
    // This test verifies deterministic (greedy) generation output.
    let model = load_distilgpt2_for_test().await?;
    let generator = DecoderGenerator::new(Arc::new(model))?;

    let prompt = "The field of Artificial Intelligence has seen a lot of progress";
    let expected_output = "The field of Artificial Intelligence has seen a lot of progress in the past few years, but it is still not clear how much improvement will be made.";

    let config = GenerationConfig {
        max_new_tokens: Some(20),
        add_bos_token: false,
        strategy: DecodingStrategy::Greedy,
        repetition_penalty: 1.1,
        ..Default::default()
    };

    let generated_text = generator.generate(prompt, &config, None).await?;
    let concat_prompt = prompt.to_string() + "" + &generated_text;
    assert_eq!(concat_prompt.trim(), expected_output.trim());

    Ok(())
}

#[tokio::test]
async fn test_distilgpt2_architectural_properties() -> Result<()> {
    // 1. Arrange: Load the model.
    let model = load_distilgpt2_for_test().await?;
    let config = model.concrete_config();

    // 2. Assert: Check architectural values directly from the config struct.
    assert_eq!(config.vocab_size, 50257);
    assert_eq!(config.n_embd, 768); // hidden size
    assert_eq!(config.n_layer, 6);
    assert_eq!(config.n_head, 12);
    assert_eq!(config.n_ctx, 1024); // max_position_embeddings

    // 3. Assert: Check that the trait implementations correctly expose these values.
    assert_eq!(model.vocab_size(), 50257);
    assert_eq!(model.hidden_size(), 768);
    assert_eq!(model.num_layers(), 6);
    assert_eq!(model.num_heads(), 12);
    assert_eq!(model.max_length(), 1024);

    // Check token IDs. GPT-2 uses the same ID for BOS, EOS, and PAD.
    assert_eq!(model.bos_token_id(), Some(50256));
    assert_eq!(model.eos_token_id(), Some(50256));
    assert_eq!(model.pad_token_id(), Some(50256));

    Ok(())
}

#[tokio::test]
async fn test_distilgpt2_generation_parity_2() -> Result<()> {
    // This test verifies that our implementation produces the exact same output
    // as the HuggingFace transformers library for a deterministic (greedy) generation task.

    // 1. Setup: Define the exact same configuration as the Python script and the working main.rs.
    let model_type = ModelType::DistilGpt2;
    let prompt = "The field of Artificial Intelligence has seen a lot of progress";

    // The "golden" output string from the Python reference script and your now-working Rust implementation.
    let expected_output = "The field of Artificial Intelligence has seen a lot of progress in the past few years, but it is still not clear how much improvement will be made.";

    // Create a config that perfectly matches the reference implementations.
    let config = GenerationConfig {
        max_new_tokens: Some(20),
        strategy: DecodingStrategy::Greedy,
        repetition_penalty: 1.1,
        add_bos_token: false, // CRITICAL for parity with default Hugging Face behavior
        ..Default::default()
    };

    // 2. Load model and create the generator.
    //    We run on CPU to match the Python script's environment.
    let gpt2_model = Gpt2Model::from_registry(model_type, None, Device::Cpu, None, None).await?;

    let generator = DecoderGenerator::new(Arc::new(gpt2_model))?;

    // 3. Execute the generation. We use the non-streaming `generate` for a simple string comparison.
    let generated_text = generator.generate(prompt, &config, None).await?;

    let concat_prompt = prompt.to_string() + "" + &generated_text;
    assert_eq!(concat_prompt.trim(), expected_output.trim());

    Ok(())
}

#[tokio::test]
async fn test_distilgpt2_generation_parity_cpu_gpu() -> Result<()> {
    // This test verifies that our implementation produces the exact same output
    // as the HuggingFace transformers library for a deterministic (greedy) generation task.

    // 1. Setup: Define the exact same configuration as the Python script and the working main.rs.
    let model_type = ModelType::DistilGpt2;
    let prompt = "The field of Artificial Intelligence has seen a lot of progress";

    // Create a config that perfectly matches the reference implementations.
    let config = GenerationConfig {
        max_new_tokens: Some(5),
        strategy: DecodingStrategy::Greedy,
        repetition_penalty: 1.1,
        add_bos_token: false, // CRITICAL for parity with default Hugging Face behavior
        ..Default::default()
    };

    // 2. Load model and create the generator.
    //    We run on CPU to match the Python script's environment.
    let gpt2_model = Gpt2Model::from_registry(model_type, None, Device::Cpu, None, None).await?;

    let generator = DecoderGenerator::new(Arc::new(gpt2_model))?;

    // 3. Execute the generation. We use the non-streaming `generate` for a simple string comparison.
    let generated_text = generator.generate(prompt, &config, None).await?;

    let ctx = WgpuContext::new().await?;
    let gpt2_model_2 =
        Gpt2Model::from_registry(model_type, None, Device::Wgpu, Some(ctx), None).await?;
    let generator_2 = DecoderGenerator::new(Arc::new(gpt2_model_2))?;
    let generated_text_2 = generator_2.generate(prompt, &config, None).await?;

    // 4. Assert that the generated output is bit-for-bit identical to the golden value.
    //    We trim both strings to avoid any potential whitespace differences at the end.
    assert_eq!(generated_text.trim(), generated_text_2.trim());

    Ok(())
}
