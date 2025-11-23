//! LLaMA-style decoder-only language model.
//!
//! This module provides the `LlamaModel`, a model container responsible for loading
//! weights and configuration for Llama and its variants.
//!
//! The actual text generation is handled by the generic `Generator` struct.

use crate::models::llama::config::LlamaConfig;
use crate::models::llama::model::LlamaModel;
use crate::models::llama::gpu_decoder::LlamaGpuDecoder;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use edgetransformers::decoder::TransformerDecoder;
use edgetransformers::gpu_ops::GpuTensor;
use edgetransformers::gpu_ops::blocks::rope::GpuRoPE;
use edgetransformers::models::base::GpuDecoder;
use edgetransformers::models::base::{AutoregressiveLoop, DecodingStrategy};
use edgetransformers::models::download_model_files;
use edgetransformers::models::{DecoderLanguageModel, LanguageModel, ModelArchitecture, ModelType};
use edgetransformers::prelude::*;
use edgetransformers::rope::RoPE;
use edgetransformers::traits::{Decoder, DecoderArchitecture, DecoderOutput, LanguageModelConfig};
use edgetransformers::weights::ModelWeights;
use ndarray::{Array2, Array3, s};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

use super::*;
use crate::generation::{GenerationConfig, Generator};


/// Helper function to load the Llama model for testing.
async fn load_llama_for_test() -> Result<LlamaModel> {
    LlamaModel::from_registry(ModelType::Llama3_2_1B, None, Device::Cpu, None).await
}

// #[tokio::test]
// async fn test_llama3_2_1b_generation_parity() -> Result<()> {
//     // This test verifies deterministic (greedy) generation output.
//     let model = load_llama_for_test().await?;
//     let generator = Generator::new(Box::new(model));

//     let prompt = "The field of Artificial Intelligence has seen a lot of progress";
//     let expected_output =
//         "The field of Artificial Intelligence has seen a lot of progress in recent years";

//     let config = GenerationConfig {
//         max_new_tokens: Some(5),
//         add_bos_token: true, // Llama models require the BOS token.
//         strategy: DecodingStrategy::Greedy,
//         ..Default::default()
//     };

//     let generated_text = generator.generate(prompt, &config).await?;
//     assert_eq!(generated_text.trim(), expected_output.trim());

//     Ok(())
// }

#[tokio::test]
async fn test_llama3_2_1b_architectural_properties() -> Result<()> {
    // 1. Arrange: Load the model.
    let model = load_llama_for_test().await?;
    let config = model.concrete_config();

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

// #[tokio::test]
// #[ignore] // This test downloads a large model and is very slow. Run with `cargo test -- --ignored`.
// async fn test_llama3_2_1b_generation_parity() -> Result<()> {
//     // This test verifies that our Llama implementation produces a known, correct output
//     // for a deterministic (greedy) generation task.

//     // 1. Setup: Define the model, prompt, and expected output.
//     let model_type = ModelType::Llama3_2_1B;
//     let prompt = "The field of Artificial Intelligence has seen a lot of progress";

//     // The "golden" output string for generating 5 new tokens, based on previous correct runs.
//     let expected_output = "The field of Artificial Intelligence has seen a lot of progress in the last few years";

//     // Create a config for deterministic, greedy decoding.
//     let config = GenerationConfig {
//         max_new_tokens: Some(1), // Generate exactly 5 new tokens.
//         sampling_strategy: SamplingStrategy::Greedy,
//         repetition_penalty: 1.0, // No penalty.
//         add_bos_token: true,     // CRITICAL for Llama models.
//         ..Default::default()
//     };

//     // 2. Load model and create the generator.
//     let llama_model = LlamaModel::from_registry(
//         model_type,
//         None,
//         Device::Cpu,
//         None
//     ).await?;

//     let generator = Generator::new(Box::new(llama_model));

//     // 3. Execute the generation.
//     let generated_text = generator.generate(prompt, &config).await?;

//     // 4. Assert that the generated output is bit-for-bit identical to the golden value.
//     //    We trim both strings to avoid any potential whitespace differences at the end.
//     assert_eq!(generated_text.trim(), expected_output.trim());

//     Ok(())
// }
