//! LLaMA-style decoder-only language model.
//!
//! This module provides the `LlamaModel`, a model container responsible for loading
//! weights and configuration for Llama and its variants.
//!
//! The actual text generation is handled by the generic `Generator` struct.

use crate::text_generation::llama_configs::LlamaConfig;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use edgetransformers::decoder::TransformerDecoder;
use edgetransformers::models::download_model_files;
use edgetransformers::models::{DecoderLanguageModel, LanguageModel, ModelArchitecture, ModelType};
use edgetransformers::models::base::GenerationStrategy;
use edgetransformers::prelude::*;
use edgetransformers::rope::RoPE;
use edgetransformers::traits::{Decoder, DecoderArchitecture, DecoderOutput, LanguageModelConfig};
use edgetransformers::weights::ModelWeights;
use ndarray::{Array2, Array3, s};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

/// A model container for LLaMA and its variants.
///
/// This struct holds the model's components (decoder, tokenizer, config) but
/// delegates the actual text generation task to the `Generator`.
pub struct LlamaModel {
    decoder: TransformerDecoder,
    tokenizer: Tokenizer,
    config: Arc<LlamaConfig>,
    /// The language modeling head, transposed for efficient projection.
    /// Shape: `[hidden_size, vocab_size]`.
    lm_head: Array2<f32>,
}

impl LlamaModel {
    /// A list of the specific model types supported by this implementation.
    const SUPPORTED_MODELS: &'static [ModelType] = &[
        ModelType::Llama3_2_1B,
        // Add other Llama variants here as you support them
    ];

    /// Creates a `LlamaModel` from the HuggingFace model registry.
    ///
    /// This will download the model files to a local cache directory if they
    /// are not already present.
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!("Unsupported LLaMA model type: {:?}", model_type));
        }
        if model_type.info().architecture != ModelArchitecture::Decoder {
            return Err(anyhow!("Model {:?} is not a decoder model.", model_type));
        }

        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::cache_dir()
                .expect("No cache directory found")
                .join("edgetransformers")
        });
        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

        download_model_files(&model_dir, &model_type.info().paths).await?;
        Self::from_pretrained(&model_dir, model_type, device, context)
    }

    /// Creates a `LlamaModel` from a local directory containing the model files.
    pub fn from_pretrained(
        model_path: &Path,
        _model_type: ModelType, // Used for registry validation, not needed here
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        let weights = ModelWeights::new(model_path)?;
        let mut tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        let config = Arc::new(LlamaConfig::from_json(&weights.config_json)?);

        // Set up tokenizer truncation, but no padding for autoregressive generation.
        let truncation_params = tokenizers::TruncationParams {
            max_length: config.max_position_embeddings(),
            ..Default::default()
        };
        tokenizer.with_truncation(Some(truncation_params)).unwrap();
        tokenizer.with_padding(None);

        // Create the RoPE module, passing the scaling config if it exists.
        let rope = Arc::new(RoPE::new_with_scaling(
            config.head_dim(),
            config.max_position_embeddings(),
            config.rope_theta,
            config.rope_scaling.as_ref(),
        ));

        // The generic TransformerDecoder will be built using the LlamaConfig.
        let decoder = TransformerDecoder::new(
            &weights,
            config.clone() as Arc<dyn DecoderArchitecture + Send + Sync>,
            device,
            context,
            Some(rope.clone()),
        )?;

        // Llama ties the embedding and LM head weights.
        let lm_head = weights.get_array2(config.get_lm_head_name())?;

        Ok(Self {
            decoder,
            tokenizer,
            config,
            lm_head,
        })
    }
}

// --- Trait Implementations ---
// These implementations make `LlamaModel` compatible with the generic `Generator`.

impl TransformerModel for LlamaModel {
    fn device(&self) -> Device {
        self.decoder.device()
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        self.decoder.context()
    }
}

impl LanguageModel for LlamaModel {
    fn new_cache(&self, batch_size: usize, max_len: usize) -> Result<Box<dyn Cache>> {
        Ok(match self.device() {
            Device::Cpu => Box::new(CpuKVCache::new(
                self.num_layers(),
                batch_size,
                max_len,
                self.config().kv_dim(), // ✅ Correctly use kv_dim for Llama
            )),
            Device::Wgpu => {
                let context = self
                    .context()
                    .ok_or_else(|| anyhow!("GPU model missing context"))?;
                let head_dim = self.hidden_size() / self.num_heads();
                Box::new(GpuKVCache::new(
                    &context,
                    self.num_layers(),
                    batch_size,
                    self.config().num_key_value_heads(), // Llama has a specific field for this
                    head_dim,
                    max_len,
                )?)
            }
        })
    }
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn config(&self) -> &dyn LanguageModelConfig {
        self.config.as_ref()
    }
    fn bos_token_id(&self) -> Option<u32> {
        Some(128000)
    }
    fn eos_token_id(&self) -> Option<u32> {
        Some(128001)
    }
    fn vocab_size(&self) -> usize {
        128256
    }
    fn hidden_size(&self) -> usize {
        2048
    }
    fn num_layers(&self) -> usize {
        16
    }
    fn num_heads(&self) -> usize {
        32
    }
}

#[async_trait]
impl DecoderLanguageModel for LlamaModel {
    fn decoder(&self) -> &dyn Decoder<Input = Array2<f32>, Output = DecoderOutput> {
        &self.decoder
    }
    fn lm_head(&self) -> &Array2<f32> {
        &self.lm_head
    }
    fn generation_strategy(&self) -> GenerationStrategy {
        GenerationStrategy::Pipelined
    }
    fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        // hidden_states shape: [batch, seq, hidden]
        // self.lm_head shape:   [vocab, hidden]

        let (batch_size, seq_len, hidden_size) = hidden_states.dim();
        assert_eq!(
            hidden_size,
            self.lm_head.shape()[1],
            "lm_head second dim must equal hidden_size"
        );
        assert_eq!(
            self.vocab_size(),
            self.lm_head.shape()[0],
            "lm_head first dim must equal vocab_size"
        );

        // Reshape hidden states for efficient matrix multiplication
        let hidden_2d = hidden_states.to_shape((batch_size * seq_len, hidden_size))?;

        // Perform the multiplication. We need to transpose the lm_head so the shapes align:
        // [batch*seq, hidden] @ [hidden, vocab] -> [batch*seq, vocab]
        let logits_2d = hidden_2d.dot(&self.lm_head.t());

        // Reshape the result back to 3D
        logits_2d
            .into_shape_with_order((batch_size, seq_len, self.vocab_size()))
            .map_err(|e| anyhow!(e))
    }
}


// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::generation::{Generator, GenerationConfig, GenerationStrategy, SamplingStrategy};

//     #[tokio::test]
//     #[ignore] // This test downloads a large model and is very slow. Run with `cargo test -- --ignored`.
//     async fn test_llama3_2_1b_generation_parity() -> Result<()> {
//         // This test verifies that our Llama implementation produces a known, correct output
//         // for a deterministic (greedy) generation task.
        
//         // 1. Setup: Define the model, prompt, and expected output.
//         let model_type = ModelType::Llama3_2_1B;
//         let prompt = "The field of Artificial Intelligence has seen a lot of progress";
        
//         // The "golden" output string for generating 5 new tokens, based on previous correct runs.
//         let expected_output = "The field of Artificial Intelligence has seen a lot of progress in the last few years";

//         // Create a config for deterministic, greedy decoding.
//         let config = GenerationConfig {
//             max_new_tokens: Some(1), // Generate exactly 5 new tokens.
//             sampling_strategy: SamplingStrategy::Greedy,
//             repetition_penalty: 1.0, // No penalty.
//             add_bos_token: true,     // CRITICAL for Llama models.
//             ..Default::default()
//         };

//         // 2. Load model and create the generator.
//         let llama_model = LlamaModel::from_registry(
//             model_type,
//             None,
//             Device::Cpu,
//             None
//         ).await?;
        
//         let generator = Generator::new(Box::new(llama_model));

//         // 3. Execute the generation.
//         let generated_text = generator.generate(prompt, &config).await?;

//         // 4. Assert that the generated output is bit-for-bit identical to the golden value.
//         //    We trim both strings to avoid any potential whitespace differences at the end.
//         assert_eq!(generated_text.trim(), expected_output.trim());
        
//         Ok(())
//     }
// }