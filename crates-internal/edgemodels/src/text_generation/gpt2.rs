//! GPT-2 style decoder-only language model.
//!
//! This module provides the `Gpt2Model`, a model container responsible for loading
//! weights and configuration for models like GPT-2, DistilGPT2, etc.
//!
//! The actual text generation is handled by the generic `Generator` struct,
//! which can operate on any model that implements the `DecoderLanguageModel` trait.

use crate::text_generation::configs::Gpt2Config;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use edgetransformers::decoder::TransformerDecoder;
use edgetransformers::models::download_model_files;
use edgetransformers::models::{DecoderLanguageModel, LanguageModel, ModelArchitecture, ModelType};
use edgetransformers::prelude::*;
use edgetransformers::traits::{Decoder, DecoderArchitecture, DecoderOutput, LanguageModelConfig};
use edgetransformers::weights::ModelWeights;
use ndarray::Array2;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

/// A model container for GPT-2 and its variants (e.g., DistilGPT2).
///
/// This struct holds the model's components (decoder, tokenizer, config) but
/// delegates the actual text generation task to the `Generator`.
pub struct Gpt2Model {
    decoder: TransformerDecoder,
    tokenizer: Tokenizer,
    config: Arc<Gpt2Config>,
    /// The language modeling head, transposed for efficient projection.
    /// Shape: `[hidden_size, vocab_size]`.
    lm_head: Array2<f32>,
}

impl Gpt2Model {
    /// A list of the specific model types supported by this implementation.
    const SUPPORTED_MODELS: &'static [ModelType] = &[
        ModelType::DistilGpt2,
        ModelType::Gpt2,
        ModelType::Gpt2Medium,
        ModelType::Gpt2Large,
        ModelType::Gpt2XL,
    ];

    /// Creates a `Gpt2Model` from the HuggingFace model registry.
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
            return Err(anyhow!("Unsupported GPT-2 model type: {:?}", model_type));
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

    /// Creates a `Gpt2Model` from a local directory containing the model files.
    pub fn from_pretrained(
        model_path: &Path,
        model_type: ModelType,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        let weights = ModelWeights::new(model_path)?;
        let tokenizer =
            Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(|e| anyhow!(e))?;

        // The DecoderArchitecture trait is key here. It provides the generic
        // TransformerDecoder with the specific tensor names for GPT-2.
        let config_arc: Arc<dyn DecoderArchitecture + Send + Sync> = {
            let mut cfg = serde_json::from_str::<Gpt2Config>(&weights.config_json)?;
            if model_type == ModelType::DistilGpt2 {
                // Special handling for DistilGPT2's unique weight naming convention.
                cfg.set_model_type("distilgpt2".to_string());
            };
            Arc::new(cfg)
        };

        // The underlying generic decoder, which can be CPU or GPU based.
        let decoder = TransformerDecoder::new(&weights, config_arc.clone(), device, context, None)?;

        // GPT-2 shares weights between embeddings and the final layer.
        // We load them and transpose for the matmul in the projection step.
        let lm_head = weights
            .get_array2(config_arc.get_lm_head_name())?
            .t()
            .to_owned();
            
        // Downcast the Arc back to the concrete Gpt2Config type for storage.
        let config = config_arc
            .as_any()
            .downcast_ref::<Gpt2Config>()
            .cloned()
            .map(Arc::new)
            .ok_or_else(|| anyhow!("Failed to downcast config to Gpt2Config"))?;

        Ok(Self {
            decoder,
            tokenizer,
            config,
            lm_head,
        })
    }
}

// --- Trait Implementations ---
// These implementations make `Gpt2Model` compatible with the generic `Generator`.

impl TransformerModel for Gpt2Model {
    fn device(&self) -> Device {
        self.decoder.device()
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        self.decoder.context()
    }
}

impl LanguageModel for Gpt2Model {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn config(&self) -> &dyn LanguageModelConfig {
        self.config.as_ref()
    }
}

#[async_trait]
impl DecoderLanguageModel for Gpt2Model {
    fn decoder(&self) -> &dyn Decoder<Input = Array2<f32>, Output = DecoderOutput> {
        &self.decoder
    }
    fn lm_head(&self) -> &Array2<f32> {
        &self.lm_head
    }
}