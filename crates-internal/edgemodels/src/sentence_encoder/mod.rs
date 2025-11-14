//! Sentence encoder for semantic similarity and embeddings
//!
//! Supports BERT-style encoder models like MiniLM, MPNet, and DistilBERT.
//! Automatically downloads models from HuggingFace using the registry.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::Array2;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

use edgetransformers::encoder::TransformerEncoder;
use edgetransformers::models::download_model_files;
use edgetransformers::models::{EncoderLanguageModel, LanguageModel, ModelArchitecture, ModelType};
use edgetransformers::prelude::*;
use edgetransformers::traits::{Encoder, EncoderArchitecture, EncoderOutput, LanguageModelConfig};
use edgetransformers::weights::ModelWeights;
use edgetransformers::models::base::{EncodingConfig, PoolingStrategy};
mod configs;
mod tests;
pub use configs::{DistilBERTConfig, MPNetConfig, MiniLMConfig};

/// Sentence encoder for semantic similarity tasks
///
/// Automatically handles model downloading and configuration.
pub struct SentenceEncoder {
    encoder: TransformerEncoder,
    tokenizer: Tokenizer,
    model_type: ModelType,
    config: Arc<dyn EncoderArchitecture + Send + Sync>,
}

impl SentenceEncoder {
    /// Supported encoder model types
    const SUPPORTED_MODELS: &'static [ModelType] = &[
        ModelType::MiniLML6V2,
        ModelType::MpnetBaseV2,
        ModelType::DistilBertBaseCased,
    ];

    /// Create encoder from HuggingFace model registry
    ///
    /// Automatically downloads model files to cache directory.
    ///
    /// # Arguments
    /// * `model_type` - Model type from registry
    /// * `cache_dir` - Optional cache directory (defaults to ~/.cache/edgetransformers)
    /// * `device` - Device to run on (CPU or GPU)
    /// * `context` - Optional WGPU context for GPU execution
    ///
    /// # Example
    /// ```no_run
    /// use edgemodels::sentence_encoder::SentenceEncoder;
    /// use edgetransformers::models::ModelType;
    /// use edgetransformers::traits::Device;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let encoder = SentenceEncoder::from_registry(
    ///     ModelType::MiniLML6V2,
    ///     None,
    ///     Device::Cpu,
    ///     None,
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        // Validate model type
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!(
                "SentenceEncoder: Unsupported model type: {:?}. Supported: {:?}",
                model_type,
                Self::SUPPORTED_MODELS
            ));
        }

        let info = model_type.info();

        if info.architecture != ModelArchitecture::Encoder {
            return Err(anyhow!(
                "Model {:?} is not an encoder (architecture: {:?})",
                model_type,
                info.architecture
            ));
        }

        // Determine cache directory
        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::cache_dir()
                .expect("No cache directory found")
                .join("edgetransformers")
        });

        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

        // Download model files if needed
        download_model_files(&model_dir, &info.paths).await?;

        // Load from local path
        Self::from_pretrained(&model_dir, model_type, device, context)
    }

    /// Create encoder from local model directory
    pub fn from_pretrained(
        model_path: &Path,
        model_type: ModelType,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        // Validate model type
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!("Unsupported model type: {:?}", model_type));
        }

        // Load weights and tokenizer
        let weights = ModelWeights::new(model_path)?;
        let mut tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Select config based on model type
        let (encoder, config): (
            TransformerEncoder,
            Arc<dyn EncoderArchitecture + Send + Sync>,
        ) = match model_type {
            ModelType::MiniLML6V2 => {
                let cfg = Arc::new(MiniLMConfig::from_json(&weights.config_json)?)
                    as Arc<dyn EncoderArchitecture + Send + Sync>;
                (
                    TransformerEncoder::new(&weights, cfg.clone(), device, context)?,
                    cfg,
                )
            }
            ModelType::MpnetBaseV2 => {
                let cfg = Arc::new(MPNetConfig::from_json(&weights.config_json)?)
                    as Arc<dyn EncoderArchitecture + Send + Sync>;
                (
                    TransformerEncoder::new(&weights, cfg.clone(), device, context)?,
                    cfg,
                )
            }
            ModelType::DistilBertBaseCased => {
                let cfg = Arc::new(DistilBERTConfig::from_json(&weights.config_json)?)
                    as Arc<dyn EncoderArchitecture + Send + Sync>;
                (
                    TransformerEncoder::new(&weights, cfg.clone(), device, context)?,
                    cfg,
                )
            }
            _ => return Err(anyhow!("Unsupported encoder model: {:?}", model_type)),
        };
        // Configure tokenizer padding and truncation using the model's config
        let truncation_params = tokenizers::TruncationParams {
            max_length: config.max_position_embeddings(),
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            ..Default::default()
        };

        _ = tokenizer.with_truncation(Some(truncation_params));

        let padding_params = tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(padding_params));
        Ok(Self {
            encoder,
            tokenizer,
            config,
            model_type,
        })
    }

    /// Get the embedding dimension (same as hidden size)
    pub fn embedding_dim(&self) -> usize {
        self.encoder.hidden_size()
    }

    /// Get the maximum sequence length
    pub fn max_seq_length(&self) -> usize {
        self.encoder.max_length()
    }

    /// Get the model type
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Encode text with default mean pooling and normalization
    pub async fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let config = EncodingConfig {
            normalize: true,
            pooling_strategy: PoolingStrategy::Mean
        };
        <Self as EncoderLanguageModel>::encode(self, text, &config).await
    }

    /// Encode text without normalization (raw embeddings)
    pub async fn encode_raw(&self, text: &str) -> Result<Vec<f32>> {
        let config = EncodingConfig {
            normalize: false,
            pooling_strategy: PoolingStrategy::Mean
        };
        <Self as EncoderLanguageModel>::encode(self, text, &config).await
    }

    /// Encode with pooling strategy and optional normalization
    pub async fn encode_with(
        &self,
        text: &str,
        pooling_strategy: Option<&str>,
        normalize: bool,
    ) -> Result<Vec<f32>> {
        let strategy = pooling_strategy.unwrap_or("mean");
        let config = EncodingConfig {
            normalize: normalize,
            pooling_strategy: match strategy {
                "mean" => PoolingStrategy::Mean,
                "cls" => PoolingStrategy::Cls,
                "lastToken" => PoolingStrategy::LastToken,
                "max" => PoolingStrategy::Max,
                _ => panic!("Unknown pooling strategy {}", strategy)
            }
        };
        <Self as EncoderLanguageModel>::encode(self, text, &config).await
    }

    /// Encode batch of texts with default mean pooling and normalization
    pub async fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let config = EncodingConfig {
            normalize: true,
            pooling_strategy: PoolingStrategy::Mean
        };
        <Self as EncoderLanguageModel>::encode_batch(self, texts, &config).await
    }

    /// Encode batch of texts without normalization (raw embeddings)
    pub async fn encode_batch_raw(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let config = EncodingConfig {
            normalize: false,
            pooling_strategy: PoolingStrategy::Mean
        };
        <Self as EncoderLanguageModel>::encode_batch(self, texts, &config).await
    }

    /// Encode batch with custom pooling strategy and normalization
    pub async fn encode_batch_with(
        &self,
        texts: &[&str],
        pooling_strategy: Option<&str>,
        normalize: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let strategy = pooling_strategy.unwrap_or("mean");
        let config = EncodingConfig {
            normalize: normalize,
            pooling_strategy: match strategy {
                "mean" => PoolingStrategy::Mean,
                "cls" => PoolingStrategy::Cls,
                "lastToken" => PoolingStrategy::LastToken,
                "max" => PoolingStrategy::Max,
                _ => panic!("Unknown pooling strategy {}", strategy)
            }
        };
        <Self as EncoderLanguageModel>::encode_batch(self, texts, &config).await
    }
}

// Implement base language model trait
impl LanguageModel for SentenceEncoder {
    fn new_cache(&self, _batch_size: usize, _max_len: usize) -> Result<Box<dyn Cache>> {
        panic!("Sentence Encoder does not support KV Cache");
    }
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn config(&self) -> &dyn LanguageModelConfig {
        self.config.as_ref()
    }
}

// Implement encoder language model trait
#[async_trait]
impl EncoderLanguageModel for SentenceEncoder {
    fn encoder(&self) -> &dyn Encoder<Input = Array2<u32>, Output = EncoderOutput> {
        &self.encoder
    }
}

// Implement transformer model trait
impl TransformerModel for SentenceEncoder {
    fn device(&self) -> Device {
        self.encoder.device()
    }
}
