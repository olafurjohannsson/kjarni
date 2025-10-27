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
use edgetransformers::models::{EncoderLanguageModel, LanguageModel, ModelArchitecture, ModelType};
use edgetransformers::prelude::*;
use edgetransformers::traits::{Encoder, EncoderArchitecture, EncoderOutput, LanguageModelConfig};
use edgetransformers::weights::ModelWeights;

mod configs;
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
                "Unsupported model type: {:?}. Supported: {:?}",
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
        Self::download_model_files(&model_dir, &info.paths).await?;

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
        let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
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

        Ok(Self {
            encoder,
            tokenizer,
            config,
            model_type,
        })
    }

    /// Get the model type
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Encode text with default mean pooling
    pub async fn encode(&self, text: &str) -> Result<Vec<f32>> {
        <Self as EncoderLanguageModel>::encode(self, text, "mean").await
    }

    /// Encode batch of texts with default mean pooling
    pub async fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        <Self as EncoderLanguageModel>::encode_batch(self, texts, "mean").await
    }

    /// Download model files from HuggingFace
    async fn download_model_files(
        model_dir: &Path,
        paths: &edgetransformers::models::ModelPaths,
    ) -> Result<()> {
        tokio::fs::create_dir_all(model_dir).await?;

        let files = [
            ("model.safetensors", &paths.weights_url),
            ("tokenizer.json", &paths.tokenizer_url),
            ("config.json", &paths.config_url),
        ];

        for (filename, url) in files {
            let local_path = model_dir.join(filename);

            if !local_path.exists() {
                println!("Downloading {}...", filename);
                let response = reqwest::get(*url).await?;

                if !response.status().is_success() {
                    return Err(anyhow!(
                        "Failed to download {}: HTTP {}",
                        filename,
                        response.status()
                    ));
                }

                let bytes = response.bytes().await?;
                tokio::fs::write(&local_path, &bytes).await?;
                println!("✓ Downloaded {}", filename);
            }
        }

        Ok(())
    }
}

// Implement base language model trait
impl LanguageModel for SentenceEncoder {
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
    fn encoder(&self) -> &dyn Encoder<Input = Array2<f32>, Output = EncoderOutput> {
        &self.encoder
    }
}

// Implement transformer model trait
impl TransformerModel for SentenceEncoder {
    fn device(&self) -> Device {
        self.encoder.device()
    }
}
