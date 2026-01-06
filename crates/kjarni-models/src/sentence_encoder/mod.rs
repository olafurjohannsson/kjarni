//! Sentence encoder for semantic similarity and embeddings
//!
//! Supports BERT-style encoder models like MiniLM, MPNet, and DistilBERT.
//! Automatically downloads models from HuggingFace using the registry.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use kjarni_transformers::models::registry::WeightsFormat;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::{Model, Tokenizer};

use kjarni_transformers::{
    cpu::encoder::{
        config::{EncodingConfig, PoolingStrategy}, traits::{
            CpuEncoder, CpuEncoderOps, EncoderLanguageModel, GpuEncoder, GpuEncoderOps,
            SentenceEncoderModel,
        },
        CpuTransformerEncoder,
        GpuTransformerEncoder,
    },
    models::{download_model_files, LanguageModel, ModelArchitecture, ModelType},
    traits::{Cache, Device},
    weights::ModelWeights,
    WgpuContext,
};

mod configs;
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::traits::{InferenceModel, ModelConfig, ModelLayout, ModelMetadata};

use crate::sentence_encoder::configs::BertConfig;

/// Sentence encoder for semantic similarity tasks
///
/// Automatically handles model downloading and configuration.
pub struct SentenceEncoder {
    cpu_encoder: Option<CpuTransformerEncoder>,
    gpu_encoder: Option<GpuTransformerEncoder>,
    tokenizer: Tokenizer,
    model_type: ModelType,
    config: Arc<dyn kjarni_transformers::traits::ModelConfig>, // Unified Trait
    device: Device,
    context: Option<Arc<WgpuContext>>,
    pub meta: ModelMetadata,
    pub layout: ModelLayout,
}

impl SentenceEncoder {
    /// Supported encoder model types
    const SUPPORTED_MODELS: &'static [ModelType] = &[
        ModelType::MiniLML6V2,
        ModelType::NomicEmbedText, // New! THIS NEEDS ROPE
        ModelType::BgeM3,          // New!
        // ModelType::MpnetBaseV2, // Add if you kept this in your registry
        // distilbertcased?
    ];

    /// Create encoder from HuggingFace model registry
    ///
    /// Automatically downloads model files to cache directory.
    ///
    /// # Arguments
    /// * `model_type` - Model type from registry
    /// * `cache_dir` - Optional cache directory (defaults to ~/.cache/kjarni)
    /// * `device` - Device to run on (CPU or GPU)
    /// * `context` - Optional WGPU context for GPU execution
    ///
    /// # Example
    /// ```no_run
    /// use kjarni_models::sentence_encoder::SentenceEncoder;
    /// use kjarni_transformers::models::ModelType;
    /// use kjarni_transformers::traits::Device;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let encoder = SentenceEncoder::from_registry(
    ///     ModelType::MiniLML6V2,
    ///     None,
    ///     Device::Cpu,
    ///     None,
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
        load_cfg: Option<ModelLoadConfig>,
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

        // 2. Validate Architecture
        // Since 'ModelArchitecture::Encoder' is gone, we check for specific Encoder families.
        match info.architecture {
            ModelArchitecture::Bert | ModelArchitecture::NomicBert => {
                // These are valid encoders
            }
            _ => {
                return Err(anyhow!(
                    "Model {:?} is not an encoder (architecture: {:?})",
                    model_type,
                    info.architecture
                ));
            }
        }

        // Determine cache directory
        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::cache_dir()
                .expect("No cache directory found")
                .join("kjarni")
        });

        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

        // Download model files if needed
        download_model_files(&model_dir, &info.paths, WeightsFormat::SafeTensors).await?;

        // Load from local path
        Self::from_pretrained(&model_dir, model_type, device, context, load_cfg)
    }

    /// Create encoder from local model directory
    pub fn from_pretrained(
        model_path: &Path,
        model_type: ModelType,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_cfg: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        // Validate model type
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!("Unsupported model type: {:?}", model_type));
        }

        // Load weights and tokenizer
        let weights = ModelWeights::new(model_path)?;
        let load_cfg = load_cfg.unwrap_or_default();
        let mut tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        let config: Arc<dyn ModelConfig> = match model_type {

            // todo UPDATE

            ModelType::MiniLML6V2 => Arc::new(BertConfig::from_json(&weights.config_json)?),

            ModelType::NomicEmbedText => Arc::new(BertConfig::from_json(&weights.config_json)?),

            _ => return Err(anyhow!("Unsupported encoder model: {:?}", model_type)),
        };
        let meta = config.metadata();
        let layout = config.layout();
        let mut cpu_encoder = None;
        let mut gpu_encoder = None;

        match device {
            Device::Cpu => {
                cpu_encoder = Some(CpuTransformerEncoder::new(
                    &weights,
                    meta.clone(),
                    layout.clone(),
                    load_cfg,
                )?);
            }
            Device::Wgpu => {
                let ctx = context
                    .clone()
                    .ok_or_else(|| anyhow!("WGPU context required"))?;
                gpu_encoder = Some(GpuTransformerEncoder::new(
                    &weights,
                    ctx,
                    meta.clone(),
                    layout.clone(),
                    load_cfg,
                )?);
            }
        }

        // Configure tokenizer padding and truncation using the model's config
        let truncation_params = tokenizers::TruncationParams {
            max_length: meta.max_seq_len,
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
            cpu_encoder,
            gpu_encoder,
            tokenizer,
            config,
            model_type,
            device,
            context,
            meta,
            layout,
        })
    }

    /// Get the embedding dimension (same as hidden size)
    pub fn embedding_dim(&self) -> usize {
        self.meta.hidden_size
    }

    /// Get the maximum sequence length
    pub fn max_seq_length(&self) -> usize {
        self.meta.max_seq_len
    }

    pub fn hidden_size(&self) -> usize {
        self.meta.hidden_size
    }

    /// Get the model type
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Encode text with default mean pooling and normalization
    pub async fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let config = EncodingConfig {
            normalize: true,
            pooling_strategy: PoolingStrategy::Mean,
        };
        <Self as SentenceEncoderModel>::encode(self, text, &config).await
    }

    /// Encode text without normalization (raw embeddings)
    pub async fn encode_raw(&self, text: &str) -> Result<Vec<f32>> {
        let config = EncodingConfig {
            normalize: false,
            pooling_strategy: PoolingStrategy::Mean,
        };
        <Self as SentenceEncoderModel>::encode(self, text, &config).await
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
                _ => panic!("Unknown pooling strategy {}", strategy),
            },
        };
        <Self as SentenceEncoderModel>::encode(self, text, &config).await
    }

    /// Encode batch of texts with default mean pooling and normalization
    pub async fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let config = EncodingConfig {
            normalize: true,
            pooling_strategy: PoolingStrategy::Mean,
        };
        <Self as SentenceEncoderModel>::encode_batch(self, texts, &config).await
    }

    /// Encode batch of texts without normalization (raw embeddings)
    pub async fn encode_batch_raw(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let config = EncodingConfig {
            normalize: false,
            pooling_strategy: PoolingStrategy::Mean,
        };
        <Self as SentenceEncoderModel>::encode_batch(self, texts, &config).await
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
                _ => panic!("Unknown pooling strategy {}", strategy),
            },
        };
        <Self as SentenceEncoderModel>::encode_batch(self, texts, &config).await
    }
}

// Implement base language model trait
impl LanguageModel for SentenceEncoder {
    fn vocab_size(&self) -> usize { self.meta.vocab_size }
    fn hidden_size(&self) -> usize { self.meta.hidden_size }
    fn num_layers(&self) -> usize { self.meta.num_layers }
    fn num_heads(&self) -> usize { self.meta.num_attention_heads }
    fn context_size(&self) -> usize { self.meta.max_seq_len }
    fn tokenizer(&self) -> &Tokenizer { &self.tokenizer }
    fn bos_token_id(&self) -> Option<u32> { self.tokenizer.token_to_id("[CLS]") }
    fn eos_token_id(&self) -> Option<u32> { self.tokenizer.token_to_id("[SEP]") }
    fn pad_token_id(&self) -> Option<u32> { self.tokenizer.token_to_id("[PAD]") }
    fn forced_bos_token_id(&self) -> Option<u32> {
        None
    }
    fn forced_eos_token_id(&self) -> Option<u32> {
        None
    }
    fn new_cache(&self, _: usize, _: usize, _: usize) -> Result<Box<dyn Cache>> {
        Err(anyhow!("Sentence Encoders do not use a KV cache."))
    }
}

impl InferenceModel for SentenceEncoder {
    fn device(&self) -> Device {
        self.device
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        self.context.clone()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CpuEncoderOps for SentenceEncoder {
    fn encoder(&self) -> &dyn CpuEncoder {
        self.cpu_encoder
            .as_ref()
            .expect("CPU encoder not initialized for this model.")
    }
}

impl GpuEncoderOps for SentenceEncoder {
    fn encoder(&self) -> &dyn GpuEncoder {
        self.gpu_encoder
            .as_ref()
            .expect("GPU encoder not initialized for this model.")
    }
}

// It fulfills the base contract of being an EncoderLanguageModel.
#[async_trait]
impl EncoderLanguageModel for SentenceEncoder {
    fn encoder_cpu_ops(&self) -> Option<&dyn CpuEncoderOps> {
        if self.device.is_cpu() {
            Some(self)
        } else {
            None
        }
    }

    fn encoder_gpu_ops(&self) -> Option<&dyn GpuEncoderOps> {
        if self.device.is_gpu() {
            Some(self)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests;
