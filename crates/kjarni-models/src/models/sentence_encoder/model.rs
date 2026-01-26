//! Sentence encoder for semantic similarity and embeddings.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use kjarni_transformers::utils::configure_threading;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

use kjarni_transformers::{
    WgpuContext,
    cpu::encoder::{
        CpuTransformerEncoder, GpuTransformerEncoder,
        config::{EncodingConfig, PoolingStrategy},
        traits::{
            CpuEncoder, CpuEncoderOps, EncoderLanguageModel, GpuEncoder, GpuEncoderOps,
            SentenceEncoderModel,
        },
    },
    models::base::ModelLoadConfig,
    models::{LanguageModel, ModelType},
    pipeline::{EncoderLoader, EncoderModelFactory, EncoderPipeline},
    traits::{Cache, Device, InferenceModel, ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};

use crate::models::sequence_classifier::configs::RobertaConfig;
use super::configs::{BertConfig, DistilBertConfig};


// =============================================================================
// Sentence Encoder
// =============================================================================

/// Sentence encoder for semantic similarity tasks.
///
/// Produces dense vector embeddings from text for use in:
/// - Semantic search
/// - Clustering
/// - Similarity comparison
pub struct SentenceEncoder {
    pipeline: EncoderPipeline,
    tokenizer: Tokenizer,
    config: Arc<dyn ModelConfig>, // Dynamic, not concrete type
    model_type: Option<ModelType>,
}

impl EncoderModelFactory for SentenceEncoder {
    fn load_config(weights: &ModelWeights) -> Result<Arc<dyn ModelConfig>> {
        // Auto-detect config type from JSON
        if weights.is_distilbert() {
            Ok(Arc::new(DistilBertConfig::from_json(
                &weights.config_json(),
            )?))
        } else if weights.is_roberta() || weights.is_distilroberta() {
            Ok(Arc::new(RobertaConfig::from_json(&weights.config_json())?))
        } else {
            // Default to BertConfig for BERT-like models
            Ok(Arc::new(BertConfig::from_json(&weights.config_json())?))
        }
    }

    fn build_backends(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        load_config: ModelLoadConfig,
        context: Option<&Arc<WgpuContext>>,
        device: Device,
    ) -> Result<(Option<Box<dyn CpuEncoder>>, Option<Box<dyn GpuEncoder>>)> {
        let mut cpu = None;
        let mut gpu = None;

        match device {
            Device::Cpu => {
                cpu = Some(Box::new(CpuTransformerEncoder::new(
                    weights,
                    meta.clone(),
                    layout.clone(),
                    load_config,
                )?) as Box<dyn CpuEncoder>);
            }
            Device::Wgpu => {
                let ctx = context.ok_or_else(|| anyhow!("GPU context required"))?;
                gpu = Some(Box::new(GpuTransformerEncoder::new(
                    weights,
                    ctx.clone(),
                    meta.clone(),
                    layout.clone(),
                    load_config,
                )?) as Box<dyn GpuEncoder>);
            }
        }

        Ok((cpu, gpu))
    }

    fn pooling_strategy() -> PoolingStrategy {
        PoolingStrategy::Mean
    }

    fn new_from_pipeline(
        pipeline: EncoderPipeline,
        tokenizer: Tokenizer,
        config: Arc<dyn ModelConfig>,
        model_type: Option<ModelType>,
    ) -> Self {
        Self {
            pipeline,
            tokenizer,
            config,
            model_type,
        }
    }
}

// =============================================================================
// Public API
// =============================================================================

impl SentenceEncoder {
    /// Create encoder from HuggingFace model registry.
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        EncoderLoader::load_from_registry::<Self>(
            model_type,
            cache_dir,
            device,
            context,
            load_config,
        )
        .await
    }

    /// Create encoder from local model directory.
    pub fn from_pretrained(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
        model_type: Option<ModelType>,
    ) -> Result<Self> {
        EncoderLoader::load_from_pretrained::<Self>(
            model_path,
            device,
            context,
            load_config,
            model_type,
        )
    }

    // ========================================================================
    // Encoding API
    // ========================================================================

    /// Encode text with default mean pooling and normalization.
    pub async fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let config = EncodingConfig {
            normalize: true,
            pooling_strategy: self.pipeline.pooling_strategy(),
        };
        <Self as SentenceEncoderModel>::encode(self, text, &config).await
    }

    /// Encode text without normalization (raw embeddings).
    pub async fn encode_raw(&self, text: &str) -> Result<Vec<f32>> {
        let config = EncodingConfig {
            normalize: false,
            pooling_strategy: self.pipeline.pooling_strategy(),
        };
        <Self as SentenceEncoderModel>::encode(self, text, &config).await
    }

    /// Encode with custom pooling strategy and optional normalization.
    ///
    /// # Arguments
    /// * `text` - The text to encode
    /// * `pooling_strategy` - Pooling method: "mean", "cls", "max", or "lastToken"
    /// * `normalize` - Whether to L2-normalize the output embedding
    ///
    /// # Example
    /// ```ignore
    /// let mean_embedding = encoder.encode_with(text, Some("mean"), true).await?;
    /// let cls_embedding = encoder.encode_with(text, Some("cls"), true).await?;
    /// ```
    pub async fn encode_with(
        &self,
        text: &str,
        pooling_strategy: Option<&str>,
        normalize: bool,
    ) -> Result<Vec<f32>> {
        let strategy = Self::parse_pooling_strategy(pooling_strategy)?;
        let config = EncodingConfig {
            normalize,
            pooling_strategy: strategy,
        };
        <Self as SentenceEncoderModel>::encode(self, text, &config).await
    }

    /// Encode batch of texts with default mean pooling and normalization.
    pub async fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let config = EncodingConfig {
            normalize: true,
            pooling_strategy: self.pipeline.pooling_strategy(),
        };
        <Self as SentenceEncoderModel>::encode_batch(self, texts, &config).await
    }

    /// Encode batch of texts without normalization (raw embeddings).
    pub async fn encode_batch_raw(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let config = EncodingConfig {
            normalize: false,
            pooling_strategy: self.pipeline.pooling_strategy(),
        };
        <Self as SentenceEncoderModel>::encode_batch(self, texts, &config).await
    }

    pub async fn encode_batch_flat(&self, texts: &[&str]) -> Result<(Vec<f32>, usize, usize)> {
        println!("Calling encode_batch_flat");
        // 1. Get raw hidden states (Array3<f32>)
        // Use your existing logic here
        let (hidden_states, attention_mask) = self.get_hidden_states_batch(texts).await?;

        // 2. Perform Pooling (e.g. Mean Pooling)
        // This usually returns Array2<f32> (batch_size, hidden_dim)
        let mut pooled = kjarni_transformers::mean_pool(&hidden_states, &attention_mask)?; // Adjust based on your pooling fn

        // 3. Normalize (in place)
        // Adjust based on config, but assuming normalize=true for this example
        kjarni_transformers::cpu::encoder::traits::l2_normalize_inplace(&mut pooled);

        // 4. THE OPTIMIZATION: Zero-copy conversion from Array2 to Vec
        let (rows, cols) = pooled.dim();

        // Ensure memory is contiguous (C-order). 
        // If it's already contiguous (which it usually is after pooling), this is free.
        // If not, it performs one efficient copy.
        let standard_layout = pooled.as_standard_layout();
        
        // Convert to Vec<f32> without re-allocating if possible
        let (flat_vec, _) = standard_layout.into_owned().into_raw_vec_and_offset();

        Ok((flat_vec, rows, cols))
    }

    /// Encode batch with custom pooling strategy and optional normalization.
    ///
    /// # Arguments
    /// * `texts` - The texts to encode
    /// * `pooling_strategy` - Pooling method: "mean", "cls", "max", or "lastToken"
    /// * `normalize` - Whether to L2-normalize the output embeddings
    ///
    /// # Example
    /// ```ignore
    /// let mean_raw = encoder.encode_batch_with(&texts, Some("mean"), false).await?;
    /// let cls_norm = encoder.encode_batch_with(&texts, Some("cls"), true).await?;
    /// ```
    pub async fn encode_batch_with(
        &self,
        texts: &[&str],
        pooling_strategy: Option<&str>,
        normalize: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let strategy = Self::parse_pooling_strategy(pooling_strategy)?;
        let config = EncodingConfig {
            normalize,
            pooling_strategy: strategy,
        };
        <Self as SentenceEncoderModel>::encode_batch(self, texts, &config).await
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Parse a string pooling strategy into the enum.
    fn parse_pooling_strategy(strategy: Option<&str>) -> Result<PoolingStrategy> {
        match strategy.unwrap_or("mean") {
            "mean" => Ok(PoolingStrategy::Mean),
            "cls" => Ok(PoolingStrategy::Cls),
            "max" => Ok(PoolingStrategy::Max),
            "lastToken" | "last_token" | "last" => Ok(PoolingStrategy::LastToken),
            other => Err(anyhow!(
                "Unknown pooling strategy '{}'. Supported: mean, cls, max, lastToken",
                other
            )),
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    pub fn pipeline(&self) -> &EncoderPipeline {
        &self.pipeline
    }

    pub fn embedding_dim(&self) -> usize {
        self.pipeline.hidden_size()
    }

    pub fn max_seq_length(&self) -> usize {
        self.pipeline.max_seq_length()
    }

    pub fn model_type(&self) -> Option<ModelType> {
        self.model_type
    }

    pub fn hidden_size(&self) -> usize {
        self.pipeline.hidden_size()
    }

    pub fn device(&self) -> Device {
        self.pipeline.plan().layers
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl LanguageModel for SentenceEncoder {
    fn vocab_size(&self) -> usize {
        self.pipeline.vocab_size()
    }
    fn hidden_size(&self) -> usize {
        self.pipeline.hidden_size()
    }
    fn num_layers(&self) -> usize {
        self.pipeline.num_layers()
    }
    fn num_heads(&self) -> usize {
        self.config.metadata().num_attention_heads
    }
    fn context_size(&self) -> usize {
        self.pipeline.max_seq_length()
    }
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn bos_token_id(&self) -> Option<u32> {
        self.tokenizer.token_to_id("[CLS]")
    }
    fn eos_token_id(&self) -> Option<u32> {
        self.tokenizer.token_to_id("[SEP]")
    }
    fn pad_token_id(&self) -> Option<u32> {
        self.tokenizer.token_to_id("[PAD]")
    }
    fn forced_bos_token_id(&self) -> Option<u32> {
        None
    }
    fn forced_eos_token_id(&self) -> Option<u32> {
        None
    }
    fn new_cache(&self, _: usize, _: usize, _: usize) -> Result<Box<dyn Cache>> {
        Err(anyhow!("Sentence encoders do not use KV cache"))
    }
}

impl InferenceModel for SentenceEncoder {
    fn device(&self) -> Device {
        self.pipeline.plan().layers
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        self.pipeline.context().cloned()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CpuEncoderOps for SentenceEncoder {
    fn encoder(&self) -> &dyn CpuEncoder {
        self.pipeline
            .cpu_encoder()
            .expect("CPU encoder not available")
    }
    fn embed_tokens(
            &self,
            input_ids: &ndarray::Array2<u32>,
            token_type_ids: Option<&ndarray::Array2<u32>>,
            pos: usize,
        ) -> Result<ndarray::Array3<f32>> {
        self.pipeline
            .embeddings()
            .embed_cpu(input_ids, token_type_ids, pos)
    }
}

impl GpuEncoderOps for SentenceEncoder {
    fn encoder(&self) -> &dyn GpuEncoder {
        self.pipeline
            .gpu_encoder()
            .expect("GPU encoder not available")
    }
}

#[async_trait]
impl EncoderLanguageModel for SentenceEncoder {
    fn encoder_cpu_ops(&self) -> Option<&dyn CpuEncoderOps> {
        if self.pipeline.cpu_encoder().is_some() {
            Some(self)
        } else {
            None
        }
    }

    fn encoder_gpu_ops(&self) -> Option<&dyn GpuEncoderOps> {
        if self.pipeline.gpu_encoder().is_some() {
            Some(self)
        } else {
            None
        }
    }
}
