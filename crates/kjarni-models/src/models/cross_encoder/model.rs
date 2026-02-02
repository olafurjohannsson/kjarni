//! Cross-encoder for semantic similarity and reranking.
//!
//! Takes query-document pairs and outputs relevance scores.
//! Unlike bi-encoders (which encode texts separately), cross-encoders
//! process the pair together for more accurate scoring.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use kjarni_transformers::gpu::{GpuFrameContext, GpuTensor};
use kjarni_transformers::models::base::ModelInput;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

use kjarni_transformers::{
    WgpuContext,
    cpu::encoder::{
        CpuTransformerEncoder, GpuTransformerEncoder,
        classifier::CpuSequenceClassificationHead,
        config::PoolingStrategy,
        traits::{CpuEncoder, CpuEncoderOps, EncoderLanguageModel, GpuEncoder, GpuEncoderOps},
    },
    models::base::ModelLoadConfig,
    models::{LanguageModel, ModelType},
    pipeline::{EncoderLoader, EncoderModelFactory, EncoderPipeline},
    traits::{Cache, Device, InferenceModel, ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};

use crate::BertConfig;
use crate::models::sequence_classifier::configs::MiniLMCrossEncoderConfig;

// =============================================================================
// Cross Encoder
// =============================================================================

/// Cross-encoder for semantic similarity and reranking.
///
/// Processes query-document pairs together through the encoder,
/// producing a single relevance score per pair.
pub struct CrossEncoder {
    pipeline: EncoderPipeline,
    tokenizer: Tokenizer,
    config: Arc<dyn ModelConfig + Send + Sync>,
    model_type: Option<ModelType>,
}

// =============================================================================
// EncoderModelFactory Implementation
// =============================================================================

impl EncoderModelFactory for CrossEncoder {
    fn load_config(weights: &ModelWeights) -> Result<Arc<dyn ModelConfig>> {
        // Cross-encoders are typically BERT-based
        if weights
            .config_json()
            .contains("cross-encoder/ms-marco-MiniLM")
        {
            Ok(Arc::new(MiniLMCrossEncoderConfig::from_json(
                weights.config_json(),
            )?))
        } else {
            // Default to BERT config
            Ok(Arc::new(BertConfig::from_json(weights.config_json())?))
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
            _ => {
                panic!("No CPU or GPU encoder on CrossEncoder, will not work!");
            }
        }

        Ok((cpu, gpu))
    }

    fn build_head(
        weights: &ModelWeights,
        load_config: &ModelLoadConfig,
    ) -> Result<Option<CpuSequenceClassificationHead>> {
        // Cross-encoders output a single score (num_labels = 1)
        // Use from_weights which auto-detects the head structure
        Ok(Some(CpuSequenceClassificationHead::from_weights(
            weights,
            load_config,
            None, // No label names for regression
        )?))
    }

    fn pooling_strategy() -> PoolingStrategy {
        PoolingStrategy::Cls
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

impl CrossEncoder {
    /// Create cross-encoder from HuggingFace model registry.
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

    /// Create cross-encoder from local model directory.
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

    // =========================================================================
    // Pair Scoring API
    // =========================================================================

    /// Score a single query-document pair.
    pub async fn predict_pair(&self, query: &str, document: &str) -> Result<f32> {
        let scores = self.predict_pairs(&[(query, document)]).await?;
        scores
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No score returned"))
    }

    /// Score multiple query-document pairs.
    pub async fn predict_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize pairs - the tokenizer handles [CLS] query [SEP] document [SEP]
        let encodings = self
            .tokenizer
            .encode_batch(pairs.to_vec(), true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let batch_size = encodings.len();
        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);

        let mut input_ids = ndarray::Array2::<u32>::zeros((batch_size, max_len));
        let mut attention_mask = ndarray::Array2::<f32>::zeros((batch_size, max_len));
        let mut token_type_ids = ndarray::Array2::<u32>::zeros((batch_size, max_len));

        for (i, encoding) in encodings.iter().enumerate() {
            for (j, &id) in encoding.get_ids().iter().enumerate() {
                input_ids[[i, j]] = id;
                attention_mask[[i, j]] = encoding.get_attention_mask()[j] as f32;
                token_type_ids[[i, j]] = encoding.get_type_ids()[j];
            }
        }

        let hidden_states = if let Some(ops) = self.encoder_cpu_ops() {
            ops.forward_tokens(
                &input_ids,
                Some(&attention_mask),
                Some(&token_type_ids),
                0
            )?.last_hidden_state
        } else if let Some(ops) = self.encoder_gpu_ops() {
            let context = self
                .context()
                .ok_or_else(|| anyhow!("GPU model missing context"))?;
            let pool = context.get_inference_pool();
            let mut pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&context, pool_guard);
            let (encoder_cmd, pool_ref) = frame.resources();

            // Upload data to GPU
            let input_ids_gpu = GpuTensor::from_ndarray(&context, &input_ids)?;
            let attention_mask_gpu = GpuTensor::from_ndarray(&context, &attention_mask)?;
            let token_types_gpu = GpuTensor::from_ndarray(&context, &token_type_ids)?;
            // Run the forward pass
            let gpu_output = ops.encoder().forward(
                encoder_cmd,
                pool_ref,
                ModelInput::TokensGpu(&input_ids_gpu),
                &attention_mask_gpu,
                Some(ModelInput::TokensGpu(&token_types_gpu)),
            )?;

            frame.finish();

            // Download the result back to CPU
            gpu_output.last_hidden_state.to_ndarray_3d().await?
        } else {
            return Err(anyhow!(
                "No available CPU or GPU encoder implementation for this model."
            ));
        };

        // Forward through head // TODO: lm head on GPU
        let logits = self
            .pipeline
            .cpu_head()
            .ok_or_else(|| anyhow!("No classification head available"))?
            .forward(&hidden_states, Some(&attention_mask))?;

        // Extract scores (first column for regression)
        Ok(logits.column(0).to_vec())
    }

    // =========================================================================
    // Reranking API
    // =========================================================================

    /// Rerank documents by relevance to a query.
    ///
    /// Returns (original_index, score) tuples sorted by score descending.
    pub async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let pairs: Vec<(&str, &str)> = documents.iter().map(|doc| (query, *doc)).collect();
        let scores = self.predict_pairs(&pairs).await?;

        let mut ranked: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(ranked)
    }

    /// Rerank and return only top-k results.
    pub async fn rerank_top_k(
        &self,
        query: &str,
        documents: &[&str],
        k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let mut ranked = self.rerank(query, documents).await?;
        ranked.truncate(k);
        Ok(ranked)
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    pub fn pipeline(&self) -> &EncoderPipeline {
        &self.pipeline
    }

    pub fn max_seq_length(&self) -> usize {
        self.pipeline.max_seq_length()
    }

    pub fn hidden_size(&self) -> usize {
        self.pipeline.hidden_size()
    }

    pub fn model_type(&self) -> Option<ModelType> {
        self.model_type
    }

    pub fn device(&self) -> Device {
        self.pipeline.plan().layers
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl LanguageModel for CrossEncoder {
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
        Err(anyhow!("CrossEncoder does not use KV cache"))
    }
}

impl InferenceModel for CrossEncoder {
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


impl CpuEncoderOps for CrossEncoder {
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

impl GpuEncoderOps for CrossEncoder {
    fn encoder(&self) -> &dyn GpuEncoder {
        self.pipeline
            .gpu_encoder()
            .expect("GPU encoder not available")
    }
}

#[async_trait]
impl EncoderLanguageModel for CrossEncoder {
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
