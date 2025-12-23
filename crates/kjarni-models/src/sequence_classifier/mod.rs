//! Cross-encoder for reranking and pairwise classification
//!
//! Takes two texts as input and outputs a relevance score.
//! Used for reranking search results or computing pairwise similarity.

use anyhow::{Result, anyhow};
use ndarray::Array2;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

use async_trait::async_trait;

use kjarni_transformers::{
    WgpuContext,
    cache::Cache,
    encoder::{
        CpuEncoder, CpuTransformerEncoder, GpuEncoder, GpuTransformerEncoder,
        classifier::{CpuSequenceClassificationHead, GpuSequenceClassificationHead},
        traits::{CpuEncoderOps, EncoderLanguageModel, GpuEncoderInput, GpuEncoderOps},
    },
    gpu_ops::{GpuFrameContext, GpuTensor},
    linear_layer::LinearLayer,
    models::{LanguageModel, ModelArchitecture, ModelType, download_model_files},
    traits::Device,
    weights::ModelWeights,
};
mod configs;
pub use configs::MiniLMCrossEncoderConfig;
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::traits::{InferenceModel, ModelConfig, ModelLayout, ModelMetadata};

/// A generic sequence classifier for running models like BERT and RoBERTa on classification tasks.
///
/// This struct acts as a "Model Container". It owns the concrete compute components
/// (e.g., `CpuTransformerEncoder`, `CpuSequenceClassificationHead`) and implements the
/// high-level `EncoderLanguageModel` trait, making it a fully functional part of the engine.
pub struct SequenceClassifier {
    cpu_encoder: Option<CpuTransformerEncoder>,
    gpu_encoder: Option<GpuTransformerEncoder>,
    cpu_head: Option<CpuSequenceClassificationHead>,
    gpu_head: Option<GpuSequenceClassificationHead>,

    tokenizer: Tokenizer,
    model_type: ModelType,
    device: Device,
    context: Option<Arc<WgpuContext>>,
    pub meta: ModelMetadata,
    pub layout: ModelLayout,
    config: Arc<dyn ModelConfig>,
}

impl SequenceClassifier {
    /// Supported cross-encoder model types
    const SUPPORTED_MODELS: &'static [ModelType] = &[ModelType::MiniLML6V2CrossEncoder];

    /// Create cross-encoder from HuggingFace registry
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_cfg: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!(
                "Unsupported model type: {:?}. Supported: {:?}",
                model_type,
                Self::SUPPORTED_MODELS
            ));
        }

        let info = model_type.info();

        if info.architecture != ModelArchitecture::CrossEncoder {
            return Err(anyhow!(
                "Model {:?} is not a cross-encoder (architecture: {:?})",
                model_type,
                info.architecture
            ));
        }

        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::cache_dir()
                .expect("No cache directory found")
                .join("kjarni")
        });

        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

        // Download files
        download_model_files(&model_dir, &info.paths).await?;

        // Load from local path
        Self::from_pretrained(&model_dir, model_type, device, context, load_cfg)
    }

    /// Create cross-encoder from local model directory
    pub fn from_pretrained(
        model_path: &Path,
        model_type: ModelType,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_cfg: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!(
                "CrossEncoder: Unsupported model type: {:?}",
                model_type
            ));
        }

        let weights = ModelWeights::new(model_path)?;

        let mut tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Load encoder
        let config: Arc<dyn ModelConfig> = match model_type {
            ModelType::MiniLML6V2CrossEncoder => {
                let config = Arc::new(MiniLMCrossEncoderConfig::from_json(&weights.config_json)?);
                config
            }
            _ => return Err(anyhow!("Unsupported cross-encoder: {:?}", model_type)),
        };
        let meta = config.metadata();
        let layout = config.layout();
        let load_cfg = load_cfg.unwrap_or_default();
        // Note: For BERT-style classification, we use the final_norm (Pooler) and lm_head (Classifier)
        let pooler = LinearLayer::from_weights(
            &weights,
            &layout.final_norm,
            None,
            load_cfg.target_dtype,
            None,
        )?;
        let classifier = LinearLayer::from_weights(
            &weights,
            &layout.lm_head,
            None,
            load_cfg.target_dtype,
            None,
        )?;

        let mut cpu_encoder = None;
        let mut gpu_encoder = None;
        let mut cpu_head = None;
        let mut gpu_head = None;

        match device {
            Device::Cpu => {
                cpu_encoder = Some(CpuTransformerEncoder::new(
                    &weights,
                    meta.clone(),
                    layout.clone(),
                    load_cfg,
                )?);
                cpu_head = Some(CpuSequenceClassificationHead::new(
                    Some(pooler),
                    classifier,
                )?);
            }
            Device::Wgpu => {
                let ctx = context
                    .clone()
                    .ok_or_else(|| anyhow!("WGPU context required"))?;
                gpu_encoder = Some(GpuTransformerEncoder::new(
                    &weights,
                    ctx.clone(),
                    meta.clone(),
                    layout.clone(),
                    load_cfg,
                )?);

                gpu_head = Some(GpuSequenceClassificationHead::new(
                    &ctx,
                    Some(pooler.to_gpu(&ctx)?),
                    pooler.bias_to_gpu(&ctx)?,
                    classifier.to_gpu(&ctx)?,
                    classifier.bias_to_gpu(&ctx)?.unwrap(),
                )?);
            }
        }
        let truncation_params = tokenizers::TruncationParams {
            max_length: meta.max_seq_len,
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
            cpu_head,
            gpu_head,
            tokenizer,
            config,
            model_type,
            device,
            context,
            meta,
            layout,
        })
    }

    /// Predicts classification scores for a batch of single texts (e.g., sentiment analysis).
    ///
    /// # Arguments
    /// * `texts`: A slice of string slices to classify.
    ///
    /// # Returns
    /// A `Vec<Vec<f32>>` where each inner vector contains the logits for each class for the corresponding input text.
    pub async fn predict(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // 1. Get hidden states from the base EncoderLanguageModel trait.
        // This is the "dumb backend" part that handles tokenization, masking, and CPU/GPU dispatch.
        let (hidden_states, _attention_mask) = self.get_hidden_states_batch(texts).await?;

        // 2. Pass the hidden states to the appropriate device-specific classification head.
        let logits = if self.device.is_cpu() {
            // CPU path is straightforward.
            self.cpu_head.as_ref().unwrap().forward(&hidden_states)?
        } else if self.device.is_gpu() {
            // GPU path requires managing the frame and data transfers.
            let context = self.context.as_ref().unwrap();
            let pool = context.get_inference_pool();
            let mut pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(context, pool_guard);

            // The `get_hidden_states_batch` already ran on GPU and downloaded.
            // For a pure GPU path, we would avoid this download/re-upload.
            // This implementation is a "simple but correct" hybrid path.
            let hidden_states_gpu = GpuTensor::from_ndarray(context, &hidden_states)?;

            let logits_gpu = self
                .gpu_head
                .as_ref()
                .unwrap()
                .forward(&mut frame, &hidden_states_gpu)?;

            frame.finish();
            logits_gpu.to_ndarray_2d().await?
        } else {
            return Err(anyhow!("No backend available for this device."));
        };

        Ok(logits.outer_iter().map(|row| row.to_vec()).collect())
    }

    /// Predicts a relevance score for a batch of text pairs (Cross-Encoder functionality).
    ///
    /// # Arguments
    /// * `pairs`: A slice of `(&str, &str)` tuples to score.
    ///
    /// # Returns
    /// A `Vec<f32>` containing the single relevance score for each pair.
    pub async fn predict_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        // 1. Tokenization logic is the same: manually create CPU tensors.
        let encodings = self.tokenizer.encode_batch(pairs.to_vec(), true).unwrap();

        let batch_size = encodings.len();
        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
        let mut batch_input_ids = Array2::<u32>::zeros((batch_size, max_len));
        let mut batch_attention_mask = Array2::<f32>::zeros((batch_size, max_len));
        let mut batch_token_type_ids = Array2::<u32>::zeros((batch_size, max_len));

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();
            // Pad to max_len
            for j in 0..ids.len() {
                batch_input_ids[[i, j]] = ids[j];
                batch_attention_mask[[i, j]] = attention_mask[j] as f32;
                batch_token_type_ids[[i, j]] = type_ids[j];
            }
        }

        // 2. Dispatch to the correct backend using the `ops` traits.
        let logits = if let Some(ops) = self.encoder_cpu_ops() {
            // --- CPU PATH ---
            // a) Get hidden states from the encoder component.
            let hidden_states = ops
                .encoder()
                .forward(
                    &batch_input_ids,
                    &batch_attention_mask,
                    Some(&batch_token_type_ids),
                )?
                .last_hidden_state;

            // b) Pass the full hidden states to the classification head.
            // The head knows how to extract the CLS token and do the rest.
            self.cpu_head.as_ref().unwrap().forward(&hidden_states)?
        } else if let Some(ops) = self.encoder_gpu_ops() {
            // --- GPU PATH ---
            let context = self.context.as_ref().unwrap();
            let pool = context.get_inference_pool();
            let mut pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(context, pool_guard);

            // a) Upload data and get hidden states from the encoder component.
            let input_ids_gpu = GpuTensor::from_ndarray(context, &batch_input_ids)?;
            let attention_mask_gpu = GpuTensor::from_ndarray(context, &batch_attention_mask)?;
            let token_type_ids_gpu = GpuTensor::from_ndarray(context, &batch_token_type_ids)?;

            let (encoder_cmd, pool_ref) = frame.resources();
            let hidden_states_gpu = ops
                .encoder()
                .forward(
                    encoder_cmd,
                    pool_ref,
                    GpuEncoderInput::TokensGpu(&input_ids_gpu),
                    &attention_mask_gpu,
                    Some(&token_type_ids_gpu),
                )?
                .last_hidden_state;

            // b) Pass the full hidden states to the GPU classification head.
            let logits_gpu = self
                .gpu_head
                .as_ref()
                .unwrap()
                .forward(&mut frame, &hidden_states_gpu)?;

            frame.finish();

            // c) Download the final result.
            logits_gpu.to_ndarray_2d().await?
        } else {
            return Err(anyhow!("No backend available"));
        };

        // 3. Extract the single score column.
        Ok(logits.column(0).to_vec())
    }

    /// Score multiple text pairs
    ///
    /// # Example
    /// ```no_run
    /// use kjarni_models::cross_encoder::CrossEncoder;
    /// # async fn example(encoder: &CrossEncoder) -> anyhow::Result<()> {
    /// let pairs = [
    ///     ("query", "relevant document"),
    ///     ("query", "irrelevant document"),
    /// ];
    /// let scores = encoder.predict_batch(&pairs).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn predict_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>> {
        self.predict_pairs(pairs).await
    }
    /// A convenience wrapper around `predict_pairs` for a single text pair.
    /// This is the method your test `test_torch_cross_encoder_predict` needs.
    pub async fn predict_pair(&self, text1: &str, text2: &str) -> Result<f32> {
        let scores = self.predict_pairs(&[(text1, text2)]).await?;
        scores
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Prediction for pair returned no score."))
    }
    /// Reranks a list of documents against a query, returning sorted (index, score) tuples.
    pub async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }
        let pairs: Vec<(&str, &str)> = documents.iter().map(|doc| (query, *doc)).collect();
        let scores = self.predict_pairs(&pairs).await?;
        let mut indexed_scores: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(indexed_scores)
    }

    pub async fn rerank_indices(&self, query: &str, documents: &[&str]) -> Result<Vec<usize>> {
        let ranked = self.rerank(query, documents).await?;
        Ok(ranked.into_iter().map(|(idx, _)| idx).collect())
    }

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

    pub async fn rerank_top_k_indices(
        &self,
        query: &str,
        documents: &[&str],
        k: usize,
    ) -> Result<Vec<usize>> {
        let top_k = self.rerank_top_k(query, documents, k).await?;
        Ok(top_k.into_iter().map(|(idx, _)| idx).collect())
    }

    /// Get the maximum sequence length
    pub fn max_seq_length(&self) -> usize {
        self.meta.max_seq_len
    }
}

impl LanguageModel for SequenceClassifier {
    // Implement same getters as SentenceEncoder using self.meta
    fn vocab_size(&self) -> usize {
        self.meta.vocab_size
    }
    fn hidden_size(&self) -> usize {
        self.meta.hidden_size
    }
    fn num_layers(&self) -> usize {
        self.meta.num_layers
    }
    fn num_heads(&self) -> usize {
        self.meta.num_attention_heads
    }
    fn context_size(&self) -> usize {
        self.meta.max_seq_len
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

    fn new_cache(&self, _: usize, _: usize, _: usize) -> Result<Box<dyn Cache>> {
        Err(anyhow!("Sequence Classifiers do not use a KV cache."))
    }
}

impl InferenceModel for SequenceClassifier {
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

impl CpuEncoderOps for SequenceClassifier {
    fn encoder(&self) -> &dyn CpuEncoder {
        self.cpu_encoder
            .as_ref()
            .expect("CPU encoder not initialized")
    }
}

impl GpuEncoderOps for SequenceClassifier {
    fn encoder(&self) -> &dyn GpuEncoder {
        self.gpu_encoder
            .as_ref()
            .expect("GPU encoder not initialized")
    }
}

#[async_trait]
impl EncoderLanguageModel for SequenceClassifier {
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
