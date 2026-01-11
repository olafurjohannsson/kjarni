use anyhow::{Result, anyhow};
use kjarni_transformers::cpu::encoder::prelude::{CpuSequenceClassificationHead, GpuSequenceClassificationHead};
use kjarni_transformers::cpu::encoder::{
    CpuEncoder, CpuEncoderOps, GpuEncoder, GpuEncoderOps, GpuTransformerEncoder,
};
use kjarni_transformers::gpu_ops::primitives::layout::clsslice::GpuClsSlice;
use kjarni_transformers::gpu_ops::primitives::linear::GpuLinearLayer;
use kjarni_transformers::gpu_ops::{GpuFrameContext, GpuTensor};
use kjarni_transformers::models::base::{ModelInput, ModelLoadConfig};
use kjarni_transformers::models::download_model_files;
use kjarni_transformers::models::registry::WeightsFormat;
use kjarni_transformers::traits::InferenceModel;
use ndarray::{Array2, Array3, s};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::{EncodeInput, Tokenizer};

use kjarni_transformers::{
    cpu::encoder::CpuTransformerEncoder,
    cpu::encoder::traits::EncoderLanguageModel,
    linear_layer::LinearLayer,
    models::{ModelArchitecture, ModelTask, ModelType},
    traits::{Device, ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};
// /// A CPU-based head for sequence classification tasks (e.g., sentiment, NLI, reranking).
// pub struct CpuSequenceClassificationHead {
//     pooler: Option<LinearLayer>,
//     classifier: LinearLayer,
// }

// impl CpuSequenceClassificationHead {
//     pub fn new(pooler: Option<LinearLayer>, classifier: LinearLayer) -> Result<Self> {
//         if let Some(p) = &pooler {
//             if p.out_features() != classifier.in_features() {
//                 return Err(anyhow!(
//                     "Dimension mismatch: Pooler output ({}) does not match Classifier input ({})",
//                     p.out_features(),
//                     classifier.in_features()
//                 ));
//             }
//         }
//         Ok(Self { pooler, classifier })
//     }

//     /// Takes the full sequence of hidden states and produces final logits.
//     ///
//     /// # Arguments
//     /// * `encoder_hidden_states`: Shape `[batch, seq_len, hidden_size]`.
//     ///
//     /// # Returns
//     /// * Logits with shape `[batch, num_classes]`.
//     pub fn forward(&self, encoder_hidden_states: &Array3<f32>) -> Result<Array2<f32>> {
//         let (batch, seq_len, _hidden_size) = encoder_hidden_states.dim();

//         if batch == 0 || seq_len == 0 {
//             return Ok(Array2::<f32>::zeros((batch, self.num_classes())));
//         }

//         let cls_embedding = encoder_hidden_states.slice(s![.., 0, ..]).to_owned();

//         let pooled_output = if let Some(p) = &self.pooler {
//             let mut pooled = p.matmul(&cls_embedding.view());
//             pooled.mapv_inplace(f32::tanh);
//             pooled
//         } else {
//             cls_embedding
//         };

//         let logits = self.classifier.matmul(&pooled_output.view());
//         Ok(logits)
//     }

//     pub fn num_classes(&self) -> usize {
//         self.classifier.out_features()
//     }
// }

/// Cross-encoder for semantic similarity and reranking. 
///     (Encoder → head → score (text pairs))
/// Takes query-document pairs and outputs relevance scores.
/// Unlike bi-encoders (which encode texts separately), cross-encoders
/// process the pair together for more accurate scoring.
pub struct CrossEncoder {
    // Encoder components
    cpu_encoder: Option<CpuTransformerEncoder>,
    gpu_encoder: Option<GpuTransformerEncoder>,
    cpu_head: Option<CpuSequenceClassificationHead>,
    gpu_head: Option<GpuSequenceClassificationHead>,
    // Regression/classification head
    head: Option<LinearLayer>,
    // gpu_head: Option<GpuLinearLayer>,
    gpu_cls_slice: Option<GpuClsSlice>,

    // Tokenizer and config
    tokenizer: Tokenizer,
    config: Arc<dyn ModelConfig>,

    // Model info
    model_type: ModelType,
    device: Device,
    context: Option<Arc<WgpuContext>>,
    meta: ModelMetadata,
    layout: ModelLayout,
}

impl CrossEncoder {
    /// Supported cross-encoder model types
    const SUPPORTED_MODELS: &'static [ModelType] = &[
        ModelType::MiniLML6V2CrossEncoder,
        // Add more cross-encoder models here as needed
    ];

    // =========================================================================
    // Construction
    // =========================================================================

    /// Create cross-encoder from HuggingFace model registry.
    ///
    /// Automatically downloads model files to cache directory.
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
                "CrossEncoder: Unsupported model type: {:?}. Supported: {:?}",
                model_type,
                Self::SUPPORTED_MODELS
            ));
        }

        let info = model_type.info();

        // Validate task
        if info.task != ModelTask::ReRanking {
            return Err(anyhow!(
                "Model {:?} is not a cross-encoder (task: {:?})",
                model_type,
                info.task
            ));
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

    /// Create cross-encoder from local model directory.
    pub fn from_pretrained(
        model_path: &Path,
        model_type: ModelType,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_cfg: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        // Validate model type
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!("Unsupported cross-encoder model: {:?}", model_type));
        }

        // Load weights
        let weights = ModelWeights::new(model_path)?;
        let load_cfg = load_cfg.unwrap_or_default();

        // Load config with auto-detected weight prefix
        let config: Arc<dyn ModelConfig> = Self::load_config(model_type, &weights)?;
        let meta = config.metadata();
        let layout = config.layout();

        // Load tokenizer
        let mut tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Configure tokenizer for pairs
        let truncation_params = tokenizers::TruncationParams {
            max_length: meta.max_seq_len,
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            ..Default::default()
        };
        let _ = tokenizer.with_truncation(Some(truncation_params));

        let padding_params = tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(padding_params));

        // Load encoder
        let mut cpu_encoder = None;
        let mut gpu_encoder = None;
        // let mut cpu_head = None;
        let mut gpu_head = None;
        let mut gpu_cls_slice = None;
        let encoder_layout = &layout.encoder.as_ref().ok_or_else(|| {
            anyhow!(
                "Model layout for {:?} missing encoder component",
                model_type
            )
        })?;
        let name = encoder_layout.final_norm_weight.as_deref().unwrap();
        let pooler = LinearLayer::builder(&weights, name)
            .with_target_dtype(load_cfg.target_dtype)
            .with_optional_bias(None)
            .build()?;

        let classifier = LinearLayer::builder(&weights, &layout.lm_head)
            .with_optional_bias(None)
            .with_target_dtype(load_cfg.target_dtype)
            .build()?;
        let mut cpu_head = None;
        match device {
            Device::Cpu => {
                cpu_encoder = Some(CpuTransformerEncoder::new(
                    &weights,
                    meta.clone(),
                    layout.clone(),
                    load_cfg.clone(),
                )?);
                cpu_head = Some(CpuSequenceClassificationHead::new(
                    Some(pooler),
                    classifier,
                )?);
                // cpu_head = Some(Self::load_head(&weights, &load_cfg)?);
            }
            Device::Wgpu => {
                let ctx = context
                    .clone()
                    .ok_or_else(|| anyhow!("WGPU context required for GPU device"))?;
                gpu_encoder = Some(GpuTransformerEncoder::new(
                    &weights,
                    ctx.clone(),
                    meta.clone(),
                    layout.clone(),
                    load_cfg.clone(),
                )?);

                gpu_head = Some(GpuSequenceClassificationHead::new(
                    &ctx,
                    Some(pooler.to_gpu(&ctx)?),
                    pooler.bias_to_gpu(&ctx)?,
                    classifier.to_gpu(&ctx)?,
                    classifier.bias_to_gpu(&ctx)?.unwrap(),
                )?);
                gpu_cls_slice = Some(GpuClsSlice::new(&ctx));
            }
        }

        Ok(Self {
            cpu_encoder,
            gpu_encoder,
            // head: cpu_head,
            head: None,
            cpu_head,
            gpu_cls_slice,
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

    /// Load model config with auto-detected weight prefix.
    fn load_config(model_type: ModelType, weights: &ModelWeights) -> Result<Arc<dyn ModelConfig>> {
        let arch = model_type.info().architecture;

        // Auto-detect weight prefix by checking what exists in the weights file
        // let prefix = Self::detect_weight_prefix(weights)?;

        match arch {
            ModelArchitecture::Bert => {
                // Check model_type in JSON to determine exact config type
                println!("weights: {:?}", weights.config_json);
                if weights
                    .config_json
                    .contains("\"_name_or_path\": \"cross-encoder/ms-marco-MiniLM-L-12-v2\"")
                    || weights
                        .config_json
                        .contains("\"_name_or_path\": \"cross-encoder/ms-marco-MiniLM-L-12-v2\"")
                {
                    let config = MiniLMCrossEncoderConfig::from_json(&weights.config_json)?;
                    Ok(Arc::new(config))
                } else {
                    panic!("Not implemented")
                }
            }
            _ => Err(anyhow!(
                "Unsupported architecture for cross-encoder: {:?}",
                arch
            )),
        }
    }

    fn load_head_name(weights: &ModelWeights) -> Result<(&str, &str)> {
        // Cross-encoders typically have a simple head: classifier.weight + classifier.bias
        // Output is usually 1 (regression score) or num_labels (classification)

        // LoadedLMHead::new(None, &weights, LMHeadConfig::new())

        if weights.contains("classifier.weight") {
            Ok(("classifier.weight", "classifier.bias"))
        } else if weights.contains("classifier.out_proj.weight") {
            Ok(("classifier.out_proj.weight", "classifier.out_proj.bias"))
        } else {
            Err(anyhow!(
                "Could not find classification head weights. Checked for 'classifier.weight' and 'classifier.out_proj.weight'"
            ))
        }
    }

    /// Auto-detect the weight prefix used in the model file.
    fn detect_weight_prefix(weights: &ModelWeights) -> Result<String> {
        // Try common prefixes
        let prefixes = ["bert.", "roberta.", "distilbert.", "model.", ""];

        for prefix in prefixes {
            let test_key = format!("{}embeddings.word_embeddings.weight", prefix);
            if weights.contains(&test_key) {
                return Ok(prefix.to_string());
            }
        }

        // Also check for encoder-specific paths
        for prefix in prefixes {
            let test_key = format!("{}encoder.embeddings.word_embeddings.weight", prefix);
            if weights.contains(&test_key) {
                return Ok(format!("{}encoder.", prefix));
            }
        }

        Err(anyhow!(
            "Could not detect weight prefix. Checked for embeddings.word_embeddings.weight with prefixes: {:?}",
            prefixes
        ))
    }

    /// Load the classification/regression head.
    fn load_head(weights: &ModelWeights, load_cfg: &ModelLoadConfig) -> Result<LinearLayer> {
        // Cross-encoders typically have a simple head: classifier.weight + classifier.bias
        // Output is usually 1 (regression score) or num_labels (classification)

        if weights.contains("classifier.weight") {
            LinearLayer::builder(weights, "classifier.weight")
                .with_optional_bias(Some("classifier.bias"))
                .with_target_dtype(load_cfg.target_dtype)
                .build()
        } else if weights.contains("classifier.out_proj.weight") {
            // Some models use out_proj naming
            LinearLayer::builder(weights, "classifier.out_proj.weight")
                .with_optional_bias(Some("classifier.out_proj.bias"))
                .with_target_dtype(load_cfg.target_dtype)
                .build()
        } else {
            Err(anyhow!(
                "Could not find classification head weights. Checked for 'classifier.weight' and 'classifier.out_proj.weight'"
            ))
        }
    }

    // =========================================================================
    // Pair Scoring API
    // =========================================================================

    pub async fn predict_pair(&self, query: &str, document: &str) -> Result<f32> {
        let scores = self.predict_pairs(&[(query, document)]).await?;
        scores
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No score returned"))
    }

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

            self.cpu_head.as_ref().unwrap().forward(&hidden_states, None)?
        } else if let Some(ops) = self.encoder_gpu_ops() {
            let context = self.context.as_ref().unwrap();
            let pool = context.get_inference_pool();
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(context, pool_guard);


            let input_ids_gpu = GpuTensor::from_ndarray(context, &batch_input_ids)?;
            let attention_mask_gpu = GpuTensor::from_ndarray(context, &batch_attention_mask)?;
            let token_type_ids_gpu = GpuTensor::from_ndarray(context, &batch_token_type_ids)?;

            let (encoder_cmd, pool_ref) = frame.resources();
            let hidden_states_gpu = ops
                .encoder()
                .forward(
                    encoder_cmd,
                    pool_ref,
                    ModelInput::TokensGpu(&input_ids_gpu),
                    &attention_mask_gpu,
                    Some(ModelInput::TokensGpu(&token_type_ids_gpu)),
                )?
                .last_hidden_state;

            let logits_gpu = self
                .gpu_head
                .as_ref()
                .unwrap()
                .forward(&mut frame, &hidden_states_gpu)?;

            frame.finish();

            logits_gpu.to_ndarray_2d().await?
        } else {
            return Err(anyhow!("No backend available"));
        };

        Ok(logits.column(0).to_vec())
    }

    // =========================================================================
    // Reranking API
    // =========================================================================

    /// Rerank documents by relevance to a query.
    ///
    /// # Arguments
    /// * `query` - The search query
    /// * `documents` - Documents to rerank
    ///
    /// # Returns
    /// Vector of (original_index, score) tuples sorted by score descending.
    pub async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // Create pairs: same query with each document
        let pairs: Vec<(&str, &str)> = documents.iter().map(|doc| (query, *doc)).collect();

        // Score all pairs
        let scores = self.predict_pairs(&pairs).await?;

        // Create (index, score) tuples and sort by score descending
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

    /// Rerank with owned strings (convenience method).
    pub async fn rerank_owned(
        &self,
        query: &str,
        documents: &[String],
    ) -> Result<Vec<(usize, f32)>> {
        let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        self.rerank(query, &doc_refs).await
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the maximum sequence length.
    pub fn max_seq_length(&self) -> usize {
        self.meta.max_seq_len
    }

    /// Get the hidden size.
    pub fn hidden_size(&self) -> usize {
        self.meta.hidden_size
    }

    /// Get the model type.
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Get the device.
    pub fn device(&self) -> Device {
        self.device
    }
}

impl CrossEncoder {
    fn encoder_cpu_ops(&self) -> Option<&dyn CpuEncoderOps> {
        if self.cpu_encoder.is_some() {
            Some(self)
        } else {
            None
        }
    }

    fn encoder_gpu_ops(&self) -> Option<&dyn GpuEncoderOps> {
        if self.gpu_encoder.is_some() {
            Some(self)
        } else {
            None
        }
    }
}

impl CpuEncoderOps for CrossEncoder {
    fn encoder(&self) -> &dyn CpuEncoder {
        self.cpu_encoder
            .as_ref()
            .expect("CPU encoder not initialized for this model.")
    }
}

impl GpuEncoderOps for CrossEncoder {
    fn encoder(&self) -> &dyn GpuEncoder {
        self.gpu_encoder
            .as_ref()
            .expect("GPU encoder not initialized for this model.")
    }
}

impl InferenceModel for CrossEncoder {
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
use kjarni_transformers::{LanguageModel, WgpuContext};

use crate::cross_encoder::MiniLMCrossEncoderConfig;
use crate::sentence_encoder::BertConfig;
use crate::sequence_classifier::RobertaConfig;

impl LanguageModel for CrossEncoder {
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

    fn forced_eos_token_id(&self) -> Option<u32> {
        None
    }

    fn new_cache(
        &self,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<Box<dyn kjarni_transformers::Cache>> {
        Err(anyhow!("CrossEncoder does not use KV cache"))
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cross_encoder_predict() -> Result<()> {
        let encoder = CrossEncoder::from_registry(
            ModelType::MiniLML6V2CrossEncoder,
            None,
            Device::Cpu,
            None,
            None,
        )
        .await?;

        let score = encoder
            .predict_pair("i love kjarni", "kjarni is a new model inference library")
            .await?;

        // Cross-encoder should return a finite score
        assert!(score.is_finite());
        println!("Score: {}", score);

        Ok(())
    }

    #[tokio::test]
    async fn test_cross_encoder_torch_parity() -> Result<()> {
        let encoder = CrossEncoder::from_registry(
            ModelType::MiniLML6V2CrossEncoder,
            None,
            Device::Cpu,
            None,
            None,
        )
        .await?;

        let score = encoder
            .predict_pair("i love edgeGPT", "edgeGPT is a new model inference library")
            .await?;

        // Golden value from PyTorch
        let torch_value = 3.1776933670043945;
        assert!(
            (score - torch_value).abs() < 1e-3,
            "Score {} doesn't match torch value {}",
            score,
            torch_value
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_cross_encoder_rerank() -> Result<()> {
        let encoder = CrossEncoder::from_registry(
            ModelType::MiniLML6V2CrossEncoder,
            None,
            Device::Cpu,
            None,
            None,
        )
        .await?;

        let query = "machine learning algorithms";
        let documents = vec![
            "Machine learning algorithms include decision trees, neural networks, and SVMs.",
            "The weather forecast predicts rain tomorrow.",
            "Deep learning is a subset of machine learning using neural networks.",
            "Cooking recipes for Italian pasta dishes.",
        ];

        let ranked = encoder.rerank(query, &documents).await?;

        // Should return all 4 documents
        assert_eq!(ranked.len(), 4);

        // ML-related docs should rank higher than unrelated ones
        // Expected order: [0, 2, 3, 1] based on torch golden values
        let expected_indices: Vec<usize> = vec![0, 2, 3, 1];
        let actual_indices: Vec<usize> = ranked.iter().map(|v| v.0).collect();
        assert_eq!(actual_indices, expected_indices);

        // Scores should be sorted descending
        for i in 1..ranked.len() {
            assert!(ranked[i - 1].1 >= ranked[i].1);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_cross_encoder_rerank_empty() -> Result<()> {
        let encoder = CrossEncoder::from_registry(
            ModelType::MiniLML6V2CrossEncoder,
            None,
            Device::Cpu,
            None,
            None,
        )
        .await?;

        let ranked = encoder.rerank("query", &[]).await?;
        assert!(ranked.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_cross_encoder_rerank_top_k() -> Result<()> {
        let encoder = CrossEncoder::from_registry(
            ModelType::MiniLML6V2CrossEncoder,
            None,
            Device::Cpu,
            None,
            None,
        )
        .await?;

        let documents = vec!["Doc 1", "Doc 2", "Doc 3", "Doc 4", "Doc 5"];

        let top_2 = encoder.rerank_top_k("query", &documents, 2).await?;
        assert_eq!(top_2.len(), 2);

        // Requesting more than available should return all
        let top_10 = encoder.rerank_top_k("query", &documents, 10).await?;
        assert_eq!(top_10.len(), 5);

        Ok(())
    }

    #[tokio::test]
    async fn test_cross_encoder_gpu() -> Result<()> {
        let context = WgpuContext::new().await?;
        let gpu_encoder = CrossEncoder::from_registry(
            ModelType::MiniLML6V2CrossEncoder,
            None,
            Device::Wgpu,
            Some(context),
            None,
        )
        .await?;

        let cpu_encoder = CrossEncoder::from_registry(
            ModelType::MiniLML6V2CrossEncoder,
            None,
            Device::Cpu,
            None,
            None,
        )
        .await?;

        let cpu_score = cpu_encoder
            .predict_pair("test query", "test document")
            .await?;
        let gpu_score = gpu_encoder
            .predict_pair("test query", "test document")
            .await?;

        // CPU and GPU should produce same results
        assert!(
            (cpu_score - gpu_score).abs() < 1e-3,
            "CPU {} vs GPU {}",
            cpu_score,
            gpu_score
        );

        Ok(())
    }
}
