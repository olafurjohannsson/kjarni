//! Generic sequence classifier for text classification tasks.
//!
//! Supports BERT, DistilBERT, RoBERTa, and other encoder models with
//! classification heads for sentiment, emotion, toxicity, etc.

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use kjarni_transformers::PoolingStrategy;
use kjarni_transformers::cpu::encoder_decoder::Seq2SeqCPUDecoder;
use kjarni_transformers::encoder_decoder::config::Seq2SeqDecoderConfig;
use kjarni_transformers::models::registry::WeightsFormat;
use ndarray::{Array1, Array2};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::{EncodeInput, Tokenizer};

use kjarni_transformers::{
    WgpuContext,
    activations::softmax_inplace,
    cache::Cache,
    cpu::encoder::{
        CpuTransformerEncoder, GpuTransformerEncoder,
        classifier::{CpuSequenceClassificationHead, GpuSequenceClassificationHead},
        traits::{CpuEncoder, CpuEncoderOps, EncoderLanguageModel, GpuEncoder, GpuEncoderOps},
    },
    gpu_ops::{GpuFrameContext, GpuTensor},
    linear_layer::LinearLayer,
    models::{
        LanguageModel, ModelArchitecture, ModelTask, ModelType,
        base::{ModelInput, ModelLoadConfig},
        download_model_files,
    },
    traits::{Device, InferenceModel, ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};

// Re-export configs
mod configs;
#[cfg(test)]
mod tests;

pub use configs::*;
pub mod zero_shot;
use crate::models::bart::config::BartConfig;
use crate::sentence_encoder::{BertConfig, DistilBertConfig};

// =============================================================================
// Generic Sequence Classifier
// =============================================================================

/// A generic sequence classifier supporting multiple encoder architectures.
///
/// Works with BERT, DistilBERT, RoBERTa, DeBERTa and other encoder models
/// for tasks like sentiment analysis, emotion detection, and zero-shot classification.

pub struct SequenceClassifier {
    // Encoder components
    pub(crate) cpu_encoder: Option<CpuTransformerEncoder>,
    pub(crate) gpu_encoder: Option<GpuTransformerEncoder>,

    pub(crate) cpu_decoder: Option<Seq2SeqCPUDecoder>,
    // pub(crate) cpu_head: Option<CpuSequenceClassificationHead>,

    // Classification head
    pub(crate) cpu_head: Option<CpuSequenceClassificationHead>,
    pub(crate) gpu_head: Option<GpuSequenceClassificationHead>,

    // Keep these private or pub(crate) as needed
    pub(crate) tokenizer: Tokenizer,
    model_type: ModelType,
    device: Device,
    context: Option<Arc<WgpuContext>>,
    pub meta: ModelMetadata,
    pub layout: ModelLayout,
    config: Arc<dyn ModelConfig>,

    num_labels: usize,
    labels: Option<Vec<String>>,
}

impl SequenceClassifier {
    /// Check if a model type is supported for sequence classification.
    pub fn is_supported(model_type: ModelType) -> bool {
        let info = model_type.info();

        // Must be an encoder architecture
        let is_encoder = matches!(
            info.architecture,
            ModelArchitecture::Bert | ModelArchitecture::NomicBert
        );

        // Must be a classification task (not embedding or reranking)
        let is_classification_task = matches!(
            info.task,
            ModelTask::SentimentAnalysis
                | ModelTask::Classification
                | ModelTask::ZeroShotClassification
        );

        let is_cross_encoder = is_encoder && matches!(
            info.task,
            ModelTask::ReRanking
        );

        let is_zero_shot = matches!(info.architecture, ModelArchitecture::Bart)
            && matches!(info.task, ModelTask::ZeroShotClassification);

        is_encoder && is_classification_task || is_zero_shot || is_cross_encoder
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Get list of supported classification models.
    pub fn supported_models() -> Vec<ModelType> {
        ModelType::all()
            .filter(|m| Self::is_supported(*m))
            .collect()
    }

    /// Create classifier from HuggingFace registry.
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_cfg: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        if !Self::is_supported(model_type) {
            let supported: Vec<_> = Self::supported_models()
                .iter()
                .map(|m| m.cli_name())
                .collect();
            return Err(anyhow!(
                "Model '{}' is not supported for classification.\n\
                 Architecture: {:?}, Task: {:?}\n\
                 Supported models: {}",
                model_type.cli_name(),
                model_type.info().architecture,
                model_type.info().task,
                supported.join(", ")
            ));
        }

        let info = model_type.info();
        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::cache_dir()
                .expect("No cache directory found")
                .join("kjarni")
        });

        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

        // Download files
        download_model_files(&model_dir, &info.paths, WeightsFormat::SafeTensors).await?;

        // Load from local path
        Self::from_pretrained(&model_dir, model_type, device, context, load_cfg)
    }

    /// Create classifier from local model directory.
    pub fn from_pretrained(
        model_path: &Path,
        model_type: ModelType,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_cfg: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        let weights = ModelWeights::new(model_path)?;
        let load_cfg = load_cfg.unwrap_or_default();

        // Load config (handles BERT, DistilBERT, RoBERTa)
        let config: Arc<dyn ModelConfig> = Self::load_config(model_type, &weights)?;
        let meta = config.metadata();
        let layout = config.layout();

        // Get labels from config
        let labels = config.id2label().map(|l| l.to_vec());

        // Load tokenizer
        let mut tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Configure truncation/padding
        let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: meta.max_seq_len,
            ..Default::default()
        }));
        let padding_params = tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            // The pad token and id should be read from the model's config/tokenizer file.
            // For BART, the pad_token_id is 1.
            pad_id: 1, //config.pad_token_id().unwrap_or(1),
            pad_token: tokenizer
                .get_vocab(true)
                .get("[PAD]")
                .map(|_| "[PAD]".to_string())
                .unwrap_or_else(|| "<pad>".to_string()),
            ..Default::default()
        };
        tokenizer.with_padding(Some(padding_params));
        // tokenizer.with_padding(Some(tokenizers::PaddingParams {
        //     strategy: tokenizers::PaddingStrategy::BatchLongest,
        //     ..Default::default()
        // }));

        let mut cpu_encoder = None;
        let mut gpu_encoder = None;
        let mut cpu_head = None;
        let mut gpu_head = None;
        let mut cpu_decoder = None;
        match device {
            Device::Cpu => {
                cpu_encoder = Some(CpuTransformerEncoder::new(
                    &weights,
                    meta.clone(),
                    layout.clone(),
                    load_cfg.clone(),
                )?);
                if model_type.info().architecture == ModelArchitecture::Bart {
                    println!("Loading Bart DECODER");
                    // Initialize the Decoder
                    let dec_config = Seq2SeqDecoderConfig::bart();
                    let bart_config = config
                        .as_any()
                        .downcast_ref::<BartConfig>()
                        .ok_or_else(|| anyhow!("Config is not BartConfig"))?;

                    let decoder = Seq2SeqCPUDecoder::new(
                        &weights,
                        bart_config, // Pass the loaded config
                        dec_config,
                        load_cfg.clone(),
                    )?;
                    cpu_decoder = Some(decoder);
                }
                
                // Use the auto-detecting head loader!
                cpu_head = Some(CpuSequenceClassificationHead::from_weights(
                    &weights,
                    &load_cfg,
                    labels.clone(),
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
                    load_cfg.clone(),
                )?);

                // GPU head would need similar treatment
                // gpu_head = Some(GpuSequenceClassificationHead::from_weights(...)?);
            }
        }

        let num_labels = cpu_head.as_ref().map(|h| h.num_classes()).unwrap_or(2);

        Ok(Self {
            cpu_encoder,
            gpu_encoder,
            cpu_decoder,
            cpu_head,
            gpu_head,
            tokenizer,
            model_type,
            device,
            context,
            meta,
            layout,
            config,
            num_labels,
            labels,
        })
    }

    /// Load model config based on architecture.
    fn load_config(model_type: ModelType, weights: &ModelWeights) -> Result<Arc<dyn ModelConfig>> {
        let arch = model_type.info().architecture;

        match arch {
            ModelArchitecture::Bert => {
                // Check if it's DistilBERT or standard BERT/RoBERTa
                if weights
                    .config_json
                    .contains("\"model_type\": \"distilbert\"")
                    || weights
                        .config_json
                        .contains("\"model_type\":\"distilbert\"")
                {
                    Ok(Arc::new(DistilBertConfig::from_json(&weights.config_json)?))
                } else if weights.config_json.contains("\"model_type\": \"roberta\"")
                    || weights.config_json.contains("\"model_type\":\"roberta\"")
                {
                    Ok(Arc::new(RobertaConfig::from_json(&weights.config_json)?))
                } else {
                    Ok(Arc::new(BertConfig::from_json(&weights.config_json)?))
                }
            }
            ModelArchitecture::Bart => {
                // BART zero shot is special, it ignores the decoder
                if weights.config_json.contains("\"model_type\": \"bart\"") {
                    Ok(Arc::new(BartConfig::from_json(&weights.config_json)?))
                } else {
                    unimplemented!()
                }
            }
            // ModelArchitecture::NomicBert => {
            //     Ok(Arc::new(NomicBertConfig::from_json(&weights.config_json)?))
            // }
            _ => Err(anyhow!(
                "Unsupported architecture for classification: {:?}",
                arch
            )),
        }
    }

    // =========================================================================
    // Public Classification API
    // =========================================================================

    /// Get the label names for this classifier.
    pub fn labels(&self) -> Option<&[String]> {
        self.labels.as_deref()
    }

    /// Get the number of labels.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    /// Classify a single text, returning (label, score) pairs sorted by score.
    pub async fn classify(&self, text: &str, top_k: usize) -> Result<Vec<(String, f32)>> {
        let results = self.classify_batch(&[text], top_k).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No results"))
    }

    /// Classify returning raw probability scores without label names.
    pub async fn classify_with_scores(&self, text: &str) -> Result<Vec<f32>> {
        let mut logits = self.predict(&[text]).await?;
        let mut probs = logits.pop().ok_or_else(|| anyhow!("No predictions"))?;
        softmax_inplace(&mut probs);
        Ok(probs)
    }

    /// Batch classify multiple texts.
    pub async fn classify_batch(
        &self,
        texts: &[&str],
        top_k: usize,
    ) -> Result<Vec<Vec<(String, f32)>>> {
        let logits = self.predict(texts).await?;

        let labels = self.labels().ok_or_else(|| {
            anyhow!("Model has no label mapping. Use classify_with_scores() instead.")
        })?;

        let mut all_results = Vec::with_capacity(texts.len());

        for mut scores in logits {
            // Apply softmax
            softmax_inplace(&mut scores);

            // Map to labels
            let mut results: Vec<(String, f32)> = labels
                .iter()
                .zip(scores.iter())
                .map(|(label, &score)| (label.clone(), score))
                .collect();

            // Sort by score descending
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(top_k);
            all_results.push(results);
        }

        Ok(all_results)
    }

    /// Get raw logits for a batch of texts.
    pub async fn predict(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // 1. Run the Encoder (Once!)
        let (encoder_hidden_states, encoder_mask) = self.get_hidden_states_batch(texts).await?;

        // 2. Prepare the Vector for the Head
        // We also determine if we need a mask for the head.
        let (vector_for_head, head_mask) = if let Some(decoder) = &self.cpu_decoder {
            // === BART PATH (Encoder -> Decoder -> Head) ===
            let batch_size = encoder_hidden_states.shape()[0];
            let eos_id = self.config.eos_token_id().unwrap_or(2);

            // Input is just the EOS token
            let decoder_input_ids = Array2::from_elem((batch_size, 1), eos_id as u32);

            let decoder_output = decoder.forward(
                &decoder_input_ids,
                &encoder_hidden_states,
                None,                // No self-attn mask needed for len 1
                Some(&encoder_mask), // Mask padding in the encoder
                None,                // No KV cache
                None,                // No Cross cache
                0,                   // Offset 0
            )?;

            // Output is [Batch, 1, Hidden].
            // We pass None for the mask because there is no padding in this 1-token sequence.
            (decoder_output.last_hidden_state, None)
        } else {
            // === BERT PATH (Encoder -> Head) ===
            // Output is [Batch, Seq, Hidden].
            // We MUST pass the encoder mask so the head knows which tokens are padding.
            (encoder_hidden_states, Some(&encoder_mask))
        };

        // 3. Run the Head
        let logits = self
            .cpu_head
            .as_ref()
            .ok_or_else(|| anyhow!("No classification head available"))?
            .forward(&vector_for_head, head_mask)?; // Pass the correct mask (None for BART, Some for BERT)

        Ok(logits.outer_iter().map(|r| r.to_vec()).collect())
    }

    /// Maximum sequence length.
    pub fn max_seq_length(&self) -> usize {
        self.meta.max_seq_len
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl LanguageModel for SequenceClassifier {
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
    fn new_cache(&self, _: usize, _: usize, _: usize) -> Result<Box<dyn Cache>> {
        Err(anyhow!("Classifiers do not use KV cache"))
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
