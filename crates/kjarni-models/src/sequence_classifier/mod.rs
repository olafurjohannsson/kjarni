//! Generic sequence classifier for text classification tasks.
//!
//! Supports BERT, DistilBERT, RoBERTa, and other encoder models with
//! classification heads for sentiment, emotion, toxicity, etc.

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use kjarni_transformers::PoolingStrategy;
use kjarni_transformers::models::registry::WeightsFormat;
use ndarray::{Array1, Array2};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

use kjarni_transformers::{
    activations::softmax_inplace,
    cache::Cache,
    cpu::encoder::{
        classifier::{CpuSequenceClassificationHead, GpuSequenceClassificationHead},
        traits::{CpuEncoder, CpuEncoderOps, EncoderLanguageModel, GpuEncoder, GpuEncoderOps},
        CpuTransformerEncoder, GpuTransformerEncoder,
    },
    gpu_ops::{GpuFrameContext, GpuTensor},
    linear_layer::LinearLayer,
    models::{
        base::{ModelInput, ModelLoadConfig},
        download_model_files, LanguageModel, ModelArchitecture, ModelTask, ModelType
    },
    traits::{Device, InferenceModel, ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
    WgpuContext,
};

// Re-export configs
mod configs;
pub use configs::*;

use crate::sentence_encoder::{BertConfig, DistilBertConfig};

// =============================================================================
// Classification Head Configuration
// =============================================================================

/// Describes the structure of a classification head for different model types.
#[derive(Debug, Clone)]
pub struct ClassificationHeadConfig {
    /// Pre-classifier projection layer (e.g., DistilBERT has this, BERT doesn't)
    pub pre_classifier: Option<HeadLayerConfig>,
    /// Pooler layer (some models use this for [CLS] projection)
    pub pooler: Option<HeadLayerConfig>,
    /// Final classifier layer
    pub classifier: HeadLayerConfig,
    /// Activation between pre_classifier and classifier
    pub activation: HeadActivation,
    /// How to extract the representation for classification
    pub pooling_strategy: PoolingStrategy,
}

#[derive(Debug, Clone)]
pub struct HeadLayerConfig {
    pub weight: String,
    pub bias: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadActivation {
    None,
    Relu,
    Tanh,
    Gelu,
}


impl Default for ClassificationHeadConfig {
    fn default() -> Self {
        Self {
            pre_classifier: None,
            pooler: None,
            classifier: HeadLayerConfig {
                weight: "classifier.weight".to_string(),
                bias: Some("classifier.bias".to_string()),
            },
            activation: HeadActivation::None,
            pooling_strategy: PoolingStrategy::Cls,
        }
    }
}

// =============================================================================
// Model-Specific Head Configurations
// =============================================================================

/// Get classification head config based on model architecture and type.
fn get_head_config(model_type: ModelType, weights: &ModelWeights) -> ClassificationHeadConfig {
    // Detect based on what weights exist
    let has_pre_classifier = weights.contains("pre_classifier.weight");
    let has_pooler = weights.contains("bert.pooler.dense.weight")
        || weights.contains("roberta.pooler.dense.weight");
    let has_classifier_dense = weights.contains("classifier.dense.weight");

    match model_type.info().architecture {
        // DistilBERT: pre_classifier -> ReLU -> classifier
        ModelArchitecture::Bert if has_pre_classifier => ClassificationHeadConfig {
            pre_classifier: Some(HeadLayerConfig {
                weight: "pre_classifier.weight".to_string(),
                bias: Some("pre_classifier.bias".to_string()),
            }),
            pooler: None,
            classifier: HeadLayerConfig {
                weight: "classifier.weight".to_string(),
                bias: Some("classifier.bias".to_string()),
            },
            activation: HeadActivation::Relu,
            pooling_strategy: PoolingStrategy::Cls,
        },

        // RoBERTa: classifier.dense -> Tanh -> classifier.out_proj
        ModelArchitecture::Bert if has_classifier_dense => ClassificationHeadConfig {
            pre_classifier: Some(HeadLayerConfig {
                weight: "classifier.dense.weight".to_string(),
                bias: Some("classifier.dense.bias".to_string()),
            }),
            pooler: None,
            classifier: HeadLayerConfig {
                weight: "classifier.out_proj.weight".to_string(),
                bias: Some("classifier.out_proj.bias".to_string()),
            },
            activation: HeadActivation::Tanh,
            pooling_strategy: PoolingStrategy::Cls,
        },

        // Standard BERT: pooler -> classifier (pooler applies Tanh internally)
        ModelArchitecture::Bert if has_pooler => {
            let prefix = if weights.contains("roberta.pooler.dense.weight") {
                "roberta"
            } else {
                "bert"
            };
            ClassificationHeadConfig {
                pre_classifier: None,
                pooler: Some(HeadLayerConfig {
                    weight: format!("{}.pooler.dense.weight", prefix),
                    bias: Some(format!("{}.pooler.dense.bias", prefix)),
                }),
                classifier: HeadLayerConfig {
                    weight: "classifier.weight".to_string(),
                    bias: Some("classifier.bias".to_string()),
                },
                activation: HeadActivation::Tanh, // Pooler uses Tanh
                pooling_strategy: PoolingStrategy::Cls,
            }
        }

        // Fallback: just classifier
        _ => ClassificationHeadConfig::default(),
    }
}


// =============================================================================
// Generic Sequence Classifier
// =============================================================================

/// A generic sequence classifier supporting multiple encoder architectures.
///
/// Works with BERT, DistilBERT, RoBERTa, DeBERTa and other encoder models
/// for tasks like sentiment analysis, emotion detection, and zero-shot classification.
pub struct SequenceClassifier {
    // Encoder components
    cpu_encoder: Option<CpuTransformerEncoder>,
    gpu_encoder: Option<GpuTransformerEncoder>,

    // Classification head (unified)
    cpu_head: Option<CpuSequenceClassificationHead>,
    gpu_head: Option<GpuSequenceClassificationHead>,

    // Model info
    tokenizer: Tokenizer,
    model_type: ModelType,
    device: Device,
    context: Option<Arc<WgpuContext>>,
    pub meta: ModelMetadata,
    pub layout: ModelLayout,
    config: Arc<dyn ModelConfig>,

    // Classification-specific
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

        is_encoder && is_classification_task
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
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

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
                    load_cfg.clone(),
                )?);

                // Use the auto-detecting head loader!
                cpu_head = Some(CpuSequenceClassificationHead::from_weights(
                    &weights,
                    &load_cfg,
                    labels.clone(),
                )?);
            }
            Device::Wgpu => {
                let ctx = context.clone().ok_or_else(|| anyhow!("WGPU context required"))?;
                
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
                if weights.config_json.contains("\"model_type\": \"distilbert\"")
                    || weights.config_json.contains("\"model_type\":\"distilbert\"")
                {
                    Ok(Arc::new(DistilBertConfig::from_json(&weights.config_json)?))
                } 
                // else if weights.config_json.contains("\"model_type\": \"roberta\"")
                //     || weights.config_json.contains("\"model_type\":\"roberta\"")
                // {
                //     Ok(Arc::new(RobertaConfig::from_json(&weights.config_json)?))
                // } 
                else {
                    Ok(Arc::new(BertConfig::from_json(&weights.config_json)?))
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
        results.into_iter().next().ok_or_else(|| anyhow!("No results"))
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

        // Get hidden states from encoder
        let (hidden_states, _attention_mask) = self.get_hidden_states_batch(texts).await?;

        // Apply classification head
        let logits = if let Some(ref head) = self.cpu_head {
            head.forward(&hidden_states)?
        } else {
            return Err(anyhow!("No classification head available"));
        };

        Ok(logits.outer_iter().map(|row| row.to_vec()).collect())
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
    fn vocab_size(&self) -> usize { self.meta.vocab_size }
    fn hidden_size(&self) -> usize { self.meta.hidden_size }
    fn num_layers(&self) -> usize { self.meta.num_layers }
    fn num_heads(&self) -> usize { self.meta.num_attention_heads }
    fn context_size(&self) -> usize { self.meta.max_seq_len }
    fn tokenizer(&self) -> &Tokenizer { &self.tokenizer }
    fn bos_token_id(&self) -> Option<u32> { self.tokenizer.token_to_id("[CLS]") }
    fn eos_token_id(&self) -> Option<u32> { self.tokenizer.token_to_id("[SEP]") }
    fn pad_token_id(&self) -> Option<u32> { self.tokenizer.token_to_id("[PAD]") }
    fn forced_bos_token_id(&self) -> Option<u32> { None }
    fn forced_eos_token_id(&self) -> Option<u32> { None }
    fn new_cache(&self, _: usize, _: usize, _: usize) -> Result<Box<dyn Cache>> {
        Err(anyhow!("Classifiers do not use KV cache"))
    }
}

impl InferenceModel for SequenceClassifier {
    fn device(&self) -> Device { self.device }
    fn context(&self) -> Option<Arc<WgpuContext>> { self.context.clone() }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

impl CpuEncoderOps for SequenceClassifier {
    fn encoder(&self) -> &dyn CpuEncoder {
        self.cpu_encoder.as_ref().expect("CPU encoder not initialized")
    }
}

impl GpuEncoderOps for SequenceClassifier {
    fn encoder(&self) -> &dyn GpuEncoder {
        self.gpu_encoder.as_ref().expect("GPU encoder not initialized")
    }
}

#[async_trait]
impl EncoderLanguageModel for SequenceClassifier {
    fn encoder_cpu_ops(&self) -> Option<&dyn CpuEncoderOps> {
        if self.device.is_cpu() { Some(self) } else { None }
    }
    fn encoder_gpu_ops(&self) -> Option<&dyn GpuEncoderOps> {
        if self.device.is_gpu() { Some(self) } else { None }
    }
}