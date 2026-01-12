//! Generic sequence classifier for text classification tasks.
//!
//! Supports BERT, DistilBERT, RoBERTa, and other encoder-only models with
//! classification heads for sentiment, emotion, toxicity, NLI, etc.
//!
//! For BART-based zero-shot classification, use `BartZeroShotClassifier` instead.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

use kjarni_transformers::{
    WgpuContext,
    activations::softmax_inplace,
    cpu::encoder::{
        CpuTransformerEncoder, GpuTransformerEncoder,
        classifier::CpuSequenceClassificationHead,
        config::PoolingStrategy,
        traits::{CpuEncoder, CpuEncoderOps, EncoderLanguageModel, GpuEncoder, GpuEncoderOps},
    },
    models::{LanguageModel, ModelType},
    models::base::ModelLoadConfig,
    pipeline::{EncoderLoader, EncoderModelFactory, EncoderPipeline},
    traits::{Cache, Device, InferenceModel, ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};

pub mod configs;
#[cfg(test)]
mod tests;

use crate::sequence_classifier::configs::RobertaConfig;
use crate::{BertConfig, DistilBertConfig};

// =============================================================================
// Sequence Classifier
// =============================================================================

/// A generic sequence classifier for encoder-only models.
///
/// Supports:
/// - BERT, DistilBERT, RoBERTa for sentiment/emotion/toxicity
/// - RoBERTa-MNLI, DeBERTa-MNLI for NLI-based zero-shot (via ZeroShotClassifier wrapper)
///
/// For BART-based zero-shot, use `BartZeroShotClassifier` instead.
pub struct SequenceClassifier {
    pipeline: EncoderPipeline,
    tokenizer: Tokenizer,
    config: Arc<dyn ModelConfig>,
    model_type: Option<ModelType>,
    labels: Option<Vec<String>>,
}

// =============================================================================
// EncoderModelFactory Implementation
// =============================================================================

impl EncoderModelFactory for SequenceClassifier {
    fn load_config(weights: &ModelWeights) -> Result<Arc<dyn ModelConfig>> {
        // Auto-detect config type
        if weights.is_distilbert() {
            Ok(Arc::new(DistilBertConfig::from_json(weights.config_json())?))
        } else if weights.is_roberta() || weights.is_distilroberta() {
            Ok(Arc::new(RobertaConfig::from_json(weights.config_json())?))
        } else {
            // Default to BERT
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
        }

        Ok((cpu, gpu))
    }

    fn build_head(
        weights: &ModelWeights,
        load_config: &ModelLoadConfig,
    ) -> Result<Option<CpuSequenceClassificationHead>> {
        // Get labels from config if available
        let config: serde_json::Value = serde_json::from_str(weights.config_json())
            .unwrap_or(serde_json::Value::Null);
        
        let labels = config.get("id2label")
            .and_then(|v| v.as_object())
            .map(|obj| {
                let mut labels: Vec<(usize, String)> = obj
                    .iter()
                    .filter_map(|(k, v)| {
                        let idx: usize = k.parse().ok()?;
                        let label = v.as_str()?.to_string();
                        Some((idx, label))
                    })
                    .collect();
                labels.sort_by_key(|(idx, _)| *idx);
                labels.into_iter().map(|(_, label)| label).collect()
            });

        Ok(Some(CpuSequenceClassificationHead::from_weights(
            weights,
            load_config,
            labels,
        )?))
    }

    fn pooling_strategy() -> PoolingStrategy {
        PoolingStrategy::Cls // Classification uses CLS token
    }

    fn new_from_pipeline(
        pipeline: EncoderPipeline,
        tokenizer: Tokenizer,
        config: Arc<dyn ModelConfig>,
        model_type: Option<ModelType>,
    ) -> Self {
        let labels = pipeline.labels().map(|l| l.to_vec());
        
        Self {
            pipeline,
            tokenizer,
            config,
            model_type,
            labels,
        }
    }
}

// =============================================================================
// Public API
// =============================================================================

impl SequenceClassifier {
    /// Create classifier from HuggingFace model registry.
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

    /// Create classifier from local model directory.
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
    // Classification API
    // =========================================================================

    /// Classify text, returning the top prediction.
    pub async fn classify(&self, text: &str) -> Result<ClassificationResult> {
        let results = self.classify_top_k(text, 1).await?;
        results.into_iter().next().ok_or_else(|| anyhow!("No results"))
    }

    /// Classify text, returning all scores (after softmax).
    /// 
    /// This is the method the higher-level API depends on.
    pub async fn classify_scores(&self, text: &str) -> Result<Vec<f32>> {
        let batch_scores = self.classify_scores_batch(&[text]).await?;
        batch_scores.into_iter().next().ok_or_else(|| anyhow!("No results"))
    }

    /// Classify text, returning top-k predictions.
    pub async fn classify_top_k(&self, text: &str, k: usize) -> Result<Vec<ClassificationResult>> {
        let scores = self.classify_scores(text).await?;
        self.scores_to_top_k(&scores, k)
    }

    /// Batch classify multiple texts, returning top-k per text.
    /// 
    /// Returns Vec<Vec<(String, f32)>> to match original API.
    pub async fn classify_batch(
        &self,
        texts: &[&str],
        top_k: usize,
    ) -> Result<Vec<Vec<(String, f32)>>> {
        let batch_scores = self.classify_scores_batch(texts).await?;
        
        let labels = self.labels.as_ref().ok_or_else(|| {
            anyhow!("Model has no label mapping. Use classify_scores_batch() instead.")
        })?;

        let mut all_results = Vec::with_capacity(texts.len());

        for scores in batch_scores {
            // Map to (label, score) tuples
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

    /// Batch classify, returning all scores.
    pub async fn classify_scores_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let logits = self.predict_logits(texts).await?;
        
        // Apply softmax to each row
        Ok(logits
            .into_iter()
            .map(|mut row| {
                softmax_inplace(&mut row);
                row
            })
            .collect())
    }

    /// Get raw logits (before softmax).
    pub async fn predict_logits(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize
        let encodings = self.tokenizer
            .encode_batch(texts.to_vec(), true)
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

        // Forward through encoder
        let hidden_states = self.pipeline
            .cpu_encoder()
            .ok_or_else(|| anyhow!("No CPU encoder available"))?
            .forward(&input_ids, &attention_mask, Some(&token_type_ids))?
            .last_hidden_state;

        // Forward through head
        let logits = self.pipeline
            .cpu_head()
            .ok_or_else(|| anyhow!("No classification head available"))?
            .forward(&hidden_states, Some(&attention_mask))?;

        Ok(logits.outer_iter().map(|row| row.to_vec()).collect())
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    fn scores_to_top_result(&self, scores: &[f32]) -> Result<ClassificationResult> {
        let (idx, &score) = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| anyhow!("Empty scores"))?;

        let label = self.labels
            .as_ref()
            .and_then(|l| l.get(idx))
            .cloned()
            .unwrap_or_else(|| format!("LABEL_{}", idx));

        Ok(ClassificationResult { label, score, index: idx })
    }

    fn scores_to_top_k(&self, scores: &[f32], k: usize) -> Result<Vec<ClassificationResult>> {
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);

        Ok(indexed
            .into_iter()
            .map(|(idx, score)| {
                let label = self.labels
                    .as_ref()
                    .and_then(|l| l.get(idx))
                    .cloned()
                    .unwrap_or_else(|| format!("LABEL_{}", idx));
                ClassificationResult { label, score, index: idx }
            })
            .collect())
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    pub fn pipeline(&self) -> &EncoderPipeline {
        &self.pipeline
    }

    pub fn labels(&self) -> Option<&[String]> {
        self.labels.as_deref()
    }

    pub fn num_labels(&self) -> usize {
        self.pipeline.num_labels().unwrap_or(2)
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
// Classification Result
// =============================================================================

/// Result of a classification.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub label: String,
    pub score: f32,
    pub index: usize,
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl LanguageModel for SequenceClassifier {
    fn vocab_size(&self) -> usize { self.pipeline.vocab_size() }
    fn hidden_size(&self) -> usize { self.pipeline.hidden_size() }
    fn num_layers(&self) -> usize { self.pipeline.num_layers() }
    fn num_heads(&self) -> usize { self.config.metadata().num_attention_heads }
    fn context_size(&self) -> usize { self.pipeline.max_seq_length() }
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
    fn device(&self) -> Device { self.pipeline.plan().layers }
    fn context(&self) -> Option<Arc<WgpuContext>> { self.pipeline.context().cloned() }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

impl CpuEncoderOps for SequenceClassifier {
    fn encoder(&self) -> &dyn CpuEncoder {
        self.pipeline.cpu_encoder().expect("CPU encoder not available")
    }
}

impl GpuEncoderOps for SequenceClassifier {
    fn encoder(&self) -> &dyn GpuEncoder {
        self.pipeline.gpu_encoder().expect("GPU encoder not available")
    }
}

#[async_trait]
impl EncoderLanguageModel for SequenceClassifier {
    fn encoder_cpu_ops(&self) -> Option<&dyn CpuEncoderOps> {
        if self.pipeline.cpu_encoder().is_some() { Some(self) } else { None }
    }
    fn encoder_gpu_ops(&self) -> Option<&dyn GpuEncoderOps> {
        if self.pipeline.gpu_encoder().is_some() { Some(self) } else { None }
    }
}