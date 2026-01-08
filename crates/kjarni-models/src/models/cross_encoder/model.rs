// =============================================================================
// kjarni-models/src/cross_encoder/model.rs
// =============================================================================

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use kjarni_transformers::cpu::encoder::CpuEncoder;
use ndarray::Array2;

use kjarni_transformers::{
    models::{base::ModelLoadConfig, ModelType},
    traits::Device,
    WgpuContext,
};

use crate::sequence_classifier::SequenceClassifier;

/// Cross-encoder for scoring query-document pairs.
///
/// Wraps SequenceClassifier and provides a reranking-focused API.
/// Cross-encoders process query and document together through the transformer,
/// enabling more accurate relevance scoring than bi-encoders.
///
/// # Example
///
/// ```ignore
/// let encoder = CrossEncoder::from_registry(
///     ModelType::MiniLmReranker,
///     None,
///     Device::Cpu,
///     None,
///     None,
/// ).await?;
///
/// // Score a single pair
/// let score = encoder.predict_pair("What is Rust?", "Rust is a programming language.").await?;
///
/// // Rerank documents
/// let ranked = encoder.rerank("rust programming", &[
///     "Rust is fast.",
///     "Python is easy.",
///     "Rust prevents bugs.",
/// ]).await?;
/// ```
pub struct CrossEncoder {
    inner: SequenceClassifier,
}

impl CrossEncoder {
    // =========================================================================
    // Construction
    // =========================================================================

    /// Create from an existing SequenceClassifier.
    pub fn from_classifier(classifier: SequenceClassifier) -> Self {
        Self { inner: classifier }
    }

    /// Load from registry.
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        let classifier = SequenceClassifier::from_registry(
            model_type,
            cache_dir,
            device,
            context,
            load_config,
        )
        .await?;

        Ok(Self::from_classifier(classifier))
    }

    /// Load from local model directory.
    pub fn from_pretrained(
        model_path: &Path,
        model_type: ModelType,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        let classifier = SequenceClassifier::from_pretrained(
            model_path,
            model_type,
            device,
            context,
            load_config,
        )?;

        Ok(Self::from_classifier(classifier))
    }

    // =========================================================================
    // Pair Scoring
    // =========================================================================

    /// Score a single query-document pair.
    ///
    /// Returns a relevance score. Higher scores indicate more relevance.
    pub async fn predict_pair(&self, query: &str, document: &str) -> Result<f32> {
        let scores = self.predict_pairs(&[(query, document)]).await?;
        scores
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No score returned"))
    }

    /// Score multiple query-document pairs in a batch.
    ///
    /// More efficient than calling `predict_pair` multiple times.
    pub async fn predict_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize pairs: [CLS] query [SEP] document [SEP]
        let encodings: Vec<_> = pairs
            .iter()
            .map(|(query, doc)| {
                self.inner
                    .tokenizer
                    .encode((query.to_string(), doc.to_string()), true)
                    .map_err(|e| anyhow!("Tokenization failed: {}", e))
            })
            .collect::<Result<Vec<_>>>()?;

        // Get max length for padding
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(self.inner.meta.max_seq_len);

        let batch_size = pairs.len();

        // Build input tensors
        let mut input_ids = vec![0u32; batch_size * max_len];
        let mut attention_mask = vec![0f32; batch_size * max_len];
        let mut token_type_ids = vec![0u32; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let types = encoding.get_type_ids();

            let len = ids.len().min(max_len);
            let offset = i * max_len;

            input_ids[offset..offset + len].copy_from_slice(&ids[..len]);

            attention_mask[offset..offset + len]
                .copy_from_slice(&mask[..len].iter().map(|&v| v as f32).collect::<Vec<f32>>());
            
            token_type_ids[offset..offset + len].copy_from_slice(&types[..len]);
        }

        // Convert to ndarray
        let input_ids = Array2::from_shape_vec((batch_size, max_len), input_ids)?;
        let attention_mask = Array2::<f32>::from_shape_vec((batch_size, max_len), attention_mask)?;
        let token_type_ids = Array2::from_shape_vec((batch_size, max_len), token_type_ids)?;

        // Run through encoder
        let encoder_output = if let Some(ref encoder) = self.inner.cpu_encoder {
            encoder.forward(&input_ids, &attention_mask, Some(&token_type_ids))?
        } else if let Some(ref _encoder) = self.inner.gpu_encoder {
            // GPU path would go here
            return Err(anyhow!("GPU cross-encoder not yet implemented"));
        } else {
            return Err(anyhow!("No encoder available"));
        };

        let hidden_states = encoder_output.last_hidden_state;

        // Run through classification head
        let logits = if let Some(ref head) = self.inner.cpu_head {
            head.forward(&hidden_states)?
        } else {
            return Err(anyhow!("No classification head available"));
        };

        // Extract scores
        // For cross-encoders, typically:
        // - 1 output: use directly as score
        // - 2 outputs (binary): use logit[1] - logit[0] or just logit[1]
        let scores: Vec<f32> = if logits.ncols() == 1 {
            logits.column(0).to_vec()
        } else {
            // Binary classification - take positive class logit
            logits.column(1).to_vec()
        };

        Ok(scores)
    }

    // =========================================================================
    // Reranking
    // =========================================================================

    /// Rerank documents by relevance to a query.
    ///
    /// Returns (original_index, score) tuples sorted by score descending.
    pub async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // Create pairs: same query with each document
        let pairs: Vec<(&str, &str)> = documents
            .iter()
            .map(|doc| (query, *doc))
            .collect();

        // Score all pairs
        let scores = self.predict_pairs(&pairs).await?;

        // Create (index, score) tuples and sort
        let mut ranked: Vec<(usize, f32)> = scores
            .into_iter()
            .enumerate()
            .collect();

        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

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

    /// Rerank with owned strings.
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

    /// Maximum sequence length.
    pub fn max_seq_length(&self) -> usize {
        self.inner.meta.max_seq_len
    }

    /// Hidden size.
    pub fn hidden_size(&self) -> usize {
        self.inner.meta.hidden_size
    }

    /// Number of output labels.
    pub fn num_labels(&self) -> usize {
        self.inner.num_labels()
    }

    /// Get the inner SequenceClassifier.
    pub fn classifier(&self) -> &SequenceClassifier {
        &self.inner
    }

    /// Device the model is running on.
    pub fn device(&self) -> Device {
        self.inner.device()
    }
}