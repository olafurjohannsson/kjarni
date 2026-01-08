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
use tokenizers::EncodeInput;

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
        scores.into_iter().next().ok_or_else(|| anyhow!("No score returned"))
    }

    /// Score multiple query-document pairs in a batch (SIMPLIFIED AND CORRECTED).
    pub async fn predict_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        // 1. Create sentence pairs for the tokenizer.
        let inputs_to_tokenize: Vec<EncodeInput> = pairs
            .iter()
            .map(|&(query, doc)| (query.to_string(), doc.to_string()).into())
            .collect();

        // 2. Let the SequenceClassifier do all the heavy lifting.
        //    This single call handles tokenization, padding, masking, and running the encoder.
        let logits = self.inner.predict_from_pairs(&inputs_to_tokenize).await?;

        // 3. Extract scores (same logic as before).
        let scores: Vec<f32> = if logits.get(0).map_or(0, |l| l.len()) == 1 {
            logits.iter().map(|l| l[0]).collect()
        } else {
            logits.iter().map(|l| l.get(1).copied().unwrap_or(0.0)).collect()
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