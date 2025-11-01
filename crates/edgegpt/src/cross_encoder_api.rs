//! High-level cross-encoder API

use anyhow::Result;
use edgemodels::cross_encoder::CrossEncoder;

/// High-level API wrapper for cross-encoding
///
/// This struct wraps a reference to a CrossEncoder
pub struct CrossEncoderAPI<'a> {
    encoder: &'a CrossEncoder,
}

impl<'a> CrossEncoderAPI<'a> {
    /// Create a new API wrapper
    pub fn new(encoder: &'a CrossEncoder) -> Self {
        Self { encoder }
    }

    /// Get a reference to the underlying encoder
    pub fn encoder(&self) -> &CrossEncoder {
        self.encoder
    }

    /// Score a single text pair
    pub async fn predict(&self, text1: &str, text2: &str) -> Result<f32> {
        self.encoder.predict(text1, text2).await
    }

    /// Score multiple text pairs
    pub async fn predict_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>> {
        self.encoder.predict_batch(pairs).await
    }

    /// Rerank documents by relevance to a query
    pub async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
        self.encoder.rerank(query, documents).await
    }

    /// Rerank and return only indices
    pub async fn rerank_indices(&self, query: &str, documents: &[&str]) -> Result<Vec<usize>> {
        self.encoder.rerank_indices(query, documents).await
    }

    /// Rerank and return only top K results
    pub async fn rerank_top_k(
        &self,
        query: &str,
        documents: &[&str],
        k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        self.encoder.rerank_top_k(query, documents, k).await
    }

    /// Rerank and return only top K indices
    pub async fn rerank_top_k_indices(
        &self,
        query: &str,
        documents: &[&str],
        k: usize,
    ) -> Result<Vec<usize>> {
        self.encoder.rerank_top_k_indices(query, documents, k).await
    }

    /// Get the maximum sequence length
    pub fn max_seq_length(&self) -> usize {
        self.encoder.max_seq_length()
    }
}