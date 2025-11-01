//! High-level sentence encoding API

use anyhow::Result;
use edgemodels::sentence_encoder::SentenceEncoder;
use crate::utils::{cosine_similarity, similarity_matrix};

/// High-level API wrapper for sentence encoding
///
/// This struct wraps a reference to a SentenceEncoder
pub struct SentenceEncoderAPI<'a> {
    encoder: &'a SentenceEncoder,
}

impl<'a> SentenceEncoderAPI<'a> {
    /// Create a new API wrapper
    pub fn new(encoder: &'a SentenceEncoder) -> Self {
        Self { encoder }
    }

    /// Get a reference to the underlying encoder
    pub fn encoder(&self) -> &SentenceEncoder {
        self.encoder
    }

    /// Encode a single sentence with default settings
    pub async fn encode(&self, text: &str) -> Result<Vec<f32>> {
        self.encoder.encode(text).await
    }

    /// Encode without normalization
    pub async fn encode_raw(&self, text: &str) -> Result<Vec<f32>> {
        self.encoder.encode_raw(text).await
    }

    /// Encode with custom pooling and normalization
    pub async fn encode_with(
        &self,
        text: &str,
        pooling: Option<&str>,
        normalize: bool,
    ) -> Result<Vec<f32>> {
        self.encoder.encode_with(text, pooling, normalize).await
    }

    /// Encode a batch of sentences
    pub async fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.encoder.encode_batch(texts).await
    }

    /// Encode batch without normalization
    pub async fn encode_batch_raw(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.encoder.encode_batch_raw(texts).await
    }

    /// Encode batch with custom settings
    pub async fn encode_batch_with(
        &self,
        texts: &[&str],
        pooling: Option<&str>,
        normalize: bool,
    ) -> Result<Vec<Vec<f32>>> {
        self.encoder.encode_batch_with(texts, pooling, normalize).await
    }

    /// Compute cosine similarity between two texts
    pub async fn similarity(&self, text1: &str, text2: &str) -> Result<f32> {
        let embeddings = self.encode_batch(&[text1, text2]).await?;
        Ok(cosine_similarity(&embeddings[0], &embeddings[1]))
    }

    /// Compute pairwise similarity matrix for a batch of texts
    pub async fn similarity_matrix(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let embeddings = self.encode_batch(texts).await?;
        Ok(similarity_matrix(&embeddings))
    }

    /// Find the most similar texts to a query from a list of candidates
    pub async fn find_similar(
        &self,
        query: &str,
        candidates: &[&str],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let query_emb = self.encode(query).await?;
        let candidate_embs = self.encode_batch(candidates).await?;
        
        let mut similarities: Vec<(usize, f32)> = candidate_embs
            .iter()
            .enumerate()
            .map(|(idx, emb)| (idx, cosine_similarity(&query_emb, emb)))
            .collect();
        
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        
        Ok(similarities)
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.encoder.embedding_dim()
    }

    /// Get the maximum sequence length
    pub fn max_seq_length(&self) -> usize {
        self.encoder.max_seq_length()
    }
}