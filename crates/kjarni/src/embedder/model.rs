//! Core Embedder implementation.

use std::sync::Arc;

use kjarni_transformers::{models::ModelType, traits::Device, WgpuContext, LanguageModel};

use crate::common::{default_cache_dir, ensure_model_downloaded};
use crate::SentenceEncoder;

use super::builder::EmbedderBuilder;
use super::types::{EmbeddingOverrides, EmbedderError, EmbedderResult, PoolingStrategy};
use super::validation::validate_for_embedding;

/// High-level text embedder.
///
/// Generates dense vector representations of text for semantic search,
/// clustering, and similarity tasks.
pub struct Embedder {
    /// The underlying encoder.
    inner: SentenceEncoder,

    /// Model type.
    model_type: ModelType,

    /// Default overrides.
    default_overrides: EmbeddingOverrides,

    /// Device.
    device: Device,

    /// GPU context.
    context: Option<Arc<WgpuContext>>,
}

impl Embedder {
    /// Create an Embedder with default settings.
    pub async fn new(model: &str) -> EmbedderResult<Self> {
        Self::builder(model).build().await
    }

    /// Internal: construct from builder.
    pub(crate) async fn from_builder(builder: EmbedderBuilder) -> EmbedderResult<Self> {
        // 1. Resolve model type
        let model_type = ModelType::from_cli_name(&builder.model)
            .ok_or_else(|| EmbedderError::UnknownModel(builder.model.clone()))?;

        // 2. Validate for embedding
        validate_for_embedding(model_type)?;

        // 3. Ensure model is downloaded
        let cache_dir = builder.cache_dir.clone().unwrap_or_else(default_cache_dir);
        ensure_model_downloaded(model_type, Some(&cache_dir), builder.download_policy, builder.quiet)
            .await
            .map_err(|e| match e {
                crate::common::KjarniError::ModelNotDownloaded(m) => {
                    EmbedderError::ModelNotDownloaded(m)
                }
                crate::common::KjarniError::DownloadFailed { model, source } => {
                    EmbedderError::DownloadFailed { model, source }
                }
                other => EmbedderError::EmbeddingFailed(other.into()),
            })?;

        // 4. Resolve device
        let device = builder.device.to_device();

        // 5. Create GPU context if needed
        let context = if device == Device::Wgpu {
            if let Some(ctx) = builder.context {
                Some(ctx)
            } else {
                Some(
                    WgpuContext::new()
                        .await
                        .map_err(|_| EmbedderError::GpuUnavailable)?,
                )
            }
        } else {
            None
        };

        // 6. Load the encoder
        let load_config = builder.load_config.map(|c| c.into_inner());
        let inner = SentenceEncoder::from_registry(
            model_type,
            Some(cache_dir),
            device,
            context.clone(),
            load_config,
        )
        .await
        .map_err(|e| EmbedderError::LoadFailed {
            model: builder.model.clone(),
            source: e,
        })?;

        Ok(Self {
            inner,
            model_type,
            default_overrides: builder.overrides,
            device,
            context,
        })
    }

    // =========================================================================
    // Embedding Methods
    // =========================================================================

    /// Embed a single text.
    pub async fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        self.embed_with_config(text, &EmbeddingOverrides::default())
            .await
    }

    /// Embed with custom overrides.
    pub async fn embed_with_config(
        &self,
        text: &str,
        overrides: &EmbeddingOverrides,
    ) -> EmbedderResult<Vec<f32>> {
        let merged = self.merge_overrides(overrides);
        let pooling = merged
            .pooling
            .as_ref()
            .map(|p| p.as_str())
            .unwrap_or("mean");
        let normalize = merged.normalize.unwrap_or(true);

        self.inner
            .encode_with(text, Some(pooling), normalize)
            .await
            .map_err(EmbedderError::EmbeddingFailed)
    }

    /// Embed multiple texts.
    pub async fn embed_batch(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        self.embed_batch_with_config(texts, &EmbeddingOverrides::default())
            .await
    }

    /// Embed multiple texts with custom overrides.
    pub async fn embed_batch_with_config(
        &self,
        texts: &[&str],
        overrides: &EmbeddingOverrides,
    ) -> EmbedderResult<Vec<Vec<f32>>> {
        let merged = self.merge_overrides(overrides);
        let pooling = merged
            .pooling
            .as_ref()
            .map(|p| p.as_str())
            .unwrap_or("mean");
        let normalize = merged.normalize.unwrap_or(true);

        self.inner
            .encode_batch_with(texts, Some(pooling), normalize)
            .await
            .map_err(EmbedderError::EmbeddingFailed)
    }

    /// Compute cosine similarity between two texts.
    pub async fn similarity(&self, text1: &str, text2: &str) -> EmbedderResult<f32> {
        let embeddings = self.embed_batch(&[text1, text2]).await?;
        Ok(cosine_similarity(&embeddings[0], &embeddings[1]))
    }

    /// Compute similarities between a query and multiple documents.
    pub async fn similarities(
        &self,
        query: &str,
        documents: &[&str],
    ) -> EmbedderResult<Vec<f32>> {
        let mut all_texts = vec![query];
        all_texts.extend(documents);

        let embeddings = self.embed_batch(&all_texts).await?;
        let query_emb = &embeddings[0];

        Ok(embeddings[1..]
            .iter()
            .map(|doc_emb| cosine_similarity(query_emb, doc_emb))
            .collect())
    }

    /// Rank documents by similarity to a query.
    pub async fn rank_by_similarity(
        &self,
        query: &str,
        documents: &[&str],
    ) -> EmbedderResult<Vec<(usize, f32)>> {
        let similarities = self.similarities(query, documents).await?;
        let mut indexed: Vec<(usize, f32)> = similarities.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(indexed)
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the model type.
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Get the model's CLI name.
    pub fn model_name(&self) -> &str {
        self.model_type.cli_name()
    }

    /// Get the device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get the embedding dimension.
    pub fn dimension(&self) -> usize {
        self.inner.hidden_size()
    }

    /// Get maximum sequence length.
    pub fn max_seq_length(&self) -> usize {
        self.inner.context_size()
    }

    // =========================================================================
    // Internal
    // =========================================================================

    fn merge_overrides(&self, runtime: &EmbeddingOverrides) -> EmbeddingOverrides {
        EmbeddingOverrides {
            pooling: runtime.pooling.clone().or(self.default_overrides.pooling.clone()),
            normalize: runtime.normalize.or(self.default_overrides.normalize),
            max_length: runtime.max_length.or(self.default_overrides.max_length),
        }
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}