//! Types for the embedder module.

use kjarni_transformers::PoolingStrategy;
use thiserror::Error;

/// Errors specific to embedding.
#[derive(Debug, Error)]
pub enum EmbedderError {
    /// Model not found.
    #[error("{0}")]
    UnknownModel(String),

    /// Model cannot embed.
    #[error("Model '{model}' cannot be used for embedding: {reason}")]
    IncompatibleModel { model: String, reason: String },

    /// Model not downloaded.
    #[error("Model '{0}' not downloaded and download policy is set to Never")]
    ModelNotDownloaded(String),

    /// Download failed.
    #[error("Failed to download model '{model}': {source}")]
    DownloadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// Load failed.
    #[error("Failed to load model '{model}': {source}")]
    LoadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// Embedding failed.
    #[error("Embedding failed: {0}")]
    EmbeddingFailed(#[from] anyhow::Error),

    /// GPU unavailable.
    #[error("GPU requested but WebGPU context could not be created")]
    GpuUnavailable,

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for embedder operations.
pub type EmbedderResult<T> = Result<T, EmbedderError>;



/// Overrides for embedding behavior.
#[derive(Debug, Clone, Default)]
pub struct EmbeddingOverrides {
    /// Pooling strategy (model default if None).
    pub pooling: Option<PoolingStrategy>,

    /// L2 normalize output (default: true for most models).
    pub normalize: Option<bool>,

    /// Truncate inputs longer than this.
    pub max_length: Option<usize>,
}

impl EmbeddingOverrides {
    /// Create overrides for search queries (normalized, mean pooling).
    pub fn for_search() -> Self {
        Self {
            pooling: Some(PoolingStrategy::Mean),
            normalize: Some(true),
            max_length: None,
        }
    }

    /// Create overrides for clustering (normalized).
    pub fn for_clustering() -> Self {
        Self {
            pooling: Some(PoolingStrategy::Mean),
            normalize: Some(true),
            max_length: None,
        }
    }

    /// Create overrides for similarity (normalized).
    pub fn for_similarity() -> Self {
        Self {
            pooling: Some(PoolingStrategy::Mean),
            normalize: Some(true),
            max_length: None,
        }
    }
}