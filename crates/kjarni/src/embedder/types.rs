//! Types for the embedder module.

use std::fmt;
use thiserror::Error;

/// Errors specific to embedding.
#[derive(Debug, Error)]
pub enum EmbedderError {
    /// Model not found.
    #[error("Unknown model: '{0}'. Run 'kjarni model list --task embedding' to see available models.")]
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

/// Pooling strategies for sequence outputs.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum PoolingStrategy {
    /// Mean pooling over all tokens (default, recommended).
    #[default]
    Mean,

    /// Max pooling over all tokens.
    Max,

    /// Use [CLS] token representation.
    Cls,

    /// Use last token representation.
    LastToken,
}

impl PoolingStrategy {
    /// Convert to string for the low-level API.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Max => "max",
            Self::Cls => "cls",
            Self::LastToken => "last_token",
        }
    }
}

impl fmt::Display for PoolingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mean => write!(f, "mean"),
            Self::Max => write!(f, "max"),
            Self::Cls => write!(f, "cls"),
            Self::LastToken => write!(f, "last_token"),
        }
    }
}

impl std::str::FromStr for PoolingStrategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mean" => Ok(Self::Mean),
            "max" => Ok(Self::Max),
            "cls" => Ok(Self::Cls),
            "last_token" | "lasttoken" | "last" => Ok(Self::LastToken),
            _ => Err(format!("Unknown pooling strategy: {}", s)),
        }
    }
}

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