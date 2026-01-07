//! Common error types for kjarni.

use std::fmt;
use thiserror::Error;

/// Errors that can occur when using kjarni.
#[derive(Debug, Error)]
pub enum KjarniError {
    /// Model not found in registry.
    #[error("Unknown model: '{0}'. Run 'kjarni model list' to see available models.")]
    UnknownModel(String),

    /// Model cannot perform the requested task.
    #[error("Model '{model}' cannot be used for {task}: {reason}")]
    IncompatibleModel {
        model: String,
        task: String,
        reason: String,
    },

    /// Model not downloaded and download policy is Never.
    #[error("Model '{0}' not downloaded and download policy is set to Never")]
    ModelNotDownloaded(String),

    /// Failed to download model.
    #[error("Failed to download model '{model}': {source}")]
    DownloadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// Failed to load model.
    #[error("Failed to load model '{model}': {source}")]
    LoadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// Inference failed.
    #[error("Inference failed: {0}")]
    InferenceFailed(#[from] anyhow::Error),

    /// GPU requested but not available.
    #[error("GPU requested but WebGPU context could not be created")]
    GpuUnavailable,

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// No labels available for classification.
    #[error("Model has no label mapping configured")]
    NoLabels,
}

/// Result type for kjarni operations.
pub type KjarniResult<T> = Result<T, KjarniError>;

/// Warning emitted for suboptimal configurations.
#[derive(Debug, Clone)]
pub struct KjarniWarning {
    pub message: String,
    pub suggestion: Option<String>,
}

impl fmt::Display for KjarniWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Warning: {}", self.message)?;
        if let Some(suggestion) = &self.suggestion {
            write!(f, " {}", suggestion)?;
        }
        Ok(())
    }
}