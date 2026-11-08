//! Errors for the ner module.

/// Errors from entity recognition.
#[derive(Debug, thiserror::Error)]
pub enum NerError {
    #[error("{0}")]
    UnknownModel(String),

    #[error("Model '{model}' cannot be used for entity recognition: {reason}")]
    IncompatibleModel { model: String, reason: String },

    #[error("Model '{0}' not downloaded and download policy is set to Never")]
    ModelNotDownloaded(String),

    #[error("Model path not found: {0}")]
    ModelPathNotFound(String),

    #[error("Failed to download model '{model}': {source}")]
    DownloadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    #[error("Failed to load model '{model}': {source:#}")]
    LoadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    #[error("Entity recognition failed: {0}")]
    Failed(#[from] anyhow::Error),

    #[error("GPU requested but WebGPU context could not be created")]
    GpuUnavailable,
}

pub type NerResult<T> = Result<T, NerError>;
