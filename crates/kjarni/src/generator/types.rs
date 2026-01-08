// =============================================================================
// kjarni/src/generator/types.rs
// =============================================================================

//! Generator types and error definitions.

use thiserror::Error;

/// Errors that can occur during generator operations.
#[derive(Debug, Error)]
pub enum GeneratorError {
    /// Model name not found in registry.
    #[error("Unknown model: '{0}'. Run 'kjarni model list --task chat' to see available models.")]
    UnknownModel(String),

    /// Model files not present locally.
    #[error("Model '{0}' not downloaded. Run: kjarni model download {0}")]
    ModelNotDownloaded(String),

    /// Download failed.
    #[error("Failed to download model '{model}': {source}")]
    DownloadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// Model loading failed.
    #[error("Failed to load model '{model}': {source}")]
    LoadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// GPU requested but unavailable.
    #[error("GPU unavailable. Use .cpu() or check your graphics drivers.")]
    GpuUnavailable,

    /// Generation failed.
    #[error("Generation failed: {0}")]
    GenerationFailed(#[from] anyhow::Error),

    /// Invalid model for generation.
    #[error("Model '{0}' is not suitable for text generation: {1}")]
    InvalidModel(String, String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for generator operations.
pub type GeneratorResult<T> = Result<T, GeneratorError>;

/// Token information returned during streaming.
#[derive(Debug, Clone)]
pub struct GeneratedToken {
    /// The token text.
    pub text: String,
    /// The token ID.
    pub id: u32,
    /// Whether this is a special token.
    pub is_special: bool,
}