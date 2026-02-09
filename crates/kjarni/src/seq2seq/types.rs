//! Types for seq2seq generation.

use thiserror::Error;


// Errors


/// Errors that can occur during seq2seq generation.
#[derive(Debug, Error)]
pub enum Seq2SeqError {
    /// Model name not found in registry.
    #[error("{0}")]
    UnknownModel(String),

    /// Model exists but hasn't been downloaded.
    #[error("Model '{0}' not downloaded. Set download_policy(DownloadPolicy::IfMissing) or download manually.")]
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

    /// GPU requested but not available.
    #[error("GPU unavailable. Use .cpu() or ensure GPU drivers are installed.")]
    GpuUnavailable,

    /// Generation failed during inference.
    #[error("Generation failed: {0}")]
    GenerationFailed(#[source] anyhow::Error),

    /// Model architecture incompatible with seq2seq.
    #[error("Model '{model}' incompatible: {reason}")]
    IncompatibleModel { model: String, reason: String },

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for seq2seq operations.
pub type Seq2SeqResult<T> = Result<T, Seq2SeqError>;


// Generated Token


/// A single token from streaming generation.
#[derive(Debug, Clone)]
pub struct Seq2SeqToken {
    /// The decoded text for this token.
    pub text: String,

    /// The token ID.
    pub id: u32,

    /// Whether this is a special token (EOS, PAD, etc.).
    pub is_special: bool,
}


// Overrides


/// User-specified overrides for seq2seq generation.
///
/// All fields are optional. `None` means "use model default".
/// Only set values the user explicitly requests - the model already has good defaults.
#[derive(Debug, Clone, Default)]
pub struct Seq2SeqOverrides {
    /// Minimum output length in tokens.
    pub min_length: Option<usize>,

    /// Maximum output length in tokens.
    pub max_length: Option<usize>,

    /// Number of beams for beam search. `1` = greedy decoding.
    pub num_beams: Option<usize>,

    /// Length penalty for beam search.
    pub length_penalty: Option<f32>,

    /// Whether to use sampling instead of beam search.
    pub do_sample: Option<bool>,

    /// Stop when `num_beams` complete sequences are found.
    pub early_stopping: Option<bool>,

    /// Prevent repeating n-grams of this size.
    pub no_repeat_ngram_size: Option<usize>,

    /// Penalty for repeating tokens (1.0 = no penalty).
    pub repetition_penalty: Option<f32>,
}

impl Seq2SeqOverrides {
    /// Use greedy decoding (fastest, single beam).
    pub fn greedy() -> Self {
        Self {
            num_beams: Some(1),
            ..Default::default()
        }
    }

    /// Check if any overrides are set.
    pub fn is_empty(&self) -> bool {
        self.min_length.is_none()
            && self.max_length.is_none()
            && self.num_beams.is_none()
            && self.length_penalty.is_none()
            && self.early_stopping.is_none()
            && self.no_repeat_ngram_size.is_none()
            && self.repetition_penalty.is_none()
            && self.do_sample.is_none()
    }

    /// Merge with another set of overrides. Values from `other` take precedence.
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            min_length: other.min_length.or(self.min_length),
            max_length: other.max_length.or(self.max_length),
            num_beams: other.num_beams.or(self.num_beams),
            length_penalty: other.length_penalty.or(self.length_penalty),
            early_stopping: other.early_stopping.or(self.early_stopping),
            no_repeat_ngram_size: other.no_repeat_ngram_size.or(self.no_repeat_ngram_size),
            repetition_penalty: other.repetition_penalty.or(self.repetition_penalty),
            do_sample: other.do_sample.or(self.do_sample),
        }
    }
}