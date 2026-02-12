//! Types for summarization.

use thiserror::Error;

use crate::seq2seq::Seq2SeqError;

/// Errors that can occur during summarization.
#[derive(Debug, Error)]
pub enum SummarizerError {
    /// Unknown model name.
    #[error("{0}")]
    UnknownModel(String),

    /// Model incompatible with summarization.
    #[error("Model '{model}' incompatible for summarization: {reason}")]
    IncompatibleModel { model: String, reason: String },

    /// Summarization failed.
    #[error("Summarization failed: {0}")]
    SummarizationFailed(#[source] anyhow::Error),

    /// Underlying seq2seq error.
    #[error(transparent)]
    Seq2Seq(#[from] Seq2SeqError),
}

/// Result type for summarization operations.
pub type SummarizerResult<T> = Result<T, SummarizerError>;