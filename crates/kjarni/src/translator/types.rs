//! Types for translation.

use thiserror::Error;

use crate::seq2seq::Seq2SeqError;

/// Errors that can occur during translation.
#[derive(Debug, Error)]
pub enum TranslatorError {
    /// Unknown model name.
    #[error("{0}")]
    UnknownModel(String),

    /// Model incompatible with translation.
    #[error("Model '{model}' incompatible for translation: {reason}")]
    IncompatibleModel { model: String, reason: String },

    /// Unknown or unsupported language.
    #[error("Unknown language: '{0}'. Use language codes (en, de, fr) or names (English, German, French).")]
    UnknownLanguage(String),

    /// Missing required language specification.
    #[error("Missing language: specify both source and target languages, or set defaults via builder.")]
    MissingLanguage,

    /// Translation failed.
    #[error("Translation failed: {0}")]
    TranslationFailed(#[source] anyhow::Error),

    /// Underlying seq2seq error.
    #[error(transparent)]
    Seq2Seq(#[from] Seq2SeqError),
}

/// Result type for translation operations.
pub type TranslatorResult<T> = Result<T, TranslatorError>;