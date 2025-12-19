// In a new (or the old, now empty) file: src/cross_encoder/mod.rs

//! A Cross-Encoder for reranking, which is a specialized `SequenceClassifier`.
//! This module provides a convenient type alias.

// Re-export the config so users can find it here.
pub use crate::sequence_classifier::MiniLMCrossEncoderConfig;

/// A type alias for `SequenceClassifier` to be used for reranking tasks.
pub type CrossEncoder = crate::sequence_classifier::SequenceClassifier;