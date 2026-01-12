// =============================================================================
// kjarni-models/src/cross_encoder/mod.rs
// =============================================================================

//! Cross-encoder for semantic similarity and reranking.
//!
//! CrossEncoder processes query-document pairs together through a transformer
//! to produce relevance scores. Unlike bi-encoders (SentenceEncoder), cross-encoders
//! see both texts simultaneously, enabling more accurate relevance judgments.
//!
//! This module provides the low-level CrossEncoder implementation.
//! For high-level reranking API, see `kjarni::reranker::Reranker`.

mod model;

pub use model::CrossEncoder;


// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests;