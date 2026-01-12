//! High-level text reranking API.
//!
//! Rerank documents by relevance to a query using cross-encoder models.
//!
//! # Quick Start
//!
//! ```ignore
//! use kjarni::reranker;
//!
//! // One-liner
//! let ranked = reranker::rerank("minilm-l6-v2-cross-encoder", "query", &docs).await?;
//!
//! // Builder pattern
//! let reranker = Reranker::builder("ms-marco-minilm")
//!     .gpu()
//!     .build()
//!     .await?;
//!
//! let ranked = reranker.rerank("What is rust?", &documents).await?;
//! let score = reranker.score("query", "document").await?;
//! ```

mod builder;
mod model;
pub mod presets;
mod types;
mod validation;

pub use builder::RerankerBuilder;
pub use model::Reranker;
pub use presets::{RerankerPreset, RerankerTier};
pub use types::{RerankOverrides, RerankResult, RerankerError, RerankerResult};

// ============================================================================
// Convenience Functions
// ============================================================================

/// Rerank documents by relevance to a query.
///
/// Returns documents sorted by relevance score (highest first).
///
/// # Example
///
/// ```ignore
/// let documents = vec!["doc1", "doc2", "doc3"];
/// let ranked = kjarni::reranker::rerank("ms-marco-minilm", "query", &documents).await?;
/// for (idx, score) in ranked {
///     println!("Doc {}: {:.4}", idx, score);
/// }
/// ```
pub async fn rerank(model: &str, query: &str, documents: &[&str]) -> RerankerResult<Vec<RerankResult>> {
    let reranker = Reranker::new(model).await?;
    reranker.rerank(query, documents).await
}

/// Score a single query-document pair.
///
/// # Example
///
/// ```ignore
/// let score = kjarni::reranker::score("ms-marco-minilm", "What is Rust?", "Rust is a programming language.").await?;
/// println!("Relevance: {:.4}", score);
/// ```
pub async fn score(model: &str, query: &str, document: &str) -> RerankerResult<f32> {
    let reranker = Reranker::new(model).await?;
    reranker.score(query, document).await
}

/// Check if a model is valid for reranking.
pub fn is_reranking_model(model: &str) -> RerankerResult<()> {
    use kjarni_transformers::models::ModelType;

    let model_type = ModelType::from_cli_name(model)
        .ok_or_else(|| RerankerError::UnknownModel(model.to_string()))?;

    validation::validate_for_reranking(model_type)?;
    Ok(())
}

/// List all available reranking models.
pub fn available_models() -> Vec<&'static str> {
    validation::get_reranking_models()
}

#[cfg(test)]
mod tests;