//! High-level text embedding API.
//!
//! Generate dense vector embeddings for semantic search, clustering, and similarity.
//!
//! # Quick Start
//!
//! ```ignore
//! use kjarni::embedder;
//!
//! // One-liner
//! let embedding = embedder::embed("minilm-l6-v2", "Hello world").await?;
//!
//! // Builder pattern
//! let embedder = Embedder::builder("nomic-embed-text")
//!     .gpu()
//!     .normalize(true)
//!     .pooling(PoolingStrategy::Mean)
//!     .build()
//!     .await?;
//!
//! let embedding = embedder.embed("Hello world").await?;
//! let similarity = embedder.similarity("text1", "text2").await?;
//! ```

mod builder;
mod model;
pub mod presets;
mod types;
mod validation;

pub use builder::EmbedderBuilder;
pub use model::Embedder;
pub use presets::{EmbedderPreset, EmbedderTier};
pub use types::{EmbeddingOverrides, EmbedderError, EmbedderResult, PoolingStrategy};

// ============================================================================
// Convenience Functions
// ============================================================================

/// Embed a single text using the specified model.
///
/// # Example
///
/// ```ignore
/// let embedding = kjarni::embedder::embed("minilm-l6-v2", "Hello world").await?;
/// println!("Dimension: {}", embedding.len());
/// ```
pub async fn embed(model: &str, text: &str) -> EmbedderResult<Vec<f32>> {
    let embedder = Embedder::new(model).await?;
    embedder.embed(text).await
}

/// Compute cosine similarity between two texts.
pub async fn similarity(model: &str, text1: &str, text2: &str) -> EmbedderResult<f32> {
    let embedder = Embedder::new(model).await?;
    embedder.similarity(text1, text2).await
}

/// Check if a model is valid for embedding.
pub fn is_embedding_model(model: &str) -> EmbedderResult<()> {
    use kjarni_transformers::models::ModelType;

    let model_type = ModelType::from_cli_name(model)
        .ok_or_else(|| EmbedderError::UnknownModel(model.to_string()))?;

    validation::validate_for_embedding(model_type)?;
    Ok(())
}

/// List all available embedding models.
pub fn available_models() -> Vec<&'static str> {
    validation::get_embedding_models()
}

#[cfg(test)]
mod tests;