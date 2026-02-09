//! Generic encoder-decoder (seq2seq) text generation.
//!
//! This module provides `Seq2SeqGenerator`, a foundation for text-to-text
//! generation using encoder-decoder models like T5 and BART.
//!
//! For most use cases, prefer the higher-level task APIs:
//! - [`crate::translator`] - Translation between languages
//! - [`crate::summarizer`] - Text summarization
//!
//! Use `Seq2SeqGenerator` directly when you need:
//! - Custom prompt formats
//! - Direct access to generation parameters
//! - Tasks not covered by specific APIs
//!
//! # Quick Start
//!
//! ```ignore
//! use kjarni::seq2seq::Seq2SeqGenerator;
//!
//! // One-liner
//! let output = kjarni::seq2seq::generate("flan-t5-base", "translate English to French: Hello").await?;
//!
//! // Reusable instance
//! let generator = Seq2SeqGenerator::new("flan-t5-base").await?;
//! let output1 = generator.generate("summarize: Long text...").await?;
//! let output2 = generator.generate("translate: More text...").await?;
//! ```
//!
//! # Configuration
//!
//! ```ignore
//! use kjarni::seq2seq::{Seq2SeqGenerator, Seq2SeqOverrides};
//!
//! // Builder pattern
//! let generator = Seq2SeqGenerator::builder("flan-t5-large")
//!     .num_beams(6)           // Higher quality
//!     .max_length(256)        // Longer outputs
//!     .gpu()                  // Use GPU
//!     .build()
//!     .await?;
//!
//! // Runtime overrides
//! let output = generator.generate_with_config(
//!     "summarize: ...",
//!     &Seq2SeqOverrides::greedy()  // Fast single-beam
//! ).await?;
//! ```
//!
//! # Streaming
//!
//! ```ignore
//! use futures::StreamExt;
//!
//! let mut stream = generator.stream("translate: Hello world").await?;
//! while let Some(token) = stream.next().await {
//!     print!("{}", token?.text);
//! }
//! ```


mod builder;
mod model;
mod resolution;
mod types;
mod validation;

// Re-exports
pub use builder::Seq2SeqGeneratorBuilder;
pub use model::Seq2SeqGenerator;
pub use types::{Seq2SeqError, Seq2SeqOverrides, Seq2SeqResult, Seq2SeqToken};

/// Generate text with default settings.
///
/// This is the simplest possible API - a one-liner for quick seq2seq generation.
///
/// # Example
///
/// ```ignore
/// let output = kjarni::seq2seq::generate(
///     "flan-t5-base",
///     "translate English to German: How are you?"
/// ).await?;
/// ```
///
/// # Notes
///
/// - Uses CPU by default
/// - Downloads model if not present
/// - For repeated generation, create a `Seq2SeqGenerator` instance instead
pub async fn generate(model: &str, input: &str) -> Seq2SeqResult<String> {
    Seq2SeqGenerator::new(model).await?.generate(input).await
}

/// Generate with custom overrides.
///
/// # Example
///
/// ```ignore
/// use kjarni::seq2seq::{generate_with_config, Seq2SeqOverrides};
///
/// let output = generate_with_config(
///     "flan-t5-base",
///     "summarize: Long article...",
///     Seq2SeqOverrides::high_quality()
/// ).await?;
/// ```
pub async fn generate_with_config(
    model: &str,
    input: &str,
    overrides: Seq2SeqOverrides,
) -> Seq2SeqResult<String> {
    Seq2SeqGenerator::builder(model)
        .with_overrides(overrides)
        .build()
        .await?
        .generate(input)
        .await
}

/// List all available seq2seq models.
///
/// Returns CLI names of models that can be used with Seq2SeqGenerator.
pub fn available_models() -> Vec<&'static str> {
    validation::get_seq2seq_models()
}

/// Get suggested models for seq2seq tasks.
pub fn suggested_models() -> Vec<&'static str> {
    validation::suggest_seq2seq_models()
}

/// Check if a model is valid for seq2seq.
///
/// Returns Ok(()) if valid, or an error describing why not.
pub fn is_seq2seq_model(model: &str) -> Seq2SeqResult<()> {
    use kjarni_transformers::models::ModelType;

    let model_type = ModelType::from_cli_name(model)
        .ok_or_else(|| Seq2SeqError::UnknownModel(model.to_string()))?;

    validation::validate_for_seq2seq(model_type)?;
    Ok(())
}


#[cfg(test)]
mod send_sync_tests {
    use super::*;

    const _: () = {
        const fn assert_send<T: Send>() {}
        const fn assert_sync<T: Sync>() {}

        assert_send::<Seq2SeqGenerator>();
        assert_sync::<Seq2SeqGenerator>();

        assert_send::<Seq2SeqGeneratorBuilder>();
        assert_sync::<Seq2SeqGeneratorBuilder>();

        assert_send::<Seq2SeqOverrides>();
        assert_sync::<Seq2SeqOverrides>();

        assert_send::<Seq2SeqToken>();
        assert_sync::<Seq2SeqToken>();

        assert_send::<Seq2SeqError>();
        assert_sync::<Seq2SeqError>();
    };
}


#[cfg(test)]
mod tests;