//! High-level summarization API.
//!
//! # Quick Start
//!
//! ```ignore
//! use kjarni::summarizer;
//!
//! // One-liner
//! let summary = summarizer::summarize("distilbart-cnn", &long_article).await?;
//!
//! // Reusable instance
//! let s = Summarizer::new("distilbart-cnn").await?;
//! let summary = s.summarize(&article).await?;
//!
//! ```
//!
//! # Streaming
//!
//! ```ignore
//! use futures::StreamExt;
//!
//! let mut stream = summarizer.stream(&article).await?;
//! while let Some(chunk) = stream.next().await {
//!     print!("{}", chunk?);
//! }
//! ```
//!
mod builder;
mod model;
pub mod presets;

mod types;
mod validation;

pub use builder::SummarizerBuilder;
pub use model::Summarizer;
pub use presets::{SummarizerPreset, SummarizerTier};
pub use types::{SummarizerError, SummarizerResult};
pub use validation::{get_summarization_models, recommended_summarization_models, validate_for_summarization};
use kjarni_transformers::models::ModelType;

/// Summarize text with default settings.
///
/// # Example
///
/// ```ignore
/// let summary = kjarni::summarizer::summarize("distilbart-cnn", &article).await?;
/// ```
pub async fn summarize(model: &str, text: &str) -> SummarizerResult<String> {
    Summarizer::new(model).await?.summarize(text).await
}

/// Summarize using a preset.
///
/// # Example
///
/// ```ignore
/// use kjarni::summarizer::presets::SUMMARIZER_FAST_V1;
///
/// let summary = summarize_preset(&SUMMARIZER_FAST_V1, &article).await?;
/// ```
pub async fn summarize_preset(preset: &SummarizerPreset, text: &str) -> SummarizerResult<String> {
    SummarizerBuilder::from_preset(preset)
        .build()
        .await?
        .summarize(text)
        .await
}

/// Summarize using a tier.
///
/// # Example
///
/// ```ignore
/// use kjarni::summarizer::SummarizerTier;
///
/// let summary = summarize_tier(SummarizerTier::Fast, &article).await?;
/// ```
pub async fn summarize_tier(tier: SummarizerTier, text: &str) -> SummarizerResult<String> {
    summarize_preset(tier.resolve(), text).await
}

/// List available summarization models.
///
/// # Example
///
/// ```ignore
/// for model in kjarni::summarizer::available_models() {
///     println!("{}", model);
/// }
/// ```
pub fn available_models() -> Vec<&'static str> {
    validation::get_summarization_models()
}

/// Get recommended summarization models.
///
/// Returns a curated list of models known to work well for summarization.
pub fn recommended_models() -> Vec<&'static str> {
    validation::recommended_summarization_models()
}

/// Check if a model can be used for summarization.
///
/// Returns `Ok(())` if valid, or an error describing why not.
///
/// # Example
///
/// ```ignore
/// if kjarni::summarizer::is_summarization_model("distilbart-cnn").is_ok() {
///     println!("Model is valid for summarization");
/// }
/// ```
pub fn is_summarization_model(model: &str) -> SummarizerResult<()> {
    let model_type = ModelType::from_cli_name(model)
        .ok_or_else(|| SummarizerError::UnknownModel(model.to_string()))?;
    validation::validate_for_summarization(model_type)
}

#[cfg(test)]
mod tests;