//! High-level translation API.
//!
//! # Quick Start
//!
//! ```ignore
//! use kjarni::translator;
//!
//! // One-liner
//! let german = translator::translate("flan-t5-base", "Hello!", "en", "de").await?;
//!
//! // Reusable instance
//! let t = Translator::new("flan-t5-base").await?;
//! let german = t.translate("Hello!", "english", "german").await?;
//! let french = t.translate("Hello!", "en", "fr").await?;
//!
//! // With defaults for batch translation
//! let t = Translator::builder("flan-t5-base")
//!     .from("english")
//!     .to("german")
//!     .build()
//!     .await?;
//!
//! for text in texts {
//!     let translated = t.translate_default(&text).await?;
//! }
//! ```
//!
//! # Streaming
//!
//! ```ignore
//! use futures::StreamExt;
//!
//! let mut stream = translator.stream("Long text...", "en", "de").await?;
//! while let Some(chunk) = stream.next().await {
//!     print!("{}", chunk?);
//! }
//! ```
//!
//! # Supported Languages
//!
//! English, German, French, Spanish, Italian, Portuguese, Dutch, Russian,
//! Chinese, Japanese, Korean, Arabic, Hindi, Turkish, Polish, Romanian
//!
//! Languages can be specified as ISO codes ("en", "de") or names ("English", "German").

mod builder;
mod languages;
mod model;
pub mod presets;
mod types;
mod validation;

pub use builder::TranslatorBuilder;
pub use languages::{normalize_language, supported_languages};
pub use model::Translator;
pub use presets::{TranslatorPreset, TranslatorTier};
pub use types::{TranslatorError, TranslatorResult};

// ============================================================================
// Convenience Functions
// ============================================================================

/// Translate text with explicit languages.
///
/// # Example
///
/// ```ignore
/// let german = kjarni::translator::translate("flan-t5-base", "Hello", "en", "de").await?;
/// ```
pub async fn translate(model: &str, text: &str, from: &str, to: &str) -> TranslatorResult<String> {
    Translator::new(model).await?.translate(text, from, to).await
}

/// Translate using a preset.
pub async fn translate_preset(
    preset: &TranslatorPreset,
    text: &str,
    from: &str,
    to: &str,
) -> TranslatorResult<String> {
    TranslatorBuilder::from_preset(preset)
        .build()
        .await?
        .translate(text, from, to)
        .await
}

/// Translate using a tier.
pub async fn translate_tier(
    tier: TranslatorTier,
    text: &str,
    from: &str,
    to: &str,
) -> TranslatorResult<String> {
    translate_preset(tier.resolve(), text, from, to).await
}

/// List available translation models.
pub fn available_models() -> Vec<&'static str> {
    validation::get_translation_models()
}

/// Check if a model can be used for translation.
pub fn is_translation_model(model: &str) -> TranslatorResult<()> {
    let model_type = ModelType::from_cli_name(model)
        .ok_or_else(|| TranslatorError::UnknownModel(model.to_string()))?;
    validation::validate_for_translation(model_type)
}

use kjarni_transformers::models::ModelType;


#[cfg(test)]
mod tests;