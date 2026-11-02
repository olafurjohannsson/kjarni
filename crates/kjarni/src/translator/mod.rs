//! High-level translation API.
//!
//! # Quick Start
//!
//! ```ignore
//! use kjarni::translator;
//!
//! // One-liner
//! let german = translator::translate("opus-mt-en-de", "Hello!", "en", "de").await?;
//!
//! // Reusable instance
//! let t = Translator::new("opus-mt-en-de").await?;
//! let german = t.translate("Hello!", "english", "german").await?;
//!
//! // Any pair on the general model, which often returns the input unchanged
//! let t = Translator::new("flan-t5-base").await?;
//! let french = t.translate("Hello!", "en", "fr").await?;
//!
//! // With defaults for batch translation
//! let t = Translator::builder("opus-mt-en-de")
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
mod segment;
mod types;
mod validation;

pub use builder::TranslatorBuilder;
pub use languages::{normalize_language, supported_languages};
pub use model::Translator;
pub use presets::{TranslatorPreset, TranslatorTier};
pub use types::{TranslatorError, TranslatorResult};

/// The general-purpose model for language pairs without a dedicated one. It often
/// returns questions and short inputs unchanged, so callers should warn when using it.
pub const FALLBACK_MODEL: &str = "flan-t5-base";

/// The best model for a language pair: Opus-MT for English and German in either
/// direction, otherwise [`FALLBACK_MODEL`]. Languages may be codes or names.
pub fn default_model(from: &str, to: &str) -> &'static str {
    match (normalize_language(from), normalize_language(to)) {
        (Some("English"), Some("German")) => "opus-mt-en-de",
        (Some("German"), Some("English")) => "opus-mt-de-en",
        _ => FALLBACK_MODEL,
    }
}

/// Translate text with explicit languages.
///
/// # Example
///
/// ```ignore
/// let german = kjarni::translator::translate("opus-mt-en-de", "Hello", "en", "de").await?;
/// ```
pub async fn translate(model: &str, text: &str, from: &str, to: &str) -> TranslatorResult<String> {
    Translator::new(model)
        .await?
        .translate(text, from, to)
        .await
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
