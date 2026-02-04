//! Core Translator implementation.

use std::sync::Arc;
use std::time::Instant;

use futures::{Stream, StreamExt};
use log::{debug, info};

use kjarni_transformers::models::ModelType;
use kjarni_transformers::traits::Device;

use crate::seq2seq::{Seq2SeqGenerator, Seq2SeqGeneratorBuilder, Seq2SeqOverrides};

use super::builder::TranslatorBuilder;
use super::languages::normalize_language;
use super::types::{TranslatorError, TranslatorResult};
use super::validation::validate_for_translation;

/// High-level translation API.
///
/// Translator wraps `Seq2SeqGenerator` and adds:
/// - Language normalization (accepts "en", "english", "English")
/// - Prompt formatting for T5 models
/// - Default language pairs for batch translation
///
/// # Example
///
/// ```ignore
/// use kjarni::translator::Translator;
///
/// // Simple usage
/// let t = Translator::new("flan-t5-base").await?;
/// let german = t.translate("Hello, how are you?", "en", "de").await?;
///
/// // With default languages for batch translation
/// let t = Translator::builder("flan-t5-base")
///     .from("english")
///     .to("german")
///     .build()
///     .await?;
///
/// let out1 = t.translate_default("Hello").await?;
/// let out2 = t.translate_default("Goodbye").await?;
/// ```
pub struct Translator {
    /// Underlying seq2seq generator.
    generator: Seq2SeqGenerator,

    /// Model type.
    model_type: ModelType,

    /// Default source language (normalized).
    default_from: Option<String>,

    /// Default target language (normalized).
    default_to: Option<String>,
}

impl Translator {
    // =========================================================================
    // Construction
    // =========================================================================

    /// Create a Translator with default settings.
    pub async fn new(model: &str) -> TranslatorResult<Self> {
        TranslatorBuilder::new(model).build().await
    }

    /// Create a builder for custom configuration.
    pub fn builder(model: &str) -> TranslatorBuilder {
        TranslatorBuilder::new(model)
    }

    /// Internal: construct from builder.
    pub(crate) async fn from_builder(builder: TranslatorBuilder) -> TranslatorResult<Self> {
        let build_start = Instant::now();

        // Resolve and validate model type
        let model_type = ModelType::from_cli_name(&builder.model)
            .ok_or_else(|| TranslatorError::UnknownModel(builder.model.clone()))?;

        debug!("Resolved translator model '{}' -> {:?}", builder.model, model_type);

        validate_for_translation(model_type)?;

        // Normalize default languages if provided
        let default_from = if let Some(ref lang) = builder.default_from {
            let normalized = normalize_language(lang)
                .ok_or_else(|| TranslatorError::UnknownLanguage(lang.clone()))?;
            debug!("Default source language: '{}' -> '{}'", lang, normalized);
            Some(normalized.to_string())
        } else {
            None
        };

        let default_to = if let Some(ref lang) = builder.default_to {
            let normalized = normalize_language(lang)
                .ok_or_else(|| TranslatorError::UnknownLanguage(lang.clone()))?;
            debug!("Default target language: '{}' -> '{}'", lang, normalized);
            Some(normalized.to_string())
        } else {
            None
        };

        // Build the underlying seq2seq generator
        let mut gen_builder = Seq2SeqGeneratorBuilder::new(&builder.model)
            .device(builder.device)
            .download_policy(builder.download_policy)
            .with_overrides(builder.overrides);

        if builder.quiet {
            gen_builder = gen_builder.quiet();
        }

        if let Some(cache_dir) = builder.cache_dir {
            gen_builder = gen_builder.cache_dir(cache_dir);
        }

        if let Some(context) = builder.context {
            gen_builder = gen_builder.with_context(context);
        }

        let generator = gen_builder.build().await?;

        let build_elapsed = build_start.elapsed();
        info!(
            "Translator ready: model='{}', device={:?}, built in {:.2}s",
            builder.model,
            generator.device(),
            build_elapsed.as_secs_f32()
        );

        Ok(Self {
            generator,
            model_type,
            default_from,
            default_to,
        })
    }

    // =========================================================================
    // Translation (explicit languages)
    // =========================================================================

    /// Translate text with explicit source and target languages.
    ///
    /// Languages can be specified as:
    /// - ISO codes: "en", "de", "fr"
    /// - Full names: "English", "German", "French"
    /// - Native names: "deutsch", "français"
    ///
    /// # Example
    ///
    /// ```ignore
    /// let german = translator.translate("Hello", "en", "de").await?;
    /// let french = translator.translate("Hello", "english", "french").await?;
    /// ```
    pub async fn translate(&self, text: &str, from: &str, to: &str) -> TranslatorResult<String> {
        let translate_start = Instant::now();

        let from_normalized = normalize_language(from)
            .ok_or_else(|| TranslatorError::UnknownLanguage(from.to_string()))?;
        let to_normalized = normalize_language(to)
            .ok_or_else(|| TranslatorError::UnknownLanguage(to.to_string()))?;

        let prompt = self.format_prompt(text, from_normalized, to_normalized);

        debug!(
            "Translation prompt: '{}' ({} chars)",
            &prompt[..prompt.len().min(80)],
            prompt.len()
        );

        info!(
            "Translating {} -> {}: {} chars",
            from_normalized,
            to_normalized,
            text.len()
        );

        let result = self
            .generator
            .generate(&prompt)
            .await
            .map_err(TranslatorError::from)?;

        let elapsed = translate_start.elapsed();
        info!(
            "Translation complete: {} -> {} chars in {:.2}s",
            text.len(),
            result.len(),
            elapsed.as_secs_f32()
        );

        Ok(result)
    }

    /// Translate with custom generation overrides.
    pub async fn translate_with_config(
        &self,
        text: &str,
        from: &str,
        to: &str,
        overrides: &Seq2SeqOverrides,
    ) -> TranslatorResult<String> {
        let from_normalized = normalize_language(from)
            .ok_or_else(|| TranslatorError::UnknownLanguage(from.to_string()))?;
        let to_normalized = normalize_language(to)
            .ok_or_else(|| TranslatorError::UnknownLanguage(to.to_string()))?;

        let prompt = self.format_prompt(text, from_normalized, to_normalized);

        debug!(
            "Translation with overrides: {} -> {}, {:?}",
            from_normalized, to_normalized, overrides
        );

        self.generator
            .generate_with_config(&prompt, overrides)
            .await
            .map_err(TranslatorError::from)
    }

    // =========================================================================
    // Translation (default languages)
    // =========================================================================

    /// Translate using default languages set via builder.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let t = Translator::builder("flan-t5-base")
    ///     .from("english")
    ///     .to("german")
    ///     .build()
    ///     .await?;
    ///
    /// let german = t.translate_default("Hello").await?;
    /// ```
    pub async fn translate_default(&self, text: &str) -> TranslatorResult<String> {
        let from = self
            .default_from
            .as_ref()
            .ok_or(TranslatorError::MissingLanguage)?;
        let to = self
            .default_to
            .as_ref()
            .ok_or(TranslatorError::MissingLanguage)?;

        let prompt = self.format_prompt(text, from, to);

        debug!("Using default languages: {} -> {}", from, to);

        self.generator
            .generate(&prompt)
            .await
            .map_err(TranslatorError::from)
    }

    /// Translate to a specific language using default source language.
    pub async fn translate_to(&self, text: &str, to: &str) -> TranslatorResult<String> {
        let from = self
            .default_from
            .as_ref()
            .ok_or(TranslatorError::MissingLanguage)?;
        let to_normalized = normalize_language(to)
            .ok_or_else(|| TranslatorError::UnknownLanguage(to.to_string()))?;

        let prompt = self.format_prompt(text, from, to_normalized);

        debug!("Translate to: {} -> {}", from, to_normalized);

        self.generator
            .generate(&prompt)
            .await
            .map_err(TranslatorError::from)
    }

    /// Translate from a specific language using default target language.
    pub async fn translate_from(&self, text: &str, from: &str) -> TranslatorResult<String> {
        let from_normalized = normalize_language(from)
            .ok_or_else(|| TranslatorError::UnknownLanguage(from.to_string()))?;
        let to = self
            .default_to
            .as_ref()
            .ok_or(TranslatorError::MissingLanguage)?;

        let prompt = self.format_prompt(text, from_normalized, to);

        debug!("Translate from: {} -> {}", from_normalized, to);

        self.generator
            .generate(&prompt)
            .await
            .map_err(TranslatorError::from)
    }

    // =========================================================================
    // Streaming
    // =========================================================================

    /// Stream translation with explicit languages.
    pub async fn stream(
        &self,
        text: &str,
        from: &str,
        to: &str,
    ) -> TranslatorResult<std::pin::Pin<Box<dyn Stream<Item = TranslatorResult<String>> + Send>>>
    {
        let from_normalized = normalize_language(from)
            .ok_or_else(|| TranslatorError::UnknownLanguage(from.to_string()))?;
        let to_normalized = normalize_language(to)
            .ok_or_else(|| TranslatorError::UnknownLanguage(to.to_string()))?;

        let prompt = self.format_prompt(text, from_normalized, to_normalized);

        info!(
            "Streaming translation {} -> {}: {} chars",
            from_normalized,
            to_normalized,
            text.len()
        );

        let inner = self.generator.stream_text(&prompt).await?;

        let mapped = inner.map(|r| r.map_err(TranslatorError::from));
        Ok(Box::pin(mapped))
    }

    /// Stream translation using default languages.
    pub async fn stream_default(
        &self,
        text: &str,
    ) -> TranslatorResult<std::pin::Pin<Box<dyn Stream<Item = TranslatorResult<String>> + Send>>>
    {
        let from = self
            .default_from
            .as_ref()
            .ok_or(TranslatorError::MissingLanguage)?;
        let to = self
            .default_to
            .as_ref()
            .ok_or(TranslatorError::MissingLanguage)?;

        let prompt = self.format_prompt(text, from, to);

        info!(
            "Streaming translation (defaults) {} -> {}: {} chars",
            from,
            to,
            text.len()
        );

        let inner = self.generator.stream_text(&prompt).await?;

        let mapped = inner.map(|r| r.map_err(TranslatorError::from));
        Ok(Box::pin(mapped))
    }

    // =========================================================================
    // Internal
    // =========================================================================

    /// Format the translation prompt for T5.
    fn format_prompt(&self, text: &str, from: &str, to: &str) -> String {
        format!("translate {} to {}: {}", from, to, text)
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the model type.
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Get the model's CLI name.
    pub fn model_name(&self) -> &str {
        self.model_type.cli_name()
    }

    /// Get the device the model is running on.
    pub fn device(&self) -> Device {
        self.generator.device()
    }

    /// Get the default source language (if set).
    pub fn default_from(&self) -> Option<&str> {
        self.default_from.as_deref()
    }

    /// Get the default target language (if set).
    pub fn default_to(&self) -> Option<&str> {
        self.default_to.as_deref()
    }

    /// Get a reference to the underlying generator.
    pub fn generator(&self) -> &Seq2SeqGenerator {
        &self.generator
    }
}


impl std::fmt::Debug for Translator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Translator")
            .field("model", &self.generator.model_name())
            .field("device", &self.generator.device())
            .field("default_from", &self.default_from)
            .field("default_to", &self.default_to)
            .finish()
    }
}