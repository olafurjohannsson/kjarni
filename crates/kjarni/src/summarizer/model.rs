//! Core Summarizer implementation.

use futures::{Stream, StreamExt};

use kjarni_transformers::models::{ModelArchitecture, ModelType};
use kjarni_transformers::traits::Device;

use crate::seq2seq::{Seq2SeqGenerator, Seq2SeqGeneratorBuilder, Seq2SeqOverrides};

use super::builder::SummarizerBuilder;
use super::types::{SummarizerError, SummarizerResult};
use super::validation::validate_for_summarization;

/// High-level summarization API
pub struct Summarizer {
    /// Underlying seq2seq generator.
    generator: Seq2SeqGenerator,

    /// Model type.
    model_type: ModelType,

    /// Whether this model needs a "summarize: " prefix (T5 vs BART).
    needs_prefix: bool,
}

impl Summarizer {
    /// Create a Summarizer with default settings.
    pub async fn new(model: &str) -> SummarizerResult<Self> {
        SummarizerBuilder::new(model).build().await
    }

    /// Create a builder for custom configuration.
    pub fn builder(model: &str) -> SummarizerBuilder {
        SummarizerBuilder::new(model)
    }

    /// Internal: construct from builder.
    pub(crate) async fn from_builder(builder: SummarizerBuilder) -> SummarizerResult<Self> {
        // Resolve and validate model type
        let model_type = ModelType::from_cli_name(&builder.model)
            .ok_or_else(|| SummarizerError::UnknownModel(builder.model.clone()))?;

        validate_for_summarization(model_type)?;

        // Check if model needs prefix
        let needs_prefix = matches!(model_type.info().architecture, ModelArchitecture::T5);

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

        Ok(Self {
            generator,
            model_type,
            needs_prefix,
        })
    }

    /// Summarize text with default settings.
    pub async fn summarize(&self, text: &str) -> SummarizerResult<String> {
        let prompt = self.format_prompt(text);
        self.generator
            .generate(&prompt)
            .await
            .map_err(SummarizerError::from)
    }

    /// Summarize with custom generation overrides.
    pub async fn summarize_with_config(
        &self,
        text: &str,
        overrides: &Seq2SeqOverrides,
    ) -> SummarizerResult<String> {
        let prompt = self.format_prompt(text);
        self.generator
            .generate_with_config(&prompt, overrides)
            .await
            .map_err(SummarizerError::from)
    }

    /// Stream summarization.
    pub async fn stream(
        &self,
        text: &str,
    ) -> SummarizerResult<std::pin::Pin<Box<dyn Stream<Item = SummarizerResult<String>> + Send>>>
    {
        let prompt = self.format_prompt(text);

        let inner = self.generator.stream_text(&prompt).await?;

        let mapped = inner.map(|r| r.map_err(SummarizerError::from));
        Ok(Box::pin(mapped))
    }

    /// Format the prompt based on model type.
    fn format_prompt(&self, text: &str) -> String {
        if self.needs_prefix {
            format!("summarize: {}", text)
        } else {
            // BART doesn't need a prefix
            text.to_string()
        }
    }

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

    /// Get a reference to the underlying generator.
    pub fn generator(&self) -> &Seq2SeqGenerator {
        &self.generator
    }
}

impl std::fmt::Debug for Summarizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Summarizer")
            .field("model", &self.generator.model_name())
            .field("device", &self.generator.device())
            .finish()
    }
}