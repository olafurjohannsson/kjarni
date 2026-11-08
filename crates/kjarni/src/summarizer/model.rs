//! Core Summarizer implementation.

use futures::{Stream, StreamExt};

use kjarni_transformers::models::{ModelArchitecture, ModelType};
use kjarni_transformers::traits::Device;

use crate::common::folder_name;
use crate::seq2seq::{
    Seq2SeqGenerator, Seq2SeqGeneratorBuilder, Seq2SeqOverrides, folder_architecture,
};

use super::builder::SummarizerBuilder;
use super::types::{SummarizerError, SummarizerResult};
use super::validation::{validate_folder_for_summarization, validate_for_summarization};

/// High-level summarization API
pub struct Summarizer {
    /// Underlying seq2seq generator.
    generator: Seq2SeqGenerator,

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
        let architecture = match &builder.model_path {
            Some(path) => {
                let architecture = folder_architecture(path)?;
                validate_folder_for_summarization(&folder_name(path), architecture)?;
                architecture
            }
            None => {
                let model_type = ModelType::from_cli_name(&builder.model)
                    .ok_or_else(|| SummarizerError::UnknownModel(builder.model.clone()))?;
                validate_for_summarization(model_type)?;
                model_type.info().architecture
            }
        };

        // T5 is told the task in the input; BART is trained for it.
        let needs_prefix = matches!(architecture, ModelArchitecture::T5);

        let mut gen_builder = Seq2SeqGeneratorBuilder::new(&builder.model)
            .device(builder.device)
            .download_policy(builder.download_policy)
            .with_overrides(builder.overrides);

        if let Some(path) = builder.model_path {
            gen_builder = gen_builder.model_path(path);
        }

        if builder.quiet {
            gen_builder = gen_builder.quiet();
        }

        if let Some(cache_dir) = builder.cache_dir {
            gen_builder = gen_builder.cache_dir(cache_dir);
        }

        if let Some(context) = builder.context {
            gen_builder = gen_builder.with_context(context);
        }

        if let Some(load_config) = builder.load_config {
            gen_builder.load_config = Some(load_config);
        }

        let generator = gen_builder.build().await?;

        Ok(Self {
            generator,
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

    /// The registry model, or `None` for a model loaded from a folder.
    pub fn model_type(&self) -> Option<ModelType> {
        self.generator.model_type()
    }

    /// The registry name, or the name of the folder the model was loaded from.
    pub fn model_name(&self) -> &str {
        self.generator.model_name()
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
