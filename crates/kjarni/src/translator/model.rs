//! Core Translator implementation.

use std::sync::Arc;
use std::time::Instant;

use futures::future::ready;
use futures::stream::{self, BoxStream};
use futures::{Stream, StreamExt};
use log::{debug, info};

use kjarni_transformers::models::{ModelArchitecture, ModelType};
use kjarni_transformers::traits::Device;

use crate::common::folder_name;
use crate::seq2seq::{
    Seq2SeqGenerator, Seq2SeqGeneratorBuilder, Seq2SeqOverrides, folder_architecture,
};

use super::builder::TranslatorBuilder;
use super::segment::{MAX_SEGMENT_TOKENS, segments};
use super::types::{TranslatorError, TranslatorResult};
use super::validation::{
    Directions, translation_input, validate_folder_for_translation, validate_for_translation,
};

/// High-level translation API
pub struct Translator {
    /// Underlying seq2seq generator.
    generator: Arc<Seq2SeqGenerator>,

    /// The directions the model translates.
    directions: Directions,

    /// Default source language (normalized).
    default_from: Option<String>,

    /// Default target language (normalized).
    default_to: Option<String>,
}

impl Translator {
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

        let (from, to) = (builder.default_from.clone(), builder.default_to.clone());

        // A Marian folder's directions need its tokenizer, so they are read
        // once it is loaded; the rest are known now.
        let known_directions = match &builder.model_path {
            Some(path) => {
                let architecture = folder_architecture(path)?;
                validate_folder_for_translation(&folder_name(path), architecture)?;
                (architecture != ModelArchitecture::Marian).then_some(Directions::Prompt)
            }
            None => {
                let model_type = ModelType::from_cli_name(&builder.model)
                    .ok_or_else(|| TranslatorError::UnknownModel(builder.model.clone()))?;
                debug!(
                    "Resolved translator model '{}' -> {:?}",
                    builder.model, model_type
                );
                validate_for_translation(model_type)?;
                Some(Directions::of(model_type))
            }
        };
        // Unknown default languages fail before the model is loaded.
        if let Some(directions) = &known_directions {
            Self::default_languages(directions, from.as_ref(), to.as_ref())?;
        }

        // Build the underlying seq2seq generator
        let mut gen_builder = Seq2SeqGeneratorBuilder::new(&builder.model)
            .device(builder.device)
            .download_policy(builder.download_policy)
            .with_overrides(builder.overrides);

        if let Some(path) = &builder.model_path {
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

        let directions = match (known_directions, &builder.model_path) {
            (Some(directions), _) => directions,
            (None, Some(path)) => {
                let tokenizer = generator.encoder_decoder().model.tokenizer();
                Directions::of_marian_folder(path, |token| tokenizer.token_to_id(token).is_some())
            }
            (None, None) => unreachable!("a registry model's directions are known"),
        };
        debug!("Translator directions: {directions:?}");
        let (default_from, default_to) =
            Self::default_languages(&directions, from.as_ref(), to.as_ref())?;

        let build_elapsed = build_start.elapsed();
        info!(
            "Translator ready: model='{}', device={:?}, built in {:.2}s",
            generator.model_name(),
            generator.device(),
            build_elapsed.as_secs_f32()
        );

        Ok(Self {
            generator: Arc::new(generator),
            directions,
            default_from,
            default_to,
        })
    }

    /// The default languages, as `directions` names them.
    fn default_languages(
        directions: &Directions,
        from: Option<&String>,
        to: Option<&String>,
    ) -> TranslatorResult<(Option<String>, Option<String>)> {
        let language = |lang: &String| {
            directions
                .language(lang)
                .ok_or_else(|| TranslatorError::UnknownLanguage(lang.clone()))
        };
        Ok((
            from.map(language).transpose()?,
            to.map(language).transpose()?,
        ))
    }

    /// The language `input` names for this model.
    fn language(&self, input: &str) -> TranslatorResult<String> {
        self.directions
            .language(input)
            .ok_or_else(|| TranslatorError::UnknownLanguage(input.to_string()))
    }

    /// Translate text with explicit source and target languages.
    ///
    /// Each sentence is translated on its own, and line breaks are kept. A `max_length`
    /// set on the builder applies to each sentence.
    pub async fn translate(&self, text: &str, from: &str, to: &str) -> TranslatorResult<String> {
        let translate_start = Instant::now();

        let from_normalized = self.language(from)?;
        let to_normalized = self.language(to)?;

        info!(
            "Translating {} -> {}: {} chars",
            from_normalized,
            to_normalized,
            text.len()
        );

        let result = self
            .translate_segments(
                text,
                &from_normalized,
                &to_normalized,
                &Seq2SeqOverrides::default(),
            )
            .await?;

        let elapsed = translate_start.elapsed();
        info!(
            "Translation complete: {} -> {} chars in {:.2}s",
            text.len(),
            result.len(),
            elapsed.as_secs_f32()
        );

        Ok(result)
    }

    /// Translate with custom generation overrides, which apply to each sentence.
    pub async fn translate_with_config(
        &self,
        text: &str,
        from: &str,
        to: &str,
        overrides: &Seq2SeqOverrides,
    ) -> TranslatorResult<String> {
        let from_normalized = self.language(from)?;
        let to_normalized = self.language(to)?;

        debug!(
            "Translation with overrides: {} -> {}, {:?}",
            from_normalized, to_normalized, overrides
        );

        self.translate_segments(text, &from_normalized, &to_normalized, overrides)
            .await
    }

    /// Translate using default languages set via builder
    pub async fn translate_default(&self, text: &str) -> TranslatorResult<String> {
        let from = self
            .default_from
            .as_ref()
            .ok_or(TranslatorError::MissingLanguage)?;
        let to = self
            .default_to
            .as_ref()
            .ok_or(TranslatorError::MissingLanguage)?;

        debug!("Using default languages: {} -> {}", from, to);

        self.translate_segments(text, from, to, &Seq2SeqOverrides::default())
            .await
    }

    /// Translate to a specific language using default source language.
    pub async fn translate_to(&self, text: &str, to: &str) -> TranslatorResult<String> {
        let from = self
            .default_from
            .as_ref()
            .ok_or(TranslatorError::MissingLanguage)?;
        let to_normalized = self.language(to)?;

        debug!("Translate to: {} -> {}", from, to_normalized);

        self.translate_segments(text, from, &to_normalized, &Seq2SeqOverrides::default())
            .await
    }

    /// Translate from a specific language using default target language.
    pub async fn translate_from(&self, text: &str, from: &str) -> TranslatorResult<String> {
        let from_normalized = self.language(from)?;
        let to = self
            .default_to
            .as_ref()
            .ok_or(TranslatorError::MissingLanguage)?;

        debug!("Translate from: {} -> {}", from_normalized, to);

        self.translate_segments(text, &from_normalized, to, &Seq2SeqOverrides::default())
            .await
    }

    /// Stream translation with explicit languages.
    pub async fn stream(
        &self,
        text: &str,
        from: &str,
        to: &str,
    ) -> TranslatorResult<std::pin::Pin<Box<dyn Stream<Item = TranslatorResult<String>> + Send>>>
    {
        let from_normalized = self.language(from)?;
        let to_normalized = self.language(to)?;

        info!(
            "Streaming translation {} -> {}: {} chars",
            from_normalized,
            to_normalized,
            text.len()
        );

        self.stream_segments(text, &from_normalized, &to_normalized)
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

        info!(
            "Streaming translation (defaults) {} -> {}: {} chars",
            from,
            to,
            text.len()
        );

        self.stream_segments(text, from, to)
    }

    /// The model input and following whitespace of each segment of `text`.
    fn prepare_segments(
        &self,
        text: &str,
        from: &str,
        to: &str,
    ) -> TranslatorResult<Vec<(String, String)>> {
        let tokenizer = self.generator.encoder_decoder().model.tokenizer();
        let count_tokens = |piece: &str| {
            tokenizer
                .encode(piece, false)
                .map_or(piece.len(), |encoding| encoding.len())
        };
        segments(text, MAX_SEGMENT_TOKENS, &count_tokens)
            .into_iter()
            .map(|segment| {
                Ok((
                    self.format_prompt(segment.text, from, to)?,
                    segment.gap.to_string(),
                ))
            })
            .collect()
    }

    async fn translate_segments(
        &self,
        text: &str,
        from: &str,
        to: &str,
        overrides: &Seq2SeqOverrides,
    ) -> TranslatorResult<String> {
        let mut result = String::new();
        for (prompt, gap) in self.prepare_segments(text, from, to)? {
            let translated = self
                .generator
                .generate_with_config(&prompt, overrides)
                .await
                .map_err(TranslatorError::from)?;
            result.push_str(&translated);
            result.push_str(&gap);
        }
        Ok(result)
    }

    /// Generates one segment at a time, as the stream is read.
    fn stream_segments(
        &self,
        text: &str,
        from: &str,
        to: &str,
    ) -> TranslatorResult<std::pin::Pin<Box<dyn Stream<Item = TranslatorResult<String>> + Send>>>
    {
        let parts = self.prepare_segments(text, from, to)?;
        let generator = self.generator.clone();

        let translated = stream::iter(parts)
            .then(move |(prompt, gap)| {
                let generator = generator.clone();
                async move {
                    let tokens: BoxStream<'static, TranslatorResult<String>> =
                        match generator.stream_text(&prompt).await {
                            Ok(tokens) => tokens.map(|r| r.map_err(TranslatorError::from)).boxed(),
                            Err(e) => stream::once(ready(Err(TranslatorError::from(e)))).boxed(),
                        };
                    let gap: BoxStream<'static, TranslatorResult<String>> = if gap.is_empty() {
                        stream::empty().boxed()
                    } else {
                        stream::once(ready(Ok(gap))).boxed()
                    };
                    tokens.chain(gap)
                }
            })
            .flatten();
        Ok(Box::pin(translated))
    }

    /// Format what the model is given for this direction.
    fn format_prompt(&self, text: &str, from: &str, to: &str) -> TranslatorResult<String> {
        translation_input(&self.directions, self.model_name(), text, from, to)
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
