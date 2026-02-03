//! Builder pattern for Translator configuration.

use std::path::PathBuf;
use std::sync::Arc;

use kjarni_transformers::WgpuContext;

use crate::common::{DownloadPolicy, KjarniDevice};
use crate::seq2seq::Seq2SeqOverrides;

use super::model::Translator;
use super::presets::TranslatorPreset;
use super::types::TranslatorResult;

/// Builder for configuring a Translator instance.
///
/// # Example
///
/// ```ignore
/// use kjarni::translator::Translator;
///
/// let translator = Translator::builder("flan-t5-base")
///     .from("english")
///     .to("german")
///     .build()
///     .await?;
/// ```
pub struct TranslatorBuilder {
    pub(crate) model: String,
    pub(crate) default_from: Option<String>,
    pub(crate) default_to: Option<String>,
    pub(crate) device: KjarniDevice,
    pub(crate) context: Option<Arc<WgpuContext>>,
    pub(crate) cache_dir: Option<PathBuf>,
    pub(crate) download_policy: DownloadPolicy,
    pub(crate) overrides: Seq2SeqOverrides,
    pub(crate) quiet: bool,
}

impl TranslatorBuilder {
    /// Create a new builder for the specified model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            default_from: None,
            default_to: None,
            device: KjarniDevice::default(),
            context: None,
            cache_dir: None,
            download_policy: DownloadPolicy::default(),
            overrides: Seq2SeqOverrides::for_translation(),
            quiet: false,
        }
    }

    /// Create a builder from a preset.
    pub fn from_preset(preset: &TranslatorPreset) -> Self {
        Self {
            model: preset.model.to_string(),
            default_from: None,
            default_to: None,
            device: preset.recommended_device,
            context: None,
            cache_dir: None,
            download_policy: DownloadPolicy::default(),
            overrides: Seq2SeqOverrides::for_translation(),
            quiet: false,
        }
    }

    // =========================================================================
    // Language Configuration
    // =========================================================================

    /// Set default source language.
    pub fn from(mut self, lang: impl Into<String>) -> Self {
        self.default_from = Some(lang.into());
        self
    }

    /// Set default target language.
    pub fn to(mut self, lang: impl Into<String>) -> Self {
        self.default_to = Some(lang.into());
        self
    }

    /// Set both default languages.
    pub fn languages(self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.from(from).to(to)
    }

    // =========================================================================
    // Device Configuration
    // =========================================================================

    /// Set the device for inference.
    pub fn device(mut self, device: KjarniDevice) -> Self {
        self.device = device;
        self
    }

    /// Use CPU for inference.
    pub fn cpu(self) -> Self {
        self.device(KjarniDevice::Cpu)
    }

    /// Use GPU for inference.
    pub fn gpu(self) -> Self {
        self.device(KjarniDevice::Gpu)
    }

    /// Provide a pre-created GPU context.
    pub fn with_context(mut self, context: Arc<WgpuContext>) -> Self {
        self.context = Some(context);
        self.device = KjarniDevice::Gpu;
        self
    }

    // =========================================================================
    // Generation Configuration
    // =========================================================================

    /// Set the number of beams for beam search.
    pub fn num_beams(mut self, n: usize) -> Self {
        self.overrides.num_beams = Some(n);
        self
    }

    /// Set the maximum output length.
    pub fn max_length(mut self, len: usize) -> Self {
        self.overrides.max_length = Some(len);
        self
    }

    /// Set the minimum output length.
    pub fn min_length(mut self, len: usize) -> Self {
        self.overrides.min_length = Some(len);
        self
    }

    /// Set length penalty for beam search.
    pub fn length_penalty(mut self, penalty: f32) -> Self {
        self.overrides.length_penalty = Some(penalty);
        self
    }

    /// Set no-repeat n-gram size.
    pub fn no_repeat_ngram_size(mut self, size: usize) -> Self {
        self.overrides.no_repeat_ngram_size = Some(size);
        self
    }

    /// Set repetition penalty.
    pub fn repetition_penalty(mut self, penalty: f32) -> Self {
        self.overrides.repetition_penalty = Some(penalty);
        self
    }

    /// Use greedy decoding (fastest, deterministic).
    pub fn greedy(mut self) -> Self {
        self.overrides.do_sample = Some(false);
        self.overrides.num_beams = Some(1);
        self
    }

    /// Use high-quality beam search.
    pub fn high_quality(mut self) -> Self {
        self.overrides.num_beams = Some(6);
        self
    }

    // =========================================================================
    // Loading Configuration
    // =========================================================================

    /// Set the cache directory for model files.
    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Set the download policy.
    pub fn download_policy(mut self, policy: DownloadPolicy) -> Self {
        self.download_policy = policy;
        self
    }

    /// Never download models automatically.
    pub fn offline(mut self) -> Self {
        self.download_policy = DownloadPolicy::Never;
        self
    }

    /// Suppress informational output.
    pub fn quiet(mut self) -> Self {
        self.quiet = true;
        self
    }

    // =========================================================================
    // Build
    // =========================================================================

    /// Build the Translator instance.
    pub async fn build(self) -> TranslatorResult<Translator> {
        Translator::from_builder(self).await
    }
}