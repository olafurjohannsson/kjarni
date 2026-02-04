//! Builder pattern for Summarizer configuration.

use std::path::PathBuf;
use std::sync::Arc;

use kjarni_transformers::WgpuContext;

use crate::common::{DownloadPolicy, KjarniDevice};
use crate::seq2seq::Seq2SeqOverrides;

use super::model::Summarizer;
use super::presets::SummarizerPreset;
use super::types::SummarizerResult;

/// Builder for configuring a Summarizer instance.
pub struct SummarizerBuilder {
    pub(crate) model: String,
    pub(crate) device: KjarniDevice,
    pub(crate) context: Option<Arc<WgpuContext>>,
    pub(crate) cache_dir: Option<PathBuf>,
    pub(crate) download_policy: DownloadPolicy,
    pub(crate) overrides: Seq2SeqOverrides,
    pub(crate) quiet: bool,
}

impl SummarizerBuilder {
    /// Create a new builder for the specified model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            device: KjarniDevice::default(),
            context: None,
            cache_dir: None,
            download_policy: DownloadPolicy::default(),
            overrides: Seq2SeqOverrides::default(),
            quiet: false,
        }
    }

    /// Create a builder from a preset.
    pub fn from_preset(preset: &SummarizerPreset) -> Self {
        let mut builder = Self::new(preset.model);
        builder.device = preset.recommended_device;
        builder.overrides.min_length = Some(preset.default_min_length);
        builder.overrides.max_length = Some(preset.default_max_length);
        builder
    }

    // =========================================================================
    // Length Presets
    // =========================================================================

    /// Short summaries (30-60 tokens).
    pub fn short(mut self) -> Self {
        self.overrides.min_length = Some(30);
        self.overrides.max_length = Some(60);
        self
    }

    /// Medium summaries (50-150 tokens).
    pub fn medium(mut self) -> Self {
        self.overrides.min_length = Some(50);
        self.overrides.max_length = Some(150);
        self
    }

    /// Long summaries (100-300 tokens).
    pub fn long(mut self) -> Self {
        self.overrides.min_length = Some(100);
        self.overrides.max_length = Some(300);
        self
    }

    // =========================================================================
    // Length Control
    // =========================================================================

    /// Set minimum summary length.
    pub fn min_length(mut self, len: usize) -> Self {
        self.overrides.min_length = Some(len);
        self
    }

    /// Set maximum summary length.
    pub fn max_length(mut self, len: usize) -> Self {
        self.overrides.max_length = Some(len);
        self
    }

// =========================================================================
    // Generation Control
    // =========================================================================

    /// Set the number of beams for beam search.
    pub fn num_beams(mut self, n: usize) -> Self {
        self.overrides.num_beams = Some(n);
        self
    }

    /// Set length penalty.
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
    // Loading Configuration
    // =========================================================================

    /// Set the cache directory.
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

    /// Build the Summarizer instance.
    pub async fn build(self) -> SummarizerResult<Summarizer> {
        Summarizer::from_builder(self).await
    }
}