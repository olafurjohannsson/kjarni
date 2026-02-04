//! Builder pattern for Seq2SeqGenerator configuration.

use std::path::PathBuf;
use std::sync::Arc;

use kjarni_transformers::WgpuContext;

use crate::common::{DownloadPolicy, KjarniDevice, LoadConfig, LoadConfigBuilder};

use super::model::Seq2SeqGenerator;
use super::types::{Seq2SeqOverrides, Seq2SeqResult};

/// Builder for configuring a Seq2SeqGenerator instance.
///
/// # Example
///
/// ```ignore
/// use kjarni::seq2seq::{Seq2SeqGenerator, Seq2SeqOverrides};
///
/// let generator = Seq2SeqGenerator::builder("flan-t5-base")
///     .num_beams(6)
///     .max_length(256)
///     .gpu()
///     .build()
///     .await?;
/// ```
pub struct Seq2SeqGeneratorBuilder {
    // Model selection
    pub(crate) model: String,
    pub(crate) model_path: Option<PathBuf>,

    // Device configuration
    pub(crate) device: KjarniDevice,
    pub(crate) context: Option<Arc<WgpuContext>>,

    // Model loading
    pub(crate) cache_dir: Option<PathBuf>,
    pub(crate) download_policy: DownloadPolicy,
    pub(crate) load_config: Option<LoadConfig>,

    // Generation overrides
    pub(crate) overrides: Seq2SeqOverrides,

    // Behavior
    pub(crate) quiet: bool,
    pub(crate) allow_warnings: bool,
}

impl Seq2SeqGeneratorBuilder {
    /// Create a new builder for the specified model.
    ///
    /// # Arguments
    ///
    /// * `model` - Model name from registry (e.g., "flan-t5-base", "distilbart-cnn")
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            model_path: None,
            device: KjarniDevice::default(),
            context: None,
            cache_dir: None,
            download_policy: DownloadPolicy::default(),
            load_config: None,
            overrides: Seq2SeqOverrides::default(),
            quiet: false,
            allow_warnings: false,
        }
    }

    /// Load model from a local path instead of registry.
    pub fn from_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(path.into());
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
    ///
    /// Useful when sharing a context across multiple models.
    pub fn with_context(mut self, context: Arc<WgpuContext>) -> Self {
        self.context = Some(context);
        self.device = KjarniDevice::Gpu;
        self
    }

    // =========================================================================
    // Generation Parameters
    // =========================================================================

    /// Set the number of beams for beam search.
    ///
    /// - `1` = greedy decoding (fastest)
    /// - `4` = typical default
    /// - `6-8` = higher quality, slower
    pub fn num_beams(mut self, n: usize) -> Self {
        self.overrides.num_beams = Some(n);
        self
    }

    /// Set the maximum output length in tokens.
    pub fn max_length(mut self, len: usize) -> Self {
        self.overrides.max_length = Some(len);
        self
    }

    /// Set the minimum output length in tokens.
    pub fn min_length(mut self, len: usize) -> Self {
        self.overrides.min_length = Some(len);
        self
    }

    /// Set the length penalty for beam search.
    ///
    /// - `< 1.0` = favor shorter outputs
    /// - `= 1.0` = neutral
    /// - `> 1.0` = favor longer outputs
    pub fn length_penalty(mut self, penalty: f32) -> Self {
        self.overrides.length_penalty = Some(penalty);
        self
    }

    /// Set early stopping for beam search.
    pub fn early_stopping(mut self, early: bool) -> Self {
        self.overrides.early_stopping = Some(early);
        self
    }

    /// Set n-gram blocking size (prevents repetition).
    ///
    /// - `0` = disabled
    /// - `3` = typical for summarization
    pub fn no_repeat_ngram_size(mut self, size: usize) -> Self {
        self.overrides.no_repeat_ngram_size = Some(size);
        self
    }

    /// Set repetition penalty.
    pub fn repetition_penalty(mut self, penalty: f32) -> Self {
        self.overrides.repetition_penalty = Some(penalty);
        self
    }

    /// Use greedy decoding (fastest, single beam).
    pub fn greedy(mut self) -> Self {
        self.overrides.num_beams = Some(1);
        self
    }

    /// Use high-quality beam search (6 beams).
    pub fn high_quality(mut self) -> Self {
        self.overrides.num_beams = Some(6);
        self
    }

    /// Apply a preset of overrides.
    pub fn with_overrides(mut self, overrides: Seq2SeqOverrides) -> Self {
        self.overrides = overrides;
        self
    }

    /// Merge additional overrides with existing ones.
    pub fn merge_overrides(mut self, overrides: Seq2SeqOverrides) -> Self {
        self.overrides = self.overrides.merge(&overrides);
        self
    }

    // =========================================================================
    // Model Loading Configuration
    // =========================================================================

    /// Configure model loading options.
    ///
    /// # Example
    ///
    /// ```ignore
    /// .with_load_config(|c| c.dtype(DType::BF16))
    /// ```
    pub fn with_load_config<F>(mut self, f: F) -> Self
    where
        F: FnOnce(LoadConfigBuilder) -> LoadConfigBuilder,
    {
        self.load_config = Some(f(LoadConfigBuilder::new()).build());
        self
    }

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

    /// Never download models automatically (offline mode).
    pub fn offline(mut self) -> Self {
        self.download_policy = DownloadPolicy::Never;
        self
    }

    // =========================================================================
    // Behavior
    // =========================================================================

    /// Suppress informational output (download progress, warnings).
    pub fn quiet(mut self) -> Self {
        self.quiet = true;
        self
    }

    /// Allow warnings without printing them.
    pub fn allow_warnings(mut self) -> Self {
        self.allow_warnings = true;
        self
    }

    // =========================================================================
    // Build
    // =========================================================================

    /// Build the Seq2SeqGenerator instance.
    ///
    /// This will:
    /// 1. Validate the model is compatible with seq2seq
    /// 2. Download the model if needed (based on download_policy)
    /// 3. Load the model onto the specified device
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The model is unknown or incompatible
    /// - The model is not downloaded and download_policy is Never
    /// - Model loading fails
    /// - GPU is requested but unavailable
    pub async fn build(self) -> Seq2SeqResult<Seq2SeqGenerator> {
        Seq2SeqGenerator::from_builder(self).await
    }
}
