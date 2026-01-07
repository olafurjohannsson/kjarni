//! Builder pattern for Classifier configuration.

use std::path::PathBuf;
use std::sync::Arc;

use kjarni_transformers::tensor::DType;
use kjarni_transformers::{WgpuContext};

use crate::common::{DownloadPolicy, KjarniDevice, LoadConfig, LoadConfigBuilder};

use super::model::Classifier;
use super::presets::ClassifierPreset;
use super::types::{
    ClassificationMode, ClassificationOverrides, ClassifierResult, LabelConfig,
};

/// Builder for configuring and constructing a Classifier.
///
/// # Example
///
/// ```ignore
/// // From registry
/// let classifier = Classifier::builder("sentiment-model")
///     .gpu()
///     .top_k(3)
///     .build()
///     .await?;
///
/// // From local path with custom labels
/// let classifier = Classifier::builder("icebert")
///     .model_path("/path/to/icebert")
///     .labels(vec!["neikvætt", "jákvætt"])
///     .build()
///     .await?;
/// ```
pub struct ClassifierBuilder {
    // Model identification
    pub(crate) model: String,
    pub(crate) model_path: Option<PathBuf>,

    // Execution environment
    pub(crate) device: KjarniDevice,
    pub(crate) context: Option<Arc<WgpuContext>>,
    pub(crate) cache_dir: Option<PathBuf>,

    // Loading configuration
    pub(crate) load_config: Option<LoadConfig>,
    pub(crate) dtype: Option<DType>,

    // Download policy
    pub(crate) download_policy: DownloadPolicy,

    // Label configuration
    pub(crate) label_config: Option<LabelConfig>,

    // Classification mode
    pub(crate) mode: ClassificationMode,

    // Classification defaults
    pub(crate) overrides: ClassificationOverrides,

    // Behavior
    pub(crate) quiet: bool,
}

impl ClassifierBuilder {
    /// Create a new builder for the specified model.
    ///
    /// The model can be a registry name (e.g., "minilm-l6-v2-cross-encoder")
    /// or a path hint when combined with `model_path()`.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            model_path: None,
            device: KjarniDevice::default(),
            context: None,
            cache_dir: None,
            load_config: None,
            dtype: None,
            download_policy: DownloadPolicy::default(),
            label_config: None,
            mode: ClassificationMode::default(),
            overrides: ClassificationOverrides::default(),
            quiet: false,
        }
    }

    /// Create a builder from a preset.
    pub fn from_preset(preset: &ClassifierPreset) -> Self {
        let mut builder = Self::new(preset.model);
        builder.device = preset.recommended_device;

        // Apply preset labels if available
        if let Some(labels) = preset.labels {
            builder.label_config = Some(LabelConfig::new(
                labels.iter().map(|s| s.to_string()).collect(),
            ));
        }

        builder
    }

    // =========================================================================
    // Device Configuration
    // =========================================================================

    /// Run on CPU (default).
    pub fn cpu(mut self) -> Self {
        self.device = KjarniDevice::Cpu;
        self
    }

    /// Run on GPU.
    pub fn gpu(mut self) -> Self {
        self.device = KjarniDevice::Gpu;
        self
    }

    /// Automatically select best device.
    pub fn auto_device(mut self) -> Self {
        self.device = KjarniDevice::Auto;
        self
    }

    /// Provide a pre-created WgpuContext.
    pub fn with_context(mut self, context: Arc<WgpuContext>) -> Self {
        self.context = Some(context);
        self.device = KjarniDevice::Gpu;
        self
    }

    // =========================================================================
    // Paths
    // =========================================================================

    /// Set custom cache directory for registry models.
    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Load model from a local path instead of registry.
    ///
    /// The path should contain:
    /// - `config.json` - Model configuration
    /// - `tokenizer.json` - Tokenizer
    /// - `model.safetensors` or `pytorch_model.bin` - Weights
    ///
    /// # Example
    ///
    /// ```ignore
    /// let classifier = Classifier::builder("icebert")
    ///     .model_path("/models/icebert-sentiment")
    ///     .labels(vec!["neikvætt", "jákvætt"])
    ///     .build()
    ///     .await?;
    /// ```
    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    // =========================================================================
    // Loading Configuration
    // =========================================================================

    /// Set model loading configuration.
    pub fn load_config(mut self, config: LoadConfig) -> Self {
        self.load_config = Some(config);
        self
    }

    /// Configure loading with a builder.
    pub fn with_load_config<F>(mut self, f: F) -> Self
    where
        F: FnOnce(LoadConfigBuilder) -> LoadConfigBuilder,
    {
        self.load_config = Some(f(LoadConfigBuilder::new()).build());
        self
    }

    /// Set target data type for model weights.
    ///
    /// Use for quantization or precision control.
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Use float16 precision.
    pub fn f16(mut self) -> Self {
        self.dtype = Some(DType::F16);
        self
    }

    /// Use bfloat16 precision.
    pub fn bf16(mut self) -> Self {
        self.dtype = Some(DType::BF16);
        self
    }

    // =========================================================================
    // Download Policy
    // =========================================================================

    /// Set download policy.
    pub fn download_policy(mut self, policy: DownloadPolicy) -> Self {
        self.download_policy = policy;
        self
    }

    /// Never download models.
    pub fn offline(mut self) -> Self {
        self.download_policy = DownloadPolicy::Never;
        self
    }

    // =========================================================================
    // Label Configuration
    // =========================================================================

    /// Set custom labels for the model.
    ///
    /// Labels must be in the same order as the model's output indices.
    /// Index 0 in output corresponds to `labels[0]`, etc.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Binary sentiment
    /// .labels(vec!["negative", "positive"])
    ///
    /// // Icelandic sentiment
    /// .labels(vec!["neikvætt", "jákvætt"])
    ///
    /// // Multi-class
    /// .labels(vec!["anger", "fear", "joy", "sadness", "surprise"])
    /// ```
    pub fn labels<S: Into<String>>(mut self, labels: Vec<S>) -> Self {
        self.label_config = Some(LabelConfig::new(
            labels.into_iter().map(|s| s.into()).collect(),
        ));
        self
    }

    /// Set labels from a comma-separated string.
    ///
    /// # Example
    ///
    /// ```ignore
    /// .labels_str("negative,neutral,positive")
    /// ```
    pub fn labels_str(mut self, labels: &str) -> Self {
        self.label_config = Some(LabelConfig::from_str(labels));
        self
    }

    /// Set a full label configuration.
    pub fn label_config(mut self, config: LabelConfig) -> Self {
        self.label_config = Some(config);
        self
    }

    /// Use standard binary sentiment labels.
    pub fn sentiment_labels(mut self) -> Self {
        self.label_config = Some(LabelConfig::sentiment());
        self
    }

    /// Use standard 3-class sentiment labels.
    pub fn sentiment_3class_labels(mut self) -> Self {
        self.label_config = Some(LabelConfig::sentiment_3class());
        self
    }

    // =========================================================================
    // Classification Mode
    // =========================================================================

    /// Set classification mode.
    pub fn mode(mut self, mode: ClassificationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Use single-label classification (softmax, default).
    pub fn single_label(mut self) -> Self {
        self.mode = ClassificationMode::SingleLabel;
        self
    }

    /// Use multi-label classification (sigmoid).
    ///
    /// Each label is predicted independently.
    pub fn multi_label(mut self) -> Self {
        self.mode = ClassificationMode::MultiLabel;
        self
    }

    // =========================================================================
    // Classification Defaults
    // =========================================================================

    /// Set default top_k for results.
    pub fn top_k(mut self, k: usize) -> Self {
        self.overrides.top_k = Some(k);
        self
    }

    /// Set default confidence threshold.
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.overrides.threshold = Some(threshold);
        self
    }

    /// Return raw logits instead of probabilities.
    pub fn return_logits(mut self, return_logits: bool) -> Self {
        self.overrides.return_logits = return_logits;
        self
    }

    /// Set maximum sequence length (truncation).
    pub fn max_length(mut self, max_length: usize) -> Self {
        self.overrides.max_length = Some(max_length);
        self
    }

    /// Set batch size for inference.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.overrides.batch_size = Some(batch_size);
        self
    }

    /// Apply full overrides.
    pub fn overrides(mut self, overrides: ClassificationOverrides) -> Self {
        self.overrides = overrides;
        self
    }

    // =========================================================================
    // Behavior
    // =========================================================================

    /// Suppress non-error output.
    pub fn quiet(mut self, quiet: bool) -> Self {
        self.quiet = quiet;
        self
    }

    // =========================================================================
    // Build
    // =========================================================================

    /// Build the Classifier.
    pub async fn build(self) -> ClassifierResult<Classifier> {
        Classifier::from_builder(self).await
    }
}

impl Classifier {
    /// Create a builder for a model.
    pub fn builder(model: impl Into<String>) -> ClassifierBuilder {
        ClassifierBuilder::new(model)
    }

    /// Create a builder from a preset.
    pub fn from_preset(preset: &ClassifierPreset) -> ClassifierBuilder {
        ClassifierBuilder::from_preset(preset)
    }
}