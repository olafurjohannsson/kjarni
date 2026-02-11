//! Builder pattern for Embedder configuration.

use std::path::PathBuf;
use std::sync::Arc;

use kjarni_transformers::{PoolingStrategy, WgpuContext};

use crate::common::{DownloadPolicy, KjarniDevice, LoadConfig, LoadConfigBuilder};

use super::model::Embedder;
use super::presets::EmbedderPreset;
use super::types::{EmbeddingOverrides, EmbedderResult};

/// Builder for configuring and constructing an Embedder.
pub struct EmbedderBuilder {
    // Model identification
    pub(crate) model: String,
    pub(crate) model_path: Option<PathBuf>,

    // Execution environment
    pub(crate) device: KjarniDevice,
    pub(crate) context: Option<Arc<WgpuContext>>,
    pub(crate) cache_dir: Option<PathBuf>,

    // Loading configuration
    pub(crate) load_config: Option<LoadConfig>,

    // Download policy
    pub(crate) download_policy: DownloadPolicy,

    // Embedding defaults
    pub(crate) overrides: EmbeddingOverrides,

    // Behavior
    pub(crate) quiet: bool,
}

impl EmbedderBuilder {
    /// Create a new builder for the specified model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            model_path: None,
            device: KjarniDevice::default(),
            context: None,
            cache_dir: None,
            load_config: None,
            download_policy: DownloadPolicy::default(),
            overrides: EmbeddingOverrides::default(),
            quiet: false,
        }
    }

    /// Create a builder from a preset.
    pub fn from_preset(preset: &EmbedderPreset) -> Self {
        Self {
            model: preset.model.to_string(),
            model_path: None,
            device: preset.recommended_device,
            context: None,
            cache_dir: None,
            load_config: None,
            download_policy: DownloadPolicy::default(),
            overrides: EmbeddingOverrides {
                pooling: Some(preset.default_pooling.clone()),
                normalize: Some(preset.normalize_default),
                max_length: None,
            },
            quiet: false,
        }
    }

    /// Run on CPU 
    pub fn cpu(mut self) -> Self {
        self.device = KjarniDevice::Cpu;
        self
    }

    /// Run on GPU.
    pub fn gpu(mut self) -> Self {
        self.device = KjarniDevice::Gpu;
        self
    }

    /// Auto-select device.
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

    /// Set custom cache directory.
    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Load model from a local path.
    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Set model name from registry.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

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

    /// Set default pooling strategy.
    pub fn pooling(mut self, strategy: PoolingStrategy) -> Self {
        self.overrides.pooling = Some(strategy);
        self
    }

    /// Set whether to normalize embeddings.
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.overrides.normalize = Some(normalize);
        self
    }

    /// Set maximum sequence length.
    pub fn max_length(mut self, max_length: usize) -> Self {
        self.overrides.max_length = Some(max_length);
        self
    }

    /// Apply full overrides.
    pub fn overrides(mut self, overrides: EmbeddingOverrides) -> Self {
        self.overrides = overrides;
        self
    }

    /// Suppress non-error output.
    pub fn quiet(mut self, quiet: bool) -> Self {
        self.quiet = quiet;
        self
    }

    /// Build the Embedder.
    pub async fn build(self) -> EmbedderResult<Embedder> {
        Embedder::from_builder(self).await
    }
}

impl Embedder {
    /// Create a builder for a model.
    pub fn builder(model: impl Into<String>) -> EmbedderBuilder {
        EmbedderBuilder::new(model)
    }

    /// Create a builder from a preset.
    pub fn from_preset(preset: &EmbedderPreset) -> EmbedderBuilder {
        EmbedderBuilder::from_preset(preset)
    }
}