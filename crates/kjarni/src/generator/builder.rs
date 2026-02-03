// =============================================================================
// kjarni/src/generator/builder.rs
// =============================================================================

//! Builder pattern for Generator configuration.

use std::path::PathBuf;
use std::sync::Arc;

use kjarni_transformers::WgpuContext;

use crate::common::{DownloadPolicy, KjarniDevice, LoadConfig, LoadConfigBuilder};
use crate::generation::GenerationOverrides;
use crate::generator::presets::GeneratorPreset;

use super::model::Generator;
use super::types::GeneratorResult;

/// Builder for configuring a Generator.
///
/// # Example
///
/// ```ignore
/// let generator = Generator::builder("gpt2")
///     .cpu()
///     .temperature(0.8)
///     .max_tokens(100)
///     .build()
///     .await?;
/// ```
pub struct GeneratorBuilder {
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

    // Generation defaults
    pub(crate) generation_overrides: GenerationOverrides,

    // Behavior
    pub(crate) quiet: bool,
    pub(crate) allow_warnings: bool,
}

impl GeneratorBuilder {
    /// Create a new builder for the specified model.
    ///
    /// # Arguments
    ///
    /// * `model` - Model name from registry (e.g., "gpt2", "llama3.2-1b")
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            model_path: None,
            device: KjarniDevice::default(),
            context: None,
            cache_dir: None,
            download_policy: DownloadPolicy::default(),
            load_config: None,
            generation_overrides: GenerationOverrides::default(),
            quiet: false,
            allow_warnings: false,
        }
    }

    // =========================================================================
    // Model Selection
    // =========================================================================

    /// Load model from a local path instead of registry.
    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
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
    pub fn with_context(mut self, context: Arc<WgpuContext>) -> Self {
        self.context = Some(context);
        self.device = KjarniDevice::Gpu;
        self
    }

    // =========================================================================
    // Generation Parameters
    // =========================================================================

    /// Create a builder from a preset.
    pub fn from_preset(preset: &GeneratorPreset) -> Self {
        let mut builder = Self::new(preset.model);
        builder.device = preset.recommended_device;

        if let Some(temp) = preset.temperature {
            builder.generation_overrides.temperature = Some(temp);
        }
        builder.generation_overrides.max_new_tokens = Some(preset.default_max_tokens);

        builder
    }

    /// Configure for creative generation (high temperature).
    pub fn creative(mut self) -> Self {
        self.generation_overrides.temperature = Some(0.9);
        self.generation_overrides.top_p = Some(0.95);
        self
    }

    /// Configure for precise generation (low temperature).
    pub fn precise(mut self) -> Self {
        self.generation_overrides.temperature = Some(0.2);
        self.generation_overrides.top_p = Some(0.9);
        self
    }

    /// Set the sampling temperature.
    ///
    /// Higher values (e.g., 1.0) make output more random.
    /// Lower values (e.g., 0.1) make output more deterministic.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.generation_overrides.temperature = Some(temp);
        self
    }

    /// Set the maximum number of tokens to generate.
    pub fn max_tokens(mut self, max: usize) -> Self {
        self.generation_overrides.max_new_tokens = Some(max);
        self
    }

    /// Set top-p (nucleus) sampling threshold.
    pub fn top_p(mut self, p: f32) -> Self {
        self.generation_overrides.top_p = Some(p);
        self
    }

    /// Set top-k sampling limit.
    pub fn top_k(mut self, k: usize) -> Self {
        self.generation_overrides.top_k = Some(k);
        self
    }

    /// Set min-p sampling threshold.
    pub fn min_p(mut self, p: f32) -> Self {
        self.generation_overrides.min_p = Some(p);
        self
    }

    /// Set repetition penalty.
    ///
    /// Values > 1.0 discourage repetition. Default is typically 1.1.
    pub fn repetition_penalty(mut self, penalty: f32) -> Self {
        self.generation_overrides.repetition_penalty = Some(penalty);
        self
    }

    /// Use greedy decoding (temperature = 0, deterministic).
    pub fn greedy(mut self) -> Self {
        self.generation_overrides.temperature = Some(0.0);
        self.generation_overrides.do_sample = Some(false);
        self
    }

    /// Set all generation overrides at once.
    pub fn generation_config(mut self, overrides: GenerationOverrides) -> Self {
        self.generation_overrides = overrides;
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
    /// Generator::builder("llama3.2-3b")
    ///     .with_load_config(|cfg| cfg
    ///         .offload_embeddings(true)
    ///         .quantize_lm_head_q8()
    ///     )
    ///     .build()
    ///     .await?;
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

    /// Never download models automatically.
    pub fn offline(mut self) -> Self {
        self.download_policy = DownloadPolicy::Never;
        self
    }

    // =========================================================================
    // Behavior
    // =========================================================================

    /// Suppress informational output.
    pub fn quiet(mut self) -> Self {
        self.quiet = true;
        self
    }

    /// Allow suboptimal model choices without warnings.
    pub fn allow_warnings(mut self) -> Self {
        self.allow_warnings = true;
        self
    }

    // =========================================================================
    // Build
    // =========================================================================

    /// Build the Generator.
    pub async fn build(self) -> GeneratorResult<Generator> {
        Generator::from_builder(self).await
    }
}
