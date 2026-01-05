//! Builder pattern for Chat configuration.
//!
//! The builder accumulates configuration and performs validation
//! when `build()` is called.

use std::path::PathBuf;
use std::sync::Arc;

use kjarni_transformers::WgpuContext;

use crate::generation::overrides::GenerationOverrides;

use super::model::Chat;
use super::presets::ModelPreset;
use super::types::{ChatDevice, ChatMode, ChatResult, DownloadPolicy};

/// Builder for configuring and constructing a Chat instance.
///
/// # Example
///
/// ```ignore
/// use kjarni::chat::{Chat, ChatMode};
///
/// // Simple usage
/// let chat = Chat::builder("llama3.2-1b")
///     .build()
///     .await?;
///
/// // Full configuration
/// let chat = Chat::builder("llama3.2-3b")
///     .gpu()
///     .mode(ChatMode::Reasoning)
///     .system_prompt("You are a helpful assistant.")
///     .temperature(0.5)
///     .max_tokens(1024)
///     .build()
///     .await?;
/// ```
pub struct ChatBuilder {
    // Model identification
    pub(crate) model: String,
    pub(crate) model_path: Option<PathBuf>,

    // Execution environment
    pub(crate) device: ChatDevice,
    pub(crate) context: Option<Arc<WgpuContext>>,
    pub(crate) cache_dir: Option<PathBuf>,

    // Behavior
    pub(crate) mode: ChatMode,
    pub(crate) system_prompt: Option<String>,
    pub(crate) download_policy: DownloadPolicy,

    // Generation defaults (applied to all generations unless overridden)
    pub(crate) generation_overrides: GenerationOverrides,

    // Validation behavior
    pub(crate) allow_suboptimal: bool,
    pub(crate) quiet: bool,
}

impl ChatBuilder {
    /// Create a new builder for the specified model.
    ///
    /// # Arguments
    ///
    /// * `model` - Model CLI name (e.g., "llama3.2-1b") or preset name
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            model_path: None,
            device: ChatDevice::default(),
            context: None,
            cache_dir: None,
            mode: ChatMode::default(),
            system_prompt: None,
            download_policy: DownloadPolicy::default(),
            generation_overrides: GenerationOverrides::default(),
            allow_suboptimal: false,
            quiet: false,
        }
    }

    /// Create a builder from a preset.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kjarni::chat::{ChatBuilder, presets::CHAT_SMALL_V1};
    ///
    /// let chat = ChatBuilder::from_preset(&CHAT_SMALL_V1)
    ///     .system_prompt("You are helpful.")
    ///     .build()
    ///     .await?;
    /// ```
    pub fn from_preset(preset: &ModelPreset) -> Self {
        Self {
            model: preset.model.to_string(),
            model_path: None,
            device: preset.recommended_device,
            context: None,
            cache_dir: None,
            mode: ChatMode::default(),
            system_prompt: None,
            download_policy: DownloadPolicy::default(),
            generation_overrides: GenerationOverrides::default(),
            allow_suboptimal: false,
            quiet: false,
        }
    }

    // =========================================================================
    // Device Configuration
    // =========================================================================

    /// Run on CPU (default).
    pub fn cpu(mut self) -> Self {
        self.device = ChatDevice::Cpu;
        self
    }

    /// Run on GPU via WebGPU.
    pub fn gpu(mut self) -> Self {
        self.device = ChatDevice::Gpu;
        self
    }

    /// Automatically select best available device.
    pub fn auto_device(mut self) -> Self {
        self.device = ChatDevice::Auto;
        self
    }

    /// Provide a pre-created WgpuContext.
    ///
    /// Implies GPU execution. Useful when sharing context across models.
    pub fn with_context(mut self, context: Arc<WgpuContext>) -> Self {
        self.context = Some(context);
        self.device = ChatDevice::Gpu;
        self
    }

    // =========================================================================
    // Paths
    // =========================================================================

    /// Set custom cache directory for model files.
    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Load model from a local path instead of registry.
    ///
    /// Bypasses download and registry lookup.
    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    // =========================================================================
    // Behavior
    // =========================================================================

    /// Set chat mode (Default, Reasoning, Creative).
    pub fn mode(mut self, mode: ChatMode) -> Self {
        self.mode = mode;
        self
    }

    /// Shorthand for reasoning mode.
    pub fn reasoning(mut self) -> Self {
        self.mode = ChatMode::Reasoning;
        self
    }

    /// Shorthand for creative mode.
    pub fn creative(mut self) -> Self {
        self.mode = ChatMode::Creative;
        self
    }

    /// Set the default system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set download policy.
    pub fn download_policy(mut self, policy: DownloadPolicy) -> Self {
        self.download_policy = policy;
        self
    }

    /// Never download models, fail if not present.
    pub fn offline(mut self) -> Self {
        self.download_policy = DownloadPolicy::Never;
        self
    }

    // =========================================================================
    // Generation Defaults
    // =========================================================================

    /// Set default temperature for generations.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.generation_overrides.temperature = Some(temp);
        self
    }

    /// Set default max tokens for generations.
    pub fn max_tokens(mut self, tokens: usize) -> Self {
        self.generation_overrides.max_new_tokens = Some(tokens);
        self
    }

    /// Set default top_p for generations.
    pub fn top_p(mut self, p: f32) -> Self {
        self.generation_overrides.top_p = Some(p);
        self
    }

    /// Set default top_k for generations.
    pub fn top_k(mut self, k: usize) -> Self {
        self.generation_overrides.top_k = Some(k);
        self
    }

    /// Set default repetition penalty.
    pub fn repetition_penalty(mut self, penalty: f32) -> Self {
        self.generation_overrides.repetition_penalty = Some(penalty);
        self
    }

    /// Apply a full set of generation overrides.
    pub fn generation_config(mut self, overrides: GenerationOverrides) -> Self {
        self.generation_overrides = overrides;
        self
    }

    // =========================================================================
    // Validation Behavior
    // =========================================================================

    /// Suppress warnings about suboptimal model choices.
    pub fn allow_suboptimal_models(mut self) -> Self {
        self.allow_suboptimal = true;
        self
    }

    /// Suppress all non-error output.
    pub fn quiet(mut self, quiet: bool) -> Self {
        self.quiet = quiet;
        self
    }

    // =========================================================================
    // Build
    // =========================================================================

    /// Build the Chat instance.
    ///
    /// This will:
    /// 1. Resolve the model from registry or preset
    /// 2. Validate the model is suitable for chat
    /// 3. Download the model if needed (respecting download policy)
    /// 4. Load the model onto the specified device
    ///
    /// # Errors
    ///
    /// - `ChatError::UnknownModel` - Model not found in registry
    /// - `ChatError::IncompatibleModel` - Model cannot perform chat
    /// - `ChatError::ModelNotDownloaded` - Model not present and policy is Never
    /// - `ChatError::DownloadFailed` - Failed to download model
    /// - `ChatError::LoadFailed` - Failed to load model
    /// - `ChatError::GpuUnavailable` - GPU requested but not available
    pub async fn build(self) -> ChatResult<Chat> {
        Chat::from_builder(self).await
    }
}

// Convenience methods on Chat for creating builders
impl Chat {
    /// Create a builder for a model.
    pub fn builder(model: impl Into<String>) -> ChatBuilder {
        ChatBuilder::new(model)
    }

    /// Create a builder from a preset.
    pub fn from_preset(preset: &ModelPreset) -> ChatBuilder {
        ChatBuilder::from_preset(preset)
    }
}
