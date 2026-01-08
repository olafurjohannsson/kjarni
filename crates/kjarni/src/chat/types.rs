// =============================================================================
// kjarni/src/chat/types.rs
// =============================================================================

//! Chat types and error definitions.

use std::fmt;
use thiserror::Error;

/// Errors that can occur during chat operations.
#[derive(Debug, Error)]
pub enum ChatError {
    /// Model name not found in registry.
    #[error("Unknown model: '{0}'. Run 'kjarni model list --task chat' to see available models.")]
    UnknownModel(String),

    /// Model files not present locally.
    #[error("Model '{0}' not downloaded. Run: kjarni model download {0}")]
    ModelNotDownloaded(String),

    /// Download failed.
    #[error("Failed to download model '{model}': {source}")]
    DownloadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// Model loading failed.
    #[error("Failed to load model '{model}': {source}")]
    LoadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// GPU requested but unavailable.
    #[error("GPU unavailable. Use .cpu() or check your graphics drivers.")]
    GpuUnavailable,

    /// Generation failed.
    #[error("Generation failed: {0}")]
    GenerationFailed(#[from] anyhow::Error),

    /// Model doesn't have a chat template.
    #[error("Model '{0}' does not have a chat template. Use Generator for raw text generation.")]
    NoChatTemplate(String),

    /// Model is incompatible with chat.
    #[error("Model '{model}' is incompatible with chat: {reason}")]
    IncompatibleModel {
        model: String,
        reason: String,
    },

    /// Invalid model for chat (alias for IncompatibleModel for backwards compat).
    #[error("Model '{0}' is not suitable for chat: {1}")]
    InvalidModel(String, String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for chat operations.
pub type ChatResult<T> = Result<T, ChatError>;

/// Warning about chat model selection or configuration.
#[derive(Debug, Clone)]
pub struct ChatWarning {
    /// Warning message.
    pub message: String,
    /// Severity level.
    pub severity: WarningSeverity,
}

impl ChatWarning {
    /// Create a new warning.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            severity: WarningSeverity::Info,
        }
    }

    /// Create a warning with severity.
    pub fn with_severity(message: impl Into<String>, severity: WarningSeverity) -> Self {
        Self {
            message: message.into(),
            severity,
        }
    }
}

impl fmt::Display for ChatWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prefix = match self.severity {
            WarningSeverity::Info => "ℹ️ ",
            WarningSeverity::Warning => "⚠️ ",
            WarningSeverity::Important => "❗",
        };
        write!(f, "{} {}", prefix, self.message)
    }
}

/// Severity level for warnings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningSeverity {
    /// Informational note.
    Info,
    /// Warning that may affect behavior.
    Warning,
    /// Important warning that should be addressed.
    Important,
}

/// Device selection for chat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatDevice {
    /// Use CPU.
    Cpu,
    /// Use GPU (WGPU).
    Gpu,
    /// Automatically select best available device.
    #[default]
    Auto,
}

impl ChatDevice {
    /// Resolve to concrete device, checking GPU availability.
    pub fn resolve(&self) -> ChatDevice {
        match self {
            Self::Auto => {
                // For now, default to CPU. GPU detection could be added here.
                // if gpu_available() { Self::Gpu } else { Self::Cpu }
                Self::Cpu
            }
            other => *other,
        }
    }

    /// Resolve to the internal Device type.
    pub fn resolve_to_device(&self) -> kjarni_transformers::traits::Device {
        match self.resolve() {
            Self::Cpu | Self::Auto => kjarni_transformers::traits::Device::Cpu,
            Self::Gpu => kjarni_transformers::traits::Device::Wgpu,
        }
    }
}

/// Chat mode affecting generation behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatMode {
    /// Balanced mode for general conversation.
    #[default]
    Default,
    /// Creative mode with higher temperature.
    Creative,
    /// Reasoning mode with lower temperature for logical tasks.
    Reasoning,
}

impl ChatMode {
    /// Get the default temperature for this mode.
    pub fn default_temperature(&self) -> f32 {
        match self {
            Self::Default => 0.7,
            Self::Creative => 0.9,
            Self::Reasoning => 0.3,
        }
    }

    /// Get the default max tokens for this mode.
    pub fn default_max_tokens(&self) -> usize {
        match self {
            Self::Default => 512,
            Self::Creative => 1024,
            Self::Reasoning => 2048,
        }
    }
}

impl fmt::Display for ChatMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => write!(f, "default"),
            Self::Creative => write!(f, "creative"),
            Self::Reasoning => write!(f, "reasoning"),
        }
    }
}

/// Role in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
        }
    }
}

/// A message in a conversation.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// Conversation history.
#[derive(Debug, Clone, Default)]
pub struct History {
    messages: Vec<Message>,
}

impl History {
    /// Create empty history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create history with a system prompt.
    pub fn with_system(system: impl Into<String>) -> Self {
        Self {
            messages: vec![Message::system(system)],
        }
    }

    /// Add a user message.
    pub fn push_user(&mut self, content: impl Into<String>) {
        self.messages.push(Message::user(content));
    }

    /// Add an assistant message.
    pub fn push_assistant(&mut self, content: impl Into<String>) {
        self.messages.push(Message::assistant(content));
    }

    /// Get all messages.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Get the number of messages.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Clear history (optionally preserving system prompt).
    pub fn clear(&mut self, keep_system: bool) {
        if keep_system {
            self.messages.retain(|m| matches!(m.role, Role::System));
        } else {
            self.messages.clear();
        }
    }
}