//! Core types for the Chat module.
//!
//! Contains enums and small types used throughout the chat API.

use std::fmt;
use thiserror::Error;

/// Chat behavior mode.
///
/// Affects generation defaults without changing the underlying model.
/// This is a behavioral hint, not a capability requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatMode {
    /// Standard conversational mode.
    /// Balanced temperature, moderate output length.
    #[default]
    Default,

    /// Reasoning mode for complex tasks.
    /// Lower temperature, longer output, encourages step-by-step thinking.
    Reasoning,

    /// Creative mode for open-ended generation.
    /// Higher temperature, more varied outputs.
    Creative,
}

impl ChatMode {
    /// Get recommended temperature for this mode.
    pub fn default_temperature(&self) -> f32 {
        match self {
            Self::Default => 0.7,
            Self::Reasoning => 0.3,
            Self::Creative => 0.9,
        }
    }

    /// Get recommended max_new_tokens for this mode.
    pub fn default_max_tokens(&self) -> usize {
        match self {
            Self::Default => 512,
            Self::Reasoning => 2048,
            Self::Creative => 1024,
        }
    }
}

/// Execution device for the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatDevice {
    /// Run on CPU (default, always available).
    #[default]
    Cpu,

    /// Run on GPU via WebGPU.
    Gpu,

    /// Automatically select best available device.
    Auto,
}

impl ChatDevice {
    /// Resolve Auto to a concrete device.
    pub fn resolve(self) -> Self {
        match self {
            Self::Auto => {
                // TODO: Check for GPU availability
                // For now, default to CPU as it's always available
                Self::Cpu
            }
            other => other,
        }
    }
}

/// Errors that can occur when using the Chat API.
#[derive(Debug, Error)]
pub enum ChatError {
    /// Model cannot perform chat/generation tasks.
    #[error("Model '{model}' cannot be used for chat: {reason}")]
    IncompatibleModel { model: String, reason: String },

    /// Model not found in registry.
    #[error("Unknown model: '{0}'. Run 'kjarni model list' to see available models.")]
    UnknownModel(String),

    /// Model not downloaded and download policy is Never.
    #[error("Model '{0}' not downloaded and download policy is set to Never")]
    ModelNotDownloaded(String),

    /// Failed to download model.
    #[error("Failed to download model '{model}': {source}")]
    DownloadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// Failed to load model.
    #[error("Failed to load model '{model}': {source}")]
    LoadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// Generation failed.
    #[error("Generation failed: {0}")]
    GenerationFailed(#[from] anyhow::Error),

    /// GPU requested but not available.
    #[error("GPU requested but WebGPU context could not be created")]
    GpuUnavailable,

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for Chat operations.
pub type ChatResult<T> = Result<T, ChatError>;

/// Warning emitted when using suboptimal model configuration.
#[derive(Debug, Clone)]
pub struct ChatWarning {
    pub message: String,
    pub suggestion: Option<String>,
}

impl fmt::Display for ChatWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Warning: {}", self.message)?;
        if let Some(suggestion) = &self.suggestion {
            write!(f, " {}", suggestion)?;
        }
        Ok(())
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

/// A single message in a conversation.
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

/// Conversation history for stateless operations.
#[derive(Debug, Clone, Default)]
pub struct History {
    messages: Vec<Message>,
}

impl History {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_system(system_prompt: impl Into<String>) -> Self {
        Self {
            messages: vec![Message::system(system_prompt)],
        }
    }

    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn push_user(&mut self, content: impl Into<String>) {
        self.push(Message::user(content));
    }

    pub fn push_assistant(&mut self, content: impl Into<String>) {
        self.push(Message::assistant(content));
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Clear but keep system message if present.
    pub fn clear_keep_system(&mut self) {
        if let Some(first) = self.messages.first() {
            if first.role == Role::System {
                let system = self.messages.remove(0);
                self.messages.clear();
                self.messages.push(system);
                return;
            }
        }
        self.messages.clear();
    }
}
