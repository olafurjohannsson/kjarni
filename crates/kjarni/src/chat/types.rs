//! Chat types and error definitions.

use std::fmt;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ChatError {
    #[error("unknown model: '{0}'. run 'kjarni model list --task chat' to see available models.")]
    UnknownModel(String),

    #[error("model '{0}' not downloaded. run: kjarni model download {0}")]
    ModelNotDownloaded(String),

    #[error("failed to download model '{model}': {source}")]
    DownloadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    #[error("failed to load model '{model}': {source}")]
    LoadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    #[error("gpu unavailable. use .cpu() or check your graphics drivers.")]
    GpuUnavailable,

    #[error("generation failed: {0}")]
    GenerationFailed(#[from] anyhow::Error),

    #[error("model '{0}' does not have a chat template. use Generator for raw text generation.")]
    NoChatTemplate(String),

    #[error("model '{model}' is incompatible with chat: {reason}")]
    IncompatibleModel { model: String, reason: String },

    #[error("model '{0}' is not suitable for chat: {1}")]
    InvalidModel(String, String),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

pub type ChatResult<T> = Result<T, ChatError>;

#[derive(Debug, Clone)]
pub struct ChatWarning {
    pub message: String,
    pub severity: WarningSeverity,
}

impl ChatWarning {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            severity: WarningSeverity::Info,
        }
    }

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
            WarningSeverity::Info => "[info]",
            WarningSeverity::Warning => "[warn]",
            WarningSeverity::Important => "[important]",
        };
        write!(f, "{} {}", prefix, self.message)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningSeverity {
    Info,
    Warning,
    Important,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatDevice {
    Cpu,
    Gpu,
    #[default]
    Auto,
}

impl ChatDevice {
    pub fn resolve(&self) -> ChatDevice {
        match self {
            Self::Auto => Self::Cpu,
            other => *other,
        }
    }

    pub fn resolve_to_device(&self) -> kjarni_transformers::traits::Device {
        match self.resolve() {
            Self::Cpu | Self::Auto => kjarni_transformers::traits::Device::Cpu,
            Self::Gpu => kjarni_transformers::traits::Device::Wgpu,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatMode {
    #[default]
    Default,
    Creative,
    Reasoning,
}

impl ChatMode {
    pub fn default_temperature(&self) -> f32 {
        match self {
            Self::Default => 0.7,
            Self::Creative => 0.9,
            Self::Reasoning => 0.3,
        }
    }

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

#[derive(Debug, Clone, Default)]
pub struct History {
    messages: Vec<Message>,
}

impl History {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_system(system: impl Into<String>) -> Self {
        Self {
            messages: vec![Message::system(system)],
        }
    }

    pub fn push_user(&mut self, content: impl Into<String>) {
        self.messages.push(Message::user(content));
    }

    pub fn push_assistant(&mut self, content: impl Into<String>) {
        self.messages.push(Message::assistant(content));
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

    pub fn clear(&mut self, keep_system: bool) {
        if keep_system {
            self.messages.retain(|m| matches!(m.role, Role::System));
        } else {
            self.messages.clear();
        }
    }
}