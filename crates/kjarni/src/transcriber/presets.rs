//! Pre-built configurations for common transcription tasks.

use kjarni_transformers::ModelType;

use super::types::Task;

/// Pre-defined transcriber configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriberPreset {
    /// Default: Whisper Small, transcribe, no timestamps, auto-detect language.
    Default,
    /// English-only: Whisper Small, language forced to English.
    English,
    /// Translation: Whisper Small, translate any language to English.
    Translate,
    /// High quality: Whisper Large V3, transcribe, auto-detect language.
    LargeV3,
}

impl TranscriberPreset {
    /// Model type for this preset.
    pub fn model_type(&self) -> ModelType {
        match self {
            Self::Default | Self::English | Self::Translate => ModelType::WhisperSmall,
            Self::LargeV3 => ModelType::WhisperLargeV3,
        }
    }

    /// Fixed language, or `None` for auto-detect.
    pub fn language(&self) -> Option<&str> {
        match self {
            Self::English => Some("en"),
            _ => None,
        }
    }

    /// Task for this preset.
    pub fn task(&self) -> Task {
        match self {
            Self::Translate => Task::Translate,
            _ => Task::Transcribe,
        }
    }

    /// Whether timestamps are enabled by default.
    pub fn timestamps(&self) -> bool {
        false
    }

    /// Model identifier string (for builder resolution).
    pub fn model_id(&self) -> &str {
        match self {
            Self::Default | Self::English | Self::Translate => "whisper-small",
            Self::LargeV3 => "whisper-large-v3",
        }
    }
}

impl Default for TranscriberPreset {
    fn default() -> Self {
        Self::Default
    }
}