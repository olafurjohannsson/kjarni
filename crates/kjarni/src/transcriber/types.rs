//! Types for the transcription API.

use std::fmt;
use std::path::PathBuf;


// Task


/// Whisper task type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Transcribe speech in the source language.
    Transcribe,
    /// Translate speech to English.
    Translate,
}

impl Default for Task {
    fn default() -> Self {
        Self::Transcribe
    }
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Task::Transcribe => write!(f, "transcribe"),
            Task::Translate => write!(f, "translate"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    /// Full stitched transcript text.
    pub text: String,
    /// Timed segments (populated when timestamps are enabled, otherwise one
    /// segment per chunk with estimated boundaries).
    pub segments: Vec<TranscriptionSegment>,
    /// Detected or forced language code (e.g. `"en"`).
    pub language: String,
    /// Total audio duration in seconds.
    pub duration_secs: f32,
}

/// A timed segment of the transcription.
#[derive(Debug, Clone)]
pub struct TranscriptionSegment {
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
    /// Transcribed text.
    pub text: String,
}


// Streaming Token


/// A single token emitted during streamed transcription.
#[derive(Debug, Clone)]
pub struct TranscribedToken {
    /// Decoded text for this token.
    pub text: String,
    /// Token ID in the Whisper vocabulary.
    pub id: u32,
    /// `true` for timestamp tokens and other special tokens.
    pub is_special: bool,
}


// Progress Reporting


/// Stage of the transcription pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptionStage {
    /// Loading and resampling audio from disk.
    LoadingAudio,
    /// Computing mel spectrogram + running the encoder for a chunk.
    Encoding,
    /// Running greedy decode for a chunk.
    Decoding,
    /// Stitching segments from all chunks.
    Stitching,
}

impl fmt::Display for TranscriptionStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TranscriptionStage::LoadingAudio => write!(f, "Loading audio"),
            TranscriptionStage::Encoding => write!(f, "Encoding"),
            TranscriptionStage::Decoding => write!(f, "Decoding"),
            TranscriptionStage::Stitching => write!(f, "Stitching"),
        }
    }
}

/// Progress update emitted during transcription.
#[derive(Debug, Clone)]
pub struct TranscriptionProgress {
    /// Current pipeline stage.
    pub stage: TranscriptionStage,
    /// Current item (e.g. chunk index, 0-based).
    pub current: usize,
    /// Total items, if known.
    pub total: Option<usize>,
}

impl TranscriptionProgress {
    pub fn new(stage: TranscriptionStage, current: usize, total: Option<usize>) -> Self {
        Self { stage, current, total }
    }

    pub fn loading_audio() -> Self {
        Self::new(TranscriptionStage::LoadingAudio, 0, None)
    }

    pub fn encoding(current: usize, total: usize) -> Self {
        Self::new(TranscriptionStage::Encoding, current, Some(total))
    }

    pub fn decoding(current: usize, total: usize) -> Self {
        Self::new(TranscriptionStage::Decoding, current, Some(total))
    }

    pub fn stitching() -> Self {
        Self::new(TranscriptionStage::Stitching, 0, None)
    }
}

/// Callback type for progress reporting.
pub type TranscriptionProgressCallback =
    Box<dyn Fn(&TranscriptionProgress, Option<&str>) + Send + Sync>;


// Errors


/// Errors returned by the transcription API.
#[derive(Debug)]
pub enum TranscriberError {
    /// Failed to load the Whisper model.
    ModelLoadFailed(anyhow::Error),
    /// Audio file path does not exist or is not a file.
    InvalidAudioPath(PathBuf),
    /// Audio format is not supported.
    UnsupportedFormat(String),
    /// Failed to load or resample audio.
    AudioLoadFailed(anyhow::Error),
    /// Error during mel / encode / decode.
    TranscriptionFailed(anyhow::Error),
    /// Invalid configuration value.
    InvalidConfig(String),
    /// GPU was requested but is unavailable.
    GpuUnavailable,
    /// Operation cancelled via callback.
    Cancelled,
}

impl fmt::Display for TranscriberError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelLoadFailed(e) => write!(f, "Model load failed: {}", e),
            Self::InvalidAudioPath(p) => {
                write!(f, "Invalid audio path: {}", p.display())
            }
            Self::UnsupportedFormat(ext) => {
                write!(f, "Unsupported audio format: {}", ext)
            }
            Self::AudioLoadFailed(e) => write!(f, "Audio load failed: {}", e),
            Self::TranscriptionFailed(e) => write!(f, "Transcription failed: {}", e),
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            Self::GpuUnavailable => write!(f, "GPU is not available"),
            Self::Cancelled => write!(f, "Transcription cancelled"),
        }
    }
}

impl std::error::Error for TranscriberError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ModelLoadFailed(e)
            | Self::AudioLoadFailed(e)
            | Self::TranscriptionFailed(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

/// Convenience alias.
pub type TranscriberResult<T> = Result<T, TranscriberError>;