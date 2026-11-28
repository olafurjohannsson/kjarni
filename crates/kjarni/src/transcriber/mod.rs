//! Speech-to-text transcription powered by Whisper.
//!
//! # Quick Start
//!
//! ```ignore
//! use kjarni::transcriber::Transcriber;
//!
//! let transcriber = Transcriber::builder("whisper-small")
//!     .language("en")
//!     .build()
//!     .await?;
//!
//! let result = transcriber.transcribe_file("audio.wav")?;
//! println!("{}", result.text);
//! ```

mod builder;
mod model;
mod presets;
mod subtitle;
mod types;
mod validation;

#[cfg(test)]
mod tests;

// Re-exports
pub use builder::TranscriberBuilder;
pub use model::Transcriber;
pub use presets::TranscriberPreset;
pub use subtitle::{DEFAULT_MAX_LINE_CHARS, DEFAULT_MAX_LINES, segments_to_srt, segments_to_vtt};
pub use types::{
    Task, TranscribedToken, TranscriberError, TranscriberResult, TranscriptionProgress,
    TranscriptionProgressCallback, TranscriptionResult, TranscriptionSegment, TranscriptionStage,
};
