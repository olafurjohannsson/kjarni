//! [`Transcriber`] — high-level speech-to-text API backed by Whisper.
//!
//! # Usage
//!
//! ```ignore
//! let transcriber = Transcriber::builder("whisper-small")
//!     .language("en")
//!     .build()
//!     .await?;
//!
//! // Full result
//! let result = transcriber.transcribe_file("meeting.wav")?;
//! println!("{}", result.text);
//!
//! // Streaming
//! let stream = transcriber.stream_file("meeting.wav").await?;
//! pin_mut!(stream);
//! while let Some(tok) = stream.next().await {
//!     print!("{}", tok?.text);
//! }
//! ```

use std::path::Path;
use std::sync::Arc;

use futures::Stream;

use kjarni_transformers::{Device, ModelType};
use kjarni_transformers::audio::{
    compute_mel_spectrogram, load_audio, AudioLoaderConfig,
};
use kjarni_models::models::whisper::WhisperModel;
use kjarni_models::models::whisper::{
    WhisperChunkResult, WhisperTask, WhisperTranscriberConfig,
    WHISPER_CHUNK_LENGTH_SECS, WHISPER_SAMPLE_RATE,
};

use super::builder::TranscriberBuilder;
use super::types::*;
use super::validation;

// =============================================================================
// Transcriber
// =============================================================================

/// High-level Whisper transcriber.
///
/// Wraps a [`WhisperModel`] and provides file-level and streaming transcription.
pub struct Transcriber {
    model: Arc<WhisperModel>,
    model_type: ModelType,
    language: Option<String>,
    task: Task,
    timestamps: bool,
    max_tokens_per_chunk: usize,
    device: Device,
    quiet: bool,
    progress_callback: Option<TranscriptionProgressCallback>,
}

impl Transcriber {
    // =========================================================================
    // Construction
    // =========================================================================

    /// Create a builder targeting the given model (e.g. `"whisper-small"`).
    pub fn builder(model: &str) -> TranscriberBuilder {
        TranscriberBuilder::new(model)
    }

    /// Internal constructor — called by [`TranscriberBuilder::build`].
    pub(crate) fn new(
        model: Arc<WhisperModel>,
        model_type: ModelType,
        language: Option<String>,
        task: Task,
        timestamps: bool,
        max_tokens_per_chunk: usize,
        device: Device,
        quiet: bool,
        progress_callback: Option<TranscriptionProgressCallback>,
    ) -> Self {
        Self {
            model,
            model_type,
            language,
            task,
            timestamps,
            max_tokens_per_chunk,
            device,
            quiet,
            progress_callback,
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Human-readable model name.
    pub fn model_name(&self) -> &str {
        self.model_type.cli_name()
    }

    /// Device the model is running on.
    pub fn device(&self) -> Device {
        self.device
    }

    // =========================================================================
    // Transcription — Full Result
    // =========================================================================

    /// Transcribe an audio file on disk.
    ///
    /// Handles loading, resampling, chunking, encoding, decoding, and stitching.
    pub fn transcribe_file(
        &self,
        path: impl AsRef<Path>,
    ) -> TranscriberResult<TranscriptionResult> {
        let path = path.as_ref();
        validation::validate_audio_path(path)?;

        self.report_progress(TranscriptionStage::LoadingAudio, 0, 0, Some(&path.display().to_string()));

        let audio = load_audio(path, &self.audio_loader_config())
            .map_err(TranscriberError::AudioLoadFailed)?;

        let duration_secs = audio.samples.len() as f32 / audio.sample_rate as f32;
        self.transcribe_audio_inner(&audio.samples, duration_secs)
    }

    /// Transcribe raw audio samples.
    ///
    /// Expects **mono 16 kHz** `f32` samples.  If your audio has a different
    /// sample rate, use [`transcribe_file`](Self::transcribe_file) which
    /// resamples automatically, or resample before calling this method.
    pub fn transcribe_audio(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> TranscriberResult<TranscriptionResult> {
        // Light resampling guard — linear interpolation for mismatched rates
        let (samples_cow, effective_rate) = if sample_rate != WHISPER_SAMPLE_RATE {
            let resampled = resample_linear(samples, sample_rate, WHISPER_SAMPLE_RATE);
            (resampled, WHISPER_SAMPLE_RATE)
        } else {
            (samples.to_vec(), sample_rate)
        };

        let duration_secs = samples_cow.len() as f32 / effective_rate as f32;
        self.transcribe_audio_inner(&samples_cow, duration_secs)
    }

    /// Core transcription loop shared by file and raw-audio entry points.
    fn transcribe_audio_inner(
        &self,
        samples: &[f32],
        duration_secs: f32,
    ) -> TranscriberResult<TranscriptionResult> {
        let config = self.build_config();
        let mel_config = self.model.expected_mel_config();

        // --- Chunk ---
        let chunks = WhisperModel::chunk_audio(samples, WHISPER_SAMPLE_RATE);
        let total_chunks = chunks.len();
        let mut chunk_results: Vec<WhisperChunkResult> = Vec::with_capacity(total_chunks);

        for (i, chunk) in chunks.iter().enumerate() {
            let offset = i as f32 * WHISPER_CHUNK_LENGTH_SECS;

            // --- Mel spectrogram ---
            self.report_progress(
                TranscriptionStage::Encoding,
                i,
                total_chunks,
                Some(&format!("Chunk {}/{}", i + 1, total_chunks)),
            );

            let mel = compute_mel_spectrogram(chunk, &mel_config)
                .map_err(TranscriberError::TranscriptionFailed)?;

            // --- Encoder ---
            let encoder_out = self
                .model
                .encode_mel(&mel)
                .map_err(TranscriberError::TranscriptionFailed)?;

            // --- Decoder ---
            self.report_progress(
                TranscriptionStage::Decoding,
                i,
                total_chunks,
                Some(&format!("Chunk {}/{}", i + 1, total_chunks)),
            );

            let result = self
                .model
                .decode_chunk(&encoder_out, &config, offset, None)
                .map_err(TranscriberError::TranscriptionFailed)?;

            chunk_results.push(result);
        }

        // --- Stitch ---
        self.report_progress(TranscriptionStage::Stitching, 0, 0, None);

        let (text, segments) = WhisperModel::stitch_segments(chunk_results);

        let language = self.language.clone().unwrap_or_else(|| "en".to_string());

        Ok(TranscriptionResult {
            text,
            segments: segments
                .into_iter()
                .map(|s| TranscriptionSegment {
                    start: s.start,
                    end: s.end,
                    text: s.text,
                })
                .collect(),
            language,
            duration_secs,
        })
    }

    // =========================================================================
    // Transcription — Streaming
    // =========================================================================

    /// Stream decoded tokens from an audio file.
    ///
    /// Tokens are emitted as they are decoded.  The stream ends when the last
    /// chunk is fully decoded.
    pub async fn stream_file(
        &self,
        path: impl AsRef<Path>,
    ) -> TranscriberResult<std::pin::Pin<Box<dyn Stream<Item = TranscriberResult<TranscribedToken>> + Send>>>
    {
        let path = path.as_ref();
        validation::validate_audio_path(path)?;

        let audio = load_audio(path, &self.audio_loader_config())
            .map_err(TranscriberError::AudioLoadFailed)?;

        self.stream_audio(audio.samples, audio.sample_rate).await
    }

    /// Stream decoded tokens from raw audio samples.
    ///
    /// The `samples` are moved into a background task.
    pub async fn stream_audio(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> TranscriberResult<std::pin::Pin<Box<dyn Stream<Item = TranscriberResult<TranscribedToken>> + Send>>>
    {
        let model = self.model.clone();
        let config = self.build_config();
        let mel_config = model.expected_mel_config();

        // Resample if needed
        let samples = if sample_rate != WHISPER_SAMPLE_RATE {
            resample_linear(&samples, sample_rate, WHISPER_SAMPLE_RATE)
        } else {
            samples
        };

        let (tx, rx) = tokio::sync::mpsc::channel::<TranscriberResult<TranscribedToken>>(64);

        tokio::task::spawn_blocking(move || {
            let chunks = WhisperModel::chunk_audio(&samples, WHISPER_SAMPLE_RATE);

            for (i, chunk) in chunks.iter().enumerate() {
                let offset = i as f32 * WHISPER_CHUNK_LENGTH_SECS;

                // Mel spectrogram
                let mel = match compute_mel_spectrogram(chunk, &mel_config) {
                    Ok(m) => m,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(TranscriberError::TranscriptionFailed(e)));
                        return;
                    }
                };

                // Encode
                let encoder_out = match model.encode_mel(&mel) {
                    Ok(e) => e,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(TranscriberError::TranscriptionFailed(e)));
                        return;
                    }
                };

                // Decode with per-token callback → channel
                let tx_ref = &tx;
                let mut on_token = |id: u32, text: &str| -> bool {
                    let token = TranscribedToken {
                        text: text.to_string(),
                        id,
                        is_special: id >= 50257, // FIRST_SPECIAL_TOKEN
                    };
                    tx_ref.blocking_send(Ok(token)).is_ok()
                };

                if let Err(e) =
                    model.decode_chunk(&encoder_out, &config, offset, Some(&mut on_token))
                {
                    let _ = tx.blocking_send(Err(TranscriberError::TranscriptionFailed(e)));
                    return;
                }
            }
            // tx drops → stream ends
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    /// Build the low-level decode config from high-level settings.
    fn build_config(&self) -> WhisperTranscriberConfig {
        WhisperTranscriberConfig {
            language: self.language.clone(),
            task: match self.task {
                Task::Transcribe => WhisperTask::Transcribe,
                Task::Translate => WhisperTask::Translate,
            },
            timestamps: self.timestamps,
            max_tokens_per_chunk: self.max_tokens_per_chunk,
        }
    }

    /// Audio loader config: 16 kHz mono, **no** 30-second truncation.
    fn audio_loader_config(&self) -> AudioLoaderConfig {
        AudioLoaderConfig {
            target_sample_rate: WHISPER_SAMPLE_RATE,
            mono: true,
            normalize: false,
            max_duration_secs: None,
            target_duration_secs: None,
        }
    }

    /// Report progress via callback or stderr.
    fn report_progress(
        &self,
        stage: TranscriptionStage,
        current: usize,
        total: usize,
        msg: Option<&str>,
    ) {
        if let Some(ref cb) = self.progress_callback {
            let progress = TranscriptionProgress::new(
                stage,
                current,
                if total > 0 { Some(total) } else { None },
            );
            cb(&progress, msg);
        } else if !self.quiet {
            self.report_stderr(stage, current, total, msg);
        }
    }

    /// Default stderr progress output (used when `quiet == false` and no callback).
    fn report_stderr(
        &self,
        stage: TranscriptionStage,
        current: usize,
        total: usize,
        msg: Option<&str>,
    ) {
        match stage {
            TranscriptionStage::LoadingAudio => {
                if let Some(m) = msg {
                    eprintln!("Loading audio: {}", m);
                }
            }
            TranscriptionStage::Encoding | TranscriptionStage::Decoding => {
                if total > 0 {
                    eprint!("\r  {} [{}/{}]", stage, current + 1, total);
                    if let Some(m) = msg {
                        eprint!(" {}", m);
                    }
                }
            }
            TranscriptionStage::Stitching => {
                eprintln!();
            }
        }
    }
}

// =============================================================================
// Simple Linear Resampling
// =============================================================================

/// Resample audio via linear interpolation.
///
/// This is a lightweight fallback so that `transcribe_audio()` works with
/// non-16 kHz input.  For production quality consider using the `rubato` crate.
fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = (samples.len() as f64 * ratio).ceil() as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src = i as f64 / ratio;
        let lo = src.floor() as usize;
        let hi = (lo + 1).min(samples.len() - 1);
        let frac = (src - lo as f64) as f32;
        output.push(samples[lo] + (samples[hi] - samples[lo]) * frac);
    }

    output
}