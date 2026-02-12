//! Whisper transcription logic: chunking, greedy decode, timestamp parsing
use anyhow::{anyhow, Result};
use ndarray::{s, Array2, Array3};

use kjarni_transformers::{
    Cache, LanguageModel, cache::CpuBeamKVCache, cpu::encoder::{CpuEncoderOps, prelude::*}, encoder_decoder::traits::{CpuEncoderDecoderOps, EncoderDecoderLanguageModel}
};

use super::WhisperModel;

/// Whisper native sample rate.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Duration of one Whisper chunk in seconds.
pub const WHISPER_CHUNK_LENGTH_SECS: f32 = 30.0;

/// Number of samples in one 30-second chunk at 16 kHz.
pub const WHISPER_CHUNK_SAMPLES: usize = 480_000;

const SOT_TOKEN: u32 = 50258;           // <|startoftranscript|>
const EOT_TOKEN: u32 = 50257;           // <|endoftext|>
const TRANSCRIBE_TOKEN: u32 = 50359;    // <|transcribe|>
const TRANSLATE_TOKEN: u32 = 50360;     // <|translate|>
const NO_TIMESTAMPS_TOKEN: u32 = 50363; // <|notimestamps|>
const TIMESTAMP_BEGIN: u32 = 50364;     // <|0.00|>
const FIRST_SPECIAL_TOKEN: u32 = 50257;

/// Each timestamp token represents 0.02 seconds.
const TIMESTAMP_RESOLUTION: f32 = 0.02;

/// Configuration for Whisper transcription, passed down from the high-level API.
#[derive(Debug, Clone)]
pub struct WhisperTranscriberConfig {
    /// Language code (e.g. `"en"`). `None` for auto-detect (stubbed as English).
    pub language: Option<String>,
    /// Task: transcribe or translate to English.
    pub task: WhisperTask,
    /// Whether to parse timestamp tokens into timed segments.
    pub timestamps: bool,
    /// Maximum tokens to generate per 30-second chunk.
    pub max_tokens_per_chunk: usize,
}

impl Default for WhisperTranscriberConfig {
    fn default() -> Self {
        Self {
            language: None,
            task: WhisperTask::Transcribe,
            timestamps: false,
            max_tokens_per_chunk: 448,
        }
    }
}

/// Whisper task type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperTask {
    /// Transcribe audio in the source language.
    Transcribe,
    /// Translate audio to English.
    Translate,
}

/// A timed segment of transcription.
#[derive(Debug, Clone)]
pub struct WhisperSegment {
    /// Start time in seconds (absolute, after chunk-offset).
    pub start: f32,
    /// End time in seconds (absolute, after chunk-offset).
    pub end: f32,
    /// Transcribed text for this segment.
    pub text: String,
}

/// Result of decoding a single 30-second chunk.
#[derive(Debug, Clone)]
pub struct WhisperChunkResult {
    /// Timed segments within this chunk.
    pub segments: Vec<WhisperSegment>,
    /// Full joined text for this chunk.
    pub text: String,
}


impl WhisperModel {
    /// Split audio samples into 30-second chunks.
    pub fn chunk_audio(samples: &[f32], sample_rate: u32) -> Vec<Vec<f32>> {
        let chunk_size = (WHISPER_CHUNK_LENGTH_SECS * sample_rate as f32) as usize;

        if samples.is_empty() {
            return vec![];
        }

        // Short audio — single padded chunk
        if samples.len() <= chunk_size {
            let mut chunk = samples.to_vec();
            chunk.resize(chunk_size, 0.0);
            return vec![chunk];
        }

        let mut chunks = Vec::new();
        let mut offset = 0;

        while offset < samples.len() {
            let end = (offset + chunk_size).min(samples.len());
            let mut chunk = samples[offset..end].to_vec();

            // Zero-pad the last chunk to a full 30 seconds
            if chunk.len() < chunk_size {
                chunk.resize(chunk_size, 0.0);
            }

            chunks.push(chunk);
            offset += chunk_size;
        }

        chunks
    }

    /// Encode a mel spectrogram into encoder hidden states
    pub fn encode_mel(&self, mel: &Array2<f32>) -> Result<Array3<f32>> {
        // [n_mels, time] -> [1, n_mels, time]
        let mel_batch = mel
            .clone()
            .insert_axis(ndarray::Axis(0))
            .as_standard_layout()
            .to_owned();

        // Convolutional frontend + positional embeddings
        let embedded = self.embed_audio(&mel_batch)?;
        let embedded = embedded.as_standard_layout().to_owned();

        // Transformer encoder
        let seq_len = embedded.dim().1;
        let attention_mask = Array2::<f32>::ones((1, seq_len));
        let encoder = CpuEncoderOps::encoder(self);
        let output = encoder.forward(&embedded, &attention_mask)?;

        Ok(output.last_hidden_state)
    }

    /// Decode encoder hidden states into text segments
    pub fn decode_chunk(
        &self,
        encoder_hidden_states: &Array3<f32>,
        config: &WhisperTranscriberConfig,
        chunk_time_offset: f32,
        mut on_token: Option<&mut dyn FnMut(u32, &str) -> bool>,
    ) -> Result<WhisperChunkResult> {
        let tokenizer = self.tokenizer();
        let eos_token_id = self.eos_token_id().unwrap_or(EOT_TOKEN);
        let prompt_tokens = self.build_prompt_tokens(config);

        let decoder_ops = self
            .encoder_decoder_cpu_ops()
            .ok_or_else(|| anyhow!("CPU decoder ops not available"))?;

        let max_len = prompt_tokens.len() + config.max_tokens_per_chunk;
        let mut cache_box = self.new_cache(1, max_len, 0)?;
        let cache = cache_box
            .as_any_mut()
            .downcast_mut::<CpuBeamKVCache>()
            .ok_or_else(|| anyhow!("Expected CpuBeamKVCache"))?;

        let decoder = decoder_ops.decoder();
        let cross_kv = decoder.precompute_cross_attention_kv(encoder_hidden_states)?;

        let enc_seq_len = encoder_hidden_states.dim().1;
        let encoder_mask = Array2::<f32>::ones((1, enc_seq_len));

        let input_ids = Array2::from_shape_vec(
            (1, prompt_tokens.len()),
            prompt_tokens.clone(),
        )?;
        let padding_mask = Array2::<f32>::ones((1, prompt_tokens.len()));

        let output = decoder.forward(
            &input_ids,
            encoder_hidden_states,
            Some(&padding_mask),
            Some(&encoder_mask),
            Some(cache),
            Some(&cross_kv),
        )?;

        for (i, (k, v)) in output.new_self_attn_kv.into_iter().enumerate() {
            cache.update(i, &k, &v)?;
        }
        cache.increment_len(prompt_tokens.len());

        // First predicted token from the last prompt position
        let logits = decoder_ops.project_to_logits(&output.last_hidden_state)?;
        let last_logits = logits.slice(s![0, -1_i32, ..]);

        let mut next_token = Self::pick_token(&last_logits, config, eos_token_id);
        let mut generated_ids: Vec<u32> = vec![next_token];

        // Notify callback
        if let Some(ref mut cb) = on_token {
            if next_token != eos_token_id {
                let text = tokenizer.decode(&[next_token], false).unwrap_or_default();
                if !cb(next_token, &text) {
                    return self.finalize_chunk(generated_ids, config, chunk_time_offset);
                }
            }
        }
        for _step in 0..config.max_tokens_per_chunk {
            if next_token == eos_token_id {
                break;
            }

            let step_ids = Array2::from_shape_vec((1, 1), vec![next_token])?;
            let step_mask = Array2::<f32>::ones((1, 1));

            let output = decoder.forward(
                &step_ids,
                encoder_hidden_states,
                Some(&step_mask),
                Some(&encoder_mask),
                Some(cache),
                Some(&cross_kv),
            )?;

            for (i, (k, v)) in output.new_self_attn_kv.into_iter().enumerate() {
                cache.update(i, &k, &v)?;
            }
            cache.increment_len(1);

            let logits = decoder_ops.project_to_logits(&output.last_hidden_state)?;
            let last_logits = logits.slice(s![0, -1_i32, ..]);

            next_token = Self::pick_token(&last_logits, config, eos_token_id);
            generated_ids.push(next_token);

            if let Some(ref mut cb) = on_token {
                if next_token != eos_token_id {
                    let text = tokenizer.decode(&[next_token], false).unwrap_or_default();
                    if !cb(next_token, &text) {
                        break;
                    }
                }
            }
        }

        self.finalize_chunk(generated_ids, config, chunk_time_offset)
    }

    /// Greedy argmax over logits, suppressing special tokens as configured.
    fn pick_token(
        logits: &ndarray::ArrayView1<f32>,
        config: &WhisperTranscriberConfig,
        eos_token_id: u32,
    ) -> u32 {
        logits
            .iter()
            .enumerate()
            .filter(|(idx, _)| {
                let id = *idx as u32;
                // Always allow regular vocabulary tokens
                if id < FIRST_SPECIAL_TOKEN {
                    return true;
                }
                // Always allow EOS
                if id == eos_token_id {
                    return true;
                }
                // Allow timestamp tokens only when timestamps are enabled
                if config.timestamps && id >= TIMESTAMP_BEGIN {
                    return true;
                }
                false
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(eos_token_id)
    }

    /// Build the decoder prompt token sequence for the given config.
    fn build_prompt_tokens(&self, config: &WhisperTranscriberConfig) -> Vec<u32> {
        let mut tokens = vec![SOT_TOKEN];

        // Language token — resolve via tokenizer, fallback to English
        let lang = config.language.as_deref().unwrap_or("en");
        tokens.push(self.resolve_language_token(lang).unwrap_or(50259));

        // Task token
        match config.task {
            WhisperTask::Transcribe => tokens.push(TRANSCRIBE_TOKEN),
            WhisperTask::Translate => tokens.push(TRANSLATE_TOKEN),
        }

        // Suppress timestamp generation unless requested
        if !config.timestamps {
            tokens.push(NO_TIMESTAMPS_TOKEN);
        }

        tokens
    }

    /// Resolve a language code
    fn resolve_language_token(&self, language: &str) -> Option<u32> {
        let tag = format!("<|{}|>", language.to_lowercase());
        self.tokenizer().token_to_id(&tag)
    }

    /// Convert generated token IDs into a WhisperChunkResult
    fn finalize_chunk(
        &self,
        generated_ids: Vec<u32>,
        config: &WhisperTranscriberConfig,
        chunk_offset: f32,
    ) -> Result<WhisperChunkResult> {
        if config.timestamps {
            let segments =
                Self::parse_timestamp_segments(&generated_ids, self.tokenizer(), chunk_offset);
            let text = segments
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join("");
            Ok(WhisperChunkResult { segments, text })
        } else {
            // No timestamps — single segment spanning the whole chunk
            let text_ids: Vec<u32> = generated_ids
                .iter()
                .filter(|&&id| id < FIRST_SPECIAL_TOKEN)
                .copied()
                .collect();

            let text = self
                .tokenizer()
                .decode(&text_ids, true)
                .map_err(|e| anyhow!("Tokenizer decode error: {}", e))?;

            let segment = WhisperSegment {
                start: chunk_offset,
                end: chunk_offset + WHISPER_CHUNK_LENGTH_SECS,
                text: text.clone(),
            };

            Ok(WhisperChunkResult {
                segments: vec![segment],
                text,
            })
        }
    }

    /// Parse a stream of token IDs
    fn parse_timestamp_segments(
        token_ids: &[u32],
        tokenizer: &tokenizers::Tokenizer,
        chunk_offset: f32,
    ) -> Vec<WhisperSegment> {
        let mut segments = Vec::new();
        let mut current_start: Option<f32> = None;
        let mut current_tokens: Vec<u32> = Vec::new();

        for &id in token_ids {
            if id >= TIMESTAMP_BEGIN {
                // Timestamp token
                let time =
                    (id - TIMESTAMP_BEGIN) as f32 * TIMESTAMP_RESOLUTION + chunk_offset;

                if current_start.is_none() {
                    // Opening timestamp
                    current_start = Some(time);
                } else {
                    // Closing timestamp — emit segment
                    let text_ids: Vec<u32> = current_tokens
                        .iter()
                        .filter(|&&t| t < FIRST_SPECIAL_TOKEN)
                        .copied()
                        .collect();

                    let text = tokenizer.decode(&text_ids, true).unwrap_or_default();

                    if !text.trim().is_empty() {
                        segments.push(WhisperSegment {
                            start: current_start.unwrap(),
                            end: time,
                            text,
                        });
                    }

                    // Next segment starts at this timestamp
                    current_start = Some(time);
                    current_tokens.clear();
                }
            } else if id < FIRST_SPECIAL_TOKEN {
                current_tokens.push(id);
            }
            // Other special tokens are ignored
        }

        // Trailing text without a closing timestamp
        if let Some(start) = current_start {
            if !current_tokens.is_empty() {
                let text_ids: Vec<u32> = current_tokens
                    .iter()
                    .filter(|&&t| t < FIRST_SPECIAL_TOKEN)
                    .copied()
                    .collect();

                let text = tokenizer.decode(&text_ids, true).unwrap_or_default();

                if !text.trim().is_empty() {
                    segments.push(WhisperSegment {
                        start,
                        end: start + WHISPER_CHUNK_LENGTH_SECS,
                        text,
                    });
                }
            }
        }

        segments
    }

    /// Stitch results from multiple chunks into a single transcription.
    pub fn stitch_segments(
        chunk_results: Vec<WhisperChunkResult>,
    ) -> (String, Vec<WhisperSegment>) {
        if chunk_results.is_empty() {
            return (String::new(), Vec::new());
        }

        let mut full_text = String::new();
        let mut all_segments: Vec<WhisperSegment> = Vec::new();

        for result in chunk_results {
            full_text.push_str(&result.text);
            all_segments.extend(result.segments);
        }

        let merged = Self::merge_boundary_segments(all_segments);
        (full_text, merged)
    }

    /// Merge segments that meet exactly at chunk boundaries (e.g. 30.0 -> 30.0).
    fn merge_boundary_segments(segments: Vec<WhisperSegment>) -> Vec<WhisperSegment> {
        if segments.len() < 2 {
            return segments;
        }

        let mut merged: Vec<WhisperSegment> = Vec::with_capacity(segments.len());

        for seg in segments {
            let should_merge = merged.last().map_or(false, |prev| {
                (prev.end - seg.start).abs() < 0.02 && is_chunk_boundary(prev.end)
            });

            if should_merge {
                let prev = merged.last_mut().unwrap();
                prev.end = seg.end;
                prev.text.push_str(&seg.text);
            } else {
                merged.push(seg);
            }
        }

        merged
    }
}


fn is_chunk_boundary(time: f32) -> bool {
    let rem = time % WHISPER_CHUNK_LENGTH_SECS;
    rem < 0.02 || (WHISPER_CHUNK_LENGTH_SECS - rem) < 0.02
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_audio_short() {
        let samples = vec![1.0; 100_000]; // ~6.25 s
        let chunks = WhisperModel::chunk_audio(&samples, 16_000);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), WHISPER_CHUNK_SAMPLES);
        assert_eq!(chunks[0][0], 1.0);
        assert_eq!(chunks[0][100_000], 0.0); // zero-padded
    }

    #[test]
    fn test_chunk_audio_exact() {
        let samples = vec![1.0; WHISPER_CHUNK_SAMPLES];
        let chunks = WhisperModel::chunk_audio(&samples, 16_000);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), WHISPER_CHUNK_SAMPLES);
    }

    #[test]
    fn test_chunk_audio_multi() {
        let samples = vec![1.0; WHISPER_CHUNK_SAMPLES + 100]; // 30s + tiny bit
        let chunks = WhisperModel::chunk_audio(&samples, 16_000);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len(), WHISPER_CHUNK_SAMPLES);
        assert_eq!(chunks[1].len(), WHISPER_CHUNK_SAMPLES);
        assert_eq!(chunks[1][99], 1.0);
        assert_eq!(chunks[1][100], 0.0); // zero-padded
    }

    #[test]
    fn test_chunk_audio_empty() {
        let chunks = WhisperModel::chunk_audio(&[], 16_000);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_parse_timestamps_basic() {
        
        let segments = WhisperModel::merge_boundary_segments(vec![
            WhisperSegment { start: 0.0, end: 30.0, text: "Hello ".into() },
            WhisperSegment { start: 30.0, end: 45.0, text: "world.".into() },
        ]);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "Hello world.");
        assert!((segments[0].start - 0.0).abs() < 0.01);
        assert!((segments[0].end - 45.0).abs() < 0.01);
    }

    #[test]
    fn test_is_chunk_boundary() {
        assert!(is_chunk_boundary(30.0));
        assert!(is_chunk_boundary(60.0));
        assert!(is_chunk_boundary(29.99));
        assert!(!is_chunk_boundary(15.0));
        assert!(!is_chunk_boundary(29.5));
    }
}