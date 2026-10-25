//! Whisper transcription logic: chunking, greedy decode, timestamp parsing
use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, Axis, s};

use kjarni_transformers::{
    Cache, LanguageModel, cache::CpuBeamKVCache, cpu::encoder::CpuEncoderOps,
    encoder_decoder::traits::EncoderDecoderLanguageModel,
};

use super::WhisperModel;

/// Whisper native sample rate.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Duration of one Whisper chunk in seconds.
pub const WHISPER_CHUNK_LENGTH_SECS: f32 = 30.0;

/// Number of samples in one 30-second chunk at 16 kHz.
pub const WHISPER_CHUNK_SAMPLES: usize = 480_000;

const SOT_TOKEN: u32 = 50258; // <|startoftranscript|>
const EOT_TOKEN: u32 = 50257; // <|endoftext|>
const TRANSCRIBE_TOKEN: u32 = 50359; // <|transcribe|>
const TRANSLATE_TOKEN: u32 = 50360; // <|translate|>
const NO_TIMESTAMPS_TOKEN: u32 = 50363; // <|notimestamps|>
const TIMESTAMP_BEGIN: u32 = 50364; // <|0.00|>
const FIRST_SPECIAL_TOKEN: u32 = 50257;

/// Whisper's language codes. Their token ids are looked up in the
/// tokenizer, since large-v3 added one and moved the rest.
const LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

/// Each timestamp token represents 0.02 seconds.
const TIMESTAMP_RESOLUTION: f32 = 0.02;

/// Configuration for Whisper transcription, passed down from the high-level API.
#[derive(Debug, Clone)]
pub struct WhisperTranscriberConfig {
    /// Language code (e.g. `"en"`). `None` decodes as English; callers that
    /// want detection run `WhisperModel::detect_language` first.
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

        let input_ids = Array2::from_shape_vec((1, prompt_tokens.len()), prompt_tokens.clone())?;
        let padding_mask = Array2::<f32>::ones((1, prompt_tokens.len()));

        #[allow(
            deprecated,
            reason = "internal caller of an API deprecated without a named replacement"
        )]
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
        if let Some(ref mut cb) = on_token
            && next_token != eos_token_id
        {
            let text = tokenizer.decode(&[next_token], false).unwrap_or_default();
            if !cb(next_token, &text) {
                return self.finalize_chunk(generated_ids, config, chunk_time_offset);
            }
        }
        for _step in 0..config.max_tokens_per_chunk {
            if next_token == eos_token_id {
                break;
            }

            let step_ids = Array2::from_shape_vec((1, 1), vec![next_token])?;
            let step_mask = Array2::<f32>::ones((1, 1));

            #[allow(
                deprecated,
                reason = "internal caller of an API deprecated without a named replacement"
            )]
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
            if let Some(keep) = looping_tail(&generated_ids) {
                generated_ids.truncate(keep);
                break;
            }

            if let Some(ref mut cb) = on_token
                && next_token != eos_token_id
            {
                let text = tokenizer.decode(&[next_token], false).unwrap_or_default();
                if !cb(next_token, &text) {
                    break;
                }
            }
        }

        self.finalize_chunk(generated_ids, config, chunk_time_offset)
    }

    /// Decodes several chunks at once, greedily, each exactly as
    /// `decode_chunk` would alone.
    ///
    /// Decoding is bound by memory, not arithmetic: every token reads all of
    /// the decoder's weights, half a gigabyte for whisper-small, to do very
    /// little with them. Chunks are independent, so a batch steps together
    /// and shares each read. A chunk that has finished is fed end-of-text
    /// until the rest are done; its result stops at its own end.
    pub fn decode_chunks(
        &self,
        encoder_hidden_states: &[Array3<f32>],
        config: &WhisperTranscriberConfig,
        chunk_time_offsets: &[f32],
    ) -> Result<Vec<WhisperChunkResult>> {
        let n = encoder_hidden_states.len();
        if n <= 1 {
            return encoder_hidden_states
                .iter()
                .zip(chunk_time_offsets)
                .map(|(e, &offset)| self.decode_chunk(e, config, offset, None))
                .collect();
        }
        let eos_token_id = self.eos_token_id().unwrap_or(EOT_TOKEN);
        let prompt_tokens = self.build_prompt_tokens(config);
        let decoder_ops = self
            .encoder_decoder_cpu_ops()
            .ok_or_else(|| anyhow!("CPU decoder ops not available"))?;
        let decoder = decoder_ops.decoder();

        let views: Vec<_> = encoder_hidden_states.iter().map(|e| e.view()).collect();
        let encoder = ndarray::concatenate(Axis(0), &views)?;
        let encoder_mask = Array2::<f32>::ones((n, encoder.dim().1));
        let cross_kv = decoder.precompute_cross_attention_kv(&encoder)?;

        let max_len = prompt_tokens.len() + config.max_tokens_per_chunk;
        let mut cache_box = self.new_cache(n, max_len, 0)?;
        let cache = cache_box
            .as_any_mut()
            .downcast_mut::<CpuBeamKVCache>()
            .ok_or_else(|| anyhow!("Expected CpuBeamKVCache"))?;

        let mut input_ids =
            Array2::from_shape_fn((n, prompt_tokens.len()), |(_, j)| prompt_tokens[j]);
        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut done = vec![false; n];
        for _ in 0..=config.max_tokens_per_chunk {
            #[allow(
                deprecated,
                reason = "internal caller of an API deprecated without a named replacement"
            )]
            // No decoder padding mask: chunks step in lockstep, so no row is
            // ever padded, and the decoder's mask only broadcasts at batch 1.
            let output = decoder.forward(
                &input_ids,
                &encoder,
                None,
                Some(&encoder_mask),
                Some(cache),
                Some(&cross_kv),
            )?;
            for (i, (k, v)) in output.new_self_attn_kv.into_iter().enumerate() {
                cache.update(i, &k, &v)?;
            }
            cache.increment_len(input_ids.ncols());

            let logits = decoder_ops.project_to_logits(&output.last_hidden_state)?;
            let mut next = vec![eos_token_id; n];
            for b in 0..n {
                if done[b] {
                    continue;
                }
                next[b] = Self::pick_token(&logits.slice(s![b, -1_i32, ..]), config, eos_token_id);
                generated[b].push(next[b]);
                done[b] = next[b] == eos_token_id;
                if let Some(keep) = looping_tail(&generated[b]) {
                    generated[b].truncate(keep);
                    done[b] = true;
                }
            }
            if done.iter().all(|&d| d) {
                break;
            }
            input_ids = Array2::from_shape_vec((n, 1), next)?;
        }

        generated
            .into_iter()
            .zip(chunk_time_offsets)
            .map(|(ids, &offset)| self.finalize_chunk(ids, config, offset))
            .collect()
    }

    /// The language spoken in a chunk, as Whisper itself decides it: given
    /// only start-of-transcript, the language token it rates highest.
    pub fn detect_language(&self, encoder_hidden_states: &Array3<f32>) -> Result<String> {
        let decoder_ops = self
            .encoder_decoder_cpu_ops()
            .ok_or_else(|| anyhow!("CPU decoder ops not available"))?;
        let decoder = decoder_ops.decoder();
        let encoder = encoder_hidden_states.slice(s![0..1, .., ..]).to_owned();
        let cross_kv = decoder.precompute_cross_attention_kv(&encoder)?;
        let mut cache_box = self.new_cache(1, 1, 0)?;
        let cache = cache_box
            .as_any_mut()
            .downcast_mut::<CpuBeamKVCache>()
            .ok_or_else(|| anyhow!("Expected CpuBeamKVCache"))?;
        let input_ids = Array2::from_elem((1, 1), SOT_TOKEN);
        #[allow(
            deprecated,
            reason = "internal caller of an API deprecated without a named replacement"
        )]
        let output = decoder.forward(
            &input_ids,
            &encoder,
            Some(&Array2::<f32>::ones((1, 1))),
            Some(&Array2::<f32>::ones((1, encoder.dim().1))),
            Some(cache),
            Some(&cross_kv),
        )?;
        let logits = decoder_ops.project_to_logits(&output.last_hidden_state)?;
        let logits = logits.slice(s![0, -1_i32, ..]);
        LANGUAGES
            .iter()
            .filter_map(|code| {
                let id = self.resolve_language_token(code)? as usize;
                (id < logits.len()).then(|| (*code, logits[id]))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(code, _)| code.to_string())
            .ok_or_else(|| anyhow!("this model has no language tokens"))
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
                let time = (id - TIMESTAMP_BEGIN) as f32 * TIMESTAMP_RESOLUTION + chunk_offset;

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
        if let Some(start) = current_start
            && !current_tokens.is_empty()
        {
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
            let should_merge = merged.last().is_some_and(|prev| {
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

/// Where a greedy decode has fallen into a loop, if it has: the length to
/// keep, which is everything before the cycle repeated plus one copy of it.
///
/// Greedy decoding can lock onto a phrase and repeat it to the token limit,
/// most often in languages the model knows less well: a minute of Icelandic
/// came out as "er því að" two hundred times, and took three times longer
/// than its sense would have. A cycle of up to `MAX_PERIOD` tokens seen
/// `LOOP_REPEATS` times running at the end ends the chunk.
fn looping_tail(ids: &[u32]) -> Option<usize> {
    const MAX_PERIOD: usize = 16;
    const LOOP_REPEATS: usize = 4;
    (1..=MAX_PERIOD).find_map(|period| {
        let span = period * LOOP_REPEATS;
        if ids.len() < span {
            return None;
        }
        let tail = &ids[ids.len() - span..];
        let cycle = &tail[..period];
        // Timestamps are never part of a real loop's evidence: each one is a
        // new moment, and a run of "<|t|> word <|t|> word" is speech.
        if cycle.iter().any(|&t| t >= TIMESTAMP_BEGIN) {
            return None;
        }
        tail.chunks(period)
            .all(|c| c == cycle)
            .then_some(ids.len() - span + period)
    })
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
            WhisperSegment {
                start: 0.0,
                end: 30.0,
                text: "Hello ".into(),
            },
            WhisperSegment {
                start: 30.0,
                end: 45.0,
                text: "world.".into(),
            },
        ]);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "Hello world.");
        assert!((segments[0].start - 0.0).abs() < 0.01);
        assert!((segments[0].end - 45.0).abs() < 0.01);
    }

    #[test]
    fn a_repeated_phrase_ends_the_chunk_after_one_copy() {
        let phrase = [10, 11, 12];
        let mut ids = vec![1, 2, 3];
        for _ in 0..3 {
            ids.extend(phrase);
        }
        assert_eq!(looping_tail(&ids), None, "three times is still speech");
        ids.extend(phrase);
        assert_eq!(looping_tail(&ids), Some(6));
        // A single token stuck.
        assert_eq!(looping_tail(&[5, 7, 7, 7, 7]), Some(2));
        // Distinct timestamps keep the same word from counting as a loop.
        let stamped: Vec<u32> = (0..8).flat_map(|i| [TIMESTAMP_BEGIN + i, 42]).collect();
        assert_eq!(looping_tail(&stamped), None);
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
