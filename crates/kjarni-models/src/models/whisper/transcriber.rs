//! Whisper transcription logic: chunking, greedy decode, timestamp parsing
use std::sync::Arc;

use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, Axis, s};

use kjarni_transformers::gpu::{GpuFrameContext, GpuTensor};
use kjarni_transformers::models::base::ModelInput;
use kjarni_transformers::{
    Cache, LanguageModel, WgpuContext,
    cache::CpuBeamKVCache,
    cpu::encoder::{CpuEncoderOps, GpuEncoderOps, prelude::EncoderLanguageModel},
    encoder_decoder::traits::{
        CpuCrossAttentionKVCache, CpuEncoderDecoderOps, EncoderDecoderLanguageModel,
        GpuCrossAttentionKVCache, GpuEncoderDecoderOps,
    },
    traits::InferenceModel,
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

/// The least audio left in a window for a new segment to open: half a second.
const MIN_SEGMENT_TOKENS: u32 = 25;

/// The timestamp token at which a window's audio ends and its padding begins.
fn audio_end_timestamp(chunk: &AudioChunk) -> u32 {
    TIMESTAMP_BEGIN + (chunk.duration_secs / TIMESTAMP_RESOLUTION).round() as u32
}

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

/// One window of audio for Whisper: up to 30 seconds of it, zero-padded to 30.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    /// Where the window starts in the whole recording.
    pub start_secs: f32,
    /// How much of the window is audio rather than padding.
    pub duration_secs: f32,
}

/// How far back from a window's 30-second limit to look for a pause to cut at.
const CUT_SEARCH_SECS: f32 = 5.0;
/// The stretch whose energy is compared when choosing where to cut.
const CUT_WINDOW_SECS: f32 = 0.2;

/// Where to cut in `samples[lo..hi]`: the middle of the latest
/// `CUT_WINDOW_SECS` stretch about as quiet as the quietest one. Taking the
/// latest pause rather than the very quietest keeps windows full, so a minute
/// of speech fits in two windows rather than spilling a few seconds into a third.
fn quietest_point(samples: &[f32], lo: usize, hi: usize, sample_rate: u32) -> usize {
    let window = (CUT_WINDOW_SECS * sample_rate as f32) as usize;
    let step = (window / 10).max(1);
    let energies: Vec<(usize, f32)> = (lo..=hi.saturating_sub(window))
        .step_by(step)
        .map(|at| (at, samples[at..at + window].iter().map(|x| x * x).sum()))
        .collect();
    let quietest = energies.iter().fold(f32::INFINITY, |m, &(_, e)| m.min(e));
    // Twice the quietest, or a level of 1e-3 (-60 dB), whichever is higher.
    let quiet_enough = (2.0 * quietest).max(1e-6 * window as f32);
    energies
        .iter()
        .rev()
        .find(|&&(_, e)| e <= quiet_enough)
        .map_or(hi, |&(at, _)| at + window / 2)
}

/// A Whisper text decoder stepped a token at a time, on the device the model is on.
pub(super) enum DecoderSteps<'a> {
    Cpu {
        ops: &'a dyn CpuEncoderDecoderOps,
        encoder: Array3<f32>,
        encoder_mask: Array2<f32>,
        cross_kv: CpuCrossAttentionKVCache,
        cache: Box<dyn Cache>,
    },
    Gpu {
        ops: &'a dyn GpuEncoderDecoderOps,
        context: Arc<WgpuContext>,
        encoder: GpuTensor,
        cross_kv: GpuCrossAttentionKVCache,
        cache: Box<dyn Cache>,
    },
}

impl DecoderSteps<'_> {
    /// Feeds `input_ids`, `[batch, tokens]`, and returns logits whose last
    /// position predicts the next token.
    pub(super) fn step(&mut self, input_ids: &Array2<u32>) -> Result<Array3<f32>> {
        match self {
            DecoderSteps::Cpu {
                ops,
                encoder,
                encoder_mask,
                cross_kv,
                cache,
            } => {
                let cache = cache
                    .as_any_mut()
                    .downcast_mut::<CpuBeamKVCache>()
                    .ok_or_else(|| anyhow!("Expected CpuBeamKVCache"))?;
                // Rows step in lockstep, so none is ever padded, and the decoder's
                // padding mask only broadcasts at batch 1.
                let padding_mask =
                    (input_ids.nrows() == 1).then(|| Array2::<f32>::ones(input_ids.dim()));
                #[allow(
                    deprecated,
                    reason = "internal caller of an API deprecated without a named replacement"
                )]
                let output = ops.decoder().forward(
                    input_ids,
                    encoder,
                    padding_mask.as_ref(),
                    Some(encoder_mask),
                    Some(cache),
                    Some(cross_kv),
                )?;
                for (i, (k, v)) in output.new_self_attn_kv.into_iter().enumerate() {
                    cache.update(i, &k, &v)?;
                }
                cache.increment_len(input_ids.ncols());
                ops.project_to_logits(&output.last_hidden_state)
            }
            DecoderSteps::Gpu {
                ops,
                context,
                encoder,
                cross_kv,
                cache,
            } => pollster::block_on(async {
                // The GPU KV cache appends one token per call, so a prompt goes in
                // one token at a time and only the last position's logits are read.
                let mut last = None;
                for column in input_ids.columns() {
                    let batch = column.len();
                    let position = cache.get_seq_length();
                    let tokens = column.to_owned().into_shape_with_order((batch, 1))?;
                    let tokens = GpuTensor::from_ndarray(context, &tokens)?;
                    let mask = Array2::<f32>::ones((batch, position + 1));
                    let mask = GpuTensor::from_ndarray(context, &mask)?;

                    let pool = context.get_inference_pool();
                    let pool_guard = pool.lock().await;
                    let mut frame = GpuFrameContext::new(context, pool_guard);
                    let (cmd, pool) = frame.resources();
                    let embedded = ops.embed_decoder_tokens(
                        cmd,
                        pool,
                        ModelInput::TokensGpu(&tokens),
                        position,
                    )?;
                    let embedded = ops.decoder().embed_norm(cmd, pool, &embedded)?;
                    let output = ops.decoder().forward_layers(
                        cmd,
                        pool,
                        &embedded,
                        encoder,
                        &mask,
                        position,
                        Some(cache.as_mut()),
                        Some(cross_kv),
                        0,
                        ops.decoder().num_layers(),
                    )?;
                    let logits = ops.project_to_logits(&mut frame, &output.last_hidden_state)?;
                    frame.finish();
                    cache.increment_len(1);
                    last = Some(logits);
                }
                last.ok_or_else(|| anyhow!("no tokens to decode"))?
                    .to_ndarray_3d()
                    .await
            }),
        }
    }
}

impl WhisperModel {
    /// Split audio into windows of at most 30 seconds, cut at pauses.
    ///
    /// A window cut mid-word loses the word: Whisper drops an unfinished phrase
    /// at the end of one window and guesses at the fragment that opens the
    /// next, so a sentence spanning a fixed 30-second cut goes missing. Each
    /// cut is made at the quietest point of the window's last five seconds.
    pub fn chunk_audio(samples: &[f32], sample_rate: u32) -> Vec<AudioChunk> {
        let chunk_size = (WHISPER_CHUNK_LENGTH_SECS * sample_rate as f32) as usize;
        let search = (CUT_SEARCH_SECS * sample_rate as f32) as usize;
        let secs = |n: usize| n as f32 / sample_rate as f32;

        let mut chunks = Vec::new();
        let mut offset = 0;
        while offset < samples.len() {
            let end = if samples.len() - offset <= chunk_size {
                samples.len()
            } else {
                let limit = offset + chunk_size;
                quietest_point(samples, limit - search, limit, sample_rate)
            };
            let mut chunk = samples[offset..end].to_vec();
            chunk.resize(chunk_size, 0.0);
            chunks.push(AudioChunk {
                samples: chunk,
                start_secs: secs(offset),
                duration_secs: secs(end - offset),
            });
            offset = end;
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
        let embedded = CpuEncoderOps::embed_audio(self, &mel_batch)?;
        let embedded = embedded.as_standard_layout().to_owned();

        // Transformer encoder
        let seq_len = embedded.dim().1;
        let attention_mask = Array2::<f32>::ones((1, seq_len));
        if self.encoder_gpu_ops().is_some() {
            return pollster::block_on(self.encode_on_gpu(&embedded, &attention_mask));
        }
        let encoder = CpuEncoderOps::encoder(self);
        let output = encoder.forward(&embedded, &attention_mask)?;

        Ok(output.last_hidden_state)
    }

    async fn encode_on_gpu(
        &self,
        embedded: &Array3<f32>,
        attention_mask: &Array2<f32>,
    ) -> Result<Array3<f32>> {
        let context = self
            .context()
            .ok_or_else(|| anyhow!("Whisper on a GPU has no GPU context"))?;
        let hidden = GpuTensor::from_ndarray(&context, embedded)?;
        let mask = GpuTensor::from_ndarray(&context, attention_mask)?;

        let pool = context.get_inference_pool();
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&context, pool_guard);
        let (cmd, pool) = frame.resources();

        let encoder = GpuEncoderOps::encoder(self);
        let layers = encoder.forward_layers(cmd, pool, &hidden, &mask, 0, encoder.num_layers())?;
        let output = encoder.final_norm(cmd, pool, &layers)?;
        frame.finish();

        output.to_ndarray_3d().await
    }

    /// A decoder ready to step over `encoder_hidden_states`, with room for
    /// `max_len` tokens per row.
    pub(super) fn decoder_steps(
        &self,
        encoder_hidden_states: &Array3<f32>,
        max_len: usize,
    ) -> Result<DecoderSteps<'_>> {
        let batch = encoder_hidden_states.dim().0;
        let cache = self.new_cache(batch, max_len, batch)?;
        if let Some(ops) = self.encoder_decoder_cpu_ops() {
            let cross_kv = ops
                .decoder()
                .precompute_cross_attention_kv(encoder_hidden_states)?;
            return Ok(DecoderSteps::Cpu {
                ops,
                encoder: encoder_hidden_states.to_owned(),
                encoder_mask: Array2::ones((batch, encoder_hidden_states.dim().1)),
                cross_kv,
                cache,
            });
        }
        let ops = self
            .encoder_decoder_gpu_ops()
            .ok_or_else(|| anyhow!("Whisper loaded without a decoder"))?;
        let context = self
            .context()
            .ok_or_else(|| anyhow!("Whisper on a GPU has no GPU context"))?;
        let encoder = GpuTensor::from_ndarray(&context, encoder_hidden_states)?;
        let cross_kv = pollster::block_on(async {
            let pool = context.get_inference_pool();
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&context, pool_guard);
            let (cmd, pool) = frame.resources();
            let kv = ops
                .decoder()
                .precompute_cross_attention_kv(cmd, pool, &encoder);
            frame.finish();
            kv
        })?;
        Ok(DecoderSteps::Gpu {
            ops,
            context,
            encoder,
            cross_kv,
            cache,
        })
    }

    /// Decode encoder hidden states into text segments
    pub fn decode_chunk(
        &self,
        encoder_hidden_states: &Array3<f32>,
        config: &WhisperTranscriberConfig,
        chunk: &AudioChunk,
        mut on_token: Option<&mut dyn FnMut(u32, &str) -> bool>,
    ) -> Result<WhisperChunkResult> {
        let chunk_time_offset = chunk.start_secs;
        let audio_end = audio_end_timestamp(chunk);
        let tokenizer = self.tokenizer();
        let eos_token_id = self.eos_token_id().unwrap_or(EOT_TOKEN);
        let prompt_tokens = self.build_prompt_tokens(config);

        let max_len = prompt_tokens.len() + config.max_tokens_per_chunk;
        let mut steps = self.decoder_steps(encoder_hidden_states, max_len)?;

        // First predicted token from the last prompt position
        let input_ids = Array2::from_shape_vec((1, prompt_tokens.len()), prompt_tokens.clone())?;
        let logits = steps.step(&input_ids)?;
        let last_logits = logits.slice(s![0, -1_i32, ..]);

        let mut next_token = Self::pick_token(&last_logits, &[], config, eos_token_id, audio_end);
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
            let logits = steps.step(&step_ids)?;
            let last_logits = logits.slice(s![0, -1_i32, ..]);

            next_token = Self::pick_token(
                &last_logits,
                &generated_ids,
                config,
                eos_token_id,
                audio_end,
            );
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
        chunks: &[AudioChunk],
    ) -> Result<Vec<WhisperChunkResult>> {
        let n = encoder_hidden_states.len();
        if n <= 1 {
            return encoder_hidden_states
                .iter()
                .zip(chunks)
                .map(|(e, chunk)| self.decode_chunk(e, config, chunk, None))
                .collect();
        }
        let eos_token_id = self.eos_token_id().unwrap_or(EOT_TOKEN);
        let prompt_tokens = self.build_prompt_tokens(config);
        let views: Vec<_> = encoder_hidden_states.iter().map(|e| e.view()).collect();
        let encoder = ndarray::concatenate(Axis(0), &views)?;
        let max_len = prompt_tokens.len() + config.max_tokens_per_chunk;
        let mut steps = self.decoder_steps(&encoder, max_len)?;

        let mut input_ids =
            Array2::from_shape_fn((n, prompt_tokens.len()), |(_, j)| prompt_tokens[j]);
        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut done = vec![false; n];
        for _ in 0..=config.max_tokens_per_chunk {
            let logits = steps.step(&input_ids)?;
            let mut next = vec![eos_token_id; n];
            for b in 0..n {
                if done[b] {
                    continue;
                }
                next[b] = Self::pick_token(
                    &logits.slice(s![b, -1_i32, ..]),
                    &generated[b],
                    config,
                    eos_token_id,
                    audio_end_timestamp(&chunks[b]),
                );
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
            .zip(chunks)
            .map(|(ids, chunk)| self.finalize_chunk(ids, config, chunk.start_secs))
            .collect()
    }

    /// The language spoken in a chunk, as Whisper itself decides it: given
    /// only start-of-transcript, the language token it rates highest.
    pub fn detect_language(&self, encoder_hidden_states: &Array3<f32>) -> Result<String> {
        let encoder = encoder_hidden_states.slice(s![0..1, .., ..]).to_owned();
        let mut steps = self.decoder_steps(&encoder, 1)?;
        let logits = steps.step(&Array2::from_elem((1, 1), SOT_TOKEN))?;
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
    /// The next token, greedily. With timestamps on, Whisper's own rules
    /// apply (`ApplyTimestampRules` in OpenAI's decoding.py): a segment opens
    /// with a timestamp in the first second, timestamps come in pairs and never
    /// go back, and one is taken whenever the timestamps together are likelier
    /// than the best text token. Left to plain argmax, Whisper rarely closes a
    /// segment, so nothing marks text it invents past the end of the audio.
    ///
    /// No segment opens within `MIN_SEGMENT_SECS` of `audio_end`, the timestamp
    /// where the window's audio gives way to padding: past the speech, Whisper
    /// fills the silence with a two-second segment after another to 30 seconds.
    fn pick_token(
        logits: &ndarray::ArrayView1<f32>,
        generated: &[u32],
        config: &WhisperTranscriberConfig,
        eos_token_id: u32,
        audio_end: u32,
    ) -> u32 {
        let argmax = |allowed: &dyn Fn(u32) -> bool| {
            logits
                .iter()
                .enumerate()
                .filter(|&(id, _)| allowed(id as u32))
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map_or(eos_token_id, |(id, _)| id as u32)
        };
        let is_text = |id: u32| id < FIRST_SPECIAL_TOKEN;
        if !config.timestamps {
            return argmax(&|id| is_text(id) || id == eos_token_id);
        }

        let is_ts = |id: u32| id >= TIMESTAMP_BEGIN;
        let last_was_ts = generated.last().is_some_and(|&t| is_ts(t));
        let penultimate_was_ts = generated.len() < 2 || is_ts(generated[generated.len() - 2]);
        let (mut text_ok, ts_ok) = match (last_was_ts, penultimate_was_ts) {
            // A segment just closed: the next opens one or ends the chunk.
            (true, false) => (false, true),
            // A segment just opened: text comes next.
            (true, true) => (true, false),
            _ => (true, true),
        };
        let first_ts = match generated.iter().rev().find(|&&t| is_ts(t)) {
            // The timestamp that opens a segment may repeat the one that
            // closed the last; any other must be later.
            Some(&t) if last_was_ts && !penultimate_was_ts => t,
            Some(&t) => t + 1,
            None => TIMESTAMP_BEGIN,
        };
        // The first segment starts within the first second.
        let last_ts = if generated.is_empty() {
            text_ok = false;
            TIMESTAMP_BEGIN + 50
        } else {
            u32::MAX
        };
        // A timestamp right after a closing one opens the next segment.
        let opening = last_was_ts && !penultimate_was_ts;
        let last_ts = if opening {
            last_ts.min(audio_end.saturating_sub(MIN_SEGMENT_TOKENS))
        } else {
            last_ts
        };
        let ts_allowed = |id: u32| ts_ok && is_ts(id) && id >= first_ts && id <= last_ts;
        // The chunk may end anywhere but before its first token.
        let text_allowed =
            |id: u32| (text_ok && is_text(id)) || (id == eos_token_id && !generated.is_empty());

        if text_ok && ts_ok {
            // Take a timestamp when they are together likelier than any text.
            let max = logits.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
            let ts_mass: f32 = logits
                .iter()
                .enumerate()
                .filter(|&(id, _)| ts_allowed(id as u32))
                .map(|(_, &x)| (x - max).exp())
                .sum();
            let best_text = logits
                .iter()
                .enumerate()
                .filter(|&(id, _)| text_allowed(id as u32))
                .fold(f32::NEG_INFINITY, |m, (_, &x)| m.max(x));
            if ts_mass > (best_text - max).exp() {
                return argmax(&ts_allowed);
            }
        }
        argmax(&|id| text_allowed(id) || ts_allowed(id))
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

        (full_text, all_segments)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_audio_short() {
        let samples = vec![1.0; 100_000]; // ~6.25 s
        let chunks = WhisperModel::chunk_audio(&samples, 16_000);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].samples.len(), WHISPER_CHUNK_SAMPLES);
        assert_eq!(chunks[0].samples[0], 1.0);
        assert_eq!(chunks[0].samples[100_000], 0.0); // zero-padded
        assert_eq!(chunks[0].start_secs, 0.0);
        assert_eq!(chunks[0].duration_secs, 6.25);
    }

    #[test]
    fn test_chunk_audio_exact() {
        let samples = vec![1.0; WHISPER_CHUNK_SAMPLES];
        let chunks = WhisperModel::chunk_audio(&samples, 16_000);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].samples.len(), WHISPER_CHUNK_SAMPLES);
    }

    #[test]
    fn test_chunk_audio_empty() {
        let chunks = WhisperModel::chunk_audio(&[], 16_000);
        assert!(chunks.is_empty());
    }

    #[test]
    fn long_audio_is_cut_at_a_pause() {
        // 50 s of sound with a half-second pause at 27 s.
        let sr = 16_000;
        let mut samples = vec![0.5f32; 50 * sr];
        samples[27 * sr..27 * sr + sr / 2].fill(0.0);
        let chunks = WhisperModel::chunk_audio(&samples, sr as u32);

        assert_eq!(chunks.len(), 2);
        let cut = chunks[1].start_secs;
        assert!((27.0..27.5).contains(&cut), "cut at {cut}");
        assert_eq!(chunks[0].duration_secs, cut);
        assert!((chunks[1].duration_secs - (50.0 - cut)).abs() < 1e-3);
        // Every sample lands in exactly one window, in order.
        let joined: Vec<f32> = chunks
            .iter()
            .flat_map(|c| c.samples[..(c.duration_secs * sr as f32).round() as usize].to_vec())
            .collect();
        assert_eq!(joined, samples);
    }

    #[test]
    fn audio_without_a_pause_still_fits_its_windows() {
        let samples = vec![0.5f32; 95 * 16_000];
        let chunks = WhisperModel::chunk_audio(&samples, 16_000);
        let (last, full) = chunks.split_last().unwrap();
        assert!(
            full.iter()
                .all(|c| (25.0..=30.0).contains(&c.duration_secs))
        );
        assert!(last.duration_secs <= 30.0);
        let total: f32 = chunks.iter().map(|c| c.duration_secs).sum();
        assert!((total - 95.0).abs() < 1e-3);
    }

    const VOCAB: usize = 51_865;

    fn pick(logits: &ndarray::Array1<f32>, generated: &[u32], audio_end_secs: f32) -> u32 {
        let config = WhisperTranscriberConfig {
            timestamps: true,
            ..Default::default()
        };
        let audio_end = TIMESTAMP_BEGIN + (audio_end_secs / TIMESTAMP_RESOLUTION) as u32;
        WhisperModel::pick_token(&logits.view(), generated, &config, EOT_TOKEN, audio_end)
    }

    #[test]
    fn a_chunk_opens_with_a_timestamp_in_its_first_second() {
        let mut logits = ndarray::Array1::<f32>::zeros(VOCAB);
        logits[400] = 10.0; // text
        logits[EOT_TOKEN as usize] = 9.0;
        logits[(TIMESTAMP_BEGIN + 200) as usize] = 5.0; // 4 s: too late
        logits[(TIMESTAMP_BEGIN + 10) as usize] = 1.0;
        assert_eq!(pick(&logits, &[], 30.0), TIMESTAMP_BEGIN + 10);
    }

    #[test]
    fn no_segment_opens_in_the_padding() {
        let mut logits = ndarray::Array1::<f32>::zeros(VOCAB);
        // A segment closed at 11 s in an 11.2 s window: Whisper would open
        // another, but half a second is not left, so the chunk ends.
        let closed = [TIMESTAMP_BEGIN, 400, TIMESTAMP_BEGIN + 550];
        logits[(TIMESTAMP_BEGIN + 550) as usize] = 10.0;
        logits[EOT_TOKEN as usize] = 1.0;
        assert_eq!(pick(&logits, &closed, 11.2), EOT_TOKEN);
        // With audio left, it opens one.
        assert_eq!(pick(&logits, &closed, 20.0), TIMESTAMP_BEGIN + 550);
    }

    #[test]
    fn timestamps_together_likelier_than_any_text_close_the_segment() {
        let mut logits = ndarray::Array1::<f32>::zeros(VOCAB);
        let open = [TIMESTAMP_BEGIN, 400];
        logits.fill(-20.0);
        logits[401] = 3.0;
        // No single timestamp beats the text token, but forty of them do.
        for t in 100..140 {
            logits[(TIMESTAMP_BEGIN + t) as usize] = 1.0;
        }
        logits[(TIMESTAMP_BEGIN + 120) as usize] = 1.5;
        assert_eq!(pick(&logits, &open, 30.0), TIMESTAMP_BEGIN + 120);
        // A segment that has just opened takes text, never a timestamp.
        assert_eq!(pick(&logits, &[TIMESTAMP_BEGIN], 30.0), 401);
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
}
