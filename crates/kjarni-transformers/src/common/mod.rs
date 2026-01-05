pub mod cancellation;
pub mod sampling;
pub mod stream;

pub use cancellation::{CancellationError, CancellationHandle, CancellationToken};
pub use sampling::{
    apply_no_repeat_ngram, apply_no_repeat_ngram_inplace, apply_repetition_penalty,
    apply_repetition_penalty_inplace, apply_repetition_penalty_mut, get_top_k_from_log_probs,
    log_softmax_1d, min_p_filtering, sample_from_probs, sample_token, softmax_1d, top_k_filtering,
    top_p_filtering,
};
pub use stream::{StreamedToken, TokenType};

use serde::Deserialize;
use serde_json;

/// Parameters for sampling-based decoding (Top-K, Top-P, Temperature).
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: Some(50),
            top_p: Some(0.9),
            min_p: Some(0.1),
        }
    }
}

/// Parameters for beam search decoding.
#[derive(Clone, Debug)]
pub struct BeamSearchParams {
    pub num_beams: usize,
    pub length_penalty: f32,
    pub early_stopping: bool,
}

impl Default for BeamSearchParams {
    fn default() -> Self {
        Self {
            num_beams: 4,
            length_penalty: 1.0,
            early_stopping: true,
        }
    }
}

/// The user-facing decoding algorithm and its specific parameters.
#[derive(Clone, Debug)]
pub enum DecodingStrategy {
    /// Select the most likely token (argmax).
    Greedy,
    /// Sample from the distribution using various techniques.
    Sample(SamplingParams),
    /// Explore multiple hypotheses to find the most likely sequence.
    BeamSearch(BeamSearchParams),
}

/// The main, unified configuration struct for text generation.
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    // --- Common Parameters for all strategies ---
    pub max_new_tokens: Option<usize>,
    pub max_length: usize,
    pub min_length: usize,
    pub repetition_penalty: f32,
    pub no_repeat_ngram_size: usize,
    pub add_bos_token: bool,
    pub strategy: DecodingStrategy,
}
/// A sensible default for decoder-only models (like GPT-2 or Llama).
impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: Some(50),
            max_length: 100,
            min_length: 0,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: 0,
            add_bos_token: true,
            strategy: DecodingStrategy::Sample(SamplingParams {
                temperature: 0.7,
                top_k: Some(50),
                top_p: Some(0.9),
                min_p: Some(0.1),
            }),
        }
    }
}

// For Llama-style models with generation_config.json
#[derive(Debug, Clone, Deserialize)]
pub struct HFGenerationDefaults {
    #[serde(default)]
    pub do_sample: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
    #[serde(default)]
    pub max_length: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
}

fn default_temperature() -> f32 {
    1.0
}

impl HFGenerationDefaults {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn into_generation_config(self, max_seq_len: usize) -> GenerationConfig {
        let strategy = if self.do_sample {
            DecodingStrategy::Sample(SamplingParams {
                temperature: self.temperature,
                top_k: self.top_k,
                top_p: self.top_p,
                min_p: None,
            })
        } else {
            DecodingStrategy::Greedy
        };

        GenerationConfig {
            max_new_tokens: self.max_new_tokens,
            max_length: self.max_length.unwrap_or(max_seq_len),
            min_length: 0,
            repetition_penalty: self.repetition_penalty.unwrap_or(1.0),
            no_repeat_ngram_size: 0,
            add_bos_token: true,
            strategy,
        }
    }
}
