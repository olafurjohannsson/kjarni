


pub mod sampling;
pub mod stream;

pub use sampling::*;
pub use stream::*;


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