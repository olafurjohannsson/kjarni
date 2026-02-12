pub mod cancellation;
pub mod sampling;
pub mod stream;

pub use cancellation::{CancellationError, CancellationHandle, CancellationToken};
pub use sampling::{
    apply_no_repeat_ngram, apply_no_repeat_ngram_inplace, apply_repetition_penalty,
    apply_repetition_penalty_inplace, apply_repetition_penalty_mut, get_top_k_from_log_probs,
    log_softmax_1d, min_p_filtering, sample_from_probs, sample_token, top_k_filtering,
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
    pub max_new_tokens: Option<usize>,
    pub max_length: usize,
    pub min_length: usize,
    pub repetition_penalty: f32,
    pub no_repeat_ngram_size: usize,
    pub add_bos_token: bool,
    pub strategy: DecodingStrategy,
    pub speculation: Option<SpeculationParams>,
}
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
            speculation: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpeculationParams {
    /// Number of tokens to speculate per iteration
    pub num_tokens: usize,
    
    /// Use probability-based acceptance to preserve exact target distribution.
    /// If false, uses greedy acceptance
    pub probabilistic: bool,
}

impl Default for SpeculationParams {
    fn default() -> Self {
        Self {
            num_tokens: 4,
            probabilistic: false,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct HFGenerationConfig {
    // Token IDs
    #[serde(default)]
    pub decoder_start_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub forced_bos_token_id: Option<u32>,
    #[serde(default)]
    pub forced_eos_token_id: Option<u32>,
    
    // Length controls
    #[serde(default)]
    pub max_length: Option<usize>,
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
    #[serde(default)]
    pub min_length: Option<usize>,
    #[serde(default)]
    pub min_new_tokens: Option<usize>,
    
    // Sampling params
    #[serde(default)]
    pub do_sample: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    
    // Beam search params
    #[serde(default)]
    pub num_beams: Option<usize>,
    #[serde(default)]
    pub num_beam_groups: Option<usize>,
    #[serde(default)]
    pub length_penalty: Option<f32>,
    #[serde(default)]
    pub early_stopping: Option<bool>,
    
    // Penalties
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub no_repeat_ngram_size: Option<usize>,
    #[serde(default)]
    pub encoder_no_repeat_ngram_size: Option<usize>,
    
    // Other
    #[serde(default)]
    pub diversity_penalty: Option<f32>,
    #[serde(default)]
    pub num_return_sequences: Option<usize>,
    
    // Metadata (ignored but parsed)
    #[serde(default)]
    pub _from_model_config: Option<bool>,
    #[serde(default)]
    pub transformers_version: Option<String>,
}

/// EOS token can be single value or array in HF configs
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

impl EosTokenId {
    pub fn primary(&self) -> u32 {
        match self {
            EosTokenId::Single(id) => *id,
            EosTokenId::Multiple(ids) => ids.first().copied().unwrap_or(1),
        }
    }
    
    pub fn all(&self) -> Vec<u32> {
        match self {
            EosTokenId::Single(id) => vec![*id],
            EosTokenId::Multiple(ids) => ids.clone(),
        }
    }
}

impl HFGenerationConfig {
    /// Load from generation_config.json
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        let config: Self = serde_json::from_reader(file)?;
        Ok(config)
    }
    
    /// Try to load from model directory, return default if not found
    pub fn load_or_default(model_dir: &std::path::Path) -> Self {
        let path = model_dir.join("generation_config.json");
        Self::load(&path).unwrap_or_default()
    }
    
    pub fn to_generation_config(&self, model_defaults: &ModelGenerationDefaults) -> GenerationConfig {
        let strategy = self.determine_strategy(model_defaults);
        
        GenerationConfig {
            max_length: self.max_length.unwrap_or(model_defaults.max_length),
            min_length: self.min_length.unwrap_or(0),
            max_new_tokens: self.max_new_tokens,
            no_repeat_ngram_size: self.no_repeat_ngram_size.unwrap_or(0),
            repetition_penalty: self.repetition_penalty.unwrap_or(1.0),
            add_bos_token: model_defaults.add_bos_token,
            strategy,
            speculation: None,
        }
    }
    
    fn determine_strategy(&self, defaults: &ModelGenerationDefaults) -> DecodingStrategy {
        let num_beams = self.num_beams.unwrap_or(defaults.num_beams);
        let do_sample = self.do_sample.unwrap_or(false);
        
        if num_beams > 1 {
            DecodingStrategy::BeamSearch(BeamSearchParams {
                num_beams,
                length_penalty: self.length_penalty.unwrap_or(1.0),
                early_stopping: self.early_stopping.unwrap_or(false),
            })
        } else if do_sample {
            DecodingStrategy::Sample(SamplingParams {
                temperature: self.temperature.unwrap_or(1.0),
                top_p: self.top_p,
                top_k: self.top_k,
                min_p: None,
            })
        } else {
            DecodingStrategy::Greedy
        }
    }
}

/// Model-specific defaults that HFGenerationConfig falls back to
#[derive(Debug, Clone)]
pub struct ModelGenerationDefaults {
    pub max_length: usize,
    pub num_beams: usize,
    pub add_bos_token: bool,
}

impl Default for ModelGenerationDefaults {
    fn default() -> Self {
        Self {
            max_length: 512,
            num_beams: 1,
            add_bos_token: false,
        }
    }
}

impl ModelGenerationDefaults {
    pub fn for_t5() -> Self {
        Self {
            max_length: 512,
            num_beams: 4,
            add_bos_token: false,
        }
    }
    
    pub fn for_llama() -> Self {
        Self {
            max_length: 4096,
            num_beams: 1,
            add_bos_token: true,
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
    #[serde(default)]
    pub decoder_start_token_id: Option<usize>,
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
            speculation: None,
        }
    }
}

#[cfg(test)]
mod common_tests {
    use super::*;

    #[test]
    fn test_default_generation_config() {
        let config = GenerationConfig::default();

        assert_eq!(config.max_new_tokens, Some(50));
        assert_eq!(config.max_length, 100);
        assert!(config.add_bos_token);
        assert_eq!(config.repetition_penalty, 1.0);

        match config.strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 0.7);
                assert_eq!(params.top_k, Some(50));
                assert_eq!(params.top_p, Some(0.9));
            }
            _ => panic!("Default strategy should be Sampling"),
        }
    }

    #[test]
    fn test_hf_defaults_greedy() {
        let json = r#"{
        "do_sample": false,
        "max_length": 2048
    }"#;

        let hf_defaults = HFGenerationDefaults::from_json(json).unwrap();
        let config = hf_defaults.into_generation_config(4096);

        match config.strategy {
            DecodingStrategy::Greedy => {} // Pass
            _ => panic!("Should be Greedy"),
        }
        assert_eq!(config.max_length, 2048);
    }

    #[test]
    fn test_hf_defaults_sampling() {
        let json = r#"{
        "do_sample": true,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.1
    }"#;

        let hf_defaults = HFGenerationDefaults::from_json(json).unwrap();
        let config = hf_defaults.into_generation_config(1024);

        match config.strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 0.8);
                assert_eq!(params.top_p, Some(0.95));
                assert_eq!(params.top_k, Some(40));
            }
            _ => panic!("Should be Sampling"),
        }
        assert_eq!(config.repetition_penalty, 1.1);
        // Should fallback to max_seq_len if max_length not in JSON
        assert_eq!(config.max_length, 1024);
    }

    #[test]
    fn test_hf_defaults_empty_json() {
        // Test robustness against missing fields
        let json = "{}";
        let hf_defaults = HFGenerationDefaults::from_json(json).unwrap();

        assert!(!hf_defaults.do_sample);
        assert_eq!(hf_defaults.temperature, 1.0); // Default via serde

        let config = hf_defaults.into_generation_config(512);
        assert_eq!(config.max_length, 512);
    }

    #[test]
    fn test_hf_defaults_invalid_json() {
        let json = "{ invalid_json }";
        assert!(HFGenerationDefaults::from_json(json).is_err());
    }
}
