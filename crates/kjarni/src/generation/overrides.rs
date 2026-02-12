//! User- and runtime-provided generation parameter overrides.
//!
//! These do NOT represent a full generation configuration.
//! They are merged with model defaults at resolution time.

use serde::{Deserialize, Serialize};

/// User- or runtime-provided overrides for text generation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerationOverrides {
    /// Sampling temperature (0.0 = deterministic, higher = more random).
    pub temperature: Option<f32>,

    /// Limit sampling to top K tokens.
    pub top_k: Option<usize>,

    /// Nucleus sampling: consider tokens with cumulative probability <= top_p.
    pub top_p: Option<f32>,

    /// Minimum probability threshold for sampling.
    pub min_p: Option<f32>,

    /// Penalty for repeating tokens (1.0 = no penalty).
    pub repetition_penalty: Option<f32>,

    /// Prevent repeating n-grams of this size.
    pub no_repeat_ngram_size: Option<usize>,

    /// Maximum number of new tokens to generate.
    pub max_new_tokens: Option<usize>,

    /// Sampling vs deterministic decoding.
    ///
    /// - `Some(false)` => Force greedy decoding
    /// - `Some(true)` => Force sampling
    /// - `None` => Keep model default
    pub do_sample: Option<bool>,

    /// Number of beams for beam search (> 1 enables beam search).
    pub num_beams: Option<usize>,

    /// Length penalty for beam search (< 1 favors shorter, > 1 favors longer).
    pub length_penalty: Option<f32>,
}

impl GenerationOverrides {
    /// Create overrides for greedy decoding.
    pub fn greedy() -> Self {
        Self {
            do_sample: Some(false),
            temperature: Some(0.0),
            ..Default::default()
        }
    }

    /// Create overrides for creative generation.
    pub fn creative() -> Self {
        Self {
            temperature: Some(0.9),
            top_p: Some(0.95),
            top_k: Some(50),
            ..Default::default()
        }
    }

    /// Create overrides for precise/factual generation.
    pub fn precise() -> Self {
        Self {
            temperature: Some(0.3),
            top_p: Some(0.9),
            repetition_penalty: Some(1.1),
            ..Default::default()
        }
    }

    /// Check if any overrides are set.
    pub fn is_empty(&self) -> bool {
        self.temperature.is_none()
            && self.top_k.is_none()
            && self.top_p.is_none()
            && self.min_p.is_none()
            && self.repetition_penalty.is_none()
            && self.no_repeat_ngram_size.is_none()
            && self.max_new_tokens.is_none()
            && self.do_sample.is_none()
            && self.num_beams.is_none()
            && self.length_penalty.is_none()
    }
}

