//! Text generation utilities

use anyhow::Result;
use ndarray::Array1;
use rand::Rng;
use std::collections::{HashMap, HashSet};

#[cfg(not(target_arch = "wasm32"))]
use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
use crate::tokenizer::wasm::BPETokenizer as Tokenizer;

use edgetransformers::models::base::{
    DecodingStrategy, GenerationConfig, SamplingParams,
};

pub fn apply_repetition_penalty(
    mut logits: Array1<f32>,
    generated_ids: &[u32],
    penalty: f32,
) -> Array1<f32> {
    if penalty == 1.0 {
        return logits;
    }
    for &id in generated_ids {
        let idx = id as usize;
        if logits[idx] < 0.0 {
            logits[idx] *= penalty;
        } else {
            logits[idx] /= penalty;
        }
    }
    logits
}

/// Efficient no-repeat n-gram blocking for generation.
///
/// This prevents generating any n-gram that has already appeared in the sequence.
/// Adapted from HuggingFace’s implementation logic.
///
/// # Arguments
/// * `logits` — the next-token logits (modified in place)
/// * `tokens` — sequence of already generated tokens
/// * `ngram_size` — size of the n-gram window
///
/// # Returns
/// Modified logits with -inf applied to blocked tokens.
pub fn apply_no_repeat_ngram(
    mut logits: Array1<f32>,
    tokens: &[u32],
    ngram_size: usize,
) -> Array1<f32> {
    if ngram_size == 0 || tokens.len() < ngram_size {
        return logits;
    }

    // Map from ngram_prefix → set of next tokens that follow that prefix.
    let mut ngram_map: HashMap<Vec<u32>, HashSet<u32>> = HashMap::new();

    // Build the prefix map efficiently.
    for window in tokens.windows(ngram_size) {
        let prefix = &window[..ngram_size - 1];
        let next_token = window[ngram_size - 1];
        ngram_map
            .entry(prefix.to_vec())
            .or_default()
            .insert(next_token);
    }

    // The prefix of the last (n-1) tokens
    let current_prefix = &tokens[tokens.len() - (ngram_size - 1)..];

    // If we’ve seen this prefix before, block all tokens that previously followed it
    if let Some(blocked_tokens) = ngram_map.get(current_prefix) {
        for &t in blocked_tokens {
            if (t as usize) < logits.len() {
                logits[t as usize] = f32::NEG_INFINITY;
            }
        }
    }

    logits
}
