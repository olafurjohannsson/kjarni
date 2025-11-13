//! Text generation utilities


use anyhow::Result;
use ndarray::{Array1};
use rand::Rng;
use std::collections::{HashSet, HashMap};


#[cfg(not(target_arch = "wasm32"))]
use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
use crate::tokenizer::wasm::BPETokenizer as Tokenizer;

use edgetransformers::models::base::{GenerationConfig, SamplingStrategy};





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
pub fn apply_repetition_penalty2(
    mut logits: Array1<f32>,
    generated_ids: &[u32],
    penalty: f32,
) -> Array1<f32> {
    if penalty == 1.0 {
        return logits;
    }
    
    // Count occurrences of each token
    let mut token_counts: HashMap<u32, usize> = HashMap::new();
    for &id in generated_ids {
        *token_counts.entry(id).or_insert(0) += 1;
    }
    
    for (&id, &count) in &token_counts {
        let idx = id as usize;
        if idx >= logits.len() {
            continue;
        }
        
        // Apply penalty once per unique token (not per occurrence)
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

/// Sample a token from logits
pub fn sample_token(mut logits: Array1<f32>, config: &GenerationConfig) -> Result<u32> {
    let mut rng = rand::thread_rng();

    match config.sampling_strategy {
        SamplingStrategy::Greedy => {
            // Argmax
            Ok(logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap())
        }

        SamplingStrategy::Temperature => {
            // Apply temperature
            logits /= config.temperature;

            // Softmax and sample
            let probs = softmax_1d(&logits);
            sample_from_probs(&probs, &mut rng)
        }

        SamplingStrategy::TopK => {
            // Apply top-k filtering
            if let Some(k) = config.top_k {
                logits = top_k_filtering(logits, k);
            }

            // Apply temperature
            logits /= config.temperature;

            // Softmax and sample
            let probs = softmax_1d(&logits);
            sample_from_probs(&probs, &mut rng)
        }

        SamplingStrategy::TopP => {
            // Apply top-p (nucleus) filtering
            if let Some(p) = config.top_p {
                logits = top_p_filtering(logits, p);
            }

            // Apply temperature
            logits /= config.temperature;

            // Softmax and sample
            let probs = softmax_1d(&logits);
            sample_from_probs(&probs, &mut rng)
        }

        SamplingStrategy::TopKTopP => {
            // Apply both top-k and top-p
            if let Some(k) = config.top_k {
                logits = top_k_filtering(logits, k);
            }
            if let Some(p) = config.top_p {
                logits = top_p_filtering(logits, p);
            }

            // Apply temperature
            logits /= config.temperature;

            // Softmax and sample
            let probs = softmax_1d(&logits);
            sample_from_probs(&probs, &mut rng)
        }

        SamplingStrategy::BeamSearch => {
            anyhow::bail!("Invalid configuration, beamsearch is not per-token sampling")
        }
    }
}

/// Apply softmax to 1D array
fn softmax_1d(logits: &Array1<f32>) -> Array1<f32> {
    let max_val = logits.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let exp_logits = (logits - max_val).mapv(f32::exp);
    let sum_exp = exp_logits.sum();
    exp_logits / sum_exp
}

/// Top-k filtering
fn top_k_filtering(mut logits: Array1<f32>, k: usize) -> Array1<f32> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

    // Set all but top-k to -inf
    for &idx in &indices[k..] {
        logits[idx] = f32::NEG_INFINITY;
    }

    logits
}

/// Top-p (nucleus) filtering
fn top_p_filtering(mut logits: Array1<f32>, p: f32) -> Array1<f32> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

    let probs = softmax_1d(&logits);
    let mut cumulative = 0.0;
    let mut cutoff_idx = 0;

    for (i, &idx) in indices.iter().enumerate() {
        cumulative += probs[idx];
        if cumulative > p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Set all but nucleus to -inf
    for &idx in &indices[cutoff_idx..] {
        logits[idx] = f32::NEG_INFINITY;
    }

    logits
}

/// Sample from probability distribution
fn sample_from_probs(probs: &Array1<f32>, rng: &mut impl Rng) -> Result<u32> {
    let uniform: f32 = rng.r#gen();
    let mut cumulative = 0.0;

    for (idx, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if cumulative >= uniform {
            return Ok(idx as u32);
        }
    }

    // Fallback to last index
    Ok((probs.len() - 1) as u32)
}
