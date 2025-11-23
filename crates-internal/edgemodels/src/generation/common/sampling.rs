use anyhow::{anyhow, Result};
use async_stream::try_stream;
use futures_core::stream::Stream;
use futures_util::TryStreamExt;
use log::{debug, error};
use ndarray::Array1;
use rand::Rng;
pub use edgetransformers::models::base::{DecodingStrategy, GenerationConfig};




pub fn apply_repetition_penalty(
    mut logits: Array1<f32>,
    past_tokens: &[u32],
    penalty: f32,
) -> Array1<f32> {
    if penalty == 1.0 { return logits; }
    for &token_id in past_tokens {
        let score = logits[token_id as usize];
        if score > 0.0 {
            logits[token_id as usize] = score / penalty;
        } else {
            logits[token_id as usize] = score * penalty;
        }
    }
    logits
}

pub fn sample_token(mut logits: Array1<f32>, strategy: &DecodingStrategy) -> Result<u32> {
    match strategy {
        DecodingStrategy::Greedy => Ok(logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap()),
        DecodingStrategy::Sample(params) => {
            if let Some(k) = params.top_k { logits = top_k_filtering(logits, k); }
            if let Some(p) = params.top_p { logits = top_p_filtering(logits, p); }
            logits /= params.temperature;
            let probs = softmax_1d(&logits);
            sample_from_probs(&probs)
        }
        DecodingStrategy::BeamSearch(_) => {
            anyhow::bail!("Beam search is not supported in this generator.")
        }
    }
}

pub fn softmax_1d(logits: &Array1<f32>) -> Array1<f32> {
    let max_val = logits.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let exp_logits = (logits - max_val).mapv(f32::exp);
    let e = exp_logits.sum();
    exp_logits / e
}

pub fn top_k_filtering(mut logits: Array1<f32>, k: usize) -> Array1<f32> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    for &idx in &indices[k..] {
        logits[idx] = f32::NEG_INFINITY;
    }
    logits
}

pub fn top_p_filtering(mut logits: Array1<f32>, p: f32) -> Array1<f32> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let probs = softmax_1d(&logits);
    let mut cumulative = 0.0;
    for (i, &idx) in indices.iter().enumerate() {
        cumulative += probs[idx];
        if cumulative > p {
            for &invalid_idx in &indices[i + 1..] {
                logits[invalid_idx] = f32::NEG_INFINITY;
            }
            break;
        }
    }
    logits
}

pub fn sample_from_probs(probs: &Array1<f32>) -> Result<u32> {
    let mut rng = rand::thread_rng();
    let uniform: f32 = rng.r#gen();
    let mut cumulative = 0.0;
    for (idx, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if cumulative >= uniform {
            return Ok(idx as u32);
        }
    }
    Ok((probs.len() - 1) as u32)
}



pub fn get_top_k_from_log_probs(log_probs: &Array1<f32>, k: usize) -> Vec<(u32, f32)> {
    let mut indexed_log_probs: Vec<(usize, f32)> = log_probs
        .iter()
        .enumerate()
        .map(|(i, &lp)| (i, lp))
        .collect();
    indexed_log_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed_log_probs.truncate(k);
    indexed_log_probs
        .into_iter()
        .map(|(i, lp)| (i as u32, lp))
        .collect()
}

pub fn log_softmax_1d(logits: &Array1<f32>) -> Array1<f32> {
    let max_val = logits.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let scaled_logits = logits - max_val;
    let exp_sum = scaled_logits.mapv(f32::exp).sum();
    scaled_logits - exp_sum.ln()
}

pub fn apply_repetition_penalty_mut(logits: &mut Array1<f32>, tokens: &[u32], penalty: f32) {
    for &token in tokens {
        let idx = token as usize;
        if idx < logits.len() {
            let score = logits[idx];
            if score < 0.0 {
                logits[idx] = score * penalty;
            } else {
                logits[idx] = score / penalty;
            }
        }
    }
}

pub fn apply_no_repeat_ngram(logits: &mut Array1<f32>, tokens: &[u32], ngram_size: usize) {
    let n = ngram_size;
    // We can't form a prefix of length n-1, so we can't complete an n-gram.
    if tokens.len() < n - 1 {
        return;
    }

    // The sequence of tokens that would form the start of a new n-gram.
    let current_prefix = &tokens[tokens.len() - (n - 1)..];

    // Iterate through all historical n-grams in the generated sequence.
    for window in tokens.windows(n) {
        // Check if a historical n-gram starts with the same prefix.
        if &window[..n - 1] == current_prefix {
            // If it does, ban the token that completed that n-gram.
            let banned_token = window[n - 1] as usize;
            if banned_token < logits.len() {
                logits[banned_token] = f32::NEG_INFINITY;
            }
        }
    }
}
