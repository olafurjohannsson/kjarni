use anyhow::{Result, anyhow};
use async_stream::try_stream;

use edgetransformers::models::{DecoderLanguageModel, project_to_vocab};
use edgetransformers::prelude::*;
use futures_core::stream::Stream;
use futures_util::TryStreamExt;
use log::{debug, error};
use ndarray::{Array1, Array2, s};
use rand::Rng;

use common::{StreamedToken, TokenType};
pub mod decoder;
pub mod common;
// pub mod generator;
// pub mod seq2seq;
// pub mod seq2seq2;
pub mod encoder_decoder;

pub use edgetransformers::models::base::{AutoregressiveLoop, DecodingStrategy, GenerationConfig};


// pub use generator::{DecoderGenerationBackend};
pub use edgetransformers::decoder::DecoderGenerationBackend;


/// A generic, model-agnostic text generator for autoregressive decoding.
pub struct Generator {
    pub model: Box<dyn DecoderLanguageModel>,
}

impl Generator {
    pub fn new(model: Box<dyn DecoderLanguageModel>) -> Self {
        Self { model }
    }

    /// Generates a complete string of text, collecting all streamed tokens.
    pub async fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        let stream = self.generate_stream(prompt, config).await?;
        let results: Vec<StreamedToken> = stream.try_collect().await?;

        let iter: Vec<&str> = results.iter().map(|v| v.text.as_str()).collect();

        Ok(iter.join(""))
    }

    pub async fn generate_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<impl Stream<Item = Result<StreamedToken>>> {
        debug!("Prompt {}", prompt);
        let tokenizer = self.model.tokenizer();
        let mut tokens = tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!(e))?
            .get_ids()
            .to_vec();

        if config.add_bos_token {
            if let Some(bos_token_id) = self.model.bos_token_id() {
                if tokens.is_empty() || tokens[0] != bos_token_id {
                    tokens.insert(0, bos_token_id);
                }
            } else {
                error!("BOS token configured to be added but not found in model");
            }
        }
        println!("[NEW] Initial Token IDs: {:?}", &tokens);
        let prompt_tokens = tokens.clone();

        let prompt_len = tokens.len();
        let batch_size = 1;
        let max_len = if let Some(new_tokens) = config.max_new_tokens {
            prompt_len + new_tokens
        } else {
            config.max_length
        };
        let mut full_attention_mask = Array2::zeros((batch_size, max_len));
        let mut cache = self.model.new_cache(batch_size, max_len, 0)?;

        let eos_token_id = self.model.eos_token_id();
        let model = &self.model;

        let mut next_token_logits: Array1<f32>;

        if prompt_len > 0 {
            let prompt_ids = Array2::from_shape_vec(
                (batch_size, prompt_len),
                tokens.iter().map(|&t| t ).collect(),
            )?;
            full_attention_mask
                .slice_mut(s![.., 0..prompt_len])
                .fill(1.0);
            let mask_for_priming = match self.model.device() {
                Device::Cpu => full_attention_mask.slice(s![.., 0..prompt_len]).to_owned(),
                Device::Wgpu => full_attention_mask.clone(),
            };
            match self.model.autoregressive_loop() {
                AutoregressiveLoop::Pipelined => {
                    // Efficient one-pass logic for Llama
                    let decoder_output = model
                        .decoder()
                        .forward(&prompt_ids, &mask_for_priming, Some(cache.as_mut()))
                        .await?;
                    let logits_3d = model.project_to_logits(&decoder_output.last_hidden_state)?;
                    next_token_logits = logits_3d.slice(s![0, -1, ..]).to_owned();
                }
                AutoregressiveLoop::Legacy => {
                    // Inefficient two-pass logic for GPT-2 parity
                    // 1. Prefill to fill the cache, discard output.
                    
                    model
                        .decoder()
                        .forward(&prompt_ids, &mask_for_priming, Some(cache.as_mut()))
                        .await?;
                    // 2. We will create a dummy logits tensor. The real logits will be calculated
                    //    on the first iteration of the loop below.
                    next_token_logits = Array1::zeros(model.config().vocab_size());
                }
            }
        } else {
            if let Some(bos_token_id) = model.bos_token_id() {
                // A BOS token exists, so we can start generation from it.
                let input_ids = Array2::from_elem((1, 1), bos_token_id);
                full_attention_mask[[0, 0]] = 1.0;
                let mask = full_attention_mask.slice(s![.., 0..1]).to_owned();

                let decoder_output = model
                    .decoder()
                    .forward(&input_ids, &mask, Some(cache.as_mut()))
                    .await?;

                let logits_3d = model.project_to_logits(&decoder_output.last_hidden_state)?;
                next_token_logits = logits_3d.slice(s![0, 0, ..]).to_owned();
                tokens.push(bos_token_id);
            } else {
                return Err(anyhow!(
                    "Cannot generate from an empty prompt because the model has no BOS token."
                ));
            }
        }
        Ok(try_stream! {

            for &token_id in &prompt_tokens {
                if let Some(bos_token_id) = self.model.bos_token_id() {
                    if token_id == bos_token_id {
                        continue
                    }
                }

                let decoded_prompt_token = tokenizer.decode(&[token_id], false).map_err(|e| anyhow!(e))?;
                yield StreamedToken {
                    text: decoded_prompt_token,
                    id: token_id,
                    token_type: TokenType::Prompt,
                };
            }
            match self.model.autoregressive_loop() {
                AutoregressiveLoop::Pipelined => {
                    loop {
                        if tokens.len() >= max_len { break; }

                        // Apply penalties and sample the FIRST token
                        let processed_logits = apply_repetition_penalty(next_token_logits, &tokens, config.repetition_penalty);
                        let next_token = sample_token(processed_logits, &config.strategy)?;

                        tokens.push(next_token);
                        // Yield the newly decoded token.
                        let decoded_token = tokenizer.decode(&[next_token], false).map_err(|e| anyhow!(e))?;

                        yield StreamedToken {
                            text: decoded_token,
                            id: next_token,
                            token_type: TokenType::Generated,
                        };

                        if eos_token_id == Some(next_token) {
                            break;
                        }

                        let current_len = tokens.len();
                        let input_ids = Array2::from_shape_vec((1, 1), vec![next_token])?;
                        full_attention_mask[[0, current_len - 1]] = 1.0;


                        let mask_to_use = match model.device() {
                            Device::Cpu => full_attention_mask.slice(s![.., 0..current_len]).to_owned(),
                            Device::Wgpu => full_attention_mask.clone(),
                        };

                        let decoder_output = model.decoder().forward(&input_ids, &mask_to_use, Some(cache.as_mut())).await?;
                        let logits_3d = model.project_to_logits(&decoder_output.last_hidden_state)?;
                        next_token_logits = logits_3d.slice(s![0, 0, ..]).to_owned();
                    }
                }
                AutoregressiveLoop::Legacy => {
                    loop {
                        if tokens.len() >= max_len { break; }

                        let last_token = *tokens.last().unwrap();
                        let input_ids = Array2::from_elem((1, 1), last_token);

                        let current_len = tokens.len();
                        full_attention_mask[[0, current_len]] = 1.0;
                        let mask_to_use = match model.device() {
                            Device::Cpu => full_attention_mask.slice(s![.., 0..current_len + 1]).to_owned(),
                            Device::Wgpu => full_attention_mask.clone(),
                        };

                        // 3. Run the forward pass to get logits.
                        let decoder_output = model.decoder().forward(&input_ids, &mask_to_use, Some(cache.as_mut())).await?;
                        let logits_3d = model.project_to_logits(&decoder_output.last_hidden_state)?;
                        let mut next_token_logits = logits_3d.slice(s![0, 0, ..]).to_owned();

                        // 4. Apply penalty (this was after logits in the old code).
                        let processed_logits = apply_repetition_penalty(next_token_logits, &tokens, config.repetition_penalty);

                        // 5. Sample the *next* token.
                        let next_token = sample_token(processed_logits, &config.strategy)?;
                        tokens.push(next_token);

                        // 6. Yield the token.
                        let decoded_token = tokenizer.decode(&[next_token], false).map_err(|e| anyhow!(e))?;
                        yield StreamedToken {
                            text: decoded_token,
                            id: next_token,
                            token_type: TokenType::Generated,
                        };

                        // 7. Check for EOS.
                        if eos_token_id == Some(next_token) {
                            break;
                        }
                    }
                }

            }
        })
    }
}
fn apply_repetition_penalty(
    mut logits: Array1<f32>,
    past_tokens: &[u32],
    penalty: f32,
) -> Array1<f32> {
    if penalty == 1.0 {
        return logits;
    }
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

/// Sample a token from logits
pub fn sample_token(mut logits: Array1<f32>, strategy: &DecodingStrategy) -> Result<u32> {

    match strategy {
        DecodingStrategy::Greedy => {
            // Argmax
            Ok(logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap())
        }
        DecodingStrategy::Sample(params) => {
            let mut rng = rand::thread_rng();
            // 1. Apply Top-K filtering if specified.
            if let Some(k) = params.top_k {
                logits = top_k_filtering(logits, k);
            }

            // 2. Apply Top-P (nucleus) filtering if specified.
            //    This can be applied after Top-K.
            if let Some(p) = params.top_p {
                logits = top_p_filtering(logits, p);
            }

            // 3. Apply temperature scaling.
            logits /= params.temperature;

            // 4. Convert logits to probabilities and sample from the distribution.
            let probs = softmax_1d(&logits);
            sample_from_probs(&probs, &mut rng)
        }

        DecodingStrategy::BeamSearch(_) => {
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
