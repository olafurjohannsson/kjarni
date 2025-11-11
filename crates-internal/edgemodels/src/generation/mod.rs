use anyhow::{anyhow, Result};
use async_stream::try_stream;
use edgetransformers::models::base::GenerationConfig;
use edgetransformers::models::{project_to_vocab, DecoderLanguageModel};
use edgetransformers::prelude::*;
use futures_core::stream::Stream;
use futures_util::StreamExt;
use futures_util::TryStreamExt;
use ndarray::{s, Array1, Array2};


/// A generic, model-agnostic text generator for autoregressive decoding.
///
/// This struct encapsulates the complex logic of token-by-token generation,
/// including KV cache management, attention mask handling, and token sampling.
///
/// It operates on any model that implements the `DecoderLanguageModel` trait,
/// making it compatible with GPT-2, Llama, and any future decoder models.
pub struct Generator {
    model: Box<dyn DecoderLanguageModel>,
}

impl Generator {
    /// Creates a new generator from any compatible decoder model.
    pub fn new(model: Box<dyn DecoderLanguageModel>) -> Self {
        Self { model }
    }

    /// Generates a complete string of text, collecting all streamed tokens.
    ///
    /// This is a convenience wrapper around `generate_stream`.
    pub async fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        
        let stream = self.generate_stream(prompt, config).await?;
        // Collect all the strings from the stream and join them.
        let results: Vec<String> = stream.try_collect().await?;
        Ok(results.join(""))
    }

    /// Generates text token-by-token as an asynchronous stream.
    ///
    /// This is the primary method for generation, allowing for real-time output.
    pub async fn generate_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<impl Stream<Item = Result<String>>> {
        // --- 1. Initialization ---
        let tokenizer = self.model.tokenizer();
        let mut tokens = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!(e))?
            .get_ids()
            .to_vec();
        let prompt_len = tokens.len();
        let batch_size = 1;

        let max_len = if let Some(new_tokens) = config.max_new_tokens {
            prompt_len + new_tokens
        } else {
            config.max_length
        };

        // This physical mask buffer is used by both CPU and GPU backends.
        let mut full_attention_mask = Array2::zeros((batch_size, max_len));

        // --- 2. KV Cache Setup ---
        let mut cache: Box<dyn Cache> = match self.model.device() {
            Device::Cpu => {
                let kv_dim = self.model.config().kv_dim();
                Box::new(CpuKVCache::new(
                    self.model.num_layers(),
                    batch_size,
                    max_len,
                    kv_dim,
                ))
            }
            Device::Wgpu => {
                let context = self
                    .model
                    .context()
                    .ok_or_else(|| anyhow!("GPU model is missing its WgpuContext"))?;
                let head_dim = self.model.hidden_size() / self.model.num_heads();
                Box::new(GpuKVCache::new(
                    &context,
                    self.model.num_layers(),
                    batch_size,
                    self.model.config().num_key_value_heads(),
                    head_dim,
                    max_len,
                )?)
            }
        };

        // --- 3. Priming Pass (Process the prompt) ---
        if prompt_len > 0 {
            let prompt_ids = Array2::from_shape_vec(
                (batch_size, prompt_len),
                tokens.iter().map(|&t| t as f32).collect(),
            )?;
            
            // Unmask the prompt section of the physical buffer.
            full_attention_mask.slice_mut(s![.., 0..prompt_len]).fill(1.0);

            // The mask passed to the forward function depends on the backend.
            let mask_for_priming = match self.model.device() {
                // CPU needs a mask that matches the input sequence length.
                Device::Cpu => full_attention_mask.slice(s![.., 0..prompt_len]).to_owned(),
                // GPU needs the full physical mask to know where to write in the cache.
                Device::Wgpu => full_attention_mask.clone(),
            };

            self.model
                .decoder()
                .forward(&prompt_ids, &mask_for_priming, Some(cache.as_mut()))
                .await?;
        }

        // --- 4. Autoregressive Generation Stream ---
        let eos_token_id = self.model.eos_token_id();
        let model = &self.model;

        Ok(try_stream! {
            for _ in 0..config.max_new_tokens.unwrap_or(max_len) {
                // Check stopping conditions
                if tokens.len() >= max_len { break; }

                let current_len = tokens.len();
                let last_token = *tokens.last().unwrap_or(&model.bos_token_id().unwrap_or(0));
                let input_ids = Array2::from_shape_vec((1, 1), vec![last_token as f32])?;

                // Unmask the position for the token we are about to generate.
                full_attention_mask[[0, current_len]] = 1.0;

                // Create the appropriate mask for this generation step.
                let mask_for_generation = match model.device() {
                    Device::Cpu => full_attention_mask.slice(s![.., 0..current_len + 1]).to_owned(),
                    Device::Wgpu => full_attention_mask.clone(),
                };

                // --- Forward pass for a single token ---
                let decoder_output = model
                    .decoder()
                    .forward(&input_ids, &mask_for_generation, Some(cache.as_mut()))
                    .await?;

                // --- Sampling ---
                let logits_3d = project_to_vocab(&decoder_output.last_hidden_state, model.lm_head())?;
                let mut next_token_logits = logits_3d.slice(s![0, 0, ..]).to_owned();

                next_token_logits = apply_repetition_penalty(next_token_logits, &tokens, config.repetition_penalty);
                let next_token = sample_token(next_token_logits, config)?;

                // --- Yield and Update State ---
                tokens.push(next_token);
                let decoded_token = tokenizer.decode(&[next_token], false).map_err(|e| anyhow!(e))?;
                yield decoded_token;

                if Some(next_token) == eos_token_id {
                    break;
                }
            }
        })
    }
}

/// Helper: Apply repetition penalty to logits
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

/// Helper: Sample a token from logits based on the generation config
fn sample_token(logits: Array1<f32>, config: &GenerationConfig) -> Result<u32> {
    // TODO: Implement the full sampling logic (Top-K, Top-P, Temperature)
    // For now, we'll use simple greedy sampling.
    let max_idx = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .ok_or_else(|| anyhow!("Cannot sample from empty logits"))?;
    Ok(max_idx as u32)
}