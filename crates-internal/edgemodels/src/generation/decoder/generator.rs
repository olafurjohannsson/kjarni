use anyhow::{anyhow, Result};
use async_stream::try_stream;
use futures_core::stream::Stream;
use futures_util::TryStreamExt;
use log::{debug, error};
use ndarray::Array1;
use rand::Rng;

use edgetransformers::models::DecoderLanguageModel;
use edgetransformers::prelude::*;

// use super::{DecoderGenerationBackend, StreamedToken, TokenType}; // Assuming backend trait is in the same module

use crate::generation::common::{StreamedToken, TokenType};
use crate::generation::common::sampling::{GenerationConfig, apply_repetition_penalty, sample_token};
use crate::generation::generator::DecoderGenerationBackend;


/// A generic, model-agnostic text generator for autoregressive decoding.
/// It is generic over a `DecoderGenerationBackend` to support different hardware.
pub struct Generator<B: DecoderGenerationBackend> {
    pub model: Box<dyn DecoderLanguageModel>,
    pub backend: B,
}

impl<B: DecoderGenerationBackend> Generator<B> {
    /// Creates a new Generator with a specific model and backend.
    pub fn new(model: Box<dyn DecoderLanguageModel>, backend: B) -> Self {
        Self { model, backend }
    }

    /// Generates a complete string of text, collecting all streamed tokens.
    pub async fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        let stream = self.generate_stream(prompt, config).await?;
        let results: Vec<StreamedToken> = stream.try_collect().await?;
        let iter: Vec<&str> = results.iter().map(|v| v.text.as_str()).collect();
        Ok(iter.join(""))
    }

    /// Generates a stream of tokens based on a prompt and generation config.
    pub async fn generate_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<impl Stream<Item = Result<StreamedToken>>> {
        debug!("Prompt: {}", prompt);
        let tokenizer = self.model.tokenizer();
        let mut tokens = tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!(e))?
            .get_ids()
            .to_vec();

        // --- Handle Beginning-Of-Sentence (BOS) token ---
        if config.add_bos_token {
            if let Some(bos_token_id) = self.model.bos_token_id() {
                if tokens.is_empty() || tokens[0] != bos_token_id {
                    tokens.insert(0, bos_token_id);
                }
            } else {
                error!("BOS token configured to be added but not found in model");
            }
        }

        let prompt_tokens = tokens.clone();

        let max_len = if let Some(new_tokens) = config.max_new_tokens {
            tokens.len() + new_tokens
        } else {
            config.max_length
        };
        let cache_capacity = match self.model.autoregressive_loop() {
            edgetransformers::models::base::AutoregressiveLoop::Legacy => max_len + 1,
            edgetransformers::models::base::AutoregressiveLoop::Pipelined => max_len,
        };
        // Create a cache sized for the full generation length
        let mut cache = self.model.new_cache(1, cache_capacity, 0)?;
        let prefill_start = std::time::Instant::now();

        // --- PREFILL / PRIMING PHASE ---
        // The backend handles the Pipelined vs. Legacy logic internally.
        let mut next_token_logits = self.backend.prefill(
            self.model.as_ref(),
            &tokens,
            cache.as_mut(),
        ).await?;

        log::info!("Prefill phase took: {:?}", prefill_start.elapsed());

        Ok(try_stream! {
            // --- 1. Yield the prompt tokens ---
            for &token_id in &prompt_tokens {
                // Optional: Skip decoding/yielding special tokens like BOS
                if Some(token_id) == self.model.bos_token_id() {
                    continue;
                }
                let decoded_prompt_token = tokenizer.decode(&[token_id], false).map_err(|e| anyhow!(e))?;
                yield StreamedToken {
                    text: decoded_prompt_token,
                    id: token_id,
                    token_type: TokenType::Prompt,
                };
            }

            // --- 2. Generation Loop ---
            for i in 0..config.max_new_tokens.unwrap_or(config.max_length) {
                if tokens.len() >= max_len {
                    debug!("Reached max length ({})", max_len);
                    break;
                }

                // Sample the next token from the logits provided by the backend
                let processed_logits = apply_repetition_penalty(next_token_logits.clone(), &tokens, config.repetition_penalty);
                let next_token = sample_token(processed_logits, &config.strategy)?;

                tokens.push(next_token);

                // Stop if the End-Of-Sentence (EOS) token is generated
                if Some(next_token) == self.model.eos_token_id() {
                    debug!("EOS token generated, stopping.");
                    break;
                }

                // Decode and yield the newly generated token
                let decoded_token = tokenizer.decode(&[next_token], false).map_err(|e| anyhow!(e))?;
                yield StreamedToken {
                    text: decoded_token,
                    id: next_token,
                    token_type: TokenType::Generated,
                };
                let decode_start = std::time::Instant::now();
                // Get the logits for the *next* iteration from the backend
                next_token_logits = self.backend.decode_one(
                    self.model.as_ref(),
                    next_token,
                    tokens.len(),
                    cache.as_mut(),
                ).await?;
                debug!("[Token {}] decode_one took: {:?}", i + 1, decode_start.elapsed());
            }
        })
    }
}