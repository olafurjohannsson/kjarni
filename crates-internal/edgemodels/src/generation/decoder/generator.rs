use crate::generation::decoder::{CpuDecoderBackend, GpuDecoderBackend};
// use crate::generation::generator::DecoderGenerationBackend;
use edgetransformers::decoder::DecoderGenerationBackend;
use anyhow::{Result, anyhow};
//
use async_stream::try_stream;
use edgetransformers::cache::Cache;
use edgetransformers::models::DecoderLanguageModel;
use edgetransformers::prelude::*;
use futures_core::stream::Stream;
use futures_util::TryStreamExt;
use log::{debug, error};
use ndarray::Array1;
use std::sync::Arc;

use crate::generation::common::sampling::{
    GenerationConfig, apply_repetition_penalty, sample_token,
};
use crate::generation::common::{StreamedToken, TokenType};

pub enum AnyDecoderBackend {
    Cpu(CpuDecoderBackend),
    Gpu(GpuDecoderBackend),
}

impl AnyDecoderBackend {
    /// Prefill phase - process initial prompt tokens
    pub async fn prefill<'a>(
        &'a self,
        model: &'a dyn DecoderLanguageModel,
        initial_tokens: &[u32],
        cache: &'a mut dyn Cache,
    ) -> Result<Array1<f32>> {
        match self {
            AnyDecoderBackend::Cpu(backend) => backend.prefill(model, initial_tokens, cache).await,
            AnyDecoderBackend::Gpu(backend) => backend.prefill(model, initial_tokens, cache).await,
        }
    }

    /// Decode one token at a time during generation
    pub async fn decode_one<'a>(
        &'a self,
        model: &'a dyn DecoderLanguageModel,
        token_id: u32,
        seq_len: usize,
        cache: &'a mut dyn Cache,
    ) -> Result<Array1<f32>> {
        match self {
            AnyDecoderBackend::Cpu(backend) => {
                backend.decode_one(model, token_id, seq_len, cache).await
            }
            AnyDecoderBackend::Gpu(backend) => {
                backend.decode_one(model, token_id, seq_len, cache).await
            }
        }
    }
}

/// Text generator for autoregressive decoding
pub struct Generator {
    pub model: Box<dyn DecoderLanguageModel>,
    backend: AnyDecoderBackend,
}

impl Generator {
    pub fn new(model: Box<dyn DecoderLanguageModel>) -> Result<Self> {
        let backend = match model.device() {
            Device::Cpu => AnyDecoderBackend::Cpu(CpuDecoderBackend),
            Device::Wgpu => {
                let context = model
                    .context()
                    .ok_or_else(|| anyhow!("GPU model missing WgpuContext"))?;

                // Clean! Uses shared pool from context
                AnyDecoderBackend::Gpu(GpuDecoderBackend::new(context, model.as_ref())?)
            }
        };

        Ok(Self { model, backend })
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
        let mut next_token_logits = self
            .backend
            .prefill(self.model.as_ref(), &tokens, cache.as_mut())
            .await?;

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
            let mut total_decode_time = std::time::Duration::new(0, 0);
            let mut tokens_generated = 0;
            let start_time = std::time::Instant::now();
            // --- 2. Generation Loop ---
            for i in 0..config.max_new_tokens.unwrap_or(config.max_length) {

                if tokens.len() >= max_len {
                    log::info!("Reached max length ({})", max_len);
                    break;
                }
                log::info!("Sampling");
                // Sample the next token from the logits provided by the backend
                let processed_logits = apply_repetition_penalty(
                    next_token_logits.clone(),
                    &tokens,
                    config.repetition_penalty
                );
                let next_token = sample_token(processed_logits, &config.strategy)?;

                tokens.push(next_token);

                // Stop if the End-Of-Sentence (EOS) token is generated
                if Some(next_token) == self.model.eos_token_id() {
                    log::info!("EOS token generated, stopping.");
                    break;
                }

                // Decode and yield the newly generated token
                let decoded_token = tokenizer.decode(&[next_token], false).map_err(|e| anyhow!(e))?;
                yield StreamedToken {
                    text: decoded_token,
                    id: next_token,
                    token_type: TokenType::Generated,
                };
                if tokens.len() >= max_len {
                    log::info!("Hit max length after token, not fetching next logits");
                    break;
                }

                let t0 = std::time::Instant::now();
                log::info!("Starting decode_one");
                // Get the logits for the *next* iteration from the backend
                next_token_logits = self.backend.decode_one(
                    self.model.as_ref(),
                    next_token,
                    tokens.len(),
                    cache.as_mut(),
                ).await?;
                let dt = t0.elapsed();
                total_decode_time += dt;
                tokens_generated += 1;

                // Log every token (or every 10)
                log::info!(
                    "Token #{}: {:.2} ms | Current Speed: {:.2} t/s | Avg Speed: {:.2} t/s",
                    i + 1,
                    dt.as_secs_f64() * 1000.0,
                    1.0 / dt.as_secs_f64(),
                    tokens_generated as f64 / total_decode_time.as_secs_f64()
                );
            }
            log::info!("Loop done");
        })
    }
}
