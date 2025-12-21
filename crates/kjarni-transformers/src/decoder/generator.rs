use crate::common::{
    GenerationConfig, StreamedToken, TokenType, apply_no_repeat_ngram,
    apply_repetition_penalty_mut, sample_token,
};
use crate::decoder::prelude::*;
use crate::models::base::AutoregressiveLoop;
use crate::prelude::*;
use anyhow::{Result, anyhow};
use async_stream::try_stream;
use futures_core::stream::Stream;
use futures_util::TryStreamExt;
use log::{debug, info, warn};
use std::time::{Duration, Instant};

/// Orchestrates autoregressive text generation for decoder-only models.
///
/// This component bridges the high-level user API with the low-level model backends.
/// It handles tokenization, KV-cache management, prefilling, and the token sampling loop.
///
/// # Example
/// ```no_run
/// use kjarni_transformers::decoder::prelude::DecoderGenerator;
/// use kjarni_transformers::common::sampling::GenerationConfig;
///
/// # async fn example(model: Box<dyn kjarni_transformers::decoder::prelude::DecoderLanguageModel>) -> anyhow::Result<()> {
/// // 1. Create the generator
/// let generator = DecoderGenerator::new(model)?;
///
/// // 2. Configure generation
/// let config = GenerationConfig {
///     max_new_tokens: Some(50),
///     ..Default::default()
/// };
///
/// // 3. Generate text
/// let output = generator.generate("Rust is a systems programming language", &config).await?;
/// println!("Generated: {}", output);
/// # Ok(())
/// # }
/// ```
pub struct DecoderGenerator {
    pub model: Box<dyn DecoderLanguageModel>,
    backend: AnyDecoderBackend,
}

impl DecoderGenerator {
    /// Creates a new generator, automatically selecting the execution backend (CPU/GPU)
    /// based on the provided model's device configuration.
    pub fn new(model: Box<dyn DecoderLanguageModel>) -> Result<Self> {
        let backend = match model.device() {
            Device::Cpu => AnyDecoderBackend::Cpu(CpuDecoderBackend),
            Device::Wgpu => {
                let context = model
                    .context()
                    .ok_or_else(|| anyhow!("GPU model requires WgpuContext, but none found."))?;
                AnyDecoderBackend::Gpu(GpuDecoderBackend::new(context)?)
            }
        };

        Ok(Self { model, backend })
    }

    /// Generates a complete string of text.
    ///
    /// This is a convenience method that collects all tokens from the stream
    /// and joins them into a single string.
    ///
    /// # Arguments
    /// * `prompt` - The input text to continue
    /// * `config` - Sampling parameters (temperature, top_p, etc.)
    pub async fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        let stream = self.generate_stream(prompt, config).await?;
        let results: Vec<StreamedToken> = stream.try_collect().await?;

        let text: String = results.iter().map(|v| v.text.as_str()).collect();
        Ok(text)
    }

    /// Generates a stream of tokens based on a prompt.
    ///
    /// This allows for real-time processing of tokens as they are generated.
    /// The stream yields `StreamedToken` objects which contain the text, token ID,
    /// and whether the token was part of the prompt or generated.
    ///
    /// # Example
    /// ```no_run
    /// use futures_util::StreamExt;
    /// # async fn example(generator: kjarni_transformers::decoder::prelude::DecoderGenerator, config: kjarni_transformers::common::sampling::GenerationConfig) -> anyhow::Result<()> {
    /// let mut stream = generator.generate_stream("Once upon a time", &config).await?;
    /// futures_util::pin_mut!(stream);
    /// while let Some(token_result) = stream.next().await {
    ///     let token = token_result?;
    ///     print!("{}", token.text);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn generate_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<impl Stream<Item = Result<StreamedToken>>> {
        debug!("Starting generation for prompt: '{}'", prompt);

        // Encode the prompt and handle special tokens
        let tokenizer = self.model.tokenizer();
        let mut tokens = tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!(e))?
            .get_ids()
            .to_vec();

        if config.add_bos_token {
            if let Some(bos_token_id) = self.model.bos_token_id() {
                if tokens.first() != Some(&bos_token_id) {
                    tokens.insert(0, bos_token_id);
                }
            } else {
                warn!("Config requested BOS token, but model has no BOS ID defined.");
            }
        }

        // Store prompt tokens to echo them back in the stream later
        let prompt_tokens = tokens.clone();
        let prompt_len = tokens.len();

        // Determine generation limits and cache size
        let max_len = if let Some(new_tokens) = config.max_new_tokens {
            prompt_len + new_tokens
        } else {
            config.max_length
        };

        // Legacy loops (like GPT-2) need +1 capacity for the calculation step
        let cache_capacity = match self.model.autoregressive_loop() {
            AutoregressiveLoop::Legacy => max_len + 1,
            AutoregressiveLoop::Pipelined => max_len,
        };

        // Allocate resources on the device
        let mut cache: Box<dyn Cache> = self.model.new_cache(1, cache_capacity, 0)?;
        let mut token_tensor = self.backend.new_token_tensor()?;

        debug!("Prefilling {} tokens...", prompt_len);
        let t_prefill = std::time::Instant::now();

        // process prompt to prime the cache
        let mut next_token_logits = self
            .backend
            .prefill(self.model.as_ref(), &tokens, cache.as_mut())
            .await?;

        log::info!(
            "Prefill complete in {:.2}ms",
            t_prefill.elapsed().as_secs_f64() * 1000.0
        );

        let context_limit = self.model.context_size();
        let stop_tokens = self.model.stop_token_ids();

        Ok(try_stream! {
            // yield the prompt tokens
            for &token_id in &prompt_tokens {
                if Some(token_id) == self.model.bos_token_id() {
                    continue;
                }
                let text = tokenizer.decode(&[token_id], false).map_err(|e| anyhow!(e))?;
                yield StreamedToken {
                    text,
                    id: token_id,
                    token_type: TokenType::Prompt,
                };
            }

            let mut total_sampling_time = Duration::new(0, 0);
            let mut total_backend_time = Duration::new(0, 0);
            let mut tokens_generated = 0;
            let generation_start_time = Instant::now();

            // Begin the autoregressive generation loop
            for i in 0..config.max_new_tokens.unwrap_or(max_len) {


                // Context window
                if tokens.len() >= context_limit {
                    warn!("Context limit reached ({}), stopping.", context_limit);
                    break;
                }

                // User max length
                if tokens.len() >= max_len {
                    info!("Generation reached max length ({})", max_len);
                    break;
                }
                let t_sampling_start = Instant::now();
                let mut logits = next_token_logits.clone();

                // Repetition Penalty
                if config.repetition_penalty != 1.0 {
                    apply_repetition_penalty_mut(
                        &mut logits,
                        &tokens,
                        config.repetition_penalty
                    );
                }

                // N-Gram Penalty
                if config.no_repeat_ngram_size > 0 {
                    apply_no_repeat_ngram(
                        &mut logits,
                        &tokens,
                        config.no_repeat_ngram_size
                    );
                }

                // 3. Sampling (Top-K, Top-P, Min-P, Temp)
                let next_token = sample_token(logits, &config.strategy)?;

                total_sampling_time += t_sampling_start.elapsed();

                tokens.push(next_token);



                // yield generated token
                let text = tokenizer.decode(&[next_token], false).map_err(|e| anyhow!(e))?;
                yield StreamedToken {
                    text,
                    id: next_token,
                    token_type: TokenType::Generated,
                };
                tokens_generated += 1;
                if stop_tokens.contains(&next_token) {
                    debug!("Stop token generated: {}", next_token);
                    break;
                }

                if tokens.len() >= max_len { break; }

                let t_backend_start = Instant::now();

                self.backend.update_token_tensor(&mut token_tensor, next_token)?;

                // This is the main compute-bound part (all the matmuls).
                next_token_logits = self.backend.decode_one(
                    self.model.as_ref(),
                    &token_tensor,
                    tokens.len(),
                    cache.as_mut(),
                ).await?;

                total_backend_time += t_backend_start.elapsed();
            }

            let total_generation_time = generation_start_time.elapsed();

            if tokens_generated > 0 && total_generation_time.as_secs_f64() > 0.0 {
                let tokens_per_sec = tokens_generated as f64 / total_generation_time.as_secs_f64();
                
                // Calculate average time per token for each stage
                let avg_total_per_token = total_generation_time / tokens_generated;
                let avg_sampling_per_token = total_sampling_time / tokens_generated;
                let avg_backend_per_token = total_backend_time / tokens_generated;

                // Log the detailed breakdown
                info!("-------------------- Generation Performance --------------------");
                info!("Total Tokens: {}", tokens_generated);
                info!("Total Time:   {:.3}s", total_generation_time.as_secs_f64());
                info!("Overall T/s:  {:.2}", tokens_per_sec);
                info!("----------------------------------------------------------------");
                info!("Avg. Time per Token Breakdown:");
                info!("  - Total:    {:?}", avg_total_per_token);
                info!("  - Sampling: {:?} (Memory-Bound Part)", avg_sampling_per_token);
                info!("  - Backend:  {:?} (Compute-Bound Part)", avg_backend_per_token);
                info!("----------------------------------------------------------------");

            } else if tokens_generated > 0 {
                 info!("Generation complete. Generated {} tokens.", tokens_generated);
            }
        })
    }
}
