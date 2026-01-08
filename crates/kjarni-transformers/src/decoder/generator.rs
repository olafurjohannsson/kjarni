//! Autoregressive text generation orchestrator for decoder-only models.
//!
//! This module provides the high-level generation API that bridges user requests
//! with the low-level CPU/GPU backends. It handles the complete generation pipeline:
//!
//! 1. **Tokenization** - Converting text to token IDs
//! 2. **Prefill** - Processing the prompt through the model
//! 3. **Decode Loop** - Autoregressive token generation with sampling
//! 4. **Detokenization** - Converting generated tokens back to text
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                       DecoderGenerator                              │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  User API                    Internal Pipeline                      │
//! │  ────────                    ─────────────────                      │
//! │                                                                     │
//! │  generate(prompt) ──────►  encode() ──► prefill() ──► decode_loop() │
//! │                                              │              │       │
//! │  chat(conversation) ───►  format() ──►      ▼              ▼       │
//! │                                         ┌────────┐    ┌────────┐   │
//! │  generate_stream() ────►               │ Cache  │◄──►│ Sample │   │
//! │                                         └────────┘    └────────┘   │
//! │                                              │              │       │
//! │                                              ▼              ▼       │
//! │                                         ┌────────────────────┐     │
//! │                                         │  Backend (CPU/GPU) │     │
//! │                                         └────────────────────┘     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Backend Selection
//!
//! The generator automatically selects the appropriate backend based on the
//! model's device configuration:
//!
//! | Model Device | Backend | Characteristics |
//! |--------------|---------|-----------------|
//! | `Device::Cpu` | `CpuDecoderBackend` | Uses ndarray, AVX2 SIMD |
//! | `Device::Wgpu` | `GpuDecoderBackend` | Uses WebGPU/Vulkan compute |
//!
//! # Generation Modes
//!
//! ## Blocking Generation
//! ```ignore
//! let output = generator.generate("Once upon a time", &config).await?;
//! ```
//!
//! ## Streaming Generation
//! ```ignore
//! let stream = generator.generate_stream("Once upon a time", &config).await?;
//! while let Some(token) = stream.next().await {
//!     print!("{}", token?.text);
//! }
//! ```
//!
//! ## Chat Generation
//! ```ignore
//! let mut conv = Conversation::new();
//! conv.push_user("What is Rust?");
//! let response = generator.chat(&conv, &config).await?;
//! ```
//!
//! # Performance Characteristics
//!
//! - **Prefill**: O(n²) attention over prompt tokens, highly parallelizable
//! - **Decode**: O(n) per token with KV cache, sequential
//! - **Memory**: KV cache grows linearly with sequence length
//! - **Bottleneck**: Typically memory bandwidth during decode phase

use crate::common::{
    apply_no_repeat_ngram, apply_repetition_penalty_mut, sample_token, CancellationToken, GenerationConfig,
    StreamedToken, TokenType,
};
use crate::decoder::prelude::*;
use crate::models::base::AutoregressiveLoop;
use crate::stats::GenerationStats;
use crate::{prelude::*, Conversation};
use anyhow::{anyhow, Result};
use futures_core::stream::Stream;
use futures_util::TryStreamExt;
use log::{debug, info, trace};
use std::sync::Arc;

/// Orchestrates autoregressive text generation for decoder-only models.
///
/// This is the primary entry point for text generation. It wraps a
/// `DecoderLanguageModel` and provides high-level methods for generating
/// text from prompts or conversations.
///
/// # Thread Safety
///
/// The generator is `!Send` due to GPU backend constraints. All generation
/// must occur on the thread that created the generator.
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::decoder::prelude::DecoderGenerator;
/// use kjarni_transformers::common::GenerationConfig;
///
/// # async fn example(model: Box<dyn kjarni_transformers::decoder::prelude::DecoderLanguageModel>) -> anyhow::Result<()> {
/// let generator = DecoderGenerator::new(model)?;
///
/// let config = GenerationConfig {
///     max_new_tokens: Some(100),
///     temperature: Some(0.7),
///     ..Default::default()
/// };
///
/// let output = generator.generate("The future of AI is", &config, None).await?;
/// println!("{}", output);
/// # Ok(())
/// # }
/// ```
///
/// # Resource Management
///
/// The generator owns the model and backend. KV caches are allocated per-generation
/// and automatically freed when the generation completes or the stream is dropped.
pub struct DecoderGenerator {
    /// The underlying language model (Llama, GPT-2, Phi, etc.)
    pub model: Arc<dyn DecoderLanguageModel + Send + Sync>,

    /// Device-specific execution backend
    backend: AnyDecoderBackend,
}

impl DecoderGenerator {
    /// Creates a new generator with automatic backend selection.
    ///
    /// The backend is chosen based on the model's device:
    /// - `Device::Cpu` → `CpuDecoderBackend`
    /// - `Device::Wgpu` → `GpuDecoderBackend`
    ///
    /// # Arguments
    ///
    /// * `model` - A boxed decoder language model
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - GPU model lacks a `WgpuContext`
    /// - Backend initialization fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use kjarni_transformers::decoder::prelude::*;
    /// # fn example(model: Box<dyn DecoderLanguageModel>) -> anyhow::Result<()> {
    /// let generator = DecoderGenerator::new(model)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(model: Arc<dyn DecoderLanguageModel + Send + Sync>) -> Result<Self> {
        let backend = match model.device() {
            Device::Cpu => {
                debug!("Initializing CPU decoder backend");
                AnyDecoderBackend::Cpu(CpuDecoderBackend)
            }
            Device::Wgpu => {
                debug!("Initializing GPU decoder backend");
                let context = model
                    .context()
                    .ok_or_else(|| anyhow!("GPU model requires WgpuContext, but none found."))?;
                AnyDecoderBackend::Gpu(Arc::new(GpuDecoderBackend::new(context)?))
            }
        };

        info!(
            "DecoderGenerator initialized: device={:?}, vocab_size={}, context_size={}",
            model.device(),
            model.vocab_size(),
            model.context_size()
        );

        Ok(Self { model, backend })
    }

    // =========================================================================
    // High-Level Generation API
    // =========================================================================

    /// Generates a complete text response from a prompt.
    ///
    /// This is a convenience method that collects all tokens from the internal
    /// stream and joins them into a single string. Use `generate_stream` for
    /// real-time token access.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Input text to continue
    /// * `config` - Generation parameters (temperature, top_p, max_tokens, etc.)
    /// * `cancellation` - Token to check for cancellation
    ///
    /// # Returns
    ///
    /// The generated text (excluding the prompt).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example(generator: kjarni_transformers::decoder::prelude::DecoderGenerator) -> anyhow::Result<()> {
    /// use kjarni_transformers::common::GenerationConfig;
    /// use kjarni_transformers::common::CancellationToken;
    ///
    /// let (token, handle) = CancellationToken::new();
    /// let config = GenerationConfig::default();
    /// let output = generator.generate("Rust is", &config, Some(token)).await?;
    /// // output: " a systems programming language..."
    /// # Ok(())
    /// # }
    /// ```
    pub async fn generate(&self, prompt: &str, config: &GenerationConfig, cancellation: Option<CancellationToken>) -> Result<String> {
        let stream = self.generate_stream(prompt, config, cancellation).await?;
        let results: Vec<StreamedToken> = stream.try_collect().await?;
        // Filter to only generated tokens (exclude prompt echo)
        let text: String = results
            .iter()
            .filter(|t| t.token_type == TokenType::Generated)
            .map(|v| v.text.as_str())
            .collect();
        Ok(text)
    }

    /// Generates a response for a multi-turn conversation.
    ///
    /// Applies the model's chat template (if available) to format the
    /// conversation before generation.
    ///
    /// # Arguments
    ///
    /// * `conversation` - The conversation history
    /// * `config` - Generation parameters
    /// * `cancellation` - Token to check for cancellation
    ///
    /// # Returns
    ///
    /// The assistant's response text.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example(generator: kjarni_transformers::decoder::prelude::DecoderGenerator) -> anyhow::Result<()> {
    /// use kjarni_transformers::Conversation;
    /// use kjarni_transformers::common::GenerationConfig;
    /// use kjarni_transformers::common::CancellationToken;
    ///
    /// let mut conv = Conversation::new();
    /// conv.push_user("What is the capital of France?");
    /// let (token, handle) = CancellationToken::new();
    /// let response = generator.chat(&conv, &GenerationConfig::default(), Some(token)).await?;
    /// // response: "The capital of France is Paris."
    /// # Ok(())
    /// # }
    /// ```
    pub async fn chat(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
        cancellation: Option<CancellationToken>,
    ) -> Result<String> {
        let prompt = self.format_conversation(conversation)?;
        trace!("Formatted chat prompt: {} chars", prompt.len());
        self.generate(&prompt, config, cancellation).await
    }

    /// Generates a stream of tokens from a text prompt.
    ///
    /// Returns immediately with a stream that yields tokens as they are generated.
    /// The stream first yields prompt tokens (echoed back), then generated tokens.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Input text to continue
    /// * `config` - Generation parameters
    /// * `cancellation` - Token to check for cancellation
    ///
    /// # Returns
    ///
    /// A stream of `StreamedToken` results.
    ///
    /// # Example
    ///
    /// ```ignore
    /// # async fn example(generator: kjarni_transformers::decoder::prelude::DecoderGenerator) -> anyhow::Result<()> {
    /// use futures_util::StreamExt;
    /// use kjarni_transformers::common::GenerationConfig;
    /// use kjarni_transformers::common::CancellationToken;
    ///
    /// let (token, handle) = CancellationToken::new();
    /// let stream = generator.generate_stream("Once upon a time", &GenerationConfig::default(), Some(token)).await?;
    /// futures_util::pin_mut!(stream);
    ///
    /// while let Some(token_result) = stream.next().await {
    ///     let token = token_result?;
    ///     print!("{}", token.text);
    ///     std::io::Write::flush(&mut std::io::stdout())?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn generate_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        cancellation: Option<CancellationToken>,
    ) -> Result<impl Stream<Item=Result<StreamedToken>>> {
        let tokens = self.encode(prompt, config)?;
        self.generate_stream_from_tokens(tokens, config, cancellation).await
    }

    /// Generates a stream of tokens for a conversation.
    ///
    /// Combines `format_conversation` and `generate_stream`.
    pub async fn chat_stream(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
        cancellation: Option<CancellationToken>,
    ) -> Result<impl Stream<Item=Result<StreamedToken>>> {
        let prompt = self.format_conversation(conversation)?;
        let tokens = self.encode(&prompt, config)?;
        self.generate_stream_from_tokens(tokens, config, cancellation).await
    }

    // =========================================================================
    // Tokenization & Formatting
    // =========================================================================

    /// Encodes a text prompt into token IDs.
    ///
    /// Optionally prepends the BOS token if `config.add_bos_token` is true
    /// and the model has a BOS token configured.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Text to encode
    /// * `config` - Generation config (for BOS token handling)
    ///
    /// # Returns
    ///
    /// Vector of token IDs.
    pub fn encode(&self, prompt: &str, config: &GenerationConfig) -> Result<Vec<u32>> {
        let tokenizer = self.model.tokenizer();
        let mut tokens = tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?
            .get_ids()
            .to_vec();

        // Prepend BOS token if configured
        if config.add_bos_token {
            if let Some(bos) = self.model.bos_token_id() {
                if tokens.first() != Some(&bos) {
                    tokens.insert(0, bos);
                    trace!("Prepended BOS token: {}", bos);
                }
            }
        }

        debug!("Encoded prompt: {} chars → {} tokens", prompt.len(), tokens.len());
        Ok(tokens)
    }

    /// Formats a conversation using the model's chat template.
    ///
    /// If the model has no chat template, falls back to using the last
    /// user message as the prompt.
    ///
    /// # Arguments
    ///
    /// * `conversation` - The conversation to format
    ///
    /// # Returns
    ///
    /// Formatted prompt string ready for tokenization.
    pub fn format_conversation(&self, conversation: &Conversation) -> Result<String> {
        if let Some(template) = self.model.chat_template() {
            Ok(template.apply(conversation))
        } else {
            // Fallback for base models: use last user message
            conversation
                .last()
                .map(|m| m.content.clone())
                .ok_or_else(|| anyhow!("Conversation is empty and model has no chat template"))
        }
    }

    /// Returns whether this model requires a chat template for proper use.
    ///
    /// Instruct/chat-tuned models typically require specific formatting.
    pub fn requires_template(&self) -> bool {
        self.model.is_instruct_model()
    }

    /// Returns the stop sequences from the model's chat template.
    ///
    /// These are strings that signal the end of generation (e.g., `<|eot_id|>`).
    pub fn stop_sequences(&self) -> Vec<String> {
        self.model
            .chat_template()
            .map(|t| t.stop_sequences())
            .unwrap_or_default()
    }

    // =========================================================================
    // Core Generation Loop
    // =========================================================================

    /// Core generation implementation that operates on pre-tokenized input.
    ///
    /// This is the main generation loop that:
    /// 1. Allocates a KV cache sized for the generation
    /// 2. Runs prefill to process the prompt
    /// 3. Iteratively samples and decodes new tokens
    /// 4. Yields tokens via an async stream
    ///
    /// # Arguments
    ///
    /// * `input_tokens` - Pre-tokenized prompt
    /// * `config` - Generation parameters
    ///
    /// # Returns
    ///
    /// Async stream yielding `StreamedToken` for each token (prompt + generated).
    ///
    /// # Generation Loop Details
    ///
    /// ```text
    /// prefill(prompt) → logits
    ///        ↓
    /// ┌──────────────────────────────────────┐
    /// │  while not done:                     │
    /// │    1. Apply repetition penalty       │
    /// │    2. Apply n-gram blocking          │
    /// │    3. Sample next token              │
    /// │    4. Check stop conditions          │
    /// │    5. Yield token to stream          │
    /// │    6. decode_one(token) → logits     │
    /// └──────────────────────────────────────┘
    /// ```
    /// Generates a stream of tokens using a dedicated blocking thread.
    pub async fn generate_stream_from_tokens(
        &self,
        input_tokens: Vec<u32>,
        config: &GenerationConfig,
        cancellation: Option<CancellationToken>,
    ) -> Result<impl Stream<Item=Result<StreamedToken>>> {
        // 1. Prepare data for transfer to the compute thread
        let model = self.model.clone();
        let backend = self.backend.clone();
        let config = config.clone();

        // 2. Create a channel to bridge Compute Thread -> Async World
        // Buffer size 32 is plenty for text generation
        let (tx, rx) = tokio::sync::mpsc::channel(32);

        // 3. Spawn the compute task on a dedicated OS thread
        tokio::task::spawn_blocking(move || {
            // We need a minimal runtime to execute the backend's async methods
            // synchronously within this blocking thread.
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();

            let local_rt = match rt {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.blocking_send(Err(anyhow!("Failed to create local runtime: {}", e)));
                    return;
                }
            };

            // Run the generation logic
            local_rt.block_on(async {
                if let Err(e) = run_generation_loop(
                    model, 
                    backend, 
                    input_tokens, 
                    config, 
                    tx.clone(), 
                    cancellation
                ).await {
                    // Send fatal errors to the stream
                    let _ = tx.send(Err(e)).await;
                }
            });
        });

        // 4. Return the stream wrapper immediately
        Ok(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
    // pub async fn generate_stream_from_tokens(
    //     &self,
    //     input_tokens: Vec<u32>,
    //     config: &GenerationConfig,
    //     cancellation: Option<CancellationToken>,
    // ) -> Result<impl Stream<Item=Result<StreamedToken>>> {

    //     // Prepare data for transfer to the compute thread
    //     let model = self.model.clone();
    //     let backend = self.backend.clone();
    //     let config = config.clone();

    //     // =====================================================================
    //     // Setup Phase
    //     // =====================================================================
    //     log::info!("Generating stream from token");
    //     let prompt_tokens = input_tokens.clone();
    //     let prompt_len = prompt_tokens.len();
    //     let mut tokens = input_tokens;

    //     // Determine generation limits
    //     let max_len = config.max_new_tokens
    //         .map(|n| prompt_len + n)
    //         .unwrap_or(config.max_length);

    //     // Legacy autoregressive loops (GPT-2 style) need +1 cache capacity
    //     // because they compute logits for position N from cache position N-1
    //     let cache_capacity = match self.model.autoregressive_loop() {
    //         AutoregressiveLoop::Legacy => max_len + 1,
    //         AutoregressiveLoop::Pipelined => max_len,
    //     };

    //     debug!(
    //         "Generation setup: prompt_len={}, max_len={}, cache_capacity={}, loop={:?}",
    //         prompt_len, max_len, cache_capacity, self.model.autoregressive_loop()
    //     );

    //     // Allocate KV cache
    //     let mut cache: Box<dyn Cache> = self.model.new_cache(1, cache_capacity, 0)?;

    //     // Pre-allocate token tensor for decode loop (avoids allocation per step)
    //     let mut token_tensor = self.backend.new_token_tensor()?;
    //     // =====================================================================
    //     // Prefill Phase
    //     // =====================================================================

    //     let mut stats = GenerationStats::new();
    //     stats.start_prefill(prompt_len);

    //     debug!("Starting prefill: {} tokens", prompt_len);
    //     let prefill_start = Instant::now();
    //     let mut next_token_logits = self
    //         .backend
    //         .prefill(self.model.as_ref(), &tokens, cache.as_mut())
    //         .await?;

    //     stats.end_prefill();
    //     info!(
    //         "Prefill complete: {} tokens in {:.2}ms ({:.2} t/s)",
    //         prompt_len,
    //         prefill_start.elapsed().as_secs_f64() * 1000.0,
    //         stats.prefill_tps()
    //     );
        

    //     // =====================================================================
    //     // Decode Phase (Async Stream)
    //     // =====================================================================

    //     let context_limit = self.model.context_size();
    //     let stop_tokens = self.model.stop_token_ids();
    //     let tokenizer = self.model.tokenizer();
    //     let max_new_tokens = config.max_new_tokens.unwrap_or(max_len - prompt_len);

    //     Ok(try_stream! {
    //         // Yield prompt tokens
    //         for &token_id in &prompt_tokens {
    //             // Check cancellation during prompt echo
    //             if let Some(token) = &cancellation {
    //                 if token.is_cancelled() {
    //                     info!("Generation cancelled during prompt echo");
    //                     return;
    //                 }
    //             }
    //             // Skip BOS token in output
    //             if Some(token_id) == self.model.bos_token_id() {
    //                 continue;
    //             }
    //             let text = tokenizer
    //                 .decode(&[token_id], false)
    //                 .map_err(|e| anyhow!("Detokenization failed: {}", e))?;
    //             yield StreamedToken {
    //                 text,
    //                 id: token_id,
    //                 token_type: TokenType::Prompt,
    //             };
    //         }

    //         // Generation loop with cancellation checks
    //         for step in 0..max_new_tokens {
    //             if let Some(token) = &cancellation {
    //                 if token.is_cancelled() {
    //                     info!("Generation cancelled at step {}", step);
    //                     return;
    //                 }
    //             }
    //             // Check context window limit
    //             if tokens.len() >= context_limit {
    //                 warn!(
    //                     "Context limit reached ({}/{}), stopping generation",
    //                     tokens.len(), context_limit
    //                 );
    //                 break;
    //             }

    //             // Check max length limit
    //             if tokens.len() >= max_len {
    //                 debug!("Reached max_len={}, stopping generation", max_len);
    //                 break;
    //             }

    //             //sampling
    //             let sampling_start = Instant::now();
    //             let mut logits = next_token_logits.clone();

    //             // Apply repetition penalty
    //             if config.repetition_penalty != 1.0 {
    //                 apply_repetition_penalty_mut(
    //                     &mut logits,
    //                     &tokens,
    //                     config.repetition_penalty
    //                 );
    //             }

    //             // Apply n-gram blocking
    //             if config.no_repeat_ngram_size > 0 {
    //                 apply_no_repeat_ngram(
    //                     &mut logits,
    //                     &tokens,
    //                     config.no_repeat_ngram_size
    //                 );
    //             }

    //             // Sample next token (applies temperature, top-k, top-p, min-p)
    //             let next_token = sample_token(logits, &config.strategy)?;
                
    //             trace!(
    //                 "Step {}: sampled token {} in {:.2}ms",
    //                 step, next_token, sampling_start.elapsed().as_secs_f64() * 1000.0
    //             );

    //             tokens.push(next_token);
                
    //             // check stop tokens
    //             if stop_tokens.contains(&next_token) {
    //                 debug!("Stop token {} generated at step {}", next_token, step);
    //                 break;
    //             }

    //             // yield token
    //             let text = tokenizer
    //                 .decode(&[next_token], false)
    //                 .map_err(|e| anyhow!("Detokenization failed: {}", e))?;
                    
    //             yield StreamedToken {
    //                 text,
    //                 id: next_token,
    //                 token_type: TokenType::Generated,
    //             };
                
    //             stats.record_token();

                
    //             if tokens.len() >= max_len {
    //                 break;
    //             }

    //             // check cancel before decode
    //             if let Some(cancellation) = &cancellation {
    //                 if cancellation.is_cancelled() {
    //                     info!("Generation cancelled at step {}", step);
    //                     break;
    //                 }
    //             }

    //             // decode in backend
    //             self.backend.update_token_tensor(&mut token_tensor, next_token)?;
    //             next_token_logits = self.backend.decode_one(
    //                 self.model.as_ref(),
    //                 &token_tensor,
    //                 tokens.len(),
    //                 cache.as_mut(),
    //             ).await?;

    //             // Periodic TPS logging
    //             if GenerationStats::is_enabled() && step > 0 && step % 20 == 0 {
    //                 debug!("Step {}: {:.2} tok/s", step, stats.decode_tps());
    //             }
    //         }

    //         // generation complete
    //         stats.print_summary();
    //     })
    // }
}

async fn run_generation_loop(
    model: Arc<dyn DecoderLanguageModel + Send + Sync>,
    backend: AnyDecoderBackend,
    input_tokens: Vec<u32>,
    config: GenerationConfig,
    tx: tokio::sync::mpsc::Sender<Result<StreamedToken>>,
    cancellation: Option<CancellationToken>,
) -> Result<()> {
    let prompt_len = input_tokens.len();
    let mut tokens = input_tokens;

    // --- Limits & Setup ---
    let context_limit = model.context_size(); 
    let max_len = config.max_new_tokens
        .map(|n| prompt_len + n)
        .unwrap_or(config.max_length);

    let cache_capacity = match model.autoregressive_loop() {
        AutoregressiveLoop::Legacy => max_len + 1,
        AutoregressiveLoop::Pipelined => max_len,
    };
    debug!(
        "Generation setup: prompt_len={}, max_len={}, cache_capacity={}, loop={:?}",
        prompt_len, max_len, cache_capacity, model.autoregressive_loop()
    );
    let mut cache = model.new_cache(1, cache_capacity, 0)?;
    let mut token_tensor = backend.new_token_tensor()?;

    // --- Prefill ---
    let mut stats = GenerationStats::new();
    stats.start_prefill(prompt_len);
    
    let mut next_token_logits = backend
        .prefill(model.as_ref(), &tokens, cache.as_mut())
        .await?;
    
    stats.end_prefill();

    // --- Emit Prompt Tokens ---
    let tokenizer = model.tokenizer();
    for &token_id in &tokens {
        // Check cancellation
        if let Some(c) = &cancellation {
            if c.is_cancelled() { return Ok(()); }
        }

        if tokens.len() >= context_limit {
            debug!("Context limit reached ({}), stopping.", context_limit);
            break;
        }

        if Some(token_id) == model.bos_token_id() { continue; }
        
        let text = tokenizer.decode(&[token_id], false).map_err(|e| anyhow!(e))?;
        
        // Send to channel
        if tx.send(Ok(StreamedToken {
            text,
            id: token_id,
            token_type: TokenType::Prompt,
        })).await.is_err() {
            return Ok(()); // Receiver dropped
        }
    }

    // --- Decode Loop ---
    let max_new_tokens = config.max_new_tokens.unwrap_or(max_len - prompt_len);
    let stop_tokens = model.stop_token_ids();

    for _step in 0..max_new_tokens {
        if let Some(c) = &cancellation {
            if c.is_cancelled() { break; }
        }
        if tokens.len() >= max_len { break; }

        // Sample
        let mut logits = next_token_logits.clone();
        if config.repetition_penalty != 1.0 {
            apply_repetition_penalty_mut(&mut logits, &tokens, config.repetition_penalty);
        }
        if config.no_repeat_ngram_size > 0 {
            apply_no_repeat_ngram(&mut logits, &tokens, config.no_repeat_ngram_size);
        }

        let next_token = sample_token(logits, &config.strategy)?;
        tokens.push(next_token);

        // Emit
        let text = tokenizer.decode(&[next_token], false).map_err(|e| anyhow!(e))?;
        
        if tx.send(Ok(StreamedToken {
            text,
            id: next_token,
            token_type: TokenType::Generated,
        })).await.is_err() {
            break;
        }

        stats.record_token();

        if stop_tokens.contains(&next_token) { break; }

        // Compute Next
        backend.update_token_tensor(&mut token_tensor, next_token)?;
        next_token_logits = backend.decode_one(
            model.as_ref(),
            &token_tensor,
            tokens.len(),
            cache.as_mut(),
        ).await?;
    }
    
    stats.print_summary();
    Ok(())
}