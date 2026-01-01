//! CPU-based decoder generation backend using ndarray.
//!
//! This module provides the CPU implementation of autoregressive text generation,
//! operating entirely on the CPU using ndarray for tensor operations and AVX2
//! SIMD for optimized matrix multiplication.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     CpuDecoderBackend                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  prefill(tokens)                                                │
//! │      │                                                          │
//! │      ▼                                                          │
//! │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
//! │  │   Embed      │───▶│   Layers     │───▶│   LM Head    │      │
//! │  │  (ndarray)   │    │   (ndarray)  │    │   (ndarray)  │      │
//! │  └──────────────┘    └──────────────┘    └──────────────┘      │
//! │                             │                    │              │
//! │                             ▼                    ▼              │
//! │                      ┌──────────────┐    ┌──────────────┐      │
//! │                      │  KV Cache    │    │   Logits     │      │
//! │                      │  (Vec<f32>)  │    │  [vocab_size]│      │
//! │                      └──────────────┘    └──────────────┘      │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Design Philosophy
//!
//! The `CpuDecoderBackend` is a **thin controller** that delegates all actual
//! computation to the model's `CpuDecoderOps`. It does not:
//! - Know how to multiply matrices
//! - Know model architecture details
//! - Create attention masks
//!
//! Instead, it orchestrates the high-level flow and lets the model handle specifics.
//!
//! # Autoregressive Loop Strategies
//!
//! The backend supports two autoregressive strategies, determined by the model:
//!
//! ## Pipelined (Modern: Llama, Mistral, Phi)
//!
//! ```text
//! Prefill:  [tok1, tok2, tok3, tok4] ─────────────────► logits[4]
//!                                                            │
//! Decode:                                    [tok5] ─────────► logits[5]
//!                                            [tok6] ─────────► logits[6]
//! ```
//!
//! - Processes entire prompt in single forward pass
//! - Cache populated during prefill
//! - Each decode step processes exactly one token
//!
//! ## Legacy (GPT-2)
//!
//! ```text
//! Prefill:  [tok1, tok2, tok3] ──► cache only (no logits)
//!                    [tok4] ──────────────────► logits[4]
//!
//! Decode:            [tok5] ──────────────────► logits[5]
//! ```
//!
//! - Prompt processed in two phases: cache fill + last token
//! - Requires +1 cache capacity for implementation quirks
//! - Kept for backward compatibility with older models
//!
//! # Performance Characteristics
//!
//! | Phase | Complexity | Bottleneck |
//! |-------|------------|------------|
//! | Prefill | O(n²) attention | Compute-bound (matmul) |
//! | Decode | O(n) per token | Memory-bound (KV cache reads) |
//!
//! Typical performance on modern CPUs:
//! - Prefill: 100-500 tokens/second (depends on prompt length)
//! - Decode: 10-50 tokens/second (depends on model size)

use crate::cache::Cache;
use crate::decoder::prelude::*;
use crate::models::base::{AutoregressiveLoop, ModelInput};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use log::{debug, trace};
use ndarray::{s, Array1, Array2};
use std::time::Instant;

/// CPU-based backend for autoregressive decoder generation.
///
/// This is a stateless controller that delegates computation to the model's
/// `CpuDecoderOps`. It handles:
/// - Prefill orchestration (prompt processing)
/// - Decode step coordination
/// - Autoregressive loop strategy selection
///
/// # Stateless Design
///
/// Unlike `GpuDecoderBackend`, this backend has no internal state. All state
/// lives in:
/// - The model (weights, configuration)
/// - The KV cache (passed as argument)
/// - The token tensors (owned by caller)
///
/// This makes the CPU backend trivially thread-safe and allows multiple
/// generations to share a single backend instance.
///
/// # Example
///
/// ```ignore
/// let backend = CpuDecoderBackend;
/// let mut cache = model.new_cache(1, 2048, 1)?;
///
/// // Prefill
/// let logits = backend.prefill(&model, &prompt_tokens, &mut cache).await?;
///
/// // Decode loop
/// let mut token_tensor = backend.new_token_tensor()?;
/// for _ in 0..max_tokens {
///     backend.update_token_tensor(&mut token_tensor, current_token)?;
///     let logits = backend.decode_one(&model, &token_tensor, seq_len, &mut cache).await?;
///     current_token = sample(&logits);
/// }
/// ```
pub struct CpuDecoderBackend;

#[async_trait(?Send)]
impl DecoderGenerationBackend for CpuDecoderBackend {
    /// Token tensor type for CPU: 2D ndarray with shape `[batch=1, seq_len]`.
    ///
    /// Using `Array2` instead of `Vec<u32>` for consistency with the rest of
    /// the ndarray-based pipeline, even though batch size is always 1.
    type Tensor = Array2<u32>;

    /// Creates a token tensor from a slice of token IDs.
    ///
    /// Used during prefill to wrap the prompt tokens.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Slice of token IDs (prompt)
    ///
    /// # Returns
    ///
    /// 2D array with shape `[1, tokens.len()]`
    fn prime_tokens(&self, tokens: &[u32]) -> Result<Self::Tensor> {
        Array2::from_shape_vec((1, tokens.len()), tokens.to_vec())
            .map_err(|e| anyhow!("Failed to create token array: {}", e))
    }

    /// Creates an empty single-token tensor for the decode loop.
    ///
    /// Allocates a `[1, 1]` array that will be reused for each decode step.
    fn new_token_tensor(&self) -> Result<Self::Tensor> {
        Ok(Array2::zeros((1, 1)))
    }

    /// Updates the single-token tensor with a new token ID.
    ///
    /// This is a simple array write - no allocation required.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The token tensor to update (must have shape `[1, 1]`)
    /// * `new_token_id` - The token ID to write
    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_token_id: u32) -> Result<()> {
        tensor[[0, 0]] = new_token_id;
        Ok(())
    }

    /// Processes the prompt and returns logits for the first generated token.
    ///
    /// # Prefill Strategies
    ///
    /// ## Pipelined (Llama, Phi, Mistral)
    ///
    /// 1. Build causal attention mask for full prompt
    /// 2. Forward pass through all layers (populates KV cache)
    /// 3. Project final hidden states to logits
    /// 4. Return logits for last position
    ///
    /// ## Legacy (GPT-2)
    ///
    /// 1. Forward pass for tokens `[0..n-1]` (cache only, no projection)
    /// 2. Forward pass for token `[n-1]` with projection
    /// 3. Return logits for that position
    ///
    /// This two-phase approach is required because GPT-2's attention
    /// implementation has different cache semantics.
    ///
    /// # Arguments
    ///
    /// * `model` - The decoder language model
    /// * `initial_tokens` - Prompt token IDs
    /// * `cache` - KV cache to populate
    ///
    /// # Returns
    ///
    /// Vocabulary logits as 1D array of shape `[vocab_size]`
    ///
    /// # Errors
    ///
    /// - Empty prompt
    /// - Model doesn't support CPU execution
    /// - Forward pass failure
    async fn prefill(
        &self,
        model: &dyn DecoderLanguageModel,
        initial_tokens: &[u32],
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        if initial_tokens.is_empty() {
            return Err(anyhow!("Cannot prefill with empty prompt"));
        }

        let ops = model
            .decoder_cpu_ops()
            .ok_or_else(|| anyhow!("Model does not support CPU execution"))?;

        let prompt_len = initial_tokens.len();
        let prefill_start = Instant::now();

        debug!(
            "CPU prefill: {} tokens, loop={:?}",
            prompt_len,
            model.autoregressive_loop()
        );

        let logits = match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => {
                self.prefill_pipelined(ops, initial_tokens, cache)?
            }
            AutoregressiveLoop::Legacy => {
                self.prefill_legacy(ops, initial_tokens, cache)?
            }
        };

        debug!(
            "CPU prefill complete: {:.2}ms",
            prefill_start.elapsed().as_secs_f64() * 1000.0
        );

        Ok(logits)
    }

    /// Processes a single token and returns logits for the next token.
    ///
    /// This is the inner loop of autoregressive generation. Uses the KV cache
    /// populated during prefill and previous decode steps.
    ///
    /// # Attention Mask Sizing
    ///
    /// The mask size depends on the autoregressive loop type:
    /// - **Pipelined**: mask size = `seq_len` (matches actual sequence)
    /// - **Legacy**: mask size = `seq_len + 1` (GPT-2 quirk)
    ///
    /// # Arguments
    ///
    /// * `model` - The decoder language model
    /// * `token_tensor` - Tensor containing the current token (`[1, 1]`)
    /// * `seq_len` - Total sequence length so far (prompt + generated)
    /// * `cache` - KV cache with previous keys/values
    ///
    /// # Returns
    ///
    /// Vocabulary logits as 1D array of shape `[vocab_size]`
    async fn decode_one(
        &self,
        model: &dyn DecoderLanguageModel,
        token_tensor: &Self::Tensor,
        seq_len: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        let ops = model
            .decoder_cpu_ops()
            .ok_or_else(|| anyhow!("Model does not support CPU execution"))?;

        // Extract token ID from tensor
        let token_id = token_tensor[[0, 0]];
        let token_slice = [token_id];
        let input = ModelInput::from_tokens(&token_slice);

        // Build attention mask
        // Mask length varies by autoregressive loop type
        let mask_len = match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => seq_len,
            AutoregressiveLoop::Legacy => seq_len + 1,
        };

        let attention_mask = ops.get_attention_mask(1, mask_len - 1)?;

        // Position offset is sequence length minus 1 (0-indexed)
        let past_len = seq_len - 1;

        trace!(
            "CPU decode: token={}, seq_len={}, mask_len={}, past_len={}",
            token_id, seq_len, mask_len, past_len
        );

        // Forward pass
        let decoder_output = ops.decoder().forward(
            input,
            &attention_mask,
            past_len,
            Some(cache),
        )?;

        // Project to logits and extract single position
        let logits_3d = ops.project_to_logits(&decoder_output)?;

        // Shape: [1, 1, vocab_size] → [vocab_size]
        Ok(logits_3d.slice(s![0, 0, ..]).to_owned())
    }
}

impl CpuDecoderBackend {
    /// Pipelined prefill: process entire prompt in one forward pass.
    fn prefill_pipelined(
        &self,
        ops: &dyn CpuDecoderOps,
        tokens: &[u32],
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        let prompt_len = tokens.len();

        // Build causal mask for full sequence
        let attention_mask = ops.get_attention_mask(prompt_len, 0)?;

        // Single forward pass
        let decoder_output = ops.decoder().forward(
            ModelInput::from_tokens(tokens),
            &attention_mask,
            0, // Position offset is 0 for prefill
            Some(cache),
        )?;

        // Project to logits
        let logits_3d = ops.project_to_logits(&decoder_output)?;

        // Extract last position: [1, seq, vocab] → [vocab]
        Ok(logits_3d.slice(s![0, -1, ..]).to_owned())
    }

    /// Legacy prefill: two-phase for GPT-2 compatibility.
    fn prefill_legacy(
        &self,
        ops: &dyn CpuDecoderOps,
        tokens: &[u32],
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        let prompt_len = tokens.len();

        // Phase 1: Fill cache with all tokens (no projection)
        let mask_full = Array2::ones((1, prompt_len));
        ops.decoder().forward(
            ModelInput::from_tokens(tokens),
            &mask_full,
            0,
            Some(cache),
        )?;

        // Phase 2: Re-process last token to get logits
        let last_token = tokens[prompt_len - 1];
        let last_token_slice = [last_token];

        let mask_step = ops.get_attention_mask(1, prompt_len)?;

        let decoder_output = ops.decoder().forward(
            ModelInput::from_tokens(&last_token_slice),
            &mask_step,
            prompt_len, // Offset
            Some(cache),
        )?;

        let logits_3d = ops.project_to_logits(&decoder_output)?;

        // Extract single position: [1, 1, vocab] → [vocab]
        Ok(logits_3d.slice(s![0, 0, ..]).to_owned())
    }
}