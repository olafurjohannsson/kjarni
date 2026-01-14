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
//! │  prefill(tokens: &Array2<u32>)                                  │
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
//! │                                                                 │
//! │  decode_one(token: &Array2<u32>)                               │
//! │      │                                                          │
//! │      └──────────────────────────────────────────────────────►  │
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
//! # Token Tensor Lifecycle
//!
//! ```text
//! Tokenizer output (CPU)
//!         │
//!         ▼
//! prefill(&Array2<u32>) ────► Populate cache, return first logits
//!         │
//!         ▼
//! new_decode_token() ────────► Allocate reusable Array2 [1, 1]
//!         │
//!         ▼
//! ┌───────────────────────────────────────┐
//! │  Decode Loop                          │
//! │  ┌─────────────────────────────────┐  │
//! │  │ update_decode_token(token_id)   │  │  ◄── Simple array write
//! │  │ decode_one(&token_tensor)       │  │  ◄── Full CPU forward
//! │  │ sample(logits) → next token_id  │  │  ◄── CPU sampling
//! │  └─────────────────────────────────┘  │
//! └───────────────────────────────────────┘
//! ```
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
use crate::models::base::AutoregressiveLoop;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use log::{debug, trace};
use ndarray::{s, Array1, Array2, Array3};
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
/// // Tokenize prompt
/// let prompt_tokens = tokenizer.encode("Hello, world!")?;
/// let tokens = Array2::from_shape_vec((1, prompt_tokens.len()), prompt_tokens)?;
///
/// // Prefill
/// let logits = backend.prefill(&model, &tokens, &mut cache).await?;
/// let first_token = sample(&logits, &config);
///
/// // Allocate reusable decode tensor
/// let mut decode_token = backend.new_decode_token()?;
///
/// // Decode loop
/// for _ in 0..max_tokens {
///     backend.update_decode_token(&mut decode_token, current_token)?;
///     let logits = backend.decode_one(&model, &decode_token, seq_len, &mut cache).await?;
///     current_token = sample(&logits);
/// }
/// ```
#[derive(Clone, Default)]
pub struct CpuDecoderBackend;

impl CpuDecoderBackend {
    /// Creates a new CPU decoder backend.
    ///
    /// This is a zero-cost operation as the backend is stateless.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DecoderGenerationBackend for CpuDecoderBackend {
    /// Token tensor type for decode loop: 2D ndarray with shape `[1, 1]`.
    ///
    /// Using `Array2` for consistency with the prefill input type and
    /// the ndarray-based pipeline, even though it always holds a single token.
    type DecodeToken = Array2<u32>;

    /// Creates a single-token tensor for the decode loop.
    ///
    /// Allocates a `[1, 1]` array that will be reused for each decode step.
    /// This avoids per-step allocation overhead in the hot loop.
    ///
    /// # Returns
    ///
    /// A 2D array of shape `[1, 1]` initialized to zero.
    fn new_decode_token(&self) -> Result<Self::DecodeToken> {
        Ok(Array2::zeros((1, 1)))
    }

    /// Updates the decode token tensor with a newly sampled token ID.
    ///
    /// This is a simple array write - no allocation required.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The token tensor to update (must have shape `[1, 1]`)
    /// * `new_token_id` - The token ID to write
    fn update_decode_token(
        &self,
        tensor: &mut Self::DecodeToken,
        new_token_id: u32,
    ) -> Result<()> {
        tensor[[0, 0]] = new_token_id;
        Ok(())
    }

    /// Processes the prompt and returns logits for the first generated token.
    ///
    /// The prompt tokens are received as a CPU array and processed directly.
    /// No device transfer is needed since everything runs on CPU.
    ///
    /// # Prefill Strategies
    ///
    /// The prefill behavior depends on the model's autoregressive loop type:
    ///
    /// ## Pipelined (Llama, Phi, Mistral)
    ///
    /// Processes all tokens in a single forward pass:
    ///
    /// 1. Build causal attention mask for full prompt
    /// 2. Embed all tokens
    /// 3. Forward pass through all layers (populates KV cache)
    /// 4. Apply final normalization
    /// 5. Project to logits
    /// 6. Return logits for last position
    ///
    /// ```text
    /// [tok1, tok2, tok3, tok4] ─────────────────► logits for position 4
    /// ```
    ///
    /// ## Legacy (GPT-2)
    ///
    /// Splits prefill into two phases:
    ///
    /// 1. Forward pass for tokens `[0..n-1]` (cache only, no projection)
    /// 2. Forward pass for token `[n-1]` with projection
    /// 3. Return logits for that position
    ///
    /// ```text
    /// [tok1, tok2, tok3] ──► cache only (no logits)
    ///              [tok4] ──► logits for position 4
    /// ```
    ///
    /// This two-phase approach is required because GPT-2's attention
    /// implementation has different cache semantics.
    ///
    /// # Arguments
    ///
    /// * `model` - The decoder language model
    /// * `tokens` - Prompt token IDs as CPU array of shape `[batch, seq_len]`
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
        tokens: &Array2<u32>,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        if tokens.is_empty() {
            return Err(anyhow!("Cannot prefill with empty prompt"));
        }

        let ops = model
            .decoder_cpu_ops()
            .ok_or_else(|| anyhow!("Model does not support CPU execution"))?;

        let prompt_len = tokens.shape()[1];
        let prefill_start = Instant::now();

        debug!(
            "CPU prefill: {} tokens, strategy={:?}",
            prompt_len,
            model.autoregressive_loop()
        );

        let logits = match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => self.prefill_pipelined(ops, tokens, cache)?,
            AutoregressiveLoop::Legacy => self.prefill_legacy(ops, tokens, cache)?,
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
    /// The decode token should be pre-allocated via `new_decode_token` and
    /// updated via `update_decode_token` to minimize allocation overhead,
    /// though the cost is minimal on CPU.
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
        token_tensor: &Self::DecodeToken,
        seq_len: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        let ops = model
            .decoder_cpu_ops()
            .ok_or_else(|| anyhow!("Model does not support CPU execution"))?;

        // Position offset is sequence length minus 1 (0-indexed)
        let past_len = seq_len - 1;

        // Embed the single token
        let hidden_states: Array3<f32> = ops.embed(token_tensor, past_len)?;

        // Build attention mask
        // Mask length varies by autoregressive loop type
        let mask_len = match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => seq_len,
            AutoregressiveLoop::Legacy => seq_len + 1,
        };

        let attention_mask = ops.get_attention_mask(1, mask_len - 1)?;

        trace!(
            "CPU decode: token={}, seq_len={}, mask_len={}, past_len={}",
            token_tensor[[0, 0]],
            seq_len,
            mask_len,
            past_len
        );

        // Forward pass through decoder layers (without final norm)
        let decoder_output =
            ops.decoder()
                .forward(&hidden_states, &attention_mask, past_len, Some(cache))?;

        // Project to logits and extract single position
        let logits_3d = ops.project_to_logits(&decoder_output)?;

        // Shape: [1, 1, vocab_size] → [vocab_size]
        Ok(logits_3d.slice(s![0, 0, ..]).to_owned())
    }
}

impl CpuDecoderBackend {
    /// Pipelined prefill: process entire prompt in one forward pass.
    ///
    /// This is the standard path for modern models (Llama, Mistral, Phi).
    fn prefill_pipelined(
        &self,
        ops: &dyn CpuDecoderOps,
        tokens: &Array2<u32>,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        let prompt_len = tokens.shape()[1];

        // Build causal mask for full sequence
        let attention_mask = ops.get_attention_mask(prompt_len, 0)?;

        // Embed all tokens
        let hidden_states = ops.embed(tokens, 0)?;

        // Single forward pass through all layers (includes final_norm)
        let decoder_output = ops.decoder().forward(
            &hidden_states,
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
    ///
    /// This path is kept for backward compatibility with older models
    /// that have different cache semantics.
    fn prefill_legacy(
        &self,
        ops: &dyn CpuDecoderOps,
        tokens: &Array2<u32>,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        let prompt_len = tokens.shape()[1];

        // Phase 1: Fill cache with all tokens (no projection)
        let hidden_states = ops.embed(tokens, 0)?;
        let mask_full = Array2::ones((1, prompt_len));

        ops.decoder().forward(&hidden_states, &mask_full, 0, Some(cache))?;

        // Phase 2: Re-process last token to get logits
        let last_token = tokens[[0, prompt_len - 1]];
        let last_token_array = Array2::from_elem((1, 1), last_token);

        let hidden_states = ops.embed(&last_token_array, prompt_len - 1)?;
        let mask_step = ops.get_attention_mask(1, prompt_len)?;

        let decoder_output = ops.decoder().forward(
            &hidden_states,
            &mask_step,
            prompt_len, // Offset
            Some(cache),
        )?;

        let logits_3d = ops.project_to_logits(&decoder_output)?;

        // Extract single position: [1, 1, vocab] → [vocab]
        Ok(logits_3d.slice(s![0, 0, ..]).to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_decode_token_shape() {
        let backend = CpuDecoderBackend::new();
        let token = backend.new_decode_token().unwrap();

        assert_eq!(token.shape(), &[1, 1]);
        assert_eq!(token[[0, 0]], 0);
    }

    #[test]
    fn test_update_decode_token() {
        let backend = CpuDecoderBackend::new();
        let mut token = backend.new_decode_token().unwrap();

        backend.update_decode_token(&mut token, 42).unwrap();
        assert_eq!(token[[0, 0]], 42);

        backend.update_decode_token(&mut token, 1337).unwrap();
        assert_eq!(token[[0, 0]], 1337);
    }

    #[test]
    fn test_backend_is_stateless() {
        // Backend should be Clone and Default since it's stateless
        let backend1 = CpuDecoderBackend::default();
        let backend2 = backend1.clone();

        // Both should work independently
        let token1 = backend1.new_decode_token().unwrap();
        let token2 = backend2.new_decode_token().unwrap();

        assert_eq!(token1, token2);
    }

    #[test]
    fn test_token_array_shape() {
        let tokens = Array2::from_shape_vec((1, 5), vec![1u32, 2, 3, 4, 5]).unwrap();

        assert_eq!(tokens.shape(), &[1, 5]);
        assert_eq!(tokens.shape()[1], 5); // seq_len
    }
}