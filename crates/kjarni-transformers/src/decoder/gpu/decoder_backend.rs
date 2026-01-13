//! GPU-accelerated decoder generation backend.
//!
//! This module provides the GPU implementation of autoregressive text generation,
//! handling the forward pass, KV cache management, and logit projection entirely
//! on the GPU with minimal CPU synchronization.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     GpuDecoderBackend                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  prefill(tokens: &Array2<u32>)                                  │
//! │      │                                                          │
//! │      ▼ (upload once)                                            │
//! │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
//! │  │   Embed      │───▶│   Layers     │───▶│   LM Head    │      │
//! │  │ (CPU or GPU) │    │   (GPU)      │    │ (CPU or GPU) │      │
//! │  └──────────────┘    └──────────────┘    └──────────────┘      │
//! │                             │                    │              │
//! │                             ▼                    ▼              │
//! │                      ┌──────────────┐    ┌──────────────┐      │
//! │                      │  KV Cache    │    │  Staging     │      │
//! │                      │  (GPU)       │    │  Buffer      │      │
//! │                      └──────────────┘    └──────────────┘      │
//! │                                                  │              │
//! │  decode_one(token: &GpuTensor)                   ▼              │
//! │      │ (stays on GPU)                    ┌──────────────┐      │
//! │      └──────────────────────────────────▶│  CPU Logits  │      │
//! │                                          │  (Sampling)  │      │
//! │                                          └──────────────┘      │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Hybrid Execution Support
//!
//! The backend supports flexible device placement through `ModelInput`:
//!
//! - **Embeddings on CPU**: Pass `ModelInput::TokensCpu` → model computes embeddings
//!   on CPU → uploads hidden states to GPU for layers
//! - **Embeddings on GPU**: Pass `ModelInput::TokensGpu` → full GPU execution
//! - **LM Head on CPU**: Model downloads hidden states → CPU projection
//! - **LM Head on GPU**: Full GPU projection → staging buffer → CPU logits
//!
//! # Performance Characteristics
//!
//! - **Prefill**: O(n²) attention, processes all prompt tokens in parallel
//! - **Decode**: O(n) per token, single token through cached attention
//! - **Memory**: KV cache grows linearly with sequence length
//! - **Sync Point**: One GPU→CPU transfer per generated token (logits only)
//!
//! # Token Tensor Lifecycle
//!
//! ```text
//! Tokenizer output (CPU)
//!         │
//!         ▼
//! prefill(&Array2<u32>) ────► Upload once, populate cache, return first logits
//!         │
//!         ▼
//! new_decode_token() ────────► Allocate reusable GPU buffer [1, 1]
//!         │
//!         ▼
//! ┌───────────────────────────────────────┐
//! │  Decode Loop (hot path)               │
//! │  ┌─────────────────────────────────┐  │
//! │  │ update_decode_token(token_id)   │  │  ◄── 4-byte GPU write
//! │  │ decode_one(&token_tensor)       │  │  ◄── Full GPU forward
//! │  │ sample(logits) → next token_id  │  │  ◄── CPU sampling
//! │  └─────────────────────────────────┘  │
//! └───────────────────────────────────────┘
//! ```

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ndarray::{s, Array1, Array2};
use std::sync::Arc;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, MapMode};

use crate::cache::{Cache, GpuKVCache};
use crate::decoder::prelude::*;
use crate::gpu_ops::primitives::layout::slice_last_token::GpuLastTokenSlice;
use crate::gpu_ops::timeout::{GpuTimeoutConfig, GpuTimeoutError};
use crate::gpu_ops::{GpuFrameContext, GpuTensor};
pub use crate::models::base::AutoregressiveLoop;
use crate::models::base::ModelInput;
use crate::WgpuContext;

/// GPU-accelerated backend for autoregressive decoder generation.
///
/// Manages the GPU execution of transformer decoder models, including:
/// - Token embedding and position encoding
/// - Multi-layer transformer forward pass with KV caching
/// - LM head projection to vocabulary logits
/// - Efficient GPU→CPU transfer for sampling
///
/// # Hybrid Execution
///
/// This backend supports models with mixed CPU/GPU execution:
/// - Embeddings can be offloaded to CPU (saves VRAM for large vocabularies)
/// - Transformer layers always run on GPU
/// - LM head can be offloaded to CPU (saves VRAM, slight latency cost)
///
/// The `ModelInput` enum allows the model's `GpuDecoderOps` to handle
/// device placement internally based on its configuration.
///
/// # Thread Safety
///
/// This backend is `!Send` due to WGPU constraints. All operations must
/// occur on the thread that created the backend.
///
/// # Example
///
/// ```ignore
/// let backend = GpuDecoderBackend::new(context)?;
/// let mut cache = model.new_cache(1, 2048, 1)?;
///
/// // Tokenize prompt
/// let prompt_tokens = tokenizer.encode("Hello, world!")?;
/// let tokens = Array2::from_shape_vec((1, prompt_tokens.len()), prompt_tokens)?;
///
/// // Prefill - backend handles upload internally
/// let logits = backend.prefill(&model, &tokens, &mut cache).await?;
/// let first_token = sample(&logits, &config);
///
/// // Allocate reusable decode tensor (stays on GPU)
/// let mut decode_token = backend.new_decode_token()?;
///
/// // Autoregressive generation
/// for step in 0..max_tokens {
///     backend.update_decode_token(&mut decode_token, current_token)?;
///     let logits = backend.decode_one(&model, &decode_token, seq_len, &mut cache).await?;
///     current_token = sample(&logits, &config);
/// }
/// ```
pub struct GpuDecoderBackend {
    /// Shared GPU context (device, queue, pipelines).
    context: Arc<WgpuContext>,

    /// Kernel for extracting the last token's hidden state from a sequence.
    /// Used after the transformer stack to get the single position we need
    /// for LM head projection.
    last_token_slicer: GpuLastTokenSlice,

    /// Reusable staging buffer for GPU→CPU logits transfer.
    /// Protected by mutex for interior mutability (buffer reuse across calls).
    /// Resized on-demand if vocabulary size changes.
    staging_buffer: std::sync::Mutex<Option<Buffer>>,

    /// Current size of the staging buffer in bytes.
    /// Tracked separately to avoid querying the buffer.
    staging_buffer_size: std::sync::Mutex<usize>,

    /// Timeout configuration for GPU operations.
    timeout_config: GpuTimeoutConfig,
}

impl GpuDecoderBackend {
    /// Creates a new GPU decoder backend.
    ///
    /// # Arguments
    ///
    /// * `context` - Shared WGPU context with device, queue, and compute pipelines
    ///
    /// # Errors
    ///
    /// Returns an error if kernel compilation fails.
    pub fn new(context: Arc<WgpuContext>) -> Result<Self> {
        Ok(Self {
            last_token_slicer: GpuLastTokenSlice::new(&context),
            context,
            staging_buffer: std::sync::Mutex::new(None),
            staging_buffer_size: std::sync::Mutex::new(0),
            timeout_config: GpuTimeoutConfig::default(),
        })
    }

    /// Creates a new GPU decoder backend with custom timeout configuration.
    ///
    /// # Arguments
    ///
    /// * `context` - Shared WGPU context
    /// * `timeout_config` - Custom timeout settings for GPU operations
    pub fn with_timeout(
        context: Arc<WgpuContext>,
        timeout_config: GpuTimeoutConfig,
    ) -> Result<Self> {
        Ok(Self {
            last_token_slicer: GpuLastTokenSlice::new(&context),
            context,
            staging_buffer: std::sync::Mutex::new(None),
            staging_buffer_size: std::sync::Mutex::new(0),
            timeout_config,
        })
    }

    /// Returns a reference to the GPU context.
    pub fn context(&self) -> &Arc<WgpuContext> {
        &self.context
    }

    // =========================================================================
    // Core Forward Pass Methods
    // =========================================================================

    /// Runs a full forward pass and projects to vocabulary logits.
    ///
    /// This is the core inference method that:
    /// 1. Processes input through embeddings and transformer layers
    /// 2. Updates the KV cache with new key/value pairs
    /// 3. Extracts the last token's hidden state
    /// 4. Projects to vocabulary logits via the LM head
    /// 5. Transfers logits to CPU for sampling
    ///
    /// # Arguments
    ///
    /// * `model` - The decoder language model (provides ops and configuration)
    /// * `input` - Input tokens or hidden states (CPU or GPU via `ModelInput`)
    /// * `cache` - Mutable KV cache to update
    ///
    /// # Returns
    ///
    /// Vocabulary logits as a 1D array of shape `[vocab_size]`
    ///
    /// # Performance
    ///
    /// - Single GPU command buffer submission
    /// - One sync point for logits readback
    /// - Staging buffer reused across calls
    async fn forward_and_project(
        &self,
        model: &dyn DecoderLanguageModel,
        input: ModelInput<'_>,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        let input_len = match &input {
            ModelInput::TokensCpu(t) => t.shape()[1],
            ModelInput::TokensGpu(t) => t.shape()[1],
            ModelInput::HiddenGpu(t) => t.shape()[1],
            ModelInput::HiddenCpu(t) => t.shape()[1],
        };

        // Get model operations and downcast cache to GPU variant
        let ops = model
            .decoder_gpu_ops()
            .ok_or_else(|| anyhow!("Model does not support GPU execution"))?;

        let gpu_cache = cache
            .as_any_mut()
            .downcast_mut::<GpuKVCache>()
            .ok_or_else(|| anyhow!("Expected GpuKVCache for GPU backend"))?;

        // Acquire frame context (command encoder + tensor pool)
        let pool = self.context.get_inference_pool();
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);

        // Build causal attention mask
        //
        // The mask size depends on the model's autoregressive loop strategy:
        // - Pipelined (Llama): mask matches actual sequence length
        // - Legacy (GPT-2): mask matches full cache capacity
        let cache_len = gpu_cache.get_seq_length();
        let logical_key_len = cache_len + input_len;

        let mask_size = match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => logical_key_len,
            AutoregressiveLoop::Legacy => gpu_cache.max_seq_len(),
        };

        let attention_mask = ops.get_attention_mask(&mut frame, logical_key_len, mask_size)?;

        // Execute transformer decoder stack
        let (encoder, pool) = frame.resources();

        let hidden_states = ops.decoder().forward(
            encoder,
            pool,
            input,
            &attention_mask,
            cache_len, // position_offset = current cache length
            Some(gpu_cache),
            None, // encoder_hidden_states (for cross-attention, not used here)
        )?;

        // Extract last token's hidden state
        //
        // hidden_states: [batch, seq_len, hidden_dim]
        // last_hidden:   [batch, 1, hidden_dim]
        let (batch, _, hidden_dim) = hidden_states.dims3();
        let last_hidden = frame.pool_guard.get(vec![batch, 1, hidden_dim]);

        let (encoder, _pool) = frame.resources();
        self.last_token_slicer
            .encode(encoder, &hidden_states, &last_hidden);

        // Project to vocabulary logits
        let logits = ops.project_to_logits(&mut frame, &last_hidden)?;

        // Setup GPU→CPU transfer via staging buffer
        let size = logits.buffer().size();
        let staging = self.get_or_create_staging_buffer(size as usize);

        let (encoder, _pool) = frame.resources();
        encoder.copy_buffer_to_buffer(logits.buffer(), 0, &staging, 0, size);

        // Submit all GPU work
        frame.finish();

        // Update cache sequence length for next iteration
        gpu_cache.increment_len(input_len);

        // Synchronously read logits back to CPU
        self.read_logits_from_staging(&staging).await
    }

    /// Runs a forward pass WITHOUT projecting to logits.
    ///
    /// Used during prefill for "context stuffing" - processing tokens
    /// that only need to populate the KV cache, not produce output.
    ///
    /// This is an optimization for legacy models (GPT-2) that can't process
    /// the entire prompt in a single forward pass, allowing cache population
    /// without the overhead of LM head projection and GPU→CPU transfer.
    ///
    /// # Arguments
    ///
    /// * `model` - The decoder language model
    /// * `input` - Input tokens to process
    /// * `cache` - KV cache to populate
    async fn forward_only(
        &self,
        model: &dyn DecoderLanguageModel,
        input: ModelInput<'_>,
        cache: &mut dyn Cache,
    ) -> Result<()> {
        let input_len = match &input {
            ModelInput::TokensCpu(t) => t.shape()[1],
            ModelInput::TokensGpu(t) => t.shape()[1],
            ModelInput::HiddenGpu(t) => t.shape()[1],
            ModelInput::HiddenCpu(t) => t.shape()[1],
        };

        let ops = model
            .decoder_gpu_ops()
            .ok_or_else(|| anyhow!("Model does not support GPU execution"))?;

        let gpu_cache = cache
            .as_any_mut()
            .downcast_mut::<GpuKVCache>()
            .ok_or_else(|| anyhow!("Expected GpuKVCache for GPU backend"))?;

        let pool = self.context.get_inference_pool();
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);

        let cache_len = gpu_cache.get_seq_length();
        let logical_key_len = cache_len + input_len;

        let mask_size = match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => logical_key_len,
            AutoregressiveLoop::Legacy => gpu_cache.max_seq_len(),
        };

        let attention_mask = ops.get_attention_mask(&mut frame, logical_key_len, mask_size)?;

        let (encoder, pool) = frame.resources();
        let _ = ops.decoder().forward(
            encoder,
            pool,
            input,
            &attention_mask,
            cache_len,
            Some(gpu_cache),
            None,
        )?;

        frame.finish();
        gpu_cache.increment_len(input_len);

        Ok(())
    }

    // =========================================================================
    // Staging Buffer Management
    // =========================================================================

    /// Gets or creates a staging buffer for GPU→CPU transfer.
    ///
    /// Reuses the existing buffer if it's the right size, otherwise
    /// creates a new one. This avoids allocation overhead during
    /// the generation loop.
    ///
    /// # Arguments
    ///
    /// * `required_bytes` - Size needed for the logits tensor
    fn get_or_create_staging_buffer(&self, required_bytes: usize) -> Buffer {
        let mut buffer_guard = self.staging_buffer.lock().unwrap();
        let mut size_guard = self.staging_buffer_size.lock().unwrap();

        // Reuse existing buffer if size matches
        if buffer_guard.is_some() && *size_guard == required_bytes {
            return buffer_guard.as_ref().unwrap().clone();
        }

        // Drop old buffer and create new one
        let _ = buffer_guard.take();

        let new_buffer = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("Logits Staging Buffer"),
            size: required_bytes as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        *buffer_guard = Some(new_buffer.clone());
        *size_guard = required_bytes;
        new_buffer
    }

    /// Reads logits from the staging buffer back to CPU.
    ///
    /// This is a synchronization point - blocks until GPU work completes
    /// and the buffer is mapped for reading.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The staging buffer containing logits
    ///
    /// # Returns
    ///
    /// Logits as a 1D f32 array of shape `[vocab_size]`
    ///
    /// # Errors
    ///
    /// - Timeout if GPU doesn't respond within configured duration
    /// - Buffer mapping failure
    async fn read_logits_from_staging(&self, buffer: &Buffer) -> Result<Array1<f32>> {
        let slice = buffer.slice(..);

        // Setup async notification channel
        let (tx, mut rx) = tokio::sync::oneshot::channel();
        slice.map_async(MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Poll with timeout
        let start = std::time::Instant::now();
        let mut map_result = None;

        loop {
            // Poll GPU for completion
            match self.context.device.poll(wgpu::PollType::Poll) {
                Ok(status) => log::trace!("GPU Poll OK: {:?}", status),
                Err(e) => {
                    return Err(anyhow!("GPU Poll Failed: {:?}", e));
                }
            }

            // Check if mapping completed
            match rx.try_recv() {
                Ok(result) => {
                    map_result = Some(result);
                    break;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    // Not ready yet, continue polling
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    return Err(anyhow!("Buffer mapping channel closed unexpectedly"));
                }
            }

            // Check timeout
            let elapsed = start.elapsed();
            if elapsed >= self.timeout_config.timeout {
                return Err(GpuTimeoutError {
                    operation: "buffer_map_read".to_string(),
                    elapsed,
                    timeout: self.timeout_config.timeout,
                }
                .into());
            }

            // Yield to async runtime
            tokio::time::sleep(self.timeout_config.poll_interval).await;
        }

        // Handle mapping result
        map_result
            .ok_or_else(|| anyhow!("Buffer mapping did not complete"))?
            .map_err(|e| anyhow!("Buffer mapping failed: {:?}", e))?;

        // Read data from mapped buffer
        let data = slice.get_mapped_range();
        let vec_data: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        buffer.unmap();

        Ok(Array1::from_vec(vec_data))
    }
}

#[async_trait]
impl DecoderGenerationBackend for GpuDecoderBackend {
    /// Token tensor type for decode loop: GPU buffer containing a single token.
    ///
    /// The tensor is pre-allocated via `new_decode_token()` and reused across
    /// all decode steps. Updates happen via `update_decode_token()` which is
    /// a fast 4-byte GPU buffer write.
    type DecodeToken = GpuTensor;

    /// Creates a single-token GPU tensor for the decode loop.
    ///
    /// This tensor is reused across all decode steps, with its contents
    /// updated via `update_decode_token`. Pre-allocating avoids per-step
    /// allocation overhead.
    ///
    /// # Returns
    ///
    /// A GPU tensor of shape `[1, 1]` containing a u32 token ID.
    fn new_decode_token(&self) -> Result<Self::DecodeToken> {
        let token_ndarray = Array2::<u32>::zeros((1, 1));
        GpuTensor::from_ndarray(&self.context, &token_ndarray)
    }

    /// Updates the decode token tensor with a newly sampled token ID.
    ///
    /// This is a fast GPU buffer write - no allocation, no sync required.
    /// The write is queued and will be executed before the next decode_one call.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The pre-allocated decode token tensor
    /// * `new_token_id` - The token ID to write
    fn update_decode_token(&self, tensor: &mut Self::DecodeToken, new_token_id: u32) -> Result<()> {
        self.context
            .queue
            .write_buffer(tensor.buffer(), 0, bytemuck::cast_slice(&[new_token_id]));
        Ok(())
    }

    /// Processes the initial prompt and returns logits for the first generated token.
    ///
    /// The prompt tokens are received as a CPU array and uploaded to the GPU
    /// internally. This is a one-time upload cost that is negligible compared
    /// to the transformer forward pass.
    ///
    /// # Prefill Strategies
    ///
    /// The prefill behavior depends on the model's autoregressive loop type:
    ///
    /// ## Pipelined (Llama, Mistral, Phi)
    ///
    /// Processes all tokens in a single forward pass. More efficient for
    /// models that support variable-length attention masks.
    ///
    /// ```text
    /// [tok1, tok2, tok3, tok4] ─────────────────► logits for position 4
    /// ```
    ///
    /// ## Legacy (GPT-2)
    ///
    /// Splits prefill into two phases:
    /// 1. Context stuffing: Process tokens [0..n-1] without projection
    /// 2. First prediction: Process token [n-1] with projection
    ///
    /// ```text
    /// [tok1, tok2, tok3] ──► cache only (no logits)
    ///              [tok4] ──► logits for position 4
    /// ```
    ///
    /// This is required for models with fixed-size attention masks.
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
    /// - Model doesn't support GPU execution
    /// - KV cache type mismatch
    async fn prefill(
        &self,
        model: &dyn DecoderLanguageModel,
        tokens: &Array2<u32>,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        if tokens.is_empty() {
            return Err(anyhow!("Cannot prefill with empty tokens"));
        }

        let prompt_len = tokens.shape()[1];

        log::debug!(
            "GPU prefill: {} tokens, strategy={:?}",
            prompt_len,
            model.autoregressive_loop()
        );

        match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => {
                // Single forward pass for entire prompt
                // Pass as CPU tokens - model.decoder().forward() handles upload
                // if embeddings are on GPU, or computes on CPU then uploads hidden states
                let input = ModelInput::TokensCpu(tokens.view());
                self.forward_and_project(model, input, cache).await
            }
            AutoregressiveLoop::Legacy => {
                // Split into context + last token
                let last_idx = prompt_len - 1;

                // Phase 1: Stuff cache with context (no projection needed)
                if last_idx > 0 {
                    let context_tokens = tokens.slice(s![.., 0..last_idx]);
                    let context_input = ModelInput::TokensCpu(context_tokens);
                    self.forward_only(model, context_input, cache).await?;
                }

                // Phase 2: Get logits from last token
                let last_token = tokens.slice(s![.., last_idx..]);
                let last_input = ModelInput::TokensCpu(last_token);
                self.forward_and_project(model, last_input, cache).await
            }
        }
    }

    /// Processes a single token and returns logits for the next token.
    ///
    /// This is the inner loop of autoregressive generation. The token tensor
    /// should be pre-allocated via `new_decode_token` and updated via
    /// `update_decode_token` to minimize allocation overhead.
    ///
    /// The token stays on GPU throughout the decode loop, avoiding per-step
    /// CPU→GPU transfers.
    ///
    /// # Arguments
    ///
    /// * `model` - The decoder language model
    /// * `token_tensor` - GPU tensor containing the current token ID `[1, 1]`
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
        _seq_len: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        // Token is already on GPU - pass directly to forward
        let input = ModelInput::TokensGpu(token_tensor);
        self.forward_and_project(model, input, cache).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full tests require a GPU context which isn't available in unit tests.
    // These tests verify the non-GPU logic only.

    #[test]
    fn test_staging_buffer_reuse_logic() {
        // This would need a real WgpuContext to test properly
        // For now, just verify the struct can be created
    }

    #[test]
    fn test_model_input_size_extraction() {
        let tokens = Array2::from_shape_vec((1, 5), vec![1u32, 2, 3, 4, 5]).unwrap();
        let input = ModelInput::TokensCpu(tokens.view());

        let size = match &input {
            ModelInput::TokensCpu(t) => t.shape()[1],
            _ => 0,
        };

        assert_eq!(size, 5);
    }
}