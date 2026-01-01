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
//! │  prefill(tokens)                                                │
//! │      │                                                          │
//! │      ▼                                                          │
//! │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
//! │  │   Embed      │───▶│   Layers     │───▶│   LM Head    │      │
//! │  │  (GPU)       │    │   (GPU)      │    │   (GPU)      │      │
//! │  └──────────────┘    └──────────────┘    └──────────────┘      │
//! │                             │                    │              │
//! │                             ▼                    ▼              │
//! │                      ┌──────────────┐    ┌──────────────┐      │
//! │                      │  KV Cache    │    │  Staging     │      │
//! │                      │  (GPU)       │    │  Buffer      │      │
//! │                      └──────────────┘    └──────────────┘      │
//! │                                                  │              │
//! │                                                  ▼              │
//! │                                          ┌──────────────┐      │
//! │                                          │  CPU Logits  │      │
//! │                                          │  (Sampling)  │      │
//! │                                          └──────────────┘      │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance Characteristics
//!
//! - **Prefill**: O(n²) attention, processes all prompt tokens in parallel
//! - **Decode**: O(n) per token, single token through cached attention
//! - **Memory**: KV cache grows linearly with sequence length
//! - **Sync Point**: One GPU→CPU transfer per generated token (logits only)

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array1, Array2};
use std::sync::Arc;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, MapMode};

use crate::WgpuContext;
use crate::cache::{Cache, GpuKVCache};
use crate::common::{CancellationToken, DecodingStrategy, GenerationConfig};
use crate::decoder::prelude::*;
use crate::gpu_ops::primitives::layout::slice_last_token::GpuLastTokenSlice;
use crate::gpu_ops::timeout::{GpuTimeoutConfig, GpuTimeoutError, poll_with_timeout_async};
use crate::gpu_ops::{GpuFrameContext, GpuTensor, Kernel};
pub use crate::models::base::AutoregressiveLoop;
use crate::models::base::ModelInput;

/// GPU-accelerated backend for autoregressive decoder generation.
///
/// Manages the GPU execution of transformer decoder models, including:
/// - Token embedding and position encoding
/// - Multi-layer transformer forward pass with KV caching
/// - LM head projection to vocabulary logits
/// - Efficient GPU→CPU transfer for sampling
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
/// // Prefill with prompt
/// let logits = backend.prefill(&model, &prompt_tokens, &mut cache).await?;
/// let first_token = sample(&logits, &config);
///
/// // Autoregressive generation
/// let mut token_tensor = backend.new_token_tensor()?;
/// for step in 0..max_tokens {
///     backend.update_token_tensor(&mut token_tensor, current_token)?;
///     let logits = backend.decode_one(&model, &token_tensor, seq_len, &mut cache).await?;
///     current_token = sample(&logits, &config);
/// }
/// ```
pub struct GpuDecoderBackend {
    /// Shared GPU context (device, queue, pipelines)
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

    /// Timeout configuration for GPU operations
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
            context: context.clone(),
            last_token_slicer: GpuLastTokenSlice::new(&context),
            staging_buffer: std::sync::Mutex::new(None),
            staging_buffer_size: std::sync::Mutex::new(0),
            timeout_config: GpuTimeoutConfig::default(),
        })
    }

    /// Creates a new GPU decoder backend with custom timeout.
    pub fn with_timeout(
        context: Arc<WgpuContext>,
        timeout_config: GpuTimeoutConfig,
    ) -> Result<Self> {
        Ok(Self {
            context: context.clone(),
            last_token_slicer: GpuLastTokenSlice::new(&context),
            staging_buffer: std::sync::Mutex::new(None),
            staging_buffer_size: std::sync::Mutex::new(0),
            timeout_config,
        })
    }

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
    /// * `input` - Input tokens or hidden states (CPU or GPU)
    /// * `seq_len` - Current sequence length (for position encoding)
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
        seq_len: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        // 1. Calculate input length BEFORE moving 'input' (ownership transfer)
        let input_len = match input {
            ModelInput::TokensCpu(t) => t.len(),
            ModelInput::TokensGpu(t) => t.shape()[1],
            ModelInput::HiddenGpu(t) => t.shape()[1],
            ModelInput::HiddenCpu(t) => t.shape()[1],
        };

        // 2. Get model operations and downcast cache to GPU variant
        let ops = model
            .decoder_gpu_ops()
            .ok_or_else(|| anyhow!("Model does not support GPU execution"))?;

        let gpu_cache = cache
            .as_any_mut()
            .downcast_mut::<GpuKVCache>()
            .ok_or_else(|| anyhow!("Expected GpuKVCache"))?;

        // 3. Acquire frame context (command encoder + tensor pool)
        let pool = self.context.get_inference_pool();
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);

        // 4. Build causal attention mask
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

        // 5. Execute transformer decoder stack
        let (encoder, pool) = frame.resources();

        let hidden_states = ops
            .decoder()
            .forward(
                encoder,
                pool,
                input,
                &attention_mask,
                gpu_cache.get_seq_length(),
                Some(gpu_cache),
                None,
            )
            .await?;

        // 6. Extract last token's hidden state
        //
        // hidden_states: [batch, seq_len, hidden_dim]
        // last_hidden:   [batch, 1, hidden_dim]
        let (batch, _, hidden_dim) = hidden_states.dims3();
        let last_hidden = frame.pool_guard.get(vec![batch, 1, hidden_dim]);

        let (encoder, _pool) = frame.resources();
        self.last_token_slicer
            .encode(encoder, &hidden_states, &last_hidden);

        // 7. Project to vocabulary logits
        let logits = ops.project_to_logits(&mut frame, &last_hidden)?;

        // 8. Setup GPU→CPU transfer via staging buffer
        let size = logits.buffer().size();
        let staging = self.get_or_create_staging_buffer(size as usize);

        let (encoder, _pool) = frame.resources();
        encoder.copy_buffer_to_buffer(logits.buffer(), 0, &staging, 0, size);

        // 9. Submit all GPU work
        frame.finish();

        // 10. Update cache sequence length for next iteration
        gpu_cache.increment_len(input_len);

        // 11. Synchronously read logits back to CPU
        self.read_logits_from_staging(&staging).await
    }

    /// Runs a forward pass WITHOUT projecting to logits.
    ///
    /// Used during prefill for "context stuffing" - processing tokens
    /// that only need to populate the KV cache, not produce output.
    ///
    /// This is an optimization for models that can't process the entire
    /// prompt in a single forward pass, allowing cache population without
    /// the overhead of LM head projection and GPU→CPU transfer.
    ///
    /// # Arguments
    ///
    /// * `model` - The decoder language model
    /// * `input` - Input tokens to process
    /// * `seq_len` - Current sequence length
    /// * `cache` - KV cache to populate
    async fn forward_only(
        &self,
        model: &dyn DecoderLanguageModel,
        input: ModelInput<'_>,
        seq_len: usize,
        cache: &mut dyn Cache,
    ) -> Result<()> {
        let input_len = match input {
            ModelInput::TokensCpu(t) => t.len(),
            ModelInput::TokensGpu(t) => t.shape()[1],
            ModelInput::HiddenGpu(t) => t.shape()[1],
            ModelInput::HiddenCpu(t) => t.shape()[1],
        };

        let ops = model
            .decoder_gpu_ops()
            .ok_or_else(|| anyhow!("No GPU Ops"))?;
        let gpu_cache = cache
            .as_any_mut()
            .downcast_mut::<GpuKVCache>()
            .ok_or_else(|| anyhow!("Expected GpuKVCache"))?;

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
        let _ = ops
            .decoder()
            .forward(
                encoder,
                pool,
                input,
                &attention_mask,
                gpu_cache.get_seq_length(),
                Some(gpu_cache),
                None,
            )
            .await?;

        frame.finish();
        gpu_cache.increment_len(input_len);

        Ok(())
    }

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
            // Poll GPU
            let _ = self.context.device.poll(wgpu::PollType::Poll);

            // Check if mapping completed
            match rx.try_recv() {
                Ok(result) => {
                    map_result = Some(result);
                    break;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    // Not ready yet
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

            // Yield to runtime
            tokio::time::sleep(self.timeout_config.poll_interval).await;
        }

        // Handle mapping result
        map_result
            .ok_or_else(|| anyhow!("Buffer mapping did not complete"))?
            .map_err(|e| anyhow!("Buffer mapping failed: {:?}", e))?;

        // Read data
        let data = slice.get_mapped_range();
        let vec_data: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        buffer.unmap();

        Ok(Array1::from_vec(vec_data))
    }
}

#[async_trait(?Send)]
impl DecoderGenerationBackend for GpuDecoderBackend {
    type Tensor = GpuTensor;

    /// Creates a GPU tensor from a slice of token IDs.
    ///
    /// Used during prefill to upload the initial prompt to GPU.
    fn prime_tokens(&self, tokens: &[u32]) -> Result<Self::Tensor> {
        let tokens_ndarray = Array2::from_shape_vec((1, tokens.len()), tokens.to_vec())?;
        GpuTensor::from_ndarray(&self.context, &tokens_ndarray)
    }

    /// Creates an empty single-token tensor for the generation loop.
    ///
    /// This tensor is reused across all decode steps, with its contents
    /// updated via `update_token_tensor`.
    fn new_token_tensor(&self) -> Result<Self::Tensor> {
        let token_ndarray = Array2::<u32>::zeros((1, 1));
        GpuTensor::from_ndarray(&self.context, &token_ndarray)
    }

    /// Updates the token tensor with a newly sampled token ID.
    ///
    /// This is a fast GPU buffer write - no allocation or sync required.
    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_token_id: u32) -> Result<()> {
        self.context
            .queue
            .write_buffer(tensor.buffer(), 0, bytemuck::cast_slice(&[new_token_id]));
        Ok(())
    }

    /// Processes the initial prompt and returns logits for the first generated token.
    ///
    /// # Prefill Strategies
    ///
    /// The prefill behavior depends on the model's autoregressive loop type:
    ///
    /// ## Pipelined (Llama-style)
    /// Processes all tokens in a single forward pass. More efficient for
    /// models that support variable-length attention masks.
    ///
    /// ## Legacy (GPT-2-style)  
    /// Splits prefill into two phases:
    /// 1. Context stuffing: Process tokens [0..n-1] without projection
    /// 2. First prediction: Process token [n-1] with projection
    ///
    /// This is required for models with fixed-size attention masks.
    async fn prefill(
        &self,
        model: &dyn DecoderLanguageModel,
        initial_tokens: &[u32],
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        if initial_tokens.is_empty() {
            return Err(anyhow!("Cannot prefill with empty tokens"));
        }

        let prompt_len = initial_tokens.len();
        let tokens = ndarray::ArrayView2::from_shape((1, prompt_len), initial_tokens)
            .map_err(|e| anyhow!("Failed to create token view: {}", e))?;

        match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => {
                // Single forward pass for entire prompt
                self.forward_and_project(model, ModelInput::TokensCpu(tokens), prompt_len, cache)
                    .await
            }
            AutoregressiveLoop::Legacy => {
                // Split into context + last token
                let last_idx = prompt_len - 1;
                let context_tokens = &initial_tokens[..last_idx];
                let last_token = &initial_tokens[last_idx..];

                // Phase 1: Stuff cache with context (no projection needed)
                if !context_tokens.is_empty() {
                    let context_view =
                        ndarray::ArrayView2::from_shape((1, context_tokens.len()), context_tokens)
                            .unwrap();
                    self.forward_only(
                        model,
                        ModelInput::TokensCpu(context_view),
                        context_tokens.len(),
                        cache,
                    )
                    .await?;
                }

                // Phase 2: Get logits from last token
                let last_view =
                    ndarray::ArrayView2::from_shape((1, last_token.len()), last_token).unwrap();
                self.forward_and_project(model, ModelInput::TokensCpu(last_view), prompt_len, cache)
                    .await
            }
        }
    }

    /// Processes a single token and returns logits for the next token.
    ///
    /// This is the inner loop of autoregressive generation. The token tensor
    /// should be pre-allocated via `new_token_tensor` and updated via
    /// `update_token_tensor` to minimize allocation overhead.
    ///
    /// # Arguments
    ///
    /// * `model` - The decoder language model
    /// * `token_tensor` - GPU tensor containing the current token ID
    /// * `seq_len` - Current sequence length (for position encoding)
    /// * `cache` - KV cache with previous keys/values
    async fn decode_one(
        &self,
        model: &dyn DecoderLanguageModel,
        token_tensor: &Self::Tensor,
        seq_len: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        self.forward_and_project(model, ModelInput::TokensGpu(token_tensor), seq_len, cache)
            .await
    }
}
