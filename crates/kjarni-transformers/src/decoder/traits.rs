//! # Decoder Traits
//!
//! This module defines the core architectural traits for Autoregressive (Decoder-only) Language Models.
//!
//! ## Architecture Overview
//!
//! To support a wide variety of models (Llama, Mistral, Phi, Gemma, GPT-2) and hybrid CPU/GPU execution,
//! the architecture is split into four distinct layers:
//!
//! 1.  **Backend Controller (`DecoderGenerationBackend`)**:
//!     -   **Role**: Orchestrator. Manages the generation loop, KV Cache state, and Memory.
//!     -   **Knowledge**: Agnostic to model math. Knows *when* to run a step, not *how*.
//!
//! 2.  **Model Container (`DecoderLanguageModel`)**:
//!     -   **Role**: Router. Holds configuration and directs execution to the correct hardware implementation.
//!     -   **Knowledge**: Knows which device the model is loaded on.
//!
//! 3.  **Operations Strategy (`CpuDecoderOps` / `GpuDecoderOps`)**:
//!     -   **Role**: Mathematician. Handles model-specific logic that varies between architectures.
//!     -   **Knowledge**: Knows how to generate masks (Causal vs Sliding Window), how t<o project logits (Norm vs Scaling), etc.
//!
//! 4.  **Compute Components (`CpuDecoder` / `GpuDecoder`)**:
//!     -   **Role**: Engine. Executes the heavy Transformer layers.
//!     -   **Knowledge**: Pure linear algebra. Embed -> Normalize -> Forward Layers.

use std::any::Any;

use crate::ChatTemplate;
use crate::cache::{Cache};
use crate::common::GenerationConfig;
use crate::gpu::{GpuFrameContext, GpuKVCache, GpuTensor, GpuTensorPool};
use crate::models::base::{AutoregressiveLoop, LanguageModel, ModelInput};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ndarray::{Array1, Array2, Array3};
use wgpu::CommandEncoder;

// ============================================================================
//  The Generation Backend
// ============================================================================

/// Defines the low-level orchestration for the generation loop.
///
/// Implementations (e.g., `GpuDecoderBackend`) manage the state of the "current token"
/// and coordinate the `prefill` (prompt processing) and `decode_one` (token generation) phases.
#[async_trait]
pub trait DecoderGenerationBackend: Send + Sync {
    /// Token tensor for decode loop (GPU: GpuTensor, CPU: Array2<u32>)
    type DecodeToken: Send + Sync;

    /// Allocate reusable single-token tensor
    fn new_decode_token(&self) -> Result<Self::DecodeToken>;

    /// Update decode tensor with new token ID (fast write)
    fn update_decode_token(&self, tensor: &mut Self::DecodeToken, token_id: u32) -> Result<()>;

    /// Process prompt, populate cache, return first logits
    async fn prefill(
        &self,
        model: &dyn DecoderLanguageModel,
        tokens: &Array2<u32>,  // Always CPU
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>>;

    /// Process single token, return next logits
    async fn decode_one(
        &self,
        model: &dyn DecoderLanguageModel,
        token: &Self::DecodeToken,
        seq_len: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>>;
}

// ============================================================================
//  Compute Components
// ============================================================================

/// Defines the asynchronous interface for a GPU-native Transformer Decoder.
///
/// Breaks the forward pass into granular steps for testability and advanced control.
pub trait GpuDecoder: Send + Sync {

    fn as_any(&self) -> &dyn std::any::Any;

    /// Step 1: Compute embeddings.
    /// Handles lookup (Tokens) or passthrough (Hidden).
    fn embed(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor>;

    /// Step 2: Apply initial normalization (Pre-Norm).
    /// Used by Llama, Mistral, etc.
    fn embed_and_normalize(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor>;

    /// Step 3: Run the stack of decoder layers.
    /// Accepts `start_layer` and `end_layer` to support pipeline parallelism or debugging.
    fn forward_layers(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        position_offset: usize,
        cache: Option<&mut GpuKVCache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<GpuTensor>;

    /// Metadata: Total number of layers.
    fn num_layers(&self) -> usize;

    /// Metadata: Hidden dimension size.
    fn hidden_size(&self) -> usize;

    /// Default full forward pass.
    /// Chains `embed_and_normalize` -> `forward_layers`.
    fn forward(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        attention_mask: &GpuTensor,
        position_offset: usize,
        cache: Option<&mut GpuKVCache>,
        _encoder_hidden_states: Option<&GpuTensor>, // Reserved for future Seq2Seq reuse
    ) -> Result<GpuTensor> {
        let hidden = self.embed_and_normalize(encoder, pool, input, position_offset)?;

        self.forward_layers(
            encoder,
            pool,
            &hidden,
            attention_mask,
            position_offset,
            cache,
            0,
            self.num_layers(),
        )
    }
}

/// Defines the synchronous interface for a CPU-native Transformer Decoder.
/// Mirrors `GpuDecoder` structure for symmetry.
pub trait CpuDecoder: Send + Sync {

    // =========================================================================
    // Downcasting
    // =========================================================================

    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;


    // =========================================================================
    // Core Forward Pass
    // =========================================================================

    /// Forward pass through transformer layers.
    /// Applies attention, feed-forward, residuals, etc.
    ///
    /// # Arguments
    /// * `hidden_states` - Input hidden states of shape `[Batch, SeqLen, Hidden]`.
    /// * `attention_mask` - Attention mask of shape `[Batch, SeqLen, SeqLen]`.
    /// * `position_offset` - Offset for positional embeddings (for generation).
    /// * `cache` - Optional KV cache for autoregressive decoding.
    /// * `start_layer` - Index of the first layer to process (inclusive).
    /// * `end_layer` - Index of the last layer to process (exclusive).
    ///
    /// # Returns
    /// * Output hidden states of shape `[Batch, SeqLen, Hidden]`.
    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>>;

    

    /// Apply final layer normalization.
    ///
    /// For Llama: RMSNorm
    /// For BERT: LayerNorm
    /// For GPT-2: LayerNorm (or none, handled in LM head)
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - Output from `forward_layers`
    ///
    /// # Returns
    ///
    /// Normalized hidden states, ready for LM head projection.
    /// Apply final layer normalization (RMSNorm/LayerNorm).
    fn final_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>>;

    // =========================================================================
    // Metadata
    // =========================================================================

    fn num_layers(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_attention_heads(&self) -> usize;

    // =========================================================================
    // Metadata (optional, with defaults)
    // =========================================================================

    fn num_kv_heads(&self) -> usize {
        self.num_attention_heads()
    }

    fn head_dim(&self) -> usize {
        self.hidden_size() / self.num_attention_heads()
    }

    // =========================================================================
    // Convenience Methods (default implementations)
    // =========================================================================

    /// Forward through all layers.
    fn forward_all_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Array3<f32>> {
        self.forward_layers(
            hidden_states,
            attention_mask,
            position_offset,
            cache,
            0,
            self.num_layers(),
        )
    }

    /// Full decoder pass: layers + final norm.
    ///
    /// This is the most common usage pattern.
    fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Array3<f32>> {
        let output = self.forward_all_layers(
            hidden_states,
            attention_mask,
            position_offset,
            cache,
        )?;
        self.final_norm(&output)
    }
}

// ============================================================================
//  Ops
// ============================================================================

/// Logic specific to CPU execution.
/// Abstracts away how to access the component and how to project final logits.
pub trait CpuDecoderOps: Send + Sync {
    /// Access the underlying CPU compute component.
    fn decoder(&self) -> &dyn CpuDecoder;

    /// Embed tokens to hidden states.
    fn embed(&self, tokens: &Array2<u32>, pos: usize) -> Result<Array3<f32>>;

    /// Handles model-specific projection logic (e.g., MatMul vs Norm+MatMul).
    fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>>;

    /// Generates the attention mask on the CPU.
    /// Allows models to implement Sliding Window or Alibi logic.
    fn get_attention_mask(&self, seq_len: usize, past_len: usize) -> Result<Array2<f32>>;

    // =========================================================================
    // Default Implementations
    // =========================================================================

    /// Full forward pass: embed → layers → norm.
    fn forward(
        &self,
        tokens: &Array2<u32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Array3<f32>> {
        let hidden = self.embed(tokens, position_offset)?;
        self.decoder().forward(&hidden, attention_mask, position_offset, cache)
    }

    /// Full inference path: tokens → hidden → logits.
    fn forward_to_logits(
        &self,
        tokens: &Array2<u32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Array3<f32>> {
        let hidden = self.forward(tokens, attention_mask, position_offset, cache)?;
        self.project_to_logits(&hidden)
    }
}

/// Logic specific to GPU execution.
/// Abstracts away masking strategies (RoPE vs ALiBi) and projection (Head Norm).
pub trait GpuDecoderOps: Send + Sync {
    /// Access the underlying GPU compute component.
    fn decoder(&self) -> &dyn GpuDecoder;

    /// Generates the attention mask on the GPU.
    ///
    /// # Arguments
    /// * `seq_len`: The length of the sequence currently being processed.
    /// * `max_len`: The total capacity (or max position) for the mask.
    fn get_attention_mask(
        &self,
        ctx: &mut GpuFrameContext,
        seq_len: usize,
        max_len: usize,
    ) -> Result<GpuTensor>;

    /// Handles model-specific projection logic.
    ///
    /// # Arguments
    /// * `last_hidden_state`: Tensor of shape `[Batch, 1, Hidden]`.
    ///
    /// # Returns
    /// * Tensor of shape `[Batch, 1, Vocab]`.
    fn project_to_logits(
        &self,
        ctx: &mut GpuFrameContext,
        last_hidden_state: &GpuTensor,
    ) -> Result<GpuTensor>;
}

// ============================================================================
//  5. The Model Container
// ============================================================================


/// The primary trait for Decoder-only Language Models (Llama, GPT, Mistral, etc.).
///
/// This trait acts as a "Router", directing the Backend to the correct
/// Operations implementation (`CpuDecoderOps` or `GpuDecoderOps`).
#[async_trait]
pub trait DecoderLanguageModel: LanguageModel {
    /// Access CPU operations strategy. Returns `None` if model is GPU-only.
    fn decoder_cpu_ops(&self) -> Option<&dyn CpuDecoderOps>;

    /// Access GPU operations strategy. Returns `None` if model is CPU-only.
    fn decoder_gpu_ops(&self) -> Option<&dyn GpuDecoderOps>;

    /// Specifies the generation loop strategy (e.g., Pipelined vs Legacy).
    fn autoregressive_loop(&self) -> AutoregressiveLoop;

    /// Returns the default generation config for this specific model type.
    fn get_default_generation_config(&self) -> GenerationConfig {
        GenerationConfig::default()
    }

    /// Get the chat template for this model (if it's an instruct model)
    fn chat_template(&self) -> Option<&dyn ChatTemplate> {
        None  // Default: no template (base model)
    }
    
    /// Check if this model requires a chat template for proper use
    fn is_instruct_model(&self) -> bool {
        self.chat_template().is_some()
    }

    // --- Helper Utilities (Direct Forward Pass) ---

    /// Convenience method to run a CPU forward pass given text input.
    /// Useful for testing, feature extraction, or sanity checks without the Generator.
    fn get_logits_cpu(&self, text: &str) -> Result<Array3<f32>> {
        let input_ids = self.tokenize(text)?;
        let seq_len = input_ids.ncols();

        let ops = self
            .decoder_cpu_ops()
            .ok_or_else(|| anyhow!("CPU Ops not available"))?;

        // Use model-specific mask generation
        let attention_mask = ops.get_attention_mask(seq_len, 0)?;

        let tokens = ops.embed(&input_ids, 0)?;

        let decoder_output = ops.decoder().forward(
            // ModelInput::from_tokens(input_ids.as_slice().unwrap()),
            &tokens,
            &attention_mask,
            0,
            None,
        )?;

        ops.project_to_logits(&decoder_output)
    }

    /// Convenience method to run a GPU forward pass given text input.
    /// Useful for testing, feature extraction, or sanity checks without the Generator.
    async fn get_logits_gpu(&self, text: &str) -> Result<Array3<f32>> {
        let input_ids = self.tokenize(text)?;
        // let input_slice = input_ids.as_slice().unwrap();
        let seq_len = input_ids.len();

        let ops = self
            .decoder_gpu_ops()
            .ok_or_else(|| anyhow!("GPU Ops not available"))?;

        let context = self
            .context()
            .ok_or_else(|| anyhow!("Model missing WgpuContext"))?;

        let pool = context.get_inference_pool();
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&context, pool_guard);

        // 1. Prepare Mask
        let attention_mask = ops.get_attention_mask(&mut frame, seq_len, seq_len)?;

        // 2. Forward Pass
        // Split borrow to satisfy borrow checker
        let (encoder, pool) = frame.resources();

        let hidden = ops
            .decoder()
            .forward(
                encoder,
                pool,
                ModelInput::TokensCpu(input_ids.view()), // Fix: Pass by value
                &attention_mask,
                0,    // offset
                None, // no cache
                None, // no encoder hidden
            )?;

        // 3. Project
        // Borrow of `frame` is available again here
        let logits = ops.project_to_logits(&mut frame, &hidden)?;

        // 4. Readback
        let logits_cpu = logits.to_ndarray_3d::<f32>().await?;

        frame.finish();

        Ok(logits_cpu)
    }
}
