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
//!     -   **Knowledge**: Knows how to generate masks (Causal vs Sliding Window), how to project logits (Norm vs Scaling), etc.
//!
//! 4.  **Compute Components (`CpuDecoder` / `GpuDecoder`)**:
//!     -   **Role**: Engine. Executes the heavy Transformer layers.
//!     -   **Knowledge**: Pure linear algebra. Embed -> Normalize -> Forward Layers.

use crate::WgpuContext;
use crate::cache::{Cache, GpuKVCache};
use crate::gpu_ops::{GpuFrameContext, GpuTensor, GpuTensorPool};
use crate::common::GenerationConfig;
use crate::models::base::{AutoregressiveLoop, LanguageModel};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array1, Array2, Array3};
use std::sync::Arc;
use wgpu::CommandEncoder;

// ============================================================================
//  1. Unified Data Types
// ============================================================================

/// Flexible input for the Decoder.
///
/// This enum enables hybrid execution strategies. The backend or decoder implementation
/// can choose to auto-upload CPU data or use resident GPU data.
///
/// It also supports `Hidden` variants, allowing for Multimodal inputs (e.g., Image Encoders)
/// or Prefix Tuning where embeddings are pre-computed.
#[derive(Debug)]
pub enum DecoderInput<'a> {
    /// Standard case: Token IDs resident on GPU (Fastest for Loop).
    /// Shape: `[batch, seq]`
    TokensGpu(&'a GpuTensor),

    /// Standard case: Token IDs on CPU. Decoder handles upload.
    /// Shape: `[batch, seq]`
    TokensCpu(&'a [u32]),

    /// Advanced: Pre-computed hidden states on GPU.
    /// Shape: `[batch, seq, hidden]`
    HiddenGpu(&'a GpuTensor),

    /// Advanced: Pre-computed hidden states on CPU.
    /// Shape: `[batch, seq, hidden]`
    HiddenCpu(&'a Array3<f32>),
}

// ============================================================================
//  2. The Generation Backend (The Controller)
// ============================================================================

/// Defines the low-level orchestration for the generation loop.
///
/// Implementations (e.g., `GpuDecoderBackend`) manage the state of the "current token"
/// and coordinate the `prefill` (prompt processing) and `decode_one` (token generation) phases.
#[async_trait(?Send)]
pub trait DecoderGenerationBackend: Send + Sync {
    /// The specific tensor type used to hold the generated token sequence.
    /// This allows the backend to keep tokens on CPU (`Array2`) or GPU (`GpuTensor`).
    type Tensor: Send + Sync;

    // --- Memory Management ---

    /// Creates the initial tensor populated with prompt tokens.
    fn prime_tokens(&self, tokens: &[u32]) -> Result<Self::Tensor>;

    /// Allocates a tensor to hold a single new token.
    fn new_token_tensor(&self) -> Result<Self::Tensor>;

    /// Efficiently updates the single-token tensor with the next ID.
    /// This avoids memory reallocation during the hot generation loop.
    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_token_id: u32) -> Result<()>;

    // --- Execution Phase ---

    /// Processes the prompt tokens to populate the KV Cache.
    /// Returns the logits for the last token in the prompt.
    async fn prefill(
        &self,
        model: &dyn DecoderLanguageModel,
        initial_tokens: &[u32],
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>>;

    /// Decodes a single step.
    /// Returns the logits for the next token prediction.
    async fn decode_one(
        &self,
        model: &dyn DecoderLanguageModel,
        // token_id: u32,
        token_tensor: &Self::Tensor,
        seq_len: usize, // Total sequence length (past + 1)
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>>;
}

// ============================================================================
//  3. Compute Components (The Stack)
// ============================================================================

/// Defines the asynchronous interface for a GPU-native Transformer Decoder.
///
/// Breaks the forward pass into granular steps for testability and advanced control.
#[async_trait(?Send)]
pub trait GpuDecoder: Send + Sync {
    /// Step 1: Compute embeddings.
    /// Handles lookup (Tokens) or passthrough (Hidden).
    async fn embed(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        input: DecoderInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor>;

    /// Step 2: Apply initial normalization (Pre-Norm).
    /// Used by Llama, Mistral, etc.
    async fn embed_and_normalize(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        input: DecoderInput<'_>,
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
    async fn forward(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        input: DecoderInput<'_>,
        attention_mask: &GpuTensor,
        position_offset: usize,
        cache: Option<&mut GpuKVCache>,
        _encoder_hidden_states: Option<&GpuTensor>, // Reserved for future Seq2Seq reuse
    ) -> Result<GpuTensor> {
        let hidden = self.embed_and_normalize(encoder, pool, input, position_offset).await?;

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
    fn embed(&self, input: DecoderInput<'_>, position_offset: usize) -> Result<Array3<f32>>;

    fn embed_and_normalize(
        &self,
        input: DecoderInput<'_>,
        position_offset: usize,
    ) -> Result<Array3<f32>>;

    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>>;

    fn num_layers(&self) -> usize;

    fn forward(
        &self,
        input: DecoderInput<'_>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Array3<f32>> {
        let hidden = self.embed_and_normalize(input, position_offset)?;
        self.forward_layers(
            &hidden,
            attention_mask,
            position_offset,
            cache,
            0,
            self.num_layers(),
        )
    }
}

// ============================================================================
//  4. Operations Strategy (The Model Logic)
// ============================================================================

/// Logic specific to CPU execution.
/// Abstracts away how to access the component and how to project final logits.
pub trait CpuDecoderOps: Send + Sync {
    /// Access the underlying CPU compute component.
    fn decoder(&self) -> &dyn CpuDecoder;

    /// Handles model-specific projection logic (e.g., MatMul vs Norm+MatMul).
    fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>>;

    /// Generates the attention mask on the CPU.
    /// Allows models to implement Sliding Window or Alibi logic.
    fn get_attention_mask(&self, seq_len: usize, past_len: usize) -> Result<Array2<f32>>;
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
#[async_trait(?Send)]
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

        let decoder_output = ops.decoder().forward(
            DecoderInput::TokensCpu(input_ids.as_slice().unwrap()),
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
        let input_slice = input_ids.as_slice().unwrap();
        let seq_len = input_slice.len();

        let ops = self
            .decoder_gpu_ops()
            .ok_or_else(|| anyhow!("GPU Ops not available"))?;

        let context = self
            .context()
            .ok_or_else(|| anyhow!("Model missing WgpuContext"))?;

        let pool = context.get_inference_pool();
        let mut pool_guard = pool.lock().await;
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
                DecoderInput::TokensCpu(input_slice), // Fix: Pass by value
                &attention_mask,
                0,    // offset
                None, // no cache
                None, // no encoder hidden
            )
            .await?;

        // 3. Project
        // Borrow of `frame` is available again here
        let logits = ops.project_to_logits(&mut frame, &hidden)?;

        // 4. Readback
        let logits_cpu = logits.to_ndarray_3d::<f32>().await?;

        frame.finish();

        Ok(logits_cpu)
    }
}
