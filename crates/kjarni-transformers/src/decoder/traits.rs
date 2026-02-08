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
use crate::cache::Cache;
use crate::common::GenerationConfig;
use crate::encoder_decoder::traits::GpuCrossAttentionKVCache;
use crate::gpu::{GpuFrameContext, GpuKVCache, GpuTensor, GpuTensorPool};
use crate::models::base::{AutoregressiveLoop, LanguageModel, ModelInput};
use anyhow::{Result, anyhow};
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
        tokens: &Array2<u32>, // Always CPU
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

    fn embed_norm(
        &self,
        cmd_encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor> {
        unimplemented!()
    }
    fn final_norm(
        &self,
        cmd_encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor> {
        unimplemented!()
    }

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

    fn forward_layers2(
        &self,
        cmd_encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        encoder_hidden_states: &GpuTensor,
        self_attention_mask: Option<&GpuTensor>,
        cross_attention_mask: Option<&GpuTensor>,
        cache: Option<&mut GpuKVCache>,
        cross_kv_cache: Option<&GpuCrossAttentionKVCache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<GpuTensor> {
        unimplemented!()
    }

    fn precompute_cross_attention_kv(
        &self,
        cmd_encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        encoder_hidden_states: &GpuTensor,
    ) -> Result<GpuCrossAttentionKVCache> {
        unimplemented!()
    }

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
        let output =
            self.forward_all_layers(hidden_states, attention_mask, position_offset, cache)?;
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
        self.decoder()
            .forward(&hidden, attention_mask, position_offset, cache)
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
        None // Default: no template (base model)
    }

    /// Check if this model requires a chat template for proper use
    fn is_instruct_model(&self) -> bool {
        self.chat_template().is_some()
    }

    fn get_logits_cpu(&self, text: &str) -> Result<Array3<f32>> {
        let input_ids = self.tokenize(text)?;
        let seq_len = input_ids.ncols();

        let ops = self
            .decoder_cpu_ops()
            .ok_or_else(|| anyhow!("CPU Ops not available"))?;

        let attention_mask = ops.get_attention_mask(seq_len, 0)?;

        let tokens = ops.embed(&input_ids, 0)?;

        let decoder_output = ops.decoder().forward(
            &tokens,
            &attention_mask,
            0,
            None,
        )?;

        ops.project_to_logits(&decoder_output)
    }
    
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

        let hidden = ops.decoder().forward(
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


#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;
    use crate::traits::InferenceModel;
    use tokenizers::Tokenizer;

    // ========================================================================
    //  Mock Cache
    // ========================================================================

    #[derive(Clone)]
    struct MockCache {
        len: usize,
    }

    impl Cache for MockCache {
        fn as_any(&self) -> &dyn Any { self }
        fn as_any_mut(&mut self) -> &mut dyn Any { self }
        fn get_seq_length(&self) -> usize { self.len }
        fn set_seq_length(&mut self, len: usize) { self.len = len; }
        fn clear(&mut self) { self.len = 0; }
        fn increment_len(&mut self, n: usize) { self.len += n; }
        fn clone_box(&self) -> Box<dyn Cache> { Box::new(self.clone()) }
    }

    // ========================================================================
    //  Mock CpuDecoder
    // ========================================================================

    struct MockCpuDecoder {
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
    }

    impl MockCpuDecoder {
        fn new(num_layers: usize, hidden_size: usize, num_heads: usize) -> Self {
            Self { num_layers, hidden_size, num_heads }
        }
    }

    impl CpuDecoder for MockCpuDecoder {
        fn as_any(&self) -> &dyn Any { self }
        fn as_any_mut(&mut self) -> &mut dyn Any { self }

        fn forward_layers(
            &self,
            hidden_states: &Array3<f32>,
            _attention_mask: &Array2<f32>,
            _position_offset: usize,
            _cache: Option<&mut dyn Cache>,
            _start_layer: usize,
            _end_layer: usize,
        ) -> Result<Array3<f32>> {
            // Mock: just return input unchanged
            Ok(hidden_states.clone())
        }

        fn final_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
            // Mock: return input unchanged
            Ok(hidden_states.clone())
        }

        fn num_layers(&self) -> usize { self.num_layers }
        fn hidden_size(&self) -> usize { self.hidden_size }
        fn num_attention_heads(&self) -> usize { self.num_heads }
    }

    // ========================================================================
    //  CpuDecoder Metadata Tests
    // ========================================================================

    #[test]
    fn test_cpu_decoder_metadata() {
        let decoder = MockCpuDecoder::new(12, 768, 12);
        
        assert_eq!(decoder.num_layers(), 12);
        assert_eq!(decoder.hidden_size(), 768);
        assert_eq!(decoder.num_attention_heads(), 12);
    }

    #[test]
    fn test_cpu_decoder_default_num_kv_heads() {
        let decoder = MockCpuDecoder::new(12, 768, 12);
        
        // Default: num_kv_heads == num_attention_heads
        assert_eq!(decoder.num_kv_heads(), 12);
    }

    #[test]
    fn test_cpu_decoder_default_head_dim() {
        let decoder = MockCpuDecoder::new(12, 768, 12);
        
        // head_dim = hidden_size / num_attention_heads
        assert_eq!(decoder.head_dim(), 64); // 768 / 12
    }

    #[test]
    fn test_cpu_decoder_head_dim_different_configs() {
        // Llama-style: 4096 hidden, 32 heads
        let decoder1 = MockCpuDecoder::new(32, 4096, 32);
        assert_eq!(decoder1.head_dim(), 128); // 4096 / 32

        // GPT-2 small: 768 hidden, 12 heads
        let decoder2 = MockCpuDecoder::new(12, 768, 12);
        assert_eq!(decoder2.head_dim(), 64); // 768 / 12

        // Custom: 512 hidden, 8 heads
        let decoder3 = MockCpuDecoder::new(6, 512, 8);
        assert_eq!(decoder3.head_dim(), 64); // 512 / 8
    }

    // ========================================================================
    //  CpuDecoder Forward Tests
    // ========================================================================

    #[test]
    fn test_cpu_decoder_forward_layers() {
        let decoder = MockCpuDecoder::new(6, 64, 4);
        
        let hidden = Array3::<f32>::ones((1, 5, 64));
        let mask = Array2::<f32>::ones((5, 5));
        
        let output = decoder.forward_layers(&hidden, &mask, 0, None, 0, 6).unwrap();
        
        assert_eq!(output.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_forward_all_layers() {
        let decoder = MockCpuDecoder::new(6, 64, 4);
        
        let hidden = Array3::<f32>::ones((1, 5, 64));
        let mask = Array2::<f32>::ones((5, 5));
        
        // Uses default implementation
        let output = decoder.forward_all_layers(&hidden, &mask, 0, None).unwrap();
        
        assert_eq!(output.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_forward_default() {
        let decoder = MockCpuDecoder::new(6, 64, 4);
        
        let hidden = Array3::<f32>::ones((1, 5, 64));
        let mask = Array2::<f32>::ones((5, 5));
        
        // Uses default implementation: forward_all_layers + final_norm
        let output = decoder.forward(&hidden, &mask, 0, None).unwrap();
        
        assert_eq!(output.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_final_norm() {
        let decoder = MockCpuDecoder::new(6, 64, 4);
        
        let hidden = Array3::<f32>::ones((1, 5, 64));
        
        let output = decoder.final_norm(&hidden).unwrap();
        
        assert_eq!(output.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_with_cache() {
        let decoder = MockCpuDecoder::new(6, 64, 4);
        
        let hidden = Array3::<f32>::ones((1, 5, 64));
        let mask = Array2::<f32>::ones((5, 5));
        let mut cache = MockCache { len: 0 };
        
        let output = decoder.forward(&hidden, &mask, 0, Some(&mut cache)).unwrap();
        
        assert_eq!(output.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_partial_layers() {
        let decoder = MockCpuDecoder::new(12, 64, 4);
        
        let hidden = Array3::<f32>::ones((1, 5, 64));
        let mask = Array2::<f32>::ones((5, 5));
        
        // Run only first half
        let output1 = decoder.forward_layers(&hidden, &mask, 0, None, 0, 6).unwrap();
        assert_eq!(output1.shape(), &[1, 5, 64]);
        
        // Run only second half
        let output2 = decoder.forward_layers(&hidden, &mask, 0, None, 6, 12).unwrap();
        assert_eq!(output2.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_position_offset() {
        let decoder = MockCpuDecoder::new(6, 64, 4);
        
        let hidden = Array3::<f32>::ones((1, 1, 64)); // Single token
        let mask = Array2::<f32>::ones((1, 10)); // Mask for position 10
        
        // Simulate decoding at position 9 (0-indexed)
        let output = decoder.forward(&hidden, &mask, 9, None).unwrap();
        
        assert_eq!(output.shape(), &[1, 1, 64]);
    }

    // ========================================================================
    //  CpuDecoder Downcasting Tests
    // ========================================================================

    #[test]
    fn test_cpu_decoder_as_any() {
        let decoder = MockCpuDecoder::new(6, 64, 4);
        
        let any_ref = decoder.as_any();
        let downcasted = any_ref.downcast_ref::<MockCpuDecoder>();
        
        assert!(downcasted.is_some());
        assert_eq!(downcasted.unwrap().num_layers, 6);
    }

    #[test]
    fn test_cpu_decoder_as_any_mut() {
        let mut decoder = MockCpuDecoder::new(6, 64, 4);
        
        {
            let any_mut = decoder.as_any_mut();
            let downcasted = any_mut.downcast_mut::<MockCpuDecoder>();
            assert!(downcasted.is_some());
            downcasted.unwrap().num_layers = 12;
        }
        
        assert_eq!(decoder.num_layers, 12);
    }

    // ========================================================================
    //  Mock CpuDecoderOps
    // ========================================================================

    struct MockCpuDecoderOps {
        decoder: MockCpuDecoder,
    }

    impl MockCpuDecoderOps {
        fn new() -> Self {
            Self {
                decoder: MockCpuDecoder::new(6, 64, 4),
            }
        }
    }

    impl CpuDecoderOps for MockCpuDecoderOps {
        fn decoder(&self) -> &dyn CpuDecoder {
            &self.decoder
        }

        fn embed(&self, tokens: &Array2<u32>, _pos: usize) -> Result<Array3<f32>> {
            let (batch, seq_len) = tokens.dim();
            Ok(Array3::<f32>::ones((batch, seq_len, 64)))
        }

        fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
            let (batch, seq_len, _) = hidden_states.dim();
            Ok(Array3::<f32>::zeros((batch, seq_len, 1000))) // vocab_size = 1000
        }

        fn get_attention_mask(&self, seq_len: usize, _past_len: usize) -> Result<Array2<f32>> {
            Ok(Array2::<f32>::ones((seq_len, seq_len)))
        }
    }

    // ========================================================================
    //  CpuDecoderOps Tests
    // ========================================================================

    #[test]
    fn test_cpu_decoder_ops_decoder_access() {
        let ops = MockCpuDecoderOps::new();
        
        let decoder = ops.decoder();
        assert_eq!(decoder.num_layers(), 6);
        assert_eq!(decoder.hidden_size(), 64);
    }

    #[test]
    fn test_cpu_decoder_ops_embed() {
        let ops = MockCpuDecoderOps::new();
        
        let tokens = Array2::from_shape_vec((1, 5), vec![1u32, 2, 3, 4, 5]).unwrap();
        let hidden = ops.embed(&tokens, 0).unwrap();
        
        assert_eq!(hidden.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_ops_project_to_logits() {
        let ops = MockCpuDecoderOps::new();
        
        let hidden = Array3::<f32>::ones((1, 5, 64));
        let logits = ops.project_to_logits(&hidden).unwrap();
        
        assert_eq!(logits.shape(), &[1, 5, 1000]);
    }

    #[test]
    fn test_cpu_decoder_ops_get_attention_mask() {
        let ops = MockCpuDecoderOps::new();
        
        let mask = ops.get_attention_mask(10, 0).unwrap();
        
        assert_eq!(mask.shape(), &[10, 10]);
    }

    #[test]
    fn test_cpu_decoder_ops_forward_default() {
        let ops = MockCpuDecoderOps::new();
        
        let tokens = Array2::from_shape_vec((1, 5), vec![1u32, 2, 3, 4, 5]).unwrap();
        let mask = Array2::<f32>::ones((5, 5));
        
        // Uses default implementation
        let hidden = ops.forward(&tokens, &mask, 0, None).unwrap();
        
        assert_eq!(hidden.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_ops_forward_to_logits_default() {
        let ops = MockCpuDecoderOps::new();
        
        let tokens = Array2::from_shape_vec((1, 5), vec![1u32, 2, 3, 4, 5]).unwrap();
        let mask = Array2::<f32>::ones((5, 5));
        
        // Uses default implementation: forward + project_to_logits
        let logits = ops.forward_to_logits(&tokens, &mask, 0, None).unwrap();
        
        assert_eq!(logits.shape(), &[1, 5, 1000]);
    }

    // ========================================================================
    //  Mock DecoderLanguageModel (partial)
    // ========================================================================

    struct MockDecoderModel {
        cpu_ops: MockCpuDecoderOps,
    }

    impl InferenceModel for MockDecoderModel {
        fn device(&self) -> crate::traits::Device { crate::traits::Device::Cpu }
        fn context(&self) -> Option<Arc<crate::WgpuContext>> { None }
        fn as_any(&self) -> &dyn Any { self }
    }

    impl LanguageModel for MockDecoderModel {
        fn vocab_size(&self) -> usize { 1000 }
        fn hidden_size(&self) -> usize { 64 }
        fn num_layers(&self) -> usize { 6 }
        fn num_heads(&self) -> usize { 4 }
        fn context_size(&self) -> usize { 2048 }
        fn tokenizer(&self) -> &Tokenizer { unimplemented!() }
        fn eos_token_id(&self) -> Option<u32> { Some(2) }
        fn bos_token_id(&self) -> Option<u32> { Some(1) }
        fn forced_bos_token_id(&self) -> Option<u32> { None }
        fn forced_eos_token_id(&self) -> Option<u32> { None }
        fn pad_token_id(&self) -> Option<u32> { Some(0) }
        fn stop_token_ids(&self) -> HashSet<u32> { HashSet::from([2]) }
        fn new_cache(&self, _: usize, _: usize, _: usize) -> Result<Box<dyn Cache>> {
            Ok(Box::new(MockCache { len: 0 }))
        }
    }

    #[async_trait]
    impl DecoderLanguageModel for MockDecoderModel {
        fn decoder_cpu_ops(&self) -> Option<&dyn CpuDecoderOps> {
            Some(&self.cpu_ops)
        }

        fn decoder_gpu_ops(&self) -> Option<&dyn GpuDecoderOps> {
            None
        }

        fn autoregressive_loop(&self) -> AutoregressiveLoop {
            AutoregressiveLoop::Pipelined
        }
    }

    // ========================================================================
    //  DecoderLanguageModel Default Method Tests
    // ========================================================================

    #[test]
    fn test_decoder_language_model_get_default_generation_config() {
        let model = MockDecoderModel {
            cpu_ops: MockCpuDecoderOps::new(),
        };
        
        let config = model.get_default_generation_config();
        
        // Should return default config
        assert!(config.max_length > 0);
    }

    #[test]
    fn test_decoder_language_model_chat_template_default() {
        let model = MockDecoderModel {
            cpu_ops: MockCpuDecoderOps::new(),
        };
        
        // Default: no chat template
        assert!(model.chat_template().is_none());
    }

    #[test]
    fn test_decoder_language_model_is_instruct_model_default() {
        let model = MockDecoderModel {
            cpu_ops: MockCpuDecoderOps::new(),
        };
        
        // Default: not an instruct model (no chat template)
        assert!(!model.is_instruct_model());
    }

    #[test]
    fn test_decoder_language_model_cpu_ops_access() {
        let model = MockDecoderModel {
            cpu_ops: MockCpuDecoderOps::new(),
        };
        
        let ops = model.decoder_cpu_ops();
        assert!(ops.is_some());
        
        let ops = ops.unwrap();
        assert_eq!(ops.decoder().num_layers(), 6);
    }

    #[test]
    fn test_decoder_language_model_gpu_ops_none() {
        let model = MockDecoderModel {
            cpu_ops: MockCpuDecoderOps::new(),
        };
        
        // CPU-only model
        assert!(model.decoder_gpu_ops().is_none());
    }

    #[test]
    fn test_decoder_language_model_autoregressive_loop() {
        let model = MockDecoderModel {
            cpu_ops: MockCpuDecoderOps::new(),
        };
        
        assert!(matches!(model.autoregressive_loop(), AutoregressiveLoop::Pipelined));
    }

    // ========================================================================
    //  AutoregressiveLoop Tests
    // ========================================================================

    #[test]
    fn test_autoregressive_loop_variants() {
        let pipelined = AutoregressiveLoop::Pipelined;
        let legacy = AutoregressiveLoop::Legacy;
        
        // They should be different
        assert!(!matches!(pipelined, AutoregressiveLoop::Legacy));
        assert!(!matches!(legacy, AutoregressiveLoop::Pipelined));
    }

    // ========================================================================
    //  Edge Cases
    // ========================================================================

    #[test]
    fn test_cpu_decoder_empty_sequence() {
        let decoder = MockCpuDecoder::new(6, 64, 4);
        
        let hidden = Array3::<f32>::zeros((1, 0, 64)); // Empty sequence
        let mask = Array2::<f32>::zeros((0, 0));
        
        let output = decoder.forward(&hidden, &mask, 0, None).unwrap();
        
        assert_eq!(output.shape(), &[1, 0, 64]);
    }

    #[test]
    fn test_cpu_decoder_batch_size_greater_than_one() {
        let decoder = MockCpuDecoder::new(6, 64, 4);
        
        let hidden = Array3::<f32>::ones((4, 5, 64)); // batch = 4
        let mask = Array2::<f32>::ones((5, 5));
        
        let output = decoder.forward(&hidden, &mask, 0, None).unwrap();
        
        assert_eq!(output.shape(), &[4, 5, 64]);
    }

    #[test]
    fn test_cpu_decoder_ops_embed_batch() {
        let ops = MockCpuDecoderOps::new();
        
        let tokens = Array2::from_shape_vec((2, 3), vec![1u32, 2, 3, 4, 5, 6]).unwrap();
        let hidden = ops.embed(&tokens, 0).unwrap();
        
        assert_eq!(hidden.shape(), &[2, 3, 64]);
    }

    #[test]
    fn test_cpu_decoder_single_layer() {
        let decoder = MockCpuDecoder::new(1, 64, 4);
        
        let hidden = Array3::<f32>::ones((1, 5, 64));
        let mask = Array2::<f32>::ones((5, 5));
        
        let output = decoder.forward(&hidden, &mask, 0, None).unwrap();
        
        assert_eq!(output.shape(), &[1, 5, 64]);
        assert_eq!(decoder.num_layers(), 1);
    }

    #[test]
    fn test_cpu_decoder_large_hidden_size() {
        let decoder = MockCpuDecoder::new(32, 4096, 32);
        
        assert_eq!(decoder.hidden_size(), 4096);
        assert_eq!(decoder.head_dim(), 128);
    }

    // ========================================================================
    //  Mask Generation Tests
    // ========================================================================

    #[test]
    fn test_attention_mask_generation_prefill() {
        let ops = MockCpuDecoderOps::new();
        
        // Prefill: seq_len = 10, past_len = 0
        let mask = ops.get_attention_mask(10, 0).unwrap();
        
        assert_eq!(mask.shape(), &[10, 10]);
    }

    #[test]
    fn test_attention_mask_generation_decode() {
        let ops = MockCpuDecoderOps::new();
        
        // Decode: seq_len = 1, past_len = 10
        let mask = ops.get_attention_mask(1, 10).unwrap();
        
        assert_eq!(mask.shape(), &[1, 1]);
    }

    // ========================================================================
    //  Full Pipeline Test
    // ========================================================================

    #[test]
    fn test_full_cpu_inference_pipeline() {
        let ops = MockCpuDecoderOps::new();
        
        // 1. Tokenize (mock)
        let tokens = Array2::from_shape_vec((1, 5), vec![1u32, 100, 200, 300, 2]).unwrap();
        
        // 2. Get mask
        let mask = ops.get_attention_mask(5, 0).unwrap();
        
        // 3. Embed
        let hidden = ops.embed(&tokens, 0).unwrap();
        assert_eq!(hidden.shape(), &[1, 5, 64]);
        
        // 4. Forward through decoder
        let output = ops.decoder().forward(&hidden, &mask, 0, None).unwrap();
        assert_eq!(output.shape(), &[1, 5, 64]);
        
        // 5. Project to logits
        let logits = ops.project_to_logits(&output).unwrap();
        assert_eq!(logits.shape(), &[1, 5, 1000]);
    }

    #[test]
    fn test_full_cpu_inference_with_forward_to_logits() {
        let ops = MockCpuDecoderOps::new();
        
        let tokens = Array2::from_shape_vec((1, 5), vec![1u32, 100, 200, 300, 2]).unwrap();
        let mask = ops.get_attention_mask(5, 0).unwrap();
        
        // Single call that does everything
        let logits = ops.forward_to_logits(&tokens, &mask, 0, None).unwrap();
        
        assert_eq!(logits.shape(), &[1, 5, 1000]);
    }
}