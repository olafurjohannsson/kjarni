//! Core traits for Encoder-Decoder (Sequence-to-Sequence) models.
//!
//! This module defines the architectural contracts for models like BART and T5,
//! separating the core compute components from the high-level model container.

use crate::cache::Cache;
use crate::common::GenerationConfig;
use crate::cpu::encoder::prelude::EncoderLanguageModel;
use crate::encoder_decoder::decoder_cross_attn_layer::CrossDecoderLayer;
use crate::gpu_ops::blocks::layers::GpuCrossDecoderLayer;
use crate::gpu_ops::{GpuFrameContext, GpuTensor, GpuTensorPool};
use crate::models::base::ModelInput;
use anyhow::Result;
use async_trait::async_trait;
use ndarray::{Array2, Array3, Array4};
use wgpu::CommandEncoder;
// ============================================================================
//  1. Compute Components (The Engines)
// ============================================================================

/// A container for the pre-computed cross-attention Key/Value cache on the CPU.
#[derive(Debug, Default)]
pub struct CpuCrossAttentionKVCache(pub Vec<(Array4<f32>, Array4<f32>)>);

/// A container for the pre-computed cross-attention Key/Value cache on the GPU.
#[derive(Debug, Default)]
pub struct GpuCrossAttentionKVCache(pub Vec<(GpuTensor, GpuTensor)>);

/// The output of a single step from a `CpuCrossDecoder`.
pub struct CpuCrossDecoderOutput {
    /// The final hidden states from the decoder stack. Shape: `[batch, seq, hidden]`.
    pub last_hidden_state: Array3<f32>,
    /// The newly computed self-attention key/value pairs for each layer.
    /// This should be used to update the KV cache for the next step.
    pub new_self_attn_kv: Vec<(Array3<f32>, Array3<f32>)>,
}

/// The output of a single step from a `GpuCrossDecoder`.
pub struct GpuCrossDecoderOutput {
    /// The final hidden states on the GPU. Shape: `[batch, seq, hidden]`.
    pub last_hidden_state: GpuTensor,
    /// The newly computed self-attention key/value pairs for each layer, resident on the GPU.
    pub new_self_attn_kv: Vec<(GpuTensor, GpuTensor)>,
}

/// Defines the synchronous interface for a CPU-native decoder that uses cross-attention.
#[async_trait]
pub trait CpuCrossDecoder: Send + Sync {
    /// Pre-computes the Key and Value matrices for cross-attention from the encoder's output.
    fn precompute_cross_attention_kv(
        &self,
        encoder_hidden_states: &Array3<f32>,
    ) -> Result<CpuCrossAttentionKVCache>;

    fn embed(&self, decoder_input_ids: &Array2<u32>, position_offset: usize) -> Array3<f32>;

    /// Compute embeddings + initial normalization.
    fn embed_and_normalize(
        &self,
        input_ids: &Array2<u32>,
        position_offset: usize,
    ) -> Result<Array3<f32>>;

    fn layers(&self) -> &Vec<CrossDecoderLayer>;

    /// Run a subset of layers `[start_layer, end_layer)`.
    ///
    /// # Arguments
    /// * `hidden_states` - Input states from embedding or previous layer block.
    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        decoder_attention_mask: Option<&Array2<f32>>,
        cache: Option<&mut dyn Cache>,
        cross_kv_cache: Option<&CpuCrossAttentionKVCache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<CpuCrossDecoderOutput>;
    fn forward_layers2(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        decoder_padding_mask: Option<&Array2<f32>>, // Padding in the decoder
        encoder_padding_mask: Option<&Array2<f32>>, // Padding in the encoder (NEW)
        cache: Option<&mut dyn Cache>,
        cross_kv_cache: Option<&CpuCrossAttentionKVCache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<CpuCrossDecoderOutput> {
        Ok(CpuCrossDecoderOutput {
            new_self_attn_kv: Vec::new(),
            last_hidden_state: Array3::<f32>::zeros((0, 0, 0)),
        })
    }
    fn forward2(
        &self,
        decoder_input_ids: &Array2<u32>,
        encoder_hidden_states: &Array3<f32>,
        decoder_padding_mask: Option<&Array2<f32>>, // Padding in the decoder
        encoder_padding_mask: Option<&Array2<f32>>, // Padding in the encoder (NEW)
        cache: Option<&mut dyn Cache>,
        cross_kv_cache: Option<&CpuCrossAttentionKVCache>,
    ) -> Result<CpuCrossDecoderOutput> {
        Ok(CpuCrossDecoderOutput {
            new_self_attn_kv: Vec::new(),
            last_hidden_state: Array3::<f32>::zeros((0, 0, 0)),
        })
    }
    /// Metadata: Total number of layers.
    fn num_layers(&self) -> usize;

    /// Metadata: Hidden dimension size.
    fn hidden_size(&self) -> usize;


    // --- High-level Default Implementation ---

    /// Performs a full forward pass through the cross-attention decoder stack.
    fn forward(
        &self,
        decoder_input_ids: &Array2<u32>,
        encoder_hidden_states: &Array3<f32>,
        decoder_attention_mask: Option<&Array2<f32>>,
        cache: Option<&mut dyn Cache>,
        cross_kv_cache: Option<&CpuCrossAttentionKVCache>,
    ) -> Result<CpuCrossDecoderOutput> {
        // 1. Calculate offset
        let position_offset = cache.as_ref().map_or(0, |c| c.get_seq_length());

        // 2. Embed
        let hidden = self.embed_and_normalize(decoder_input_ids, position_offset)?;

        // 3. Run all layers
        self.forward_layers(
            &hidden,
            encoder_hidden_states,
            decoder_attention_mask,
            cache,
            cross_kv_cache,
            0,
            self.num_layers(),
        )
    }
}

#[async_trait]
pub trait GpuCrossDecoder: Send + Sync {
    /// Pre-computes the Key and Value matrices for cross-attention.
    fn precompute_cross_attention_kv(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        encoder_hidden_states: &GpuTensor,
    ) -> Result<GpuCrossAttentionKVCache>;

    fn layers(&self) -> &Vec<GpuCrossDecoderLayer>;

    fn embed(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor>;

    /// Step 2: Apply initial normalization.
    fn embed_and_normalize(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor>;

    /// Step 3: Run the decoder layers with cross-attention.
    fn forward_layers(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        encoder_hidden_states: &GpuTensor,
        decoder_attention_mask: &GpuTensor,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
        cross_kv_cache: Option<&GpuCrossAttentionKVCache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<GpuCrossDecoderOutput>;

    fn num_layers(&self) -> usize;
    fn hidden_size(&self) -> usize;

    /// Default full forward pass.
    fn forward(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        decoder_input: ModelInput<'_>,
        encoder_hidden_states: &GpuTensor,
        decoder_attention_mask: &GpuTensor,
        cache: Option<&mut dyn Cache>,
        cross_kv_cache: Option<&GpuCrossAttentionKVCache>,
    ) -> Result<GpuCrossDecoderOutput> {
        let position_offset = cache.as_ref().map_or(0, |c| c.get_seq_length());

        // 1 & 2. Embed and Norm
        let hidden = self.embed_and_normalize(encoder, pool, decoder_input, position_offset)?;

        // 3. Run Layers
        self.forward_layers(
            encoder,
            pool,
            &hidden,
            encoder_hidden_states,
            decoder_attention_mask,
            position_offset,
            cache,
            cross_kv_cache,
            0,
            self.num_layers(),
        )
    }
}

// ============================================================================
//  2. Operations Strategy (The Model Logic)
// ============================================================================

/// Defines the CPU-specific computational graph for an encoder-decoder model.
pub trait CpuEncoderDecoderOps: Send + Sync {
    // fn encoder(&self) -> &dyn CpuEncoder;
    fn decoder(&self) -> &dyn CpuCrossDecoder;
    fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>>;
    fn broadcast_encoder_states(
        &self,
        encoder_hidden_states: &Array3<f32>,
        num_beams: usize,
    ) -> Result<Array3<f32>>;
    fn get_decoder_mask(&self, seq_len: usize, past_len: usize) -> Option<Array2<f32>>;
}

/// Defines the GPU-specific computational graph for an encoder-decoder model.
pub trait GpuEncoderDecoderOps: Send + Sync {
    // fn encoder(&self) -> &dyn GpuEncoder;
    fn decoder(&self) -> &dyn GpuCrossDecoder;
    fn project_to_logits(
        &self,
        frame: &mut GpuFrameContext,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor>;
    fn broadcast_encoder_states(
        &self,
        frame: &mut GpuFrameContext,
        encoder_hidden_states: &GpuTensor,
        num_beams: usize,
    ) -> Result<GpuTensor>;
}

// ============================================================================
//  3. The Model Container
// ============================================================================

/// The primary trait for Encoder-Decoder Language Models (BART, T5, etc.).
///
/// This trait composes the `EncoderLanguageModel` trait, signifying that any
/// encoder-decoder model also has a fully functional encoder that can be used
/// on its own.
#[async_trait]
pub trait EncoderDecoderLanguageModel: EncoderLanguageModel {
    /// Provides access to the CPU-specific operations for the combined model.
    fn encoder_decoder_cpu_ops(&self) -> Option<&dyn CpuEncoderDecoderOps>;
    /// Provides access to the GPU-specific operations for the combined model.
    fn encoder_decoder_gpu_ops(&self) -> Option<&dyn GpuEncoderDecoderOps>;

    /// The token ID that should be used to start the decoding process.
    fn decoder_start_token_id(&self) -> u32;
    /// Returns the default generation configuration for this model.
    fn get_default_generation_config(&self) -> GenerationConfig;
}

// ============================================================================
//  4. The Generation Backend (The Controller)
// ============================================================================

/// Defines the low-level orchestration for the encoder-decoder generation loop.
/// This will be refactored to be "dumb".
#[async_trait]
pub trait EncoderDecoderGenerationBackend: Send + Sync {
    type Tensor: Send + Sync;

    async fn encode(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        tokens: &[u32],
        num_beams: usize,
    ) -> Result<Self::Tensor>;

    async fn decode_step(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        decoder_tokens: &Self::Tensor,
        encoder_state: &Self::Tensor,
        cache: &mut dyn Cache,
    ) -> Result<Array3<f32>>;

    fn create_token_tensor(&self, tokens: &[u32], num_beams: usize) -> Result<Self::Tensor>;
    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_tokens: &[u32]) -> Result<()>;
    fn reorder_cache(&self, cache: &mut dyn Cache, indices: &[usize]) -> Result<()>;
}
