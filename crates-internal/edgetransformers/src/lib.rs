//! Core transformer components for building transformer-based models
//!
//! This crate provides the fundamental building blocks for transformer architectures
//! without model-specific implementations.

//! EdgeTransformers: Fast transformer models for Rust

pub mod activations;
pub mod attention;
pub mod cache;
pub mod decoder;
pub mod embeddings;
pub mod encoder;
pub mod encoder_decoder;
pub mod feedforward;
pub mod gpu_context;
pub mod gpu_ops;
pub mod gpu_pipeline;
pub mod layer_norm;
pub mod models;
pub mod pooling;
pub mod traits;
pub mod utils;
pub mod tests;
pub mod weights;
pub mod decoder_layer;

// Re-export commonly used items
pub use crate::{
    attention::MultiHeadAttention,
    embeddings::Embeddings,
    feedforward::FeedForward,
    layer_norm::LayerNorm,
    pooling::{PoolingStrategy, cls_pool, last_token_pool, max_pool, mean_pool},
    weights::ModelWeights,
};
pub use cache::{Cache, CpuKVCache, GpuKVCache};
pub use gpu_context::WgpuContext;
pub use traits::{
    CrossAttentionDecoder, Decoder, Device, Encoder, TransformerConfig, TransformerModel,
};

use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, Axis};

// Re-export model traits and registry
pub use models::{
    DecoderLanguageModel, EncoderLanguageModel, LanguageModel, ModelArchitecture, ModelType,
    Seq2SeqLanguageModel,
};

// Prelude for easy imports
pub mod prelude {
    pub use crate::cache::{Cache, CpuKVCache, GpuKVCache};
    pub use crate::gpu_context::WgpuContext;
    pub use crate::models::{DecoderLanguageModel, EncoderLanguageModel, LanguageModel};
    pub use crate::traits::{Decoder, Device, Encoder, TransformerModel};
}

/// A generic transformer layer combining attention and feedforward.
/// This universal struct can represent an encoder layer, a decoder layer,
/// or an encoder-decoder's decoder layer.
pub struct TransformerLayer {
    // Self-Attention Components (always present)
    pub self_attn: MultiHeadAttention,
    pub self_attn_layer_norm: LayerNorm,

    // Cross-Attention Components (only for encoder-decoder models)
    pub cross_attn: Option<MultiHeadAttention>,
    pub cross_attn_layer_norm: Option<LayerNorm>,

    // Feed-Forward Components (always present)
    pub feedforward: FeedForward,
    pub ffn_layer_norm: LayerNorm,
}

// NOTE: THIS IS THE OLD LAYER, ONLY USED BY CPU ENCODER AND CROSS-ENCODER
// CPU AND GPU DECODER USE THE NEW SPECIFIC GPU/CPU DECODER LAYER, 
// TODO: REMOVE
impl TransformerLayer {
    /// Forward pass for an encoder or decoder-only layer with KV caching.

    pub fn forward_with_cache(
        &self,
        mut hidden: Array3<f32>,
        attention_mask: &Array2<f32>,
        config: &dyn TransformerConfig,
        layer_idx: usize,
        cache: Option<&mut CpuKVCache>,
    ) -> Result<Array3<f32>> {
        let is_prenorm = config.is_prenorm();
        let is_causal = config.is_causal();

        if is_prenorm {
            // === PRE-NORM (e.g., GPT-2) ===

            // --- 1. First Sub-layer: Self-Attention ---
            let residual_1 = hidden.clone();
            let ln1_out = self.self_attn_layer_norm.forward_3d(&hidden);

            let cached_kv = cache.as_ref().and_then(|c| c.get(layer_idx));

            let (attn_out, new_k, new_v) = self.self_attn.forward_with_cache(
                &ln1_out,
                None,
                Some(attention_mask),
                is_causal,
                cached_kv,
            )?;

            if let Some(c) = cache {
                c.update(layer_idx, &new_k, &new_v)?;
            }

            // First residual connection. The result is `attn_block_output`.
            let attn_block_output = residual_1 + attn_out;

            // --- 2. Second Sub-layer: Feed-Forward Network (MLP) ---
            let residual_2 = attn_block_output.clone();

            // ============================ THE FIX ============================
            // Ensure the tensor is in a standard memory layout before passing it
            // to the next layer norm and FFN. This is the missing piece.
            let attn_block_output_contiguous = attn_block_output.as_standard_layout().to_owned();
            let ln2_out = self
                .ffn_layer_norm
                .forward_3d(&attn_block_output_contiguous);
            // ===============================================================

            let ffn_out = self.feedforward.forward(&ln2_out)?;

            // Second residual connection
            let block_output = residual_2.as_standard_layout().to_owned() + 
                ffn_out.as_standard_layout().to_owned();

            hidden = block_output;
        } else {
            // === POST-NORM (e.g., BERT, BART) ===
            let residual = hidden.clone();
            let cached_kv = cache.as_ref().and_then(|c| c.get(layer_idx));

            let (attn_out, new_k, new_v) = self.self_attn.forward_with_cache(
                &hidden,
                None,
                Some(attention_mask),
                is_causal,
                cached_kv,
            )?;

            if let Some(cache) = cache {
                cache.update(layer_idx, &new_k, &new_v)?;
            }

            hidden = residual + attn_out;
            hidden = self.self_attn_layer_norm.forward_3d(&hidden);

            let residual = hidden.clone();
            let ffn_out = self.feedforward.forward(&hidden)?;
            hidden = residual + ffn_out;
            hidden = self.ffn_layer_norm.forward_3d(&hidden);
        }

        Ok(hidden)
    }


    pub fn forward_cross_attention(
        &self,
        mut hidden_states: Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        decoder_attention_mask: &Array2<f32>,
        encoder_attention_mask: &Array2<f32>,
        config: &dyn TransformerConfig,
        layer_idx: usize,
        cache: Option<&mut CpuKVCache>,
    ) -> Result<Array3<f32>> {
        let cross_attn = self.cross_attn.as_ref().ok_or_else(|| {
            anyhow!("Layer is not configured for cross-attention (cross_attn is None)")
        })?;
        let cross_attn_ln = self.cross_attn_layer_norm.as_ref().ok_or_else(|| {
            anyhow!("Layer is not configured for cross-attention (cross_attn_layer_norm is None)")
        })?;

        anyhow::ensure!(
            !config.is_prenorm(),
            "Cross-attention forward pass currently only supports post-norm architectures like BART."
        );

        // === 1. Self-Attention Block (cached) ===
        let residual = hidden_states.clone();
        let cached_kv = cache.as_ref().and_then(|c| c.get(layer_idx));

        let (self_attn_output, new_k, new_v) = self.self_attn.forward_with_cache(
            &hidden_states,
            None,
            Some(decoder_attention_mask),
            true,
            cached_kv,
        )?;

        if let Some(cache) = cache {
            cache.update(layer_idx, &new_k, &new_v)?;
        }

        hidden_states = residual + &self_attn_output;
        hidden_states = self.self_attn_layer_norm.forward_3d(&hidden_states);

        // === 2. Cross-Attention Block ===
        let residual = hidden_states.clone();

        let (cross_attn_output, _, _) = cross_attn.forward_with_cache(
            &hidden_states,
            Some(encoder_hidden_states),
            Some(encoder_attention_mask),
            false,
            None,
        )?;

        hidden_states = residual + &cross_attn_output;
        hidden_states = cross_attn_ln.forward_3d(&hidden_states);

        // === 3. Feed-Forward Block ===
        let residual = hidden_states.clone();
        let ffn_output = self.feedforward.forward(&hidden_states)?;
        hidden_states = residual + &ffn_output;
        hidden_states = self.ffn_layer_norm.forward_3d(&hidden_states);

        Ok(hidden_states)
    }

    /// Original forward without cache (for compatibility with encoders).
    pub fn forward(
        &self,
        hidden: Array3<f32>,
        attention_mask: &Array2<f32>,
        config: &dyn TransformerConfig,
    ) -> Result<Array3<f32>> {
        self.forward_with_cache(hidden, attention_mask, config, 0, None)
    }

    /// Performs a forward pass through the transformer layer.
    ///
    /// This method is now stateless with respect to the KV cache. It takes the past
    /// key/value state for this layer as an input and returns the new key/value
    /// state it generated as an output. The caller (the decoder loop) is responsible for managing the cache.
    ///
    /// # Arguments
    /// * `hidden_states`: The input from the previous layer or embeddings.
    /// * `attention_mask`: The attention mask for the full sequence.
    /// * `position_offset`: The number of tokens already in the cache, used for causal masking.
    /// * `config`: The model's configuration trait to check for pre-norm, etc.
    /// * `past_kv`: An optional tuple containing the readonly `(K, V)` caches from previous steps for this layer.
    ///
    /// # Returns
    /// A tuple containing:
    /// 1. The output hidden states for this layer.
    /// 2. A tuple `(new_k, new_v)` representing the key/value states generated in this pass.
    pub fn forward_2(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        config: &dyn TransformerConfig,
        past_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        // We only implement the pre-norm logic as that's what the original code used.
        // The post-norm path can be refactored similarly if needed.
        if !config.is_prenorm() {
            anyhow::bail!("This refactored forward pass currently only supports pre-norm architectures like GPT-2.");
        }

        // === PRE-NORM (e.g., GPT-2) ===

        // --- 1. First Sub-layer: Self-Attention ---
        let residual = hidden_states.clone();
        let ln1_out = self.self_attn_layer_norm.forward_3d(hidden_states);

        // Step 1a: Project the current hidden states to get the K/V for THIS token.
        // This is the "new" state that will be returned to the cache manager.
        let (new_k, new_v) = self.self_attn.project_kv(&ln1_out);

        // Step 1b: If a cache exists, combine the past state with the new state
        // to create the full K/V context for the attention calculation.
        let (full_k, full_v) = if let Some((past_k, past_v)) = past_kv {
            (
                ndarray::concatenate(Axis(1), &[past_k.view(), new_k.view()])?,
                ndarray::concatenate(Axis(1), &[past_v.view(), new_v.view()])?,
            )
        } else {
            // No cache (e.g., priming pass), so the "full" context is just the new state.
            (new_k.clone(), new_v.clone())
        };

        // Step 1c: Perform the attention calculation with the complete K/V state.
        let attn_out = self.self_attn.attend(
            &ln1_out,
            &full_k,
            &full_v,
            Some(attention_mask),
            config.is_causal(),
            position_offset,
        )?;

        // First residual connection.
        let attn_block_output = residual + attn_out;

        // --- 2. Second Sub-layer: Feed-Forward Network (MLP) ---
        let residual = attn_block_output.clone();
        let ln2_out = self.ffn_layer_norm.forward_3d(&attn_block_output);
        let ffn_out = self.feedforward.forward(&ln2_out)?;

        // Second residual connection
        let final_output = residual + ffn_out;

        // Finally, return the output hidden state and the NEW K/V slices.
        // The caller (`CpuTransformerDecoder`) is now responsible for storing these in the cache.
        Ok((final_output, (new_k, new_v)))
    }

}
