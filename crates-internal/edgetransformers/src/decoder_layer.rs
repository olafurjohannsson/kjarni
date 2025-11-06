
use crate::TransformerLayer;
use crate::attention::MultiHeadAttention;
use crate::layer_norm::LayerNorm;
use crate::feedforward::FeedForward;
use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, Axis};

/// Represents a single layer for a decoder-only transformer model (e.g., GPT-2).
///
/// This layer is specialized for causal self-attention and follows a specific
/// sub-layer order (e.g., pre-LayerNorm). It is designed to be stateless
/// with respect to the KV cache, making the data flow explicit and suitable for
/// high-performance backends like GPUs.

pub struct PreNormDecoderLayer {
    pub self_attn: MultiHeadAttention,
    pub self_attn_layer_norm: LayerNorm,
    pub feedforward: FeedForward,
    pub ffn_layer_norm: LayerNorm,
}
pub struct PostNormDecoderLayer {
    pub self_attn: MultiHeadAttention,
    pub self_attn_layer_norm: LayerNorm,
    pub feedforward: FeedForward,
    pub ffn_layer_norm: LayerNorm,
}

pub enum DecoderLayer {
    PreNorm(PreNormDecoderLayer),
    PostNorm(PostNormDecoderLayer),
}

impl DecoderLayer {
    // This is the single, unified method the decoder will call.
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        past_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        match self {
            DecoderLayer::PreNorm(layer) => {
                layer.forward(hidden_states, attention_mask, position_offset, past_kv)
            }
            DecoderLayer::PostNorm(layer) => {
                unimplemented!("PostNorm not implemented")
            }
        }
    }
}

impl PreNormDecoderLayer {


    /// Performs a forward pass through the decoder layer.
    ///
    /// This method is stateless with respect to the KV cache. It takes the past
    /// key/value state for this layer as an input and returns the new key/value
    /// state it generated as an output. The caller (the decoder loop) is responsible
    /// for managing the cache.
    ///
    /// # Returns
    /// A tuple containing:
    /// 1. The output hidden states for this layer.
    /// 2. A tuple `(new_k, new_v)` representing the key/value states generated in this pass.
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        past_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        // This implementation is specific to a pre-norm architecture like GPT-2.
        // If you need to support post-norm decoders, a similar specialized layer
        // could be created, or a config flag could be used here.

        // --- 1. First Sub-layer: Self-Attention (Pre-Norm) ---
        let residual = hidden_states.clone();
        let ln1_out = self.self_attn_layer_norm.forward_3d(hidden_states);

        // Project the current hidden states to get the K/V for THIS token.
        let (new_k, new_v) = self.self_attn.project_kv(&ln1_out);

        // Combine the past state with the new state to create the full K/V context.
        let (full_k, full_v) = if let Some((past_k, past_v)) = past_kv {
            (
                ndarray::concatenate(Axis(1), &[past_k.view(), new_k.view()])?,
                ndarray::concatenate(Axis(1), &[past_v.view(), new_v.view()])?,
            )
        } else {
            // No cache (e.g., priming pass), the context is just the new state.
            (new_k.clone(), new_v.clone())
        };

        // Perform the attention calculation with the complete K/V state.
        let attn_out = self.self_attn.attend(
            &ln1_out,
            &full_k,
            &full_v,
            Some(attention_mask),
            true, // is_causal is always true for a DecoderLayer
            position_offset,
        )?;

        // First residual connection.
        let attn_block_output = residual + attn_out;

        // --- 2. Second Sub-layer: Feed-Forward Network (Pre-Norm) ---
        let residual = attn_block_output.clone();
        let ln2_out = self.ffn_layer_norm.forward_3d(&attn_block_output);
        let ffn_out = self.feedforward.forward(&ln2_out)?;

        // Second residual connection.
        let final_output = residual + ffn_out;

        // Return the final output and the newly generated K/V state for the cache manager.
        Ok((final_output, (new_k, new_v)))
    }
}