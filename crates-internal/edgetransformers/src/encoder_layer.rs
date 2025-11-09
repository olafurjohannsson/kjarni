// Re-export commonly used items
pub use crate::{
    attention::MultiHeadAttention,
    embeddings::Embeddings,
    feedforward::FeedForward,
    normalization::LayerNorm,
    pooling::{PoolingStrategy, cls_pool, last_token_pool, max_pool, mean_pool},
    weights::ModelWeights,
    traits::TransformerConfig,
    cache::CpuKVCache,
};

use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3};


/// A generic transformer layer combining attention and feedforward.
/// This universal struct can represent an encoder layer, a decoder layer,
/// or an encoder-decoder's decoder layer.
pub struct EncoderLayer {
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

impl EncoderLayer {
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
            let residual_1 = hidden.clone();
            let ln1_out = self.self_attn_layer_norm.forward_3d(&hidden);

            let cached_kv = cache.as_ref().and_then(|c| c.get(layer_idx));

            let (attn_out, new_k, new_v) = self.self_attn.forward_with_cache(
                &ln1_out,
                None,
                Some(attention_mask),
                is_causal,
                cached_kv,
                None,
            )?;

            if let Some(c) = cache {
                c.update(layer_idx, &new_k, &new_v)?;
            }

            let attn_block_output = residual_1 + attn_out;
            let residual_2 = attn_block_output.clone();
            let attn_block_output_contiguous = attn_block_output.as_standard_layout().to_owned();
            let ln2_out = self
                .ffn_layer_norm
                .forward_3d(&attn_block_output_contiguous);
            let ffn_out = self.feedforward.forward(&ln2_out)?;
            let block_output = residual_2.as_standard_layout().to_owned() + 
                ffn_out.as_standard_layout().to_owned();
            hidden = block_output;
        } else {
            let residual = hidden.clone();
            let cached_kv = cache.as_ref().and_then(|c| c.get(layer_idx));
            let (attn_out, new_k, new_v) = self.self_attn.forward_with_cache(
                &hidden,
                None,
                Some(attention_mask),
                is_causal,
                cached_kv,
                None,
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
            None,
        )?;

        if let Some(cache) = cache {
            cache.update(layer_idx, &new_k, &new_v)?;
        }

        hidden_states = residual + &self_attn_output;
        hidden_states = self.self_attn_layer_norm.forward_3d(&hidden_states);
        let residual = hidden_states.clone();

        let (cross_attn_output, _, _) = cross_attn.forward_with_cache(
            &hidden_states,
            Some(encoder_hidden_states),
            Some(encoder_attention_mask),
            false,
            None,
            None,
        )?;

        hidden_states = residual + &cross_attn_output;
        hidden_states = cross_attn_ln.forward_3d(&hidden_states);
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


}
