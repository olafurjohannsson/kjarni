// Re-export commonly used items
pub use crate::{
    attention::MultiHeadAttention,
    cache::CpuKVCache,
    embeddings::Embeddings,
    feedforward::FeedForward,
    normalization::LayerNorm,
    pooling::{PoolingStrategy, cls_pool, last_token_pool, max_pool, mean_pool},
    traits::TransformerConfig,
    weights::ModelWeights,
};
use crate::utils::linear_algebra::matmul_3d_2d;
use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, ArrayView3, Axis};

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
        println!("EncoderLayer prenorm: {} - causal: {}", is_prenorm, is_causal);
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
            let block_output = residual_2.as_standard_layout().to_owned()
                + ffn_out.as_standard_layout().to_owned();
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
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        self_attention_mask: Option<&Array2<f32>>,
        cross_attention_mask: Option<&Array2<f32>>,
        past_kv: Option<(ArrayView3<f32>, ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        // This is a standard post-layernorm decoder layer (like BART)

        // --- 1. Self-Attention Block ---
        let residual = hidden_states.clone();
        let (attn_output, new_k, new_v) = self.self_attn.forward_with_cache(
            &hidden_states,
            None, // Key/Value source is the same as query for self-attention
            self_attention_mask,
            true, // Causal mask for self-attention
            past_kv,
            None, // No RoPE for BART
        )?;
        let mut hidden_states = residual + attn_output;
        hidden_states = self.self_attn_layer_norm.forward_3d(&hidden_states);

        // --- 2. Cross-Attention Block ---
        let residual = hidden_states.clone();
        let cross_attn = self.cross_attn.as_ref().unwrap();

        // Project Q from decoder state, K and V from encoder state
        let q = matmul_3d_2d(&hidden_states, &cross_attn.q_weight) + &cross_attn.q_bias;
        let k = matmul_3d_2d(encoder_hidden_states, &cross_attn.k_weight) + &cross_attn.k_bias;
        let v = matmul_3d_2d(encoder_hidden_states, &cross_attn.v_weight) + &cross_attn.v_bias;

        // Compute attention using the projected Q, K, V
        let cross_attn_output = cross_attn.attend(
            &q,
            &k,
            &v,
            cross_attention_mask,
            false, // Not causal
            0,     // No position offset
            None,  // No RoPE
        )?;
        hidden_states = residual + cross_attn_output;
        hidden_states = self
            .cross_attn_layer_norm
            .as_ref()
            .unwrap()
            .forward_3d(&hidden_states);

        // --- 3. Feed-Forward Block ---
        let residual = hidden_states.clone();
        let ffn_output = self.feedforward.forward(&hidden_states)?;
        hidden_states = residual + ffn_output;
        hidden_states = self.ffn_layer_norm.forward_3d(&hidden_states);

        // Return the final hidden state and the NEW self-attention K/V pair to be cached.
        Ok((hidden_states, (new_k, new_v)))
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
