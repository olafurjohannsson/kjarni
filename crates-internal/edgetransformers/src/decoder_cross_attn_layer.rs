// Re-export commonly used items
use crate::utils::linear_algebra::matmul_3d_2d;
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
use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, ArrayView3, Axis};

/// A generic transformer layer combining attention and feedforward.
/// This universal struct can represent an encoder layer, a decoder layer,
/// or an encoder-decoder's decoder layer.
pub struct DecoderCrossAttentionLayer {
    // Self-Attention Components
    pub self_attn: MultiHeadAttention,
    pub self_attn_layer_norm: LayerNorm,
    // Cross-Attention Components
    pub cross_attn: MultiHeadAttention,
    pub cross_attn_layer_norm: LayerNorm,
    // Feed-Forward Components
    pub feedforward: FeedForward,
    pub ffn_layer_norm: LayerNorm,
}

impl DecoderCrossAttentionLayer {
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        self_attention_mask: Option<&Array2<f32>>,
        cross_attention_mask: Option<&Array2<f32>>,
        past_kv: Option<(ArrayView3<f32>, ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        // Step 1: Self-Attention
        let (mut hidden_states, (new_k, new_v)) =
            self.self_attention_block(hidden_states, self_attention_mask, past_kv)?;

        // Step 2: Cross-Attention
        hidden_states = self.cross_attention_block(
            &hidden_states,
            encoder_hidden_states,
            cross_attention_mask,
        )?;

        // Step 3: Feed-Forward
        hidden_states = self.feed_forward_block(&hidden_states)?;

        Ok((hidden_states, (new_k, new_v)))
        // let residual = hidden_states.clone();
        // let (attn_output, new_k, new_v) = self.self_attn.forward_with_cache(
        //     hidden_states,
        //     None, // Key/Value source is the same as query for self-attention
        //     self_attention_mask,
        //     true, // Causal mask for self-attention
        //     past_kv,
        //     None,
        // )?;
        //
        // let mut hidden_states = residual + attn_output;
        // hidden_states = self.self_attn_layer_norm.forward_3d(&hidden_states);
        //
        // let residual = hidden_states.clone();
        //
        // let (cross_attn_output, _, _) = self.cross_attn.forward_with_cache(
        //     &hidden_states,              // Query source
        //     Some(encoder_hidden_states), // Key/Value source
        //     cross_attention_mask,
        //     false, // Not causal
        //     None,  // No KV cache for cross-attention
        //     None,
        // )?;
        //
        // hidden_states = residual + cross_attn_output;
        // hidden_states = self.cross_attn_layer_norm.forward_3d(&hidden_states);
        //
        // // --- 3. Feed-Forward Block ---
        // let residual = hidden_states.clone();
        // let ffn_output = self.feedforward.forward(&hidden_states)?;
        // hidden_states = residual + ffn_output;
        // hidden_states = self.ffn_layer_norm.forward_3d(&hidden_states);
        //
        // // Return the final hidden state and the NEW self-attention K/V pair to be cached.
        // Ok((hidden_states, (new_k, new_v)))
    }
    pub fn self_attention_block(
        &self,
        hidden_states: &Array3<f32>,
        self_attention_mask: Option<&Array2<f32>>,
        past_kv: Option<(ArrayView3<f32>, ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        let residual = hidden_states.clone();
        let (attn_output, new_k, new_v) = self.self_attn.forward_with_cache(
            hidden_states,
            None,
            self_attention_mask,
            true,
            past_kv,
            None,
        )?;

        let mut hidden_states = residual + attn_output;
        hidden_states = self.self_attn_layer_norm.forward_3d(&hidden_states);
        Ok((hidden_states, (new_k, new_v)))
    }

    /// Executes the cross-attention block (Sublayer -> Add -> Norm).
    pub fn cross_attention_block(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        cross_attention_mask: Option<&Array2<f32>>,
    ) -> Result<Array3<f32>> {
        let residual = hidden_states.clone();
        let (cross_attn_output, _, _) = self.cross_attn.forward_with_cache(
            hidden_states,
            Some(encoder_hidden_states),
            cross_attention_mask,
            false,
            None,
            None,
        )?;

        let mut hidden_states = residual + cross_attn_output;
        hidden_states = self.cross_attn_layer_norm.forward_3d(&hidden_states);
        Ok(hidden_states)
    }

    /// Executes the feed-forward block (Sublayer -> Add -> Norm).
    pub fn feed_forward_block(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        let residual = hidden_states.clone();
        let ffn_output = self.feedforward.forward(hidden_states)?;
        let mut hidden_states = residual + ffn_output;
        hidden_states = self.ffn_layer_norm.forward_3d(&hidden_states);
        Ok(hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::MultiHeadAttention;
    use crate::feedforward::{FeedForward, StdFeedForward};
    use crate::normalization::LayerNorm;
    use ndarray::{Array1, Array2, Array3, ArrayView3};

    fn create_mock_cross_attention_layer(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
    ) -> DecoderCrossAttentionLayer {
        let q_weight = Array2::from_shape_fn((hidden_size, hidden_size), |(i, j)| {
            if i == j { 1.1 } else { (i + j) as f32 * 0.001 }
        });
        let o_weight = Array2::from_shape_fn((hidden_size, hidden_size), |(i, j)| {
            if i == j { 0.9 } else { (i + j) as f32 * -0.001 }
        });
        let fc1_weight = Array2::from_shape_fn((hidden_size, intermediate_size), |(i, j)| {
            if i == j { 1.05 } else { 0.001 }
        });
        let fc2_weight = Array2::from_shape_fn((intermediate_size, hidden_size), |(i, j)| {
            if i == j { 0.95 } else { -0.001 }
        });

        let self_attn = MultiHeadAttention::new(
            hidden_size,
            num_heads,
            q_weight.clone(),
            Array1::from_elem(hidden_size, 0.1),
            q_weight.clone(),
            Array1::zeros(hidden_size),
            q_weight.clone(),
            Array1::zeros(hidden_size),
            o_weight.clone(),
            Array1::zeros(hidden_size),
            None,
        );
        let self_attn_layer_norm =
            LayerNorm::new(Array1::ones(hidden_size), Array1::zeros(hidden_size), 1e-5);

        let cross_attn = MultiHeadAttention::new(
            hidden_size,
            num_heads,
            q_weight.clone(),
            Array1::from_elem(hidden_size, 0.1),
            q_weight.clone(),
            Array1::zeros(hidden_size),
            q_weight.clone(),
            Array1::zeros(hidden_size),
            o_weight.clone(),
            Array1::zeros(hidden_size),
            None,
        );
        let cross_attn_layer_norm =
            LayerNorm::new(Array1::ones(hidden_size), Array1::zeros(hidden_size), 1e-5);

        let feedforward = FeedForward::Standard(StdFeedForward::new(
            fc1_weight,
            Array1::zeros(intermediate_size),
            fc2_weight,
            Array1::zeros(hidden_size),
            crate::activations::Activation::Gelu, // TODO CONFIG!!
        ));
        let ffn_layer_norm =
            LayerNorm::new(Array1::ones(hidden_size), Array1::zeros(hidden_size), 1e-5);

        DecoderCrossAttentionLayer {
            self_attn,
            self_attn_layer_norm,
            cross_attn,
            cross_attn_layer_norm,
            feedforward,
            ffn_layer_norm,
        }
    }

    #[test]
    fn test_decoder_cross_attention_layer_forward() -> Result<()> {
        let (batch_size, dec_len, enc_len, hidden, inter, heads) = (2, 5, 20, 64, 128, 4);
        let layer = create_mock_cross_attention_layer(hidden, inter, heads);

        let hidden_states = Array3::<f32>::ones((batch_size, dec_len, hidden));
        let encoder_hidden_states = Array3::<f32>::ones((batch_size, enc_len, hidden));
        let self_mask = Array2::<f32>::ones((batch_size, dec_len));
        let cross_mask = Array2::<f32>::ones((batch_size, enc_len));

        let result = layer.forward(
            &hidden_states,
            &encoder_hidden_states,
            Some(&self_mask),
            Some(&cross_mask),
            None,
        );

        assert!(result.is_ok(), "Forward pass failed: {:?}", result.err());
        let (output, (new_k, new_v)) = result.unwrap();

        assert_eq!(output.shape(), &[batch_size, dec_len, hidden]);
        assert_eq!(new_k.shape(), &[batch_size, dec_len, hidden]);
        assert_eq!(new_v.shape(), &[batch_size, dec_len, hidden]);

        // CORRECT ASSERTION for post-norm: The mean should be very close to 0.
        assert!(
            output.mean().unwrap().abs() < 1e-6,
            "Post-norm output mean should be near zero"
        );
        // The standard deviation should be very close to 1.
        assert!(
            (output.std(0.0) - 1.0).abs() < 1e-5,
            "Post-norm output std dev should be near one"
        );

        Ok(())
    }
}
