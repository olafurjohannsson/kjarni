use crate::encoder_decoder::{DecoderCrossAttention, DecoderSelfAttention};
pub use crate::{
    attention::MultiHeadAttention,
    cache::CpuKVCache,
    embeddings::Embeddings,
    feedforward::FeedForward,
    normalization::LayerNorm,
    pooling::{cls_pool, last_token_pool, max_pool, mean_pool, PoolingStrategy},
    traits::TransformerConfig,
    weights::ModelWeights,
};
use anyhow::Result;
use ndarray::{Array2, Array3, Array4, ArrayView3};
use std::time::Instant;
/// A generic transformer layer combining attention and feedforward.
/// This universal struct can represent an encoder layer, a decoder layer,
/// or an encoder-decoder's decoder layer.
pub struct DecoderCrossAttentionLayer {
    // Self-Attention Components
    pub self_attn: DecoderSelfAttention,
    pub self_attn_layer_norm: LayerNorm,
    // Cross-Attention Components
    pub cross_attn: DecoderCrossAttention,
    pub cross_attn_layer_norm: LayerNorm,
    // Feed-Forward Components
    pub feedforward: FeedForward,
    pub ffn_layer_norm: LayerNorm,
}

// Add this import

impl DecoderCrossAttentionLayer {
    pub fn precompute_cross_kv(
        &self,
        encoder_hidden_states: &Array3<f32>,
    ) -> Result<(Array4<f32>, Array4<f32>)> {
        // Delegate the call to the cross-attention component, which already has this logic.
        self.cross_attn.precompute_encoder_kv(encoder_hidden_states)
    }

    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        self_mask: Option<&Array2<f32>>,
        cross_mask: Option<&Array2<f32>>,
        past_kv: Option<(ArrayView3<f32>, ArrayView3<f32>)>,
        // OPTIONAL: Pass pre-computed Cross KV here if you have it
        cross_kv_cache: Option<&(ndarray::Array4<f32>, ndarray::Array4<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        // 1. Self Attention
        let t_sa = Instant::now();
        let residual = hidden_states.clone();
        let (attn_out, new_k, new_v) = self.self_attn.forward(hidden_states, self_mask, past_kv)?;

        let hidden_states = self.self_attn_layer_norm.forward_3d(&(residual + attn_out));
        log::info!("[CpuLayer] Self-Attention block took: {:?}", t_sa.elapsed());

        // 2. Cross Attention
        let t_ca = Instant::now();
        let residual = hidden_states.clone();

        let cross_out = if let Some((k_static, v_static)) = cross_kv_cache {
            // Fast Path: Zero Copy
            self.cross_attn
                .forward(&hidden_states, k_static, v_static, cross_mask)?
        } else {
            // Slow Path: Re-calculate K/V on the fly (Backwards Compatible)
            let (k_static, v_static) = self
                .cross_attn
                .precompute_encoder_kv(encoder_hidden_states)?;
            self.cross_attn
                .forward(&hidden_states, &k_static, &v_static, cross_mask)?
        };

        let hidden_states = self
            .cross_attn_layer_norm
            .forward_3d(&(residual + cross_out));
        log::info!(
            "[CpuLayer] Cross-Attention block took: {:?}",
            t_ca.elapsed()
        );

        // 3. FFN
        let t_ffn = Instant::now();
        let residual = hidden_states.clone();
        let ffn_out = self.feedforward.forward(&hidden_states)?;
        let hidden_states = self.ffn_layer_norm.forward_3d(&(residual + ffn_out));
        log::info!("[CpuLayer] FFN block took: {:?}", t_ffn.elapsed());
        Ok((hidden_states, (new_k, new_v)))
    }

    pub fn cross_attention(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        cross_attention_mask: Option<&Array2<f32>>,
        precomputed_kv: Option<&(ndarray::Array4<f32>, ndarray::Array4<f32>)>,
    ) -> Result<Array3<f32>> {
        let residual = hidden_states.clone();
        let cross_attn_output = if let Some((k, v)) = precomputed_kv {
            self.cross_attn
                .forward(hidden_states, k, v, cross_attention_mask)?
        } else {
            let (k, v) = self
                .cross_attn
                .precompute_encoder_kv(encoder_hidden_states)?;
            self.cross_attn
                .forward(hidden_states, &k, &v, cross_attention_mask)?
        };
        let hidden_states_after_add = residual + cross_attn_output;
        let final_output = self
            .cross_attn_layer_norm
            .forward_3d(&hidden_states_after_add);
        Ok(final_output)
    }
    pub fn feed_forward(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        let residual = hidden_states.clone();
        let ffn_output = self.feedforward.forward(hidden_states)?;
        let hidden_states_after_add = residual + ffn_output;
        let final_output = self.ffn_layer_norm.forward_3d(&hidden_states_after_add);
        Ok(final_output)
    }
    pub fn self_attention(
        &self,
        hidden_states: &Array3<f32>,
        self_attention_mask: Option<&Array2<f32>>,
        past_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        let residual = hidden_states.clone();
        let (attn_output, new_k, new_v) =
            self.self_attn
                .forward(hidden_states, self_attention_mask, past_kv)?;
        let hidden_states_after_add = residual + attn_output;
        let final_output = self
            .self_attn_layer_norm
            .forward_3d(&hidden_states_after_add);
        Ok((final_output, (new_k, new_v)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feedforward::{FeedForward, LegacyFeedForward};
    use crate::linear_layer::LinearLayer;
    use crate::normalization::LayerNorm;
    use ndarray::{Array1, Array2, Array3};

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

        let self_attn = DecoderSelfAttention::new(
            hidden_size,
            num_heads,
            LinearLayer::from(q_weight.clone()),
            LinearLayer::from(q_weight.clone()),
            LinearLayer::from(q_weight.clone()),
            LinearLayer::from(o_weight.clone()),
        );
        let self_attn_layer_norm =
            LayerNorm::new(Array1::ones(hidden_size), Array1::zeros(hidden_size), 1e-5);

        let cross_attn = DecoderCrossAttention::new(
            hidden_size,
            num_heads,
            LinearLayer::from(q_weight.clone()),
            LinearLayer::from(q_weight.clone()),
            LinearLayer::from(q_weight.clone()),
            LinearLayer::from(o_weight.clone()),
        );
        let cross_attn_layer_norm =
            LayerNorm::new(Array1::ones(hidden_size), Array1::zeros(hidden_size), 1e-5);

        let feedforward = FeedForward::Legacy(LegacyFeedForward::new(
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
