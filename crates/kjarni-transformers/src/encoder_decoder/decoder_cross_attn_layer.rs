use crate::encoder_decoder::{DecoderCrossAttention, DecoderSelfAttention};
use crate::Normalization;
pub use crate::{
    attention::MultiHeadAttention,
    cache::CpuKVCache,
    embeddings::Embeddings,
    feedforward::FeedForward,
    normalization::LayerNorm,
    pooling::{cls_pool, last_token_pool, max_pool, mean_pool, PoolingStrategy},
    weights::ModelWeights,
};
use anyhow::Result;
use ndarray::{Array2, Array3, Array4, ArrayView3};

/// A generic transformer layer combining attention and feedforward.
/// This universal struct can represent an encoder layer, a decoder layer,
/// or an encoder-decoder's decoder layer.
/// Unified decoder layer with cross-attention.
///
/// Supports both pre-norm (T5, Whisper) and post-norm (BART) architectures.
pub struct CrossDecoderLayer {
    // Self-Attention Components
    pub self_attn: DecoderSelfAttention,
    pub self_attn_layer_norm: Normalization,

    // Cross-Attention Components
    pub cross_attn: DecoderCrossAttention,
    pub cross_attn_layer_norm: Normalization,

    // Feed-Forward Components
    pub feedforward: FeedForward,
    pub ffn_layer_norm: Normalization,

    // Architecture flags
    pub pre_norm: bool,
}

impl CrossDecoderLayer {
    /// Create a new decoder layer.
    pub fn new(
        self_attn: DecoderSelfAttention,
        self_attn_layer_norm: Normalization,
        cross_attn: DecoderCrossAttention,
        cross_attn_layer_norm: Normalization,
        feedforward: FeedForward,
        ffn_layer_norm: Normalization,
        pre_norm: bool,
    ) -> Self {
        Self {
            self_attn,
            self_attn_layer_norm,
            cross_attn,
            cross_attn_layer_norm,
            feedforward,
            ffn_layer_norm,
            pre_norm,
        }
    }

    /// Pre-compute cross-attention K/V from encoder hidden states.
    ///
    /// Call once per generation, then reuse for all decode steps.
    pub fn precompute_cross_kv(
        &self,
        encoder_hidden_states: &Array3<f32>,
    ) -> Result<(Array4<f32>, Array4<f32>)> {
        self.cross_attn.precompute_encoder_kv(encoder_hidden_states)
    }

    /// Full forward pass through the layer.
    ///
    /// # Arguments
    /// * `hidden_states` - Input hidden states [batch, seq, hidden]
    /// * `encoder_hidden_states` - Encoder output (used if cross_kv_cache is None)
    /// * `self_mask` - Causal mask for self-attention
    /// * `cross_mask` - Mask for cross-attention
    /// * `past_kv` - Cached self-attention K/V from previous steps
    /// * `cross_kv_cache` - Pre-computed cross-attention K/V
    /// * `position_bias` - Optional relative position bias (T5)
    ///
    /// # Returns
    /// (output_hidden_states, (new_self_k, new_self_v))
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        self_mask: Option<&Array2<f32>>,
        cross_mask: Option<&Array2<f32>>,
        past_kv: Option<(ArrayView3<f32>, ArrayView3<f32>)>,
        cross_kv_cache: Option<&(Array4<f32>, Array4<f32>)>,
        position_bias: Option<&Array4<f32>>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        let (hidden_states, new_kv) = if self.pre_norm {
            self.forward_prenorm(
                hidden_states,
                encoder_hidden_states,
                self_mask,
                cross_mask,
                past_kv,
                cross_kv_cache,
                position_bias,
            )?
        } else {
            self.forward_postnorm(
                hidden_states,
                encoder_hidden_states,
                self_mask,
                cross_mask,
                past_kv,
                cross_kv_cache,
            )?
        };

        Ok((hidden_states, new_kv))
    }

    /// Post-norm forward (BART style): sublayer -> add -> norm
    fn forward_postnorm(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        self_mask: Option<&Array2<f32>>,
        cross_mask: Option<&Array2<f32>>,
        past_kv: Option<(ArrayView3<f32>, ArrayView3<f32>)>,
        cross_kv_cache: Option<&(Array4<f32>, Array4<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        // 1. Self-Attention: attn -> add -> norm
        let residual = hidden_states.clone();
        let (attn_out, new_k, new_v) = self.self_attn.forward(hidden_states, self_mask, past_kv, None)?;
        let hidden_states = self.self_attn_layer_norm.forward(&(residual + attn_out));

        // 2. Cross-Attention: attn -> add -> norm
        let residual = hidden_states.clone();
        let cross_out = self.compute_cross_attention(
            &hidden_states,
            encoder_hidden_states,
            cross_mask,
            cross_kv_cache,
        )?;
        let hidden_states = self.cross_attn_layer_norm.forward(&(residual + cross_out));

        // 3. FFN: ffn -> add -> norm
        let residual = hidden_states.clone();
        let ffn_out = self.feedforward.forward(&hidden_states)?;
        let hidden_states = self.ffn_layer_norm.forward(&(residual + ffn_out));

        Ok((hidden_states, (new_k, new_v)))
    }

    /// Pre-norm forward (T5/Whisper style): norm -> sublayer -> add
    fn forward_prenorm(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        self_mask: Option<&Array2<f32>>,
        cross_mask: Option<&Array2<f32>>,
        past_kv: Option<(ArrayView3<f32>, ArrayView3<f32>)>,
        cross_kv_cache: Option<&(Array4<f32>, Array4<f32>)>,
        position_bias: Option<&Array4<f32>>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        // 1. Self-Attention: norm -> attn -> add
        let normed = self.self_attn_layer_norm.forward(hidden_states);
        let (attn_out, new_k, new_v) = self.self_attn.forward(
            &normed,
            self_mask,
            past_kv,
            position_bias,
        )?;
        let hidden_states = hidden_states + &attn_out;

        // 2. Cross-Attention: norm -> attn -> add
        let normed = self.cross_attn_layer_norm.forward(&hidden_states);
        let cross_out = self.compute_cross_attention(
            &normed,
            encoder_hidden_states,
            cross_mask,
            cross_kv_cache,
        )?;
        let hidden_states = hidden_states + &cross_out;

        // 3. FFN: norm -> ffn -> add
        let normed = self.ffn_layer_norm.forward(&hidden_states);
        let ffn_out = self.feedforward.forward(&normed)?;
        let hidden_states = hidden_states + &ffn_out;

        Ok((hidden_states, (new_k, new_v)))
    }

    /// Compute cross-attention, using cache if available.
    fn compute_cross_attention(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        cross_mask: Option<&Array2<f32>>,
        cross_kv_cache: Option<&(Array4<f32>, Array4<f32>)>,
    ) -> Result<Array3<f32>> {
        if let Some((k_cache, v_cache)) = cross_kv_cache {
            // Fast path: use pre-computed K/V
            self.cross_attn.forward(hidden_states, k_cache, v_cache, cross_mask)
        } else {
            // Slow path: compute K/V on the fly
            let (k, v) = self.cross_attn.precompute_encoder_kv(encoder_hidden_states)?;
            self.cross_attn.forward(hidden_states, &k, &v, cross_mask)
        }
    }

    // =========================================================================
    // Individual component access (for fine-grained control)
    // =========================================================================

    pub fn self_attention(
        &self,
        hidden_states: &Array3<f32>,
        mask: Option<&Array2<f32>>,
        past_kv: Option<(ArrayView3<f32>, ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        if self.pre_norm {
            let normed = self.self_attn_layer_norm.forward(hidden_states);
            let (attn_out, new_k, new_v) = self.self_attn.forward(&normed, mask, past_kv, None)?;
            Ok((hidden_states + &attn_out, (new_k, new_v)))
        } else {
            let residual = hidden_states.clone();
            let (attn_out, new_k, new_v) = self.self_attn.forward(hidden_states, mask, past_kv, None)?;
            let out = self.self_attn_layer_norm.forward(&(residual + attn_out));
            Ok((out, (new_k, new_v)))
        }
    }

    pub fn cross_attention(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        cross_mask: Option<&Array2<f32>>,
        cross_kv_cache: Option<&(Array4<f32>, Array4<f32>)>,
    ) -> Result<Array3<f32>> {
        if self.pre_norm {
            let normed = self.cross_attn_layer_norm.forward(hidden_states);
            let cross_out = self.compute_cross_attention(
                &normed,
                encoder_hidden_states,
                cross_mask,
                cross_kv_cache,
            )?;
            Ok(hidden_states + &cross_out)
        } else {
            let residual = hidden_states.clone();
            let cross_out = self.compute_cross_attention(
                hidden_states,
                encoder_hidden_states,
                cross_mask,
                cross_kv_cache,
            )?;
            Ok(self.cross_attn_layer_norm.forward(&(residual + cross_out)))
        }
    }

    pub fn feed_forward(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        if self.pre_norm {
            let normed = self.ffn_layer_norm.forward(hidden_states);
            let ffn_out = self.feedforward.forward(&normed)?;
            Ok(hidden_states + &ffn_out)
        } else {
            let residual = hidden_states.clone();
            let ffn_out = self.feedforward.forward(hidden_states)?;
            Ok(self.ffn_layer_norm.forward(&(residual + ffn_out)))
        }
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
    ) -> CrossDecoderLayer {
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
        let self_attn_layer_norm = Normalization::LayerNorm(LayerNorm::new(
            Array1::ones(hidden_size),
            Array1::zeros(hidden_size),
            1e-5,
        ));

        let cross_attn = DecoderCrossAttention::new(
            hidden_size,
            num_heads,
            LinearLayer::from(q_weight.clone()),
            LinearLayer::from(q_weight.clone()),
            LinearLayer::from(q_weight.clone()),
            LinearLayer::from(o_weight.clone()),
        );
        let cross_attn_layer_norm = Normalization::LayerNorm(LayerNorm::new(
            Array1::ones(hidden_size),
            Array1::zeros(hidden_size),
            1e-5,
        ));

        let feedforward = FeedForward::Legacy(LegacyFeedForward::new(
            fc1_weight,
            Array1::zeros(intermediate_size),
            fc2_weight,
            Array1::zeros(hidden_size),
            crate::activations::Activation::Gelu, // TODO CONFIG!!
        ));
        let ffn_layer_norm = Normalization::LayerNorm(LayerNorm::new(
            Array1::ones(hidden_size),
            Array1::zeros(hidden_size),
            1e-5,
        ));

        CrossDecoderLayer {
            self_attn,
            self_attn_layer_norm,
            cross_attn,
            cross_attn_layer_norm,
            feedforward,
            ffn_layer_norm,
            pre_norm: false,
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
