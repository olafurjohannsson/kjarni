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
pub struct EncoderLayer {
    // Self-Attention Components (always present)
    pub self_attn: MultiHeadAttention,
    pub self_attn_layer_norm: LayerNorm,

    // Feed-Forward Components (always present)
    pub feedforward: FeedForward,
    pub ffn_layer_norm: LayerNorm,
}

impl EncoderLayer {
    pub fn forward(
        &self,
        mut hidden: Array3<f32>,
        attention_mask: &Array2<f32>,
        config: &dyn TransformerConfig,
    ) -> Result<Array3<f32>> {

        let hidden = if config.is_prenorm() {
            self.forward_prenorm(hidden, attention_mask, config)
        } else {
            self.forward_postnorm(hidden, attention_mask, config)
        };

        hidden
    }
    pub fn forward_prenorm(
        &self,
        mut hidden: Array3<f32>,
        attention_mask: &Array2<f32>,
        config: &dyn TransformerConfig,
    ) -> Result<Array3<f32>> {
        let residual_1 = hidden.clone();
        let ln1_out = self.self_attn_layer_norm.forward_3d(&hidden);

        let (attn_out, new_k, new_v) = self.self_attn.forward_with_cache(
            &ln1_out,
            None,
            Some(attention_mask),
            config.is_causal(),
            None,
            None,
        )?;
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
        Ok(hidden)
    }
    pub fn forward_postnorm(
        &self,
        mut hidden: Array3<f32>,
        attention_mask: &Array2<f32>,
        config: &dyn TransformerConfig,
    ) -> Result<Array3<f32>> {
        let residual = hidden.clone();
        let (attn_out, new_k, new_v) = self.self_attn.forward_with_cache(
            &hidden,
            None,
            Some(attention_mask),
            config.is_causal(),
            None,
            None,
        )?;
        hidden = residual + attn_out;
        hidden = self.self_attn_layer_norm.forward_3d(&hidden);
        let residual = hidden.clone();
        let ffn_out = self.feedforward.forward(&hidden)?;
        hidden = residual + ffn_out;
        hidden = self.ffn_layer_norm.forward_3d(&hidden);

        Ok(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::MultiHeadAttention;
    use crate::feedforward::{FeedForward, StdFeedForward};
    use crate::normalization::LayerNorm;
    use ndarray::{Array1, Array2, Array3};

    // A mock config for testing purposes.
    struct TestConfig {
        is_prenorm: bool,
    }
    impl TransformerConfig for TestConfig {
        fn is_prenorm(&self) -> bool {
            self.is_prenorm
        }
        fn is_causal(&self) -> bool {
            false
        }
        // Unused methods
        fn hidden_size(&self) -> usize {
            unimplemented!()
        }
        fn num_attention_heads(&self) -> usize {
            unimplemented!()
        }
        fn num_hidden_layers(&self) -> usize {
            unimplemented!()
        }
        fn layer_norm_eps(&self) -> f32 {
            unimplemented!()
        }
    }

    fn create_mock_encoder_layer(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
    ) -> EncoderLayer {
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

        let feedforward = FeedForward::Standard(StdFeedForward::new(
            fc1_weight,
            Array1::zeros(intermediate_size),
            fc2_weight,
            Array1::zeros(hidden_size),
            crate::activations::Activation::GeluNew,
        ));
        let ffn_layer_norm =
            LayerNorm::new(Array1::ones(hidden_size), Array1::zeros(hidden_size), 1e-5);

        EncoderLayer {
            self_attn,
            self_attn_layer_norm,
            feedforward,
            ffn_layer_norm,
        }
    }

    #[test]
    fn test_encoder_layer_forward_postnorm() -> Result<()> {
        let (batch_size, seq_len, hidden_size, intermediate_size, num_heads) = (2, 10, 64, 128, 4);
        let layer = create_mock_encoder_layer(hidden_size, intermediate_size, num_heads);
        let config = TestConfig { is_prenorm: false };

        let hidden_states = Array3::<f32>::ones((batch_size, seq_len, hidden_size));
        let attention_mask = Array2::<f32>::ones((batch_size, seq_len));

        let output = layer.forward(hidden_states.clone(), &attention_mask, &config)?;

        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
        assert!(!output.iter().any(|&x| x.is_nan()), "Output contains NaNs");

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

    #[test]
    fn test_encoder_layer_forward_prenorm() -> Result<()> {
        let (batch_size, seq_len, hidden_size, intermediate_size, num_heads) = (2, 10, 64, 128, 4);
        let layer = create_mock_encoder_layer(hidden_size, intermediate_size, num_heads);
        let config = TestConfig { is_prenorm: true };

        let hidden_states = Array3::<f32>::ones((batch_size, seq_len, hidden_size));
        let attention_mask = Array2::<f32>::ones((batch_size, seq_len));

        let output = layer.forward(hidden_states.clone(), &attention_mask, &config)?;

        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
        assert!(!output.iter().any(|&x| x.is_nan()), "Output contains NaNs");

        // CORRECT ASSERTION for pre-norm: The final operation is an addition, so the mean will NOT be zero.
        assert!(
            output.mean().unwrap().abs() > 0.1,
            "Pre-norm output mean should not be zero"
        );

        Ok(())
    }
}
