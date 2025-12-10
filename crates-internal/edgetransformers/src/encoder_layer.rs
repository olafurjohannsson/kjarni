use crate::encoder::encoder_self_attention::EncoderSelfAttention;
use crate::feedforward::FeedForward;
use crate::normalization::LayerNorm;
use anyhow::Result;
use ndarray::{Array2, Array3, Array4};

/// A generic encoder transformer layer supporting:
/// - BERT, RoBERTa, BART (post-norm)
/// - T5 encoder (pre-norm with relative position bias)
/// - Any bidirectional encoder architecture
pub struct EncoderLayer {
    pub self_attn: EncoderSelfAttention,
    pub self_attn_layer_norm: LayerNorm,
    pub feedforward: FeedForward,
    pub ffn_layer_norm: LayerNorm,
}

impl EncoderLayer {
    pub fn new(
        self_attn: EncoderSelfAttention,
        self_attn_layer_norm: LayerNorm,
        feedforward: FeedForward,
        ffn_layer_norm: LayerNorm,
    ) -> Self {
        Self {
            self_attn,
            self_attn_layer_norm,
            feedforward,
            ffn_layer_norm,
        }
    }

    /// Forward pass with configurable norm order
    ///
    /// # Arguments
    /// * `hidden` - Input tensor [batch, seq, hidden]
    /// * `attention_mask` - Padding mask [batch, seq], 0 = masked
    /// * `position_bias` - Optional relative position bias [1, heads, seq, seq] (T5/ALiBi)
    /// * `is_prenorm` - If true, use pre-norm (T5/GPT), else post-norm (BERT/BART)
    pub fn forward(
        &self,
        hidden: Array3<f32>,
        attention_mask: &Array2<f32>,
        position_bias: Option<&Array4<f32>>,
        is_prenorm: bool,
    ) -> Result<Array3<f32>> {
        if is_prenorm {
            self.forward_prenorm(hidden, attention_mask, position_bias)
        } else {
            self.forward_postnorm(hidden, attention_mask, position_bias)
        }
    }

    /// Pre-norm: LN → Sublayer → Residual (T5, GPT-2, LLaMA style)
    ///
    /// ```text
    /// x ──┬── LN ──► Attention ──┬──► + ──┬── LN ──► FFN ──┬──► + ──► out
    ///     └─────────────────────►┘        └────────────────►┘
    /// ```
    fn forward_prenorm(
        &self,
        hidden: Array3<f32>,
        attention_mask: &Array2<f32>,
        position_bias: Option<&Array4<f32>>,
    ) -> Result<Array3<f32>> {
        // Attention block
        let residual = &hidden;
        let normed = self.self_attn_layer_norm.forward_3d(&hidden);
        let attn_out = self
            .self_attn
            .forward(&normed, attention_mask, position_bias)?;
        let hidden = residual + &attn_out;

        // FFN block
        let residual = &hidden;
        let normed = self.ffn_layer_norm.forward_3d(&hidden);
        let ffn_out = self.feedforward.forward(&normed)?;
        let hidden = residual + &ffn_out;

        Ok(hidden)
    }

    /// Post-norm: Sublayer → Residual → LN (BERT, BART, RoBERTa style)
    ///
    /// ```text
    /// x ──┬──► Attention ──┬──► + ──► LN ──┬──► FFN ──┬──► + ──► LN ──► out
    ///     └────────────────►┘              └──────────►┘
    /// ```
    fn forward_postnorm(
        &self,
        hidden: Array3<f32>,
        attention_mask: &Array2<f32>,
        position_bias: Option<&Array4<f32>>,
    ) -> Result<Array3<f32>> {
        // Attention block
        let residual = &hidden;
        let attn_out = self
            .self_attn
            .forward(&hidden, attention_mask, position_bias)?;
        let hidden = self
            .self_attn_layer_norm
            .forward_3d(&(residual + &attn_out));

        // FFN block
        let residual = &hidden;
        let ffn_out = self.feedforward.forward(&hidden)?;
        let hidden = self.ffn_layer_norm.forward_3d(&(residual + &ffn_out));

        Ok(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feedforward::StdFeedForward;
    use crate::linear_layer::LinearLayer;
    use crate::{activations::Activation, feedforward::LegacyFeedForward};
    use ndarray::{Array1, Array2, Array3, Array4};

    fn create_test_layer(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
    ) -> EncoderLayer {
        // Attention weights [Out, In] - square so symmetric works
        let proj_weight =
            Array2::from_shape_fn(
                (hidden_size, hidden_size),
                |(i, j)| {
                    if i == j { 1.0 } else { 0.01 }
                },
            );

        // FFN weights - LinearLayer expects [Out, In]
        let fc1_weight = Array2::from_shape_fn((intermediate_size, hidden_size), |(i, j)| {
            if i == j { 1.0 } else { 0.01 }
        }); // [128, 64] = [Out, In]

        let fc2_weight = Array2::from_shape_fn((hidden_size, intermediate_size), |(i, j)| {
            if i == j { 1.0 } else { 0.01 }
        }); // [64, 128] = [Out, In]

        let q = LinearLayer::new_f32(proj_weight.clone(), Some(Array1::zeros(hidden_size)));
        let k = LinearLayer::new_f32(proj_weight.clone(), Some(Array1::zeros(hidden_size)));
        let v = LinearLayer::new_f32(proj_weight.clone(), Some(Array1::zeros(hidden_size)));
        let o = LinearLayer::new_f32(proj_weight.clone(), Some(Array1::zeros(hidden_size)));

        let self_attn = EncoderSelfAttention::new(hidden_size, num_heads, q, k, v, o);

        let self_attn_layer_norm =
            LayerNorm::new(Array1::ones(hidden_size), Array1::zeros(hidden_size), 1e-5);

        // Use StdFeedForward with [Out, In] weights
        let feedforward = FeedForward::Standard(StdFeedForward::new(
            fc1_weight,
            Array1::zeros(intermediate_size),
            fc2_weight,
            Array1::zeros(hidden_size),
            Activation::GeluNew,
        ));

        let ffn_layer_norm =
            LayerNorm::new(Array1::ones(hidden_size), Array1::zeros(hidden_size), 1e-5);

        EncoderLayer::new(self_attn, self_attn_layer_norm, feedforward, ffn_layer_norm)
    }
    #[test]
    fn test_std_feedforward_matches_legacy_feedforward() -> Result<()> {
        use crate::activations::Activation;
        use crate::feedforward::{LegacyFeedForward, StdFeedForward};
        use ndarray::{Array1, Array2, Array3};

        let hidden_size = 64;
        let intermediate_size = 128;
        let batch_size = 2;
        let seq_len = 10;

        // Create random-ish weights (deterministic)
        let fc1_data: Vec<f32> = (0..hidden_size * intermediate_size)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.01)
            .collect();
        let fc2_data: Vec<f32> = (0..intermediate_size * hidden_size)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.01)
            .collect();

        // LegacyFeedForward expects [In, Out]
        let fc1_in_out =
            Array2::from_shape_vec((hidden_size, intermediate_size), fc1_data.clone())?;
        let fc2_in_out =
            Array2::from_shape_vec((intermediate_size, hidden_size), fc2_data.clone())?;

        // StdFeedForward (LinearLayer) expects [Out, In]
        let fc1_out_in = fc1_in_out.t().as_standard_layout().to_owned(); // [intermediate, hidden]
        let fc2_out_in = fc2_in_out.t().as_standard_layout().to_owned(); // [hidden, intermediate]

        let bias1 = Array1::<f32>::zeros(intermediate_size);
        let bias2 = Array1::<f32>::zeros(hidden_size);

        // === LegacyFeedForward ([In, Out] weights) ===
        let legacy = LegacyFeedForward::new(
            fc1_in_out,
            bias1.clone(),
            fc2_in_out,
            bias2.clone(),
            Activation::GeluNew,
        );

        // === StdFeedForward ([Out, In] weights) ===
        let std_ffn =
            StdFeedForward::new(fc1_out_in, bias1, fc2_out_in, bias2, Activation::GeluNew);

        // Create test input
        let input: Array3<f32> =
            Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(b, s, h)| {
                ((b * 100 + s * 10 + h) % 29) as f32 * 0.1 - 1.0
            });

        // Run both
        let legacy_output = legacy.forward(&input)?;
        let std_output = std_ffn.forward(&input)?;

        // Compare
        let diff = (&legacy_output - &std_output).mapv(|x| x.abs());
        let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);
        let mean_diff = diff.mean().unwrap();

        println!("Max diff: {}, Mean diff: {}", max_diff, mean_diff);

        assert!(
            max_diff < 1e-5,
            "Outputs differ! Max diff: {}, Mean diff: {}",
            max_diff,
            mean_diff
        );

        println!("✓ StdFeedForward matches LegacyFeedForward");
        Ok(())
    }
    #[test]
    fn test_forward_postnorm_shape() -> Result<()> {
        let (batch, seq, hidden, intermediate, heads) = (2, 10, 64, 128, 4);
        let layer = create_test_layer(hidden, intermediate, heads);

        let input = Array3::<f32>::ones((batch, seq, hidden));
        let mask = Array2::<f32>::ones((batch, seq));

        let output = layer.forward(input, &mask, None, false)?;

        assert_eq!(output.shape(), &[batch, seq, hidden]);
        assert!(!output.iter().any(|x| x.is_nan()), "Output contains NaNs");

        Ok(())
    }

    #[test]
    fn test_forward_prenorm_shape() -> Result<()> {
        let (batch, seq, hidden, intermediate, heads) = (2, 10, 64, 128, 4);
        let layer = create_test_layer(hidden, intermediate, heads);

        let input = Array3::<f32>::ones((batch, seq, hidden));
        let mask = Array2::<f32>::ones((batch, seq));

        let output = layer.forward(input, &mask, None, true)?;

        assert_eq!(output.shape(), &[batch, seq, hidden]);
        assert!(!output.iter().any(|x| x.is_nan()), "Output contains NaNs");

        Ok(())
    }

#[test]
fn test_postnorm_output_is_normalized() -> Result<()> {
    let (batch, seq, hidden, intermediate, heads) = (2, 10, 64, 128, 4);
    let layer = create_test_layer(hidden, intermediate, heads);

    // Input with actual variance along hidden dimension
    let input = Array3::from_shape_fn((batch, seq, hidden), |(b, s, h)| {
        (h as f32) + (b * 100 + s * 10) as f32 * 0.01  // h varies from 0 to 63
    });
    let mask = Array2::<f32>::ones((batch, seq));

    let output = layer.forward(input, &mask, None, false)?;

    // Post-norm: LayerNorm normalizes EACH POSITION along hidden dim
    for b in 0..batch {
        for s in 0..seq {
            let position = output.slice(ndarray::s![b, s, ..]);
            let mean = position.mean().unwrap();
            let std = position.std(0.0);

            assert!(
                mean.abs() < 1e-4,
                "Position [{},{}] mean should be ~0, got {}",
                b, s, mean
            );
            assert!(
                (std - 1.0).abs() < 0.1,
                "Position [{},{}] std should be ~1, got {}",
                b, s, std
            );
        }
    }

    Ok(())
}
    #[test]
    fn test_encoder_self_attention_matches_multihead_attention() -> Result<()> {
        use crate::attention::MultiHeadAttention;
        use crate::encoder::encoder_self_attention::EncoderSelfAttention;
        use crate::linear_layer::LinearLayer;
        use ndarray::{Array1, Array2, Array3};

        let hidden_size = 64;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 10;

        // Create random-ish weights (deterministic for reproducibility)
        let weight_data: Vec<f32> = (0..hidden_size * hidden_size)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();

        // Base weight in [In, Out] layout (for MultiHeadAttention)
        let weight_in_out =
            Array2::from_shape_vec((hidden_size, hidden_size), weight_data.clone())?;

        // Transposed weight in [Out, In] layout (for EncoderSelfAttention/LinearLayer)
        let weight_out_in = weight_in_out.t().as_standard_layout().to_owned();

        let bias = Array1::<f32>::zeros(hidden_size);

        // === MultiHeadAttention (old style, [In, Out] weights) ===
        let mha = MultiHeadAttention::new(
            hidden_size,
            num_heads,
            weight_in_out.clone(), // q
            bias.clone(),
            weight_in_out.clone(), // k
            bias.clone(),
            weight_in_out.clone(), // v
            bias.clone(),
            weight_in_out.clone(), // output
            bias.clone(),
            None, // num_kv_heads
        );

        // === EncoderSelfAttention (new style, [Out, In] weights via LinearLayer) ===
        let esa = EncoderSelfAttention::new(
            hidden_size,
            num_heads,
            LinearLayer::new_f32(weight_out_in.clone(), Some(bias.clone())), // q
            LinearLayer::new_f32(weight_out_in.clone(), Some(bias.clone())), // k
            LinearLayer::new_f32(weight_out_in.clone(), Some(bias.clone())), // v
            LinearLayer::new_f32(weight_out_in.clone(), Some(bias.clone())), // output
        );

        // Create test input
        let input: Array3<f32> =
            Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(b, s, h)| {
                ((b * 100 + s * 10 + h) % 23) as f32 * 0.1 - 1.0
            });
        let mask = Array2::<f32>::ones((batch_size, seq_len));

        // Run both
        let mha_output = {
            // MultiHeadAttention needs manual Q projection for attend()
            let q_proj =
                crate::utils::linear_algebra::matmul_3d_2d(&input, &mha.q_weight) + &mha.q_bias;
            let (k, v) = mha.project_kv(&input);
            let context = mha.attend(&q_proj, &k, &v, Some(&mask), false, 0)?;
            crate::utils::linear_algebra::matmul_3d_2d(&context, &mha.output_weight)
                + &mha.output_bias
        };

        let esa_output = esa.forward(&input, &mask, None)?;

        // Compare
        let diff = (&mha_output - &esa_output).mapv(|x| x.abs());
        let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);
        let mean_diff = diff.mean().unwrap();

        println!("Max diff: {}, Mean diff: {}", max_diff, mean_diff);

        assert!(
            max_diff < 1e-5,
            "Outputs differ! Max diff: {}, Mean diff: {}",
            max_diff,
            mean_diff
        );

        println!("✓ EncoderSelfAttention matches MultiHeadAttention");
        Ok(())
    }
    #[test]
    fn test_prenorm_output_not_normalized() -> Result<()> {
        let (batch, seq, hidden, intermediate, heads) = (2, 10, 64, 128, 4);
        let layer = create_test_layer(hidden, intermediate, heads);

        let input = Array3::<f32>::ones((batch, seq, hidden)) * 5.0;
        let mask = Array2::<f32>::ones((batch, seq));

        let output = layer.forward(input, &mask, None, true)?;

        // Pre-norm ends with residual addition, NOT LayerNorm
        // So output won't be normalized to mean=0, std=1
        let mean = output.mean().unwrap();

        assert!(
            mean.abs() > 0.5,
            "Pre-norm mean should NOT be near zero (residual preserved), got {}",
            mean
        );

        Ok(())
    }

    #[test]
    fn test_attention_mask_applied() -> Result<()> {
        let (batch, seq, hidden, intermediate, heads) = (1, 4, 32, 64, 2);
        let layer = create_test_layer(hidden, intermediate, heads);

        let input = Array3::<f32>::ones((batch, seq, hidden));

        // Mask out last two positions
        let mask = Array2::from_shape_vec((1, 4), vec![1.0, 1.0, 0.0, 0.0])?;

        let output = layer.forward(input, &mask, None, false)?;

        // Output should still be valid (no NaNs/Infs)
        assert!(output.iter().all(|x| x.is_finite()));

        Ok(())
    }

    #[test]
    fn test_with_position_bias() -> Result<()> {
        let (batch, seq, hidden, intermediate, heads) = (2, 8, 64, 128, 4);
        let layer = create_test_layer(hidden, intermediate, heads);

        let input = Array3::<f32>::ones((batch, seq, hidden));
        let mask = Array2::<f32>::ones((batch, seq));

        // T5-style position bias: [1, heads, seq_q, seq_k]
        let position_bias = Array4::<f32>::zeros((1, heads, seq, seq));

        let output = layer.forward(input, &mask, Some(&position_bias), true)?;

        assert_eq!(output.shape(), &[batch, seq, hidden]);
        assert!(!output.iter().any(|x| x.is_nan()));

        Ok(())
    }

    #[test]
    fn test_single_token() -> Result<()> {
        let (batch, seq, hidden, intermediate, heads) = (1, 1, 64, 128, 4);
        let layer = create_test_layer(hidden, intermediate, heads);

        let input = Array3::<f32>::ones((batch, seq, hidden));
        let mask = Array2::<f32>::ones((batch, seq));

        let output = layer.forward(input, &mask, None, false)?;

        assert_eq!(output.shape(), &[batch, seq, hidden]);

        Ok(())
    }

    #[test]
    fn test_large_batch() -> Result<()> {
        let (batch, seq, hidden, intermediate, heads) = (32, 16, 64, 128, 4);
        let layer = create_test_layer(hidden, intermediate, heads);

        let input = Array3::<f32>::ones((batch, seq, hidden));
        let mask = Array2::<f32>::ones((batch, seq));

        let output = layer.forward(input, &mask, None, false)?;

        assert_eq!(output.shape(), &[batch, seq, hidden]);

        Ok(())
    }

    #[test]
    fn test_residual_connection() -> Result<()> {
        // Create layer with zero weights to isolate residual behavior
        let hidden = 32;
        let heads = 2;

        let zero_proj = Array2::<f32>::zeros((hidden, hidden));
        let zero_fc1 = Array2::<f32>::zeros((hidden, 64));
        let zero_fc2 = Array2::<f32>::zeros((64, hidden));

        let q = LinearLayer::new_f32(zero_proj.clone(), Some(Array1::zeros(hidden)));
        let k = LinearLayer::new_f32(zero_proj.clone(), Some(Array1::zeros(hidden)));
        let v = LinearLayer::new_f32(zero_proj.clone(), Some(Array1::zeros(hidden)));
        let o = LinearLayer::new_f32(zero_proj.clone(), Some(Array1::zeros(hidden)));

        let self_attn = EncoderSelfAttention::new(hidden, heads, q, k, v, o);
        let self_attn_layer_norm =
            LayerNorm::new(Array1::ones(hidden), Array1::zeros(hidden), 1e-5);

        let feedforward = FeedForward::Legacy(LegacyFeedForward::new(
            zero_fc1,
            Array1::zeros(64),
            zero_fc2,
            Array1::zeros(hidden),
            Activation::GeluNew,
        ));
        let ffn_layer_norm = LayerNorm::new(Array1::ones(hidden), Array1::zeros(hidden), 1e-5);

        let layer = EncoderLayer::new(self_attn, self_attn_layer_norm, feedforward, ffn_layer_norm);

        // With zero attention/FFN weights, pre-norm output should equal input
        // (residual passes through, sublayers contribute nothing)
        let input = Array3::from_shape_fn((1, 4, hidden), |(_, _, h)| h as f32);
        let mask = Array2::<f32>::ones((1, 4));

        let output = layer.forward(input.clone(), &mask, None, true)?;

        // Pre-norm: input passes through residual unchanged when sublayers are zero
        for (inp, out) in input.iter().zip(output.iter()) {
            assert!(
                (inp - out).abs() < 1e-5,
                "Residual should preserve input when sublayers are zero"
            );
        }

        Ok(())
    }
}
