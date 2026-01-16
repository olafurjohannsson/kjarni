use std::time::Instant;

use crate::feedforward::FeedForward;
use crate::rope::RoPE;
use crate::{Normalization, cpu::encoder::encoder_self_attention::EncoderSelfAttention};
use anyhow::Result;
use ndarray::{Array2, Array3, Array4, ArrayView3};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};

/// A generic encoder transformer layer supporting:
/// - BERT, RoBERTa, BART (post-norm)
/// - T5 encoder (pre-norm with relative position bias)
/// - Any bidirectional encoder architecture
pub struct EncoderLayer {
    pub self_attn: EncoderSelfAttention,
    pub self_attn_layer_norm: Normalization,
    pub feedforward: FeedForward,
    pub ffn_layer_norm: Normalization,
}

/// Parallel in-place addition: a += b
pub fn add_inplace(a: &mut Array3<f32>, b: &ArrayView3<f32>) {
    let a_slice = a.as_slice_mut().expect("A must be contiguous");
    let b_slice = b.as_slice().expect("B must be contiguous");
    
    // Parallelize the loop using Rayon
    a_slice.par_iter_mut()
           .zip(b_slice.par_iter())
           .for_each(|(x, y)| *x += *y);
}

impl EncoderLayer {
    pub fn new(
        self_attn: EncoderSelfAttention,
        self_attn_layer_norm: Normalization,
        feedforward: FeedForward,
        ffn_layer_norm: Normalization,
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
        rope: Option<&RoPE>,
    ) -> Result<Array3<f32>> {
        if is_prenorm {
            self.forward_prenorm(hidden, attention_mask, position_bias, rope)
        } else {
            self.forward_postnorm(hidden, attention_mask, position_bias, rope)
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
        mut hidden: Array3<f32>,
        attention_mask: &Array2<f32>,
        position_bias: Option<&Array4<f32>>,
        rope: Option<&RoPE>,
    ) -> Result<Array3<f32>> {
        let layer_start = Instant::now();
        let (b, s, d) = hidden.dim();
        // 1. Attention Block
        // let residual = &hidden;
        let normed = self.self_attn_layer_norm.forward(&hidden);
        
        let t_attn_start = Instant::now();
        let attn_out = self
            .self_attn
            .forward(&normed, attention_mask, position_bias, rope)?;
        let t_attn = t_attn_start.elapsed();

        // Note: &residual + &attn_out allocates a NEW array. 
        // This is a memory copy operation!
        // let hidden = residual + &attn_out;
        add_inplace(&mut hidden, &attn_out.view());

        // 2. FFN Block
        let residual = &hidden;
        let normed = self.ffn_layer_norm.forward(&hidden);

        let t_ffn_start = Instant::now();
        let ffn_out = self.feedforward.forward(&normed)?;
        let t_ffn = t_ffn_start.elapsed();

        let hidden = residual + &ffn_out;

        let total = layer_start.elapsed();

        // Only print if it's taking a significant amount of time (e.g. > 1ms)
        // to avoid spamming for small inputs.
        if total.as_millis() > 5 {
             println!(
                "SHAPE: [{}, {}, {}] | Total: {:>3}ms | Attn: {:>3}ms | FFN: {:>3}ms | Overhead: {:>3}ms",
                b, s, d, // <--- PRINT SHAPE
                total.as_millis(),
                t_attn.as_millis(),
                t_ffn.as_millis(),
                (total - t_attn - t_ffn).as_millis()
            );
        }

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
        mut hidden: Array3<f32>,
        attention_mask: &Array2<f32>,
        position_bias: Option<&Array4<f32>>,
        rope: Option<&RoPE>,
    ) -> Result<Array3<f32>> {
        let layer_start = Instant::now();
        let (b, s, d) = hidden.dim();
        // 1. Attention Block
        // let attn_residual = &hidden;
        
        let t_attn_start = Instant::now();
        let attn_out =
            self.self_attn
                .forward(&hidden, attention_mask, position_bias, rope)?;
        let t_attn = t_attn_start.elapsed();

        // Overhead: Addition + LayerNorm
        // let i = &(attn_residual + &attn_out);
        add_inplace(&mut hidden, &attn_out.view());
        let ffn_input = self
            .self_attn_layer_norm
            .forward(&hidden);

        // 2. FFN Block
        let ffn_residual = &ffn_input;
        
        let t_ffn_start = Instant::now();
        let ffn_out = self.feedforward.forward(&ffn_residual)?;
        let t_ffn = t_ffn_start.elapsed();

        // Overhead: Addition + LayerNorm
        let hidden = self.ffn_layer_norm.forward(&(ffn_residual + &ffn_out));
        
        let total = layer_start.elapsed();

        if total.as_millis() > 5 {
             println!(
                "SHAPE: [{}, {}, {}] | Total: {:>3}ms | Attn: {:>3}ms | FFN: {:>3}ms | Overhead: {:>3}ms",
                b, s, d, // <--- PRINT SHAPE
                total.as_millis(),
                t_attn.as_millis(),
                t_ffn.as_millis(),
                (total - t_attn - t_ffn).as_millis()
            );
        }

        Ok(hidden)
    }
}

#[cfg(test)]
mod encoder_layer_tests {
    use super::*;
    use crate::feedforward::StdFeedForward;
    use crate::linear_layer::LinearLayer;
    use crate::normalization::LayerNorm;
    use crate::{activations::Activation, feedforward::LegacyFeedForward};
    use ndarray::{Array1, Array2, Array3, Array4};
    fn create_deterministic_layer(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
    ) -> EncoderLayer {
        let mut count = 1;

        // Python: fixed_vals = torch.arange(count, count + num_elements) * 0.001
        let mut get_linear_weights = |rows: usize, cols: usize| -> Array2<f32> {
            let size = rows * cols;
            let data: Vec<f32> = (count..count + size).map(|x| x as f32 * 0.001).collect();
            count += size;
            // PyTorch Linear is [Out, In], Array2::from_shape_vec fills row-major
            Array2::from_shape_vec((rows, cols), data).unwrap()
        };

        let bias_val = 0.01;

        // 1. Self Attention (Q, K, V, Out)
        // Note: PyTorch named_parameters() order for the mocks provided
        let q_w = get_linear_weights(hidden_size, hidden_size);
        let q_b = Array1::from_elem(hidden_size, bias_val);

        let k_w = get_linear_weights(hidden_size, hidden_size);
        let k_b = Array1::from_elem(hidden_size, bias_val);

        let v_w = get_linear_weights(hidden_size, hidden_size);
        let v_b = Array1::from_elem(hidden_size, bias_val);

        let o_w = get_linear_weights(hidden_size, hidden_size);
        let o_b = Array1::from_elem(hidden_size, bias_val);

        let self_attn = EncoderSelfAttention::new(
            hidden_size,
            num_heads,
            LinearLayer::new_f32(q_w, Some(q_b)),
            LinearLayer::new_f32(k_w, Some(k_b)),
            LinearLayer::new_f32(v_w, Some(v_b)),
            LinearLayer::new_f32(o_w, Some(o_b)),
        );

        // 2. Attn LayerNorm
        // Python: weights (gamma) = 1.0, bias (beta) = 0.01
        let ln1 = Normalization::LayerNorm(LayerNorm::new(
            Array1::ones(hidden_size),
            Array1::from_elem(hidden_size, bias_val),
            1e-5,
        ));

        // 3. FeedForward (FC1, FC2)
        let fc1_w = get_linear_weights(intermediate_size, hidden_size);
        let fc1_b = Array1::from_elem(intermediate_size, bias_val);

        let fc2_w = get_linear_weights(hidden_size, intermediate_size);
        let fc2_b = Array1::from_elem(hidden_size, bias_val);

        let feedforward = FeedForward::Standard(StdFeedForward::new(
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
            Activation::Gelu, // Using Standard Gelu (not New/Tanh) to match Python default
        ));

        // 4. FFN LayerNorm
        let ln2 = Normalization::LayerNorm(LayerNorm::new(
            Array1::ones(hidden_size),
            Array1::from_elem(hidden_size, bias_val),
            1e-5,
        ));

        EncoderLayer::new(self_attn, ln1, feedforward, ln2)
    }

    #[test]
    fn test_golden_prenorm() -> Result<()> {
        let (batch, seq, hidden, intermediate, heads) = (2, 3, 4, 8, 2);

        // 1. Initialize Layer with exact deterministic weights
        let layer = create_deterministic_layer(hidden, intermediate, heads);

        // 2. Prepare Inputs
        let input_hidden_data = vec![
            0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
            0.800000, 0.900000, 1.000000, 1.100000, 1.200000, 1.300000, 1.400000, 1.500000,
            1.600000, 1.700000, 1.800000, 1.900000, 2.000000, 2.100000, 2.200000, 2.300000,
        ];
        let input = Array3::from_shape_vec((batch, seq, hidden), input_hidden_data)?;

        let input_mask_data = vec![1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.000000];
        let mask = Array2::from_shape_vec((batch, seq), input_mask_data)?;

        let pos_bias_data = vec![
            0.000000, 0.010000, 0.020000, 0.030000, 0.040000, 0.050000, 0.060000, 0.070000,
            0.080000, 0.090000, 0.100000, 0.110000, 0.120000, 0.130000, 0.140000, 0.150000,
            0.160000, 0.170000,
        ];
        // Note: Python broadcasts [1, heads, seq, seq] to batch automatically.
        // Rust implementation likely handles the broadcast inside forward or expects matching dims.
        // Assuming implementation handles broadcasting of dimension 0.
        let pos_bias = Array4::from_shape_vec((1, heads, seq, seq), pos_bias_data)?;

        // 3. Run Forward (Pre-Norm = true)
        let output = layer.forward(input, &mask, Some(&pos_bias), true, None)?;

        // 4. Validate against Golden Values
        let golden_output_prenorm_data = vec![
            0.030466, 0.131298, 0.232129, 0.332961, 0.430466, 0.531298, 0.632130, 0.732961,
            0.830467, 0.931298, 1.032130, 1.132961, 1.230467, 1.331298, 1.432130, 1.532961,
            1.630466, 1.731298, 1.832129, 1.932961, 2.030467, 2.131298, 2.232130, 2.332961,
        ];
        let golden = Array3::from_shape_vec((batch, seq, hidden), golden_output_prenorm_data)?;

        let diff = &output - &golden;
        let max_diff = diff.mapv(|x| x.abs()).fold(0.0f32, |a, b| f32::max(a, *b));

        println!("Pre-Norm Max Diff: {:.6}", max_diff);

        // Allow small float error (accumulated via multiple matmuls)
        assert!(
            max_diff < 1e-4,
            "Pre-norm golden value mismatch. Max diff: {}",
            max_diff
        );

        Ok(())
    }

    #[test]
    fn test_golden_postnorm() -> Result<()> {
        let (batch, seq, hidden, intermediate, heads) = (2, 3, 4, 8, 2);

        let layer = create_deterministic_layer(hidden, intermediate, heads);

        let input_hidden_data = vec![
            0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
            0.800000, 0.900000, 1.000000, 1.100000, 1.200000, 1.300000, 1.400000, 1.500000,
            1.600000, 1.700000, 1.800000, 1.900000, 2.000000, 2.100000, 2.200000, 2.300000,
        ];
        let input = Array3::from_shape_vec((batch, seq, hidden), input_hidden_data)?;

        let input_mask_data = vec![1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.000000];
        let mask = Array2::from_shape_vec((batch, seq), input_mask_data)?;

        // Even in post-norm (usually BERT), the python test passed the position bias
        // to generate the golden values, so we must pass it here too.
        let pos_bias_data = vec![
            0.000000, 0.010000, 0.020000, 0.030000, 0.040000, 0.050000, 0.060000, 0.070000,
            0.080000, 0.090000, 0.100000, 0.110000, 0.120000, 0.130000, 0.140000, 0.150000,
            0.160000, 0.170000,
        ];
        let pos_bias = Array4::from_shape_vec((1, heads, seq, seq), pos_bias_data)?;

        // 3. Run Forward (Pre-Norm = false)
        let output = layer.forward(input, &mask, Some(&pos_bias), false, None)?;

        // 4. Validate against Golden Values
        let golden_output_postnorm_data = vec![
            -1.331634, -0.437211, 0.457211, 1.351634, -1.331634, -0.437212, 0.457212, 1.351634,
            -1.331634, -0.437211, 0.457212, 1.351634, -1.331634, -0.437211, 0.457211, 1.351634,
            -1.331634, -0.437211, 0.457211, 1.351634, -1.331634, -0.437211, 0.457211, 1.351634,
        ];
        let golden = Array3::from_shape_vec((batch, seq, hidden), golden_output_postnorm_data)?;

        let diff = &output - &golden;
        let max_diff = diff.mapv(|x| x.abs()).fold(0.0f32, |a, b| f32::max(a, *b));

        println!("Post-Norm Max Diff: {:.6}", max_diff);

        assert!(
            max_diff < 1e-4,
            "Post-norm golden value mismatch. Max diff: {}",
            max_diff
        );

        Ok(())
    }
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

        let self_attn_layer_norm = Normalization::LayerNorm(LayerNorm::new(
            Array1::ones(hidden_size),
            Array1::zeros(hidden_size),
            1e-5,
        ));

        // Use StdFeedForward with [Out, In] weights
        let feedforward = FeedForward::Standard(StdFeedForward::new(
            fc1_weight,
            Array1::zeros(intermediate_size),
            fc2_weight,
            Array1::zeros(hidden_size),
            Activation::GeluNew,
        ));

        let ffn_layer_norm = Normalization::LayerNorm(LayerNorm::new(
            Array1::ones(hidden_size),
            Array1::zeros(hidden_size),
            1e-5,
        ));

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

        let output = layer.forward(input, &mask, None, false, None)?;

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

        let output = layer.forward(input, &mask, None, true, None)?;

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
            (h as f32) + (b * 100 + s * 10) as f32 * 0.01 // h varies from 0 to 63
        });
        let mask = Array2::<f32>::ones((batch, seq));

        let output = layer.forward(input, &mask, None, false, None)?;

        // Post-norm: LayerNorm normalizes EACH POSITION along hidden dim
        for b in 0..batch {
            for s in 0..seq {
                let position = output.slice(ndarray::s![b, s, ..]);
                let mean = position.mean().unwrap();
                let std = position.std(0.0);

                assert!(
                    mean.abs() < 1e-4,
                    "Position [{},{}] mean should be ~0, got {}",
                    b,
                    s,
                    mean
                );
                assert!(
                    (std - 1.0).abs() < 0.1,
                    "Position [{},{}] std should be ~1, got {}",
                    b,
                    s,
                    std
                );
            }
        }

        Ok(())
    }
    #[test]
    fn test_encoder_self_attention_matches_multihead_attention() -> Result<()> {
        use crate::attention::MultiHeadAttention;
        use crate::cpu::encoder::encoder_self_attention::EncoderSelfAttention;
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

        let esa_output = esa.forward(&input, &mask, None, None)?;

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

        let output = layer.forward(input, &mask, None, true, None)?;

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

        let output = layer.forward(input, &mask, None, false, None)?;

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

        let output = layer.forward(input, &mask, Some(&position_bias), true, None)?;

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

        let output = layer.forward(input, &mask, None, false, None)?;

        assert_eq!(output.shape(), &[batch, seq, hidden]);

        Ok(())
    }

    #[test]
    fn test_large_batch() -> Result<()> {
        let (batch, seq, hidden, intermediate, heads) = (32, 16, 64, 128, 4);
        let layer = create_test_layer(hidden, intermediate, heads);

        let input = Array3::<f32>::ones((batch, seq, hidden));
        let mask = Array2::<f32>::ones((batch, seq));

        let output = layer.forward(input, &mask, None, false, None)?;

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
        let self_attn_layer_norm = Normalization::LayerNorm(LayerNorm::new(
            Array1::ones(hidden),
            Array1::zeros(hidden),
            1e-5,
        ));

        let feedforward = FeedForward::Legacy(LegacyFeedForward::new(
            zero_fc1,
            Array1::zeros(64),
            zero_fc2,
            Array1::zeros(hidden),
            Activation::GeluNew,
        ));
        let ffn_layer_norm = Normalization::LayerNorm(LayerNorm::new(
            Array1::ones(hidden),
            Array1::zeros(hidden),
            1e-5,
        ));

        let layer = EncoderLayer::new(self_attn, self_attn_layer_norm, feedforward, ffn_layer_norm);

        // With zero attention/FFN weights, pre-norm output should equal input
        // (residual passes through, sublayers contribute nothing)
        let input = Array3::from_shape_fn((1, 4, hidden), |(_, _, h)| h as f32);
        let mask = Array2::<f32>::ones((1, 4));

        let output = layer.forward(input.clone(), &mask, None, true, None)?;

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
