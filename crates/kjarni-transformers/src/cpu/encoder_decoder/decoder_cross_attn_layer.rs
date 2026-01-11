use crate::Normalization;
use crate::cpu::encoder_decoder::DecoderCrossAttention;
use crate::encoder_decoder::DecoderSelfAttention;
pub use crate::feedforward::FeedForward;
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
        let (attn_out, new_k, new_v) =
            self.self_attn
                .forward(hidden_states, self_mask, past_kv, None)?;
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
        let (attn_out, new_k, new_v) =
            self.self_attn
                .forward(&normed, self_mask, past_kv, position_bias)?;
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
            self.cross_attn
                .forward(hidden_states, k_cache, v_cache, cross_mask)
        } else {
            // Slow path: compute K/V on the fly
            let (k, v) = self
                .cross_attn
                .precompute_encoder_kv(encoder_hidden_states)?;
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
            let (attn_out, new_k, new_v) =
                self.self_attn.forward(hidden_states, mask, past_kv, None)?;
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
mod cross_decoder_layer_tests {
    use super::*;
    use crate::{
        activations::Activation,
        feedforward::{FeedForward, LegacyFeedForward, StdFeedForward},
        linear_layer::LinearLayer,
        normalization::{LayerNorm, Normalization},
    };
    use anyhow::Result;
    use ndarray::{Array1, Array2, Array3, Array4, s};

    /// Helper to create LinearLayer from weight and bias vecs
    fn load_linear(
        out_features: usize,
        in_features: usize,
        weight_data: Vec<f32>,
        bias_data: Vec<f32>,
    ) -> LinearLayer {
        let weight = Array2::from_shape_vec((out_features, in_features), weight_data).unwrap();
        let bias = Array1::from_vec(bias_data);
        LinearLayer::new_f32(weight, Some(bias))
    }

    /// Create deterministic cross decoder layer for testing
    fn create_test_cross_decoder_layer(pre_norm: bool) -> CrossDecoderLayer {
        let hidden_size = 4;
        let num_heads = 2;
        let intermediate_size = 8;

        // === Self Attention Weights ===
        let layer_sa_q_weight: Vec<f32> = vec![
            0.010000, 0.020000, 0.030000, 0.040000, 0.050000, 0.060000, 0.070000, 0.080000,
            0.090000, 0.100000, 0.110000, 0.120000, 0.130000, 0.140000, 0.150000, 0.160000,
        ];
        let layer_sa_q_bias: Vec<f32> = vec![0.010000, 0.010000, 0.010000, 0.010000];

        let layer_sa_k_weight: Vec<f32> = vec![
            0.170000, 0.180000, 0.190000, 0.200000, 0.210000, 0.220000, 0.230000, 0.240000,
            0.250000, 0.260000, 0.270000, 0.280000, 0.290000, 0.300000, 0.310000, 0.320000,
        ];
        let layer_sa_k_bias: Vec<f32> = vec![0.010000, 0.010000, 0.010000, 0.010000];

        let layer_sa_v_weight: Vec<f32> = vec![
            0.330000, 0.340000, 0.350000, 0.360000, 0.370000, 0.380000, 0.390000, 0.400000,
            0.410000, 0.420000, 0.430000, 0.440000, 0.450000, 0.460000, 0.470000, 0.480000,
        ];
        let layer_sa_v_bias: Vec<f32> = vec![0.010000, 0.010000, 0.010000, 0.010000];

        let layer_sa_o_weight: Vec<f32> = vec![
            0.490000, 0.500000, 0.510000, 0.520000, 0.530000, 0.540000, 0.550000, 0.560000,
            0.570000, 0.580000, 0.590000, 0.600000, 0.610000, 0.620000, 0.630000, 0.640000,
        ];
        let layer_sa_o_bias: Vec<f32> = vec![0.010000, 0.010000, 0.010000, 0.010000];

        let sa_q = load_linear(hidden_size, hidden_size, layer_sa_q_weight, layer_sa_q_bias);
        let sa_k = load_linear(hidden_size, hidden_size, layer_sa_k_weight, layer_sa_k_bias);
        let sa_v = load_linear(hidden_size, hidden_size, layer_sa_v_weight, layer_sa_v_bias);
        let sa_o = load_linear(hidden_size, hidden_size, layer_sa_o_weight, layer_sa_o_bias);

        let self_attn = DecoderSelfAttention::new(hidden_size, num_heads, sa_q, sa_k, sa_v, sa_o);

        // Self-attention LayerNorm
        let layer_sa_ln_weight: Vec<f32> = vec![1.000000, 1.000000, 1.000000, 1.000000];
        let layer_sa_ln_bias: Vec<f32> = vec![0.010000, 0.010000, 0.010000, 0.010000];
        let self_attn_layer_norm = Normalization::LayerNorm(LayerNorm::new(
            Array1::from_vec(layer_sa_ln_weight),
            Array1::from_vec(layer_sa_ln_bias),
            1e-5,
        ));

        // === Cross Attention Weights ===
        let layer_ca_q_weight: Vec<f32> = vec![
            0.650000, 0.660000, 0.670000, 0.680000, 0.690000, 0.700000, 0.710000, 0.720000,
            0.730000, 0.740000, 0.750000, 0.760000, 0.770000, 0.780000, 0.790000, 0.800000,
        ];
        let layer_ca_q_bias: Vec<f32> = vec![0.010000, 0.010000, 0.010000, 0.010000];

        let layer_ca_k_weight: Vec<f32> = vec![
            0.810000, 0.820000, 0.830000, 0.840000, 0.850000, 0.860000, 0.870000, 0.880000,
            0.890000, 0.900000, 0.910000, 0.920000, 0.930000, 0.940000, 0.950000, 0.960000,
        ];
        let layer_ca_k_bias: Vec<f32> = vec![0.010000, 0.010000, 0.010000, 0.010000];

        let layer_ca_v_weight: Vec<f32> = vec![
            0.970000, 0.980000, 0.990000, 1.000000, 1.010000, 1.020000, 1.030000, 1.040000,
            1.050000, 1.060000, 1.070000, 1.080000, 1.090000, 1.100000, 1.110000, 1.120000,
        ];
        let layer_ca_v_bias: Vec<f32> = vec![0.010000, 0.010000, 0.010000, 0.010000];

        let layer_ca_o_weight: Vec<f32> = vec![
            1.130000, 1.140000, 1.150000, 1.160000, 1.170000, 1.180000, 1.190000, 1.200000,
            1.210000, 1.220000, 1.230000, 1.240000, 1.250000, 1.260000, 1.270000, 1.280000,
        ];
        let layer_ca_o_bias: Vec<f32> = vec![0.010000, 0.010000, 0.010000, 0.010000];

        let ca_q = load_linear(hidden_size, hidden_size, layer_ca_q_weight, layer_ca_q_bias);
        let ca_k = load_linear(hidden_size, hidden_size, layer_ca_k_weight, layer_ca_k_bias);
        let ca_v = load_linear(hidden_size, hidden_size, layer_ca_v_weight, layer_ca_v_bias);
        let ca_o = load_linear(hidden_size, hidden_size, layer_ca_o_weight, layer_ca_o_bias);

        let cross_attn = DecoderCrossAttention::new(hidden_size, num_heads, ca_q, ca_k, ca_v, ca_o);

        // Cross-attention LayerNorm
        let layer_ca_ln_weight: Vec<f32> = vec![1.000000, 1.000000, 1.000000, 1.000000];
        let layer_ca_ln_bias: Vec<f32> = vec![0.010000, 0.010000, 0.010000, 0.010000];
        let cross_attn_layer_norm = Normalization::LayerNorm(LayerNorm::new(
            Array1::from_vec(layer_ca_ln_weight),
            Array1::from_vec(layer_ca_ln_bias),
            1e-5,
        ));

        // === FFN Weights ===
        let layer_ffn_fc1_weight: Vec<f32> = vec![
            1.290000, 1.300000, 1.310000, 1.320000, 1.330000, 1.340000, 1.350000, 1.360000,
            1.370000, 1.380000, 1.390000, 1.400000, 1.410000, 1.420000, 1.430000, 1.440000,
            1.450000, 1.460000, 1.470000, 1.480000, 1.490000, 1.500000, 1.510000, 1.520000,
            1.530000, 1.540000, 1.550000, 1.560000, 1.570000, 1.580000, 1.590000, 1.600000,
        ];
        let layer_ffn_fc1_bias: Vec<f32> = vec![
            0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000, 0.010000,
        ];

        let layer_ffn_fc2_weight: Vec<f32> = vec![
            1.610000, 1.620000, 1.630000, 1.640000, 1.650000, 1.660000, 1.670000, 1.680000,
            1.690000, 1.700000, 1.710000, 1.720000, 1.730000, 1.740000, 1.750000, 1.760000,
            1.770000, 1.780000, 1.790000, 1.800000, 1.810000, 1.820000, 1.830000, 1.840000,
            1.850000, 1.860000, 1.870000, 1.880000, 1.890000, 1.900000, 1.910000, 1.920000,
        ];
        let layer_ffn_fc2_bias: Vec<f32> = vec![0.010000, 0.010000, 0.010000, 0.010000];

        let feedforward = FeedForward::Standard(StdFeedForward::new(
            Array2::from_shape_vec((intermediate_size, hidden_size), layer_ffn_fc1_weight).unwrap(),
            Array1::from_vec(layer_ffn_fc1_bias),
            Array2::from_shape_vec((hidden_size, intermediate_size), layer_ffn_fc2_weight).unwrap(),
            Array1::from_vec(layer_ffn_fc2_bias),
            Activation::Gelu,
        ));

        // FFN LayerNorm
        let layer_ffn_ln_weight: Vec<f32> = vec![1.000000, 1.000000, 1.000000, 1.000000];
        let layer_ffn_ln_bias: Vec<f32> = vec![0.010000, 0.010000, 0.010000, 0.010000];
        let ffn_layer_norm = Normalization::LayerNorm(LayerNorm::new(
            Array1::from_vec(layer_ffn_ln_weight),
            Array1::from_vec(layer_ffn_ln_bias),
            1e-5,
        ));

        CrossDecoderLayer::new(
            self_attn,
            self_attn_layer_norm,
            cross_attn,
            cross_attn_layer_norm,
            feedforward,
            ffn_layer_norm,
            pre_norm,
        )
    }

    #[test]
    fn test_cross_decoder_layer_postnorm_no_masks() -> Result<()> {
        let layer = create_test_cross_decoder_layer(false); // post-norm

        // === TEST 1: Post-norm (BART), no masks ===
        let layer_decoder_hidden: Vec<f32> = vec![
            0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
        ];
        let layer_encoder_hidden: Vec<f32> = vec![
            0.500000, 0.600000, 0.700000, 0.800000, 0.900000, 1.000000, 1.100000, 1.200000,
            1.300000, 1.400000, 1.500000, 1.600000,
        ];

        let decoder_hidden = Array3::from_shape_vec((1, 2, 4), layer_decoder_hidden)?;
        let encoder_hidden = Array3::from_shape_vec((1, 3, 4), layer_encoder_hidden)?;

        let (output, new_kv) = layer.forward(
            &decoder_hidden,
            &encoder_hidden,
            None, // self_mask
            None, // cross_mask
            None, // past_kv
            None, // cross_kv_cache
            None, // position_bias
        )?;

        // Golden output
        let layer_test1_output_postnorm: Vec<f32> = vec![
            -1.331635, -0.437212, 0.457212, 1.351635, -1.331635, -0.437212, 0.457212, 1.351634,
        ];
        let golden_output = Array3::from_shape_vec((1, 2, 4), layer_test1_output_postnorm)?;

        let diff = (&output - &golden_output).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Test 1 (Post-norm, no masks) Max Diff: {:.6}", max_diff);
        assert!(
            max_diff < 1e-4,
            "Post-norm layer mismatch. Max diff: {}",
            max_diff
        );

        // Verify self-attention K/V cache shapes
        let (new_k, new_v) = new_kv;
        assert_eq!(new_k.ndim(), 3, "K cache should be 3D");
        assert_eq!(new_v.ndim(), 3, "V cache should be 3D");

        Ok(())
    }

    #[test]
    fn test_cross_decoder_layer_prenorm_no_masks() -> Result<()> {
        let layer = create_test_cross_decoder_layer(true); // pre-norm

        // === TEST 2: Pre-norm (T5), no masks ===
        let layer_decoder_hidden: Vec<f32> = vec![
            0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
        ];
        let layer_encoder_hidden: Vec<f32> = vec![
            0.500000, 0.600000, 0.700000, 0.800000, 0.900000, 1.000000, 1.100000, 1.200000,
            1.300000, 1.400000, 1.500000, 1.600000,
        ];

        let decoder_hidden = Array3::from_shape_vec((1, 2, 4), layer_decoder_hidden)?;
        let encoder_hidden = Array3::from_shape_vec((1, 3, 4), layer_encoder_hidden)?;

        let (output, _) = layer.forward(
            &decoder_hidden,
            &encoder_hidden,
            None,
            None,
            None,
            None,
            None,
        )?;

        // Golden output
        let layer_test2_output_prenorm: Vec<f32> = vec![
            22.014660, 22.899734, 23.784811, 24.669884, 22.414780, 23.299860, 24.184940, 25.070023,
        ];
        let golden_output = Array3::from_shape_vec((1, 2, 4), layer_test2_output_prenorm)?;

        let diff = (&output - &golden_output).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Test 2 (Pre-norm, no masks) Max Diff: {:.6}", max_diff);
        assert!(
            max_diff < 1e-4,
            "Pre-norm layer mismatch. Max diff: {}",
            max_diff
        );

        Ok(())
    }

    #[test]
    fn test_cross_decoder_layer_postnorm_with_cross_mask() -> Result<()> {
        let layer = create_test_cross_decoder_layer(false); // post-norm

        // === TEST 3: Post-norm with cross mask ===
        let layer_decoder_hidden: Vec<f32> = vec![
            0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
        ];
        let layer_encoder_hidden: Vec<f32> = vec![
            0.500000, 0.600000, 0.700000, 0.800000, 0.900000, 1.000000, 1.100000, 1.200000,
            1.300000, 1.400000, 1.500000, 1.600000,
        ];
        let layer_test3_cross_mask: Vec<f32> = vec![1.000000, 1.000000, 0.000000];

        let decoder_hidden = Array3::from_shape_vec((1, 2, 4), layer_decoder_hidden)?;
        let encoder_hidden = Array3::from_shape_vec((1, 3, 4), layer_encoder_hidden)?;
        let cross_mask = Array2::from_shape_vec((1, 3), layer_test3_cross_mask)?;

        let (output, _) = layer.forward(
            &decoder_hidden,
            &encoder_hidden,
            None,
            Some(&cross_mask),
            None,
            None,
            None,
        )?;

        // Golden output
        let layer_test3_output: Vec<f32> = vec![
            -1.331635, -0.437211, 0.457212, 1.351634, -1.331635, -0.437211, 0.457211, 1.351635,
        ];
        let golden_output = Array3::from_shape_vec((1, 2, 4), layer_test3_output)?;

        let diff = (&output - &golden_output).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!(
            "Test 3 (Post-norm with cross mask) Max Diff: {:.6}",
            max_diff
        );
        assert!(
            max_diff < 1e-4,
            "Post-norm with cross mask mismatch. Max diff: {}",
            max_diff
        );

        Ok(())
    }

    #[test]
    fn test_cross_decoder_layer_postnorm_with_precomputed_cross_kv() -> Result<()> {
        let layer = create_test_cross_decoder_layer(false); // post-norm

        // === TEST 4: Post-norm with precomputed cross K/V ===
        let layer_decoder_hidden: Vec<f32> = vec![
            0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
        ];
        let layer_encoder_hidden: Vec<f32> = vec![
            0.500000, 0.600000, 0.700000, 0.800000, 0.900000, 1.000000, 1.100000, 1.200000,
            1.300000, 1.400000, 1.500000, 1.600000,
        ];

        let decoder_hidden = Array3::from_shape_vec((1, 2, 4), layer_decoder_hidden)?;
        let encoder_hidden = Array3::from_shape_vec((1, 3, 4), layer_encoder_hidden)?;

        // Precompute cross-attention K/V
        let (cross_k, cross_v) = layer.precompute_cross_kv(&encoder_hidden)?;

        let (output, _) = layer.forward(
            &decoder_hidden,
            &encoder_hidden,
            None,
            None,
            None,
            Some(&(cross_k, cross_v)),
            None,
        )?;

        // Should match non-cached output
        let layer_test4_output: Vec<f32> = vec![
            -1.331635, -0.437212, 0.457212, 1.351635, -1.331635, -0.437212, 0.457212, 1.351634,
        ];
        let golden_output = Array3::from_shape_vec((1, 2, 4), layer_test4_output)?;

        let diff = (&output - &golden_output).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!(
            "Test 4 (Post-norm with precomputed cross K/V) Max Diff: {:.6}",
            max_diff
        );
        assert!(
            max_diff < 1e-4,
            "Precomputed cross K/V mismatch. Max diff: {}",
            max_diff
        );

        Ok(())
    }

    #[test]
    fn test_cross_decoder_layer_batched() -> Result<()> {
        let layer = create_test_cross_decoder_layer(false); // post-norm

        // === TEST 5: Batched (batch=2), post-norm ===
        let layer_test5_decoder_hidden: Vec<f32> = vec![
            0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
            0.800000, 0.900000, 1.000000, 1.100000, 1.200000, 1.300000, 1.400000, 1.500000,
        ];
        let layer_test5_encoder_hidden: Vec<f32> = vec![
            0.500000, 0.600000, 0.700000, 0.800000, 0.900000, 1.000000, 1.100000, 1.200000,
            1.300000, 1.400000, 1.500000, 1.600000, 1.700000, 1.800000, 1.900000, 2.000000,
            2.100000, 2.200000, 2.300000, 2.400000, 2.500000, 2.600000, 2.700000, 2.800000,
        ];
        let layer_test5_cross_mask: Vec<f32> =
            vec![1.000000, 1.000000, 1.000000, 1.000000, 0.000000, 0.000000];

        let decoder_hidden = Array3::from_shape_vec((2, 2, 4), layer_test5_decoder_hidden)?;
        let encoder_hidden = Array3::from_shape_vec((2, 3, 4), layer_test5_encoder_hidden)?;
        let cross_mask = Array2::from_shape_vec((2, 3), layer_test5_cross_mask)?;

        let (output, _) = layer.forward(
            &decoder_hidden,
            &encoder_hidden,
            None,
            Some(&cross_mask),
            None,
            None,
            None,
        )?;

        // Golden output
        let layer_test5_output: Vec<f32> = vec![
            -1.331635, -0.437212, 0.457212, 1.351634, -1.331635, -0.437211, 0.457211, 1.351635,
            -1.331634, -0.437212, 0.457211, 1.351635, -1.331635, -0.437212, 0.457212, 1.351635,
        ];
        let golden_output = Array3::from_shape_vec((2, 2, 4), layer_test5_output)?;

        let diff = (&output - &golden_output).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Test 5 (Batched) Max Diff: {:.6}", max_diff);
        assert!(
            max_diff < 1e-4,
            "Batched layer mismatch. Max diff: {}",
            max_diff
        );

        Ok(())
    }

    #[test]
    fn test_cross_decoder_layer_single_token_decode() -> Result<()> {
        let layer = create_test_cross_decoder_layer(false); // post-norm

        // === TEST 6: Single token decode, post-norm ===
        let layer_test6_decoder_hidden: Vec<f32> = vec![0.100000, 0.200000, 0.300000, 0.400000];
        let layer_test6_encoder_hidden: Vec<f32> = vec![
            0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
            0.800000, 0.900000, 1.000000, 1.100000,
        ];

        let decoder_hidden = Array3::from_shape_vec((1, 1, 4), layer_test6_decoder_hidden)?;
        let encoder_hidden = Array3::from_shape_vec((1, 3, 4), layer_test6_encoder_hidden)?;

        let (output, (new_k, new_v)) = layer.forward(
            &decoder_hidden,
            &encoder_hidden,
            None,
            None,
            None,
            None,
            None,
        )?;

        // Golden output
        let layer_test6_output: Vec<f32> = vec![-1.331635, -0.437212, 0.457212, 1.351635];
        let golden_output = Array3::from_shape_vec((1, 1, 4), layer_test6_output)?;

        let diff = (&output - &golden_output).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Test 6 (Single token decode) Max Diff: {:.6}", max_diff);
        assert!(
            max_diff < 1e-4,
            "Single token decode mismatch. Max diff: {}",
            max_diff
        );

        // Verify self-attention K/V cache shapes for single token
        // Should be [batch=1, seq=1, hidden=4] but stored in head format
        println!("K cache shape: {:?}", new_k.shape());
        println!("V cache shape: {:?}", new_v.shape());

        Ok(())
    }

    #[test]
    fn test_cross_decoder_layer_prenorm_vs_postnorm_differ() -> Result<()> {
        // Verify that pre-norm and post-norm produce different outputs
        let layer_prenorm = create_test_cross_decoder_layer(true);
        let layer_postnorm = create_test_cross_decoder_layer(false);

        let decoder_hidden: Vec<f32> = vec![
            0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
        ];
        let encoder_hidden: Vec<f32> = vec![
            0.500000, 0.600000, 0.700000, 0.800000, 0.900000, 1.000000, 1.100000, 1.200000,
            1.300000, 1.400000, 1.500000, 1.600000,
        ];

        let decoder_hidden = Array3::from_shape_vec((1, 2, 4), decoder_hidden)?;
        let encoder_hidden = Array3::from_shape_vec((1, 3, 4), encoder_hidden)?;

        let (output_prenorm, _) = layer_prenorm.forward(
            &decoder_hidden,
            &encoder_hidden,
            None,
            None,
            None,
            None,
            None,
        )?;
        let (output_postnorm, _) = layer_postnorm.forward(
            &decoder_hidden,
            &encoder_hidden,
            None,
            None,
            None,
            None,
            None,
        )?;

        let diff = (&output_prenorm - &output_postnorm).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Pre-norm vs Post-norm diff: {:.6}", max_diff);
        assert!(
            max_diff > 1.0,
            "Pre-norm and post-norm should differ significantly! Diff: {}",
            max_diff
        );

        Ok(())
    }

    #[test]
    fn test_cross_decoder_layer_cross_mask_effect() -> Result<()> {
        // Verify that cross-attention mask actually changes output
        let layer = create_test_cross_decoder_layer(false);

        let decoder_hidden: Vec<f32> = vec![
            0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
        ];
        let encoder_hidden: Vec<f32> = vec![
            0.500000, 0.600000, 0.700000, 0.800000, 0.900000, 1.000000, 1.100000, 1.200000,
            1.300000, 1.400000, 1.500000, 1.600000,
        ];

        let decoder_hidden = Array3::from_shape_vec((1, 2, 4), decoder_hidden)?;
        let encoder_hidden = Array3::from_shape_vec((1, 3, 4), encoder_hidden)?;

        // Output without mask
        let (output_no_mask, _) = layer.forward(
            &decoder_hidden,
            &encoder_hidden,
            None,
            None,
            None,
            None,
            None,
        )?;

        // Output with aggressive mask (only first encoder position visible)
        let mask = Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 0.0])?;
        let (output_with_mask, _) = layer.forward(
            &decoder_hidden,
            &encoder_hidden,
            None,
            Some(&mask),
            None,
            None,
            None,
        )?;

        let diff = (&output_no_mask - &output_with_mask).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Cross mask effect - Output diff: {:.6}", max_diff);
        // With LayerNorm, the diff might be small but should still be non-zero
        assert!(
            max_diff > 1e-7,
            "Cross mask should change output! Diff was only {}",
            max_diff
        );

        Ok(())
    }

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
