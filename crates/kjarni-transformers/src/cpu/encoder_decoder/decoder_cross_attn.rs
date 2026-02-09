// Import Zip
use crate::activations::softmax_4d_inplace;
use crate::linear_layer::LinearLayer;
use crate::utils::linear_algebra::apply_attention_mask;
use crate::utils::linear_algebra::{matmul_4d, matmul_4d_context, matmul_4d_decode};
use anyhow::Result;
use ndarray::{Array2, Array3, Array4};

// Standard large negative value for masking (avoids NaN in softmax)
const MASK_VALUE: f32 = -1e9;

// ============================================================================
//  2. Decoder Cross-Attention
// ============================================================================

pub struct DecoderCrossAttention {
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub o_proj: LinearLayer,

    pub num_heads: usize,
    pub head_dim: usize,
    pub scale_factor: f32,
    pub scale_qk: bool,
}

impl DecoderCrossAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        q: LinearLayer,
        k: LinearLayer,
        v: LinearLayer,
        o: LinearLayer,
    ) -> Self {
        let head_dim = hidden_size / num_heads;
        Self {
            q_proj: q,
            k_proj: k,
            v_proj: v,
            o_proj: o,
            num_heads,
            head_dim,
            scale_factor: 1.0 / (head_dim as f32).sqrt(),
            scale_qk: true,
        }
    }
    pub fn with_no_qk_scaling(mut self) -> Self {
        self.scale_qk = false;
        self
    }
    /// Pre-computes K and V from the encoder output.
    pub fn precompute_encoder_kv(
        &self,
        encoder_hidden_states: &Array3<f32>,
    ) -> Result<(Array4<f32>, Array4<f32>)> {
        
        // println!("=== PRECOMPUTE K/V DEBUG ===");
        // println!(
        //     "encoder_hidden_states [0,0,:5]: {:?}",
        //     encoder_hidden_states.slice(s![0, 0, ..5])
        // );

        let (batch, seq_len, _) = encoder_hidden_states.dim();
        let enc_2d = encoder_hidden_states
            .view()
            .into_shape_with_order((batch * seq_len, self.num_heads * self.head_dim))?;

        let k = self.k_proj.matmul(&enc_2d);
        // println!("K after projection [0,:5]: {:?}", k.slice(s![0, ..5]));
        // println!(
        //     "K min/max: {:?} / {:?}",
        //     k.iter().cloned().fold(f32::INFINITY, f32::min),
        //     k.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        // );
        let v = self.v_proj.matmul(&enc_2d);

        // K: Transpose to [B, H, D, S] for efficient MatMul with Q
        let k_heads_t = k
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 3, 1])
            .as_standard_layout()
            .to_owned();

        // V: Keep as [B, H, S, D] for context computation
        let v_heads = v
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .to_owned();

        Ok((k_heads_t, v_heads))
    }

    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        encoder_k_t: &Array4<f32>,
        encoder_v: &Array4<f32>,
        attention_mask: Option<&Array2<f32>>,
    ) -> Result<Array3<f32>> {
        
        let (batch, seq_len, _) = hidden_states.dim();

        // println!("=== CROSS-ATTN DEBUG ===");
        // println!("hidden_states shape: {:?}", hidden_states.dim());
        // println!("encoder_k_t shape: {:?}", encoder_k_t.dim());
        // println!("encoder_v shape: {:?}", encoder_v.dim());
        // if let Some(m) = attention_mask {
        //     println!("attention_mask shape: {:?}", m.dim());
        //     println!(
        //         "attention_mask [0,:5]: {:?}",
        //         m.slice(s![0, ..5.min(m.dim().1)])
        //     );
        // }

        // Project Q from Decoder Hidden States
        let hidden_2d = hidden_states
            .view()
            .into_shape_with_order((batch * seq_len, self.num_heads * self.head_dim))?;
        let q = self.q_proj.matmul(&hidden_2d);

        let q_heads = q
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3])
            .to_owned();
        // println!("q_heads shape: {:?}", q_heads.dim());
        // println!("q_heads [0,0,0,:5]: {:?}", q_heads.slice(s![0, 0, 0, ..5]));

        // Scores
        let mut scores = if seq_len == 1 {
            matmul_4d_decode(&q_heads, encoder_k_t)
        } else {
            matmul_4d(&q_heads, encoder_k_t)
        };
        if self.scale_qk {
            // println!("scores BEFORE scaling: {:?}", scores);
            scores.mapv_inplace(|x| x * self.scale_factor);
            // println!("scores AFTER scaling: {:?}", scores);
        }

        // println!("scores shape: {:?}", scores.dim());
        // println!(
        //     "scores BEFORE mask [0,0,0,:5]: {:?}",
        //     scores.slice(s![0, 0, 0, ..5.min(scores.dim().3)])
        // );
        // println!(
        //     "scores min/max: {:?} / {:?}",
        //     scores.iter().cloned().fold(f32::INFINITY, f32::min),
        //     scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        // );

        // Apply Padding Mask (Using safe Zip broadcasting)
        if let Some(mask) = attention_mask {
            scores = apply_attention_mask(scores, mask);
            // println!(
            //     "scores AFTER mask [0,0,0,:5]: {:?}",
            //     scores.slice(s![0, 0, 0, ..5.min(scores.dim().3)])
            // );
        }

        softmax_4d_inplace(&mut scores);
        // println!(
        //     "attn_weights [0,0,0,:5]: {:?}",
        //     scores.slice(s![0, 0, 0, ..5.min(scores.dim().3)])
        // );
        // println!(
        //     "attn_weights sum (should be ~1): {:?}",
        //     scores.slice(s![0, 0, 0, ..]).sum()
        // );

        // Context
        let context = if seq_len == 1 {
            matmul_4d_context(&scores, encoder_v)
        } else {
            matmul_4d(&scores, encoder_v)
        };

        // Output
        let context_flat = context
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .into_shape_with_order((batch * seq_len, self.num_heads * self.head_dim))?
            .to_owned();

        let output = self.o_proj.matmul(&context_flat.view());
        let output_3d =
            output.into_shape_with_order((batch, seq_len, self.num_heads * self.head_dim))?;

        Ok(output_3d)
    }
}

#[cfg(test)]
mod cross_attention_tests {
    use super::*;
    use crate::linear_layer::LinearLayer;
    use anyhow::Result;
    use ndarray::{Array1, Array2, Array3, Array4};

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

    /// Create deterministic cross-attention for testing
    fn create_test_cross_attention() -> DecoderCrossAttention {
        let hidden_size = 4;
        let num_heads = 2;

        // === CROSS ATTENTION WEIGHTS ===
        let cross_attn_q_weight: Vec<f32> = vec![
            0.010000, 0.020000, 0.030000, 0.040000, 0.050000, 0.060000, 0.070000, 0.080000,
            0.090000, 0.100000, 0.110000, 0.120000, 0.130000, 0.140000, 0.150000, 0.160000,
        ];
        let cross_attn_q_bias: Vec<f32> = vec![0.010000, 0.010000, 0.010000, 0.010000];

        let cross_attn_k_weight: Vec<f32> = vec![
            0.170000, 0.180000, 0.190000, 0.200000, 0.210000, 0.220000, 0.230000, 0.240000,
            0.250000, 0.260000, 0.270000, 0.280000, 0.290000, 0.300000, 0.310000, 0.320000,
        ];
        let cross_attn_k_bias: Vec<f32> = vec![0.010000, 0.010000, 0.010000, 0.010000];

        let cross_attn_v_weight: Vec<f32> = vec![
            0.330000, 0.340000, 0.350000, 0.360000, 0.370000, 0.380000, 0.390000, 0.400000,
            0.410000, 0.420000, 0.430000, 0.440000, 0.450000, 0.460000, 0.470000, 0.480000,
        ];
        let cross_attn_v_bias: Vec<f32> = vec![0.010000, 0.010000, 0.010000, 0.010000];

        let cross_attn_o_weight: Vec<f32> = vec![
            0.490000, 0.500000, 0.510000, 0.520000, 0.530000, 0.540000, 0.550000, 0.560000,
            0.570000, 0.580000, 0.590000, 0.600000, 0.610000, 0.620000, 0.630000, 0.640000,
        ];
        let cross_attn_o_bias: Vec<f32> = vec![0.010000, 0.010000, 0.010000, 0.010000];

        let q = load_linear(
            hidden_size,
            hidden_size,
            cross_attn_q_weight,
            cross_attn_q_bias,
        );
        let k = load_linear(
            hidden_size,
            hidden_size,
            cross_attn_k_weight,
            cross_attn_k_bias,
        );
        let v = load_linear(
            hidden_size,
            hidden_size,
            cross_attn_v_weight,
            cross_attn_v_bias,
        );
        let o = load_linear(
            hidden_size,
            hidden_size,
            cross_attn_o_weight,
            cross_attn_o_bias,
        );

        DecoderCrossAttention::new(hidden_size, num_heads, q, k, v, o)
    }

    #[test]
    fn test_cross_attention_basic_no_mask() -> Result<()> {
        let attn = create_test_cross_attention();

        // === TEST 1: Basic Cross-Attention (no mask) ===
        let test1_decoder_hidden: Vec<f32> = vec![
            0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
        ];
        let test1_encoder_hidden: Vec<f32> = vec![
            1.000000, 1.100000, 1.200000, 1.300000, 1.400000, 1.500000, 1.600000, 1.700000,
            1.800000, 1.900000, 2.000000, 2.100000,
        ];

        let decoder_hidden = Array3::from_shape_vec((1, 2, 4), test1_decoder_hidden)?;
        let encoder_hidden = Array3::from_shape_vec((1, 3, 4), test1_encoder_hidden)?;

        // Precompute K/V from encoder
        let (k_cache, v_cache) = attn.precompute_encoder_kv(&encoder_hidden)?;

        // Run forward
        let output = attn.forward(&decoder_hidden, &k_cache, &v_cache, None)?;

        // Golden output
        let test1_output: Vec<f32> = vec![
            5.161280, 5.568286, 5.975293, 6.382300, 5.237971, 5.650974, 6.063977, 6.476981,
        ];
        let golden_output = Array3::from_shape_vec((1, 2, 4), test1_output)?;

        let diff = (&output - &golden_output).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Test 1 (Basic, no mask) Max Diff: {:.6}", max_diff);
        assert!(
            max_diff < 1e-4,
            "Basic cross-attention mismatch. Max diff: {}",
            max_diff
        );

        Ok(())
    }

    #[test]
    fn test_cross_attention_with_mask() -> Result<()> {
        let attn = create_test_cross_attention();

        // Inputs
        let test1_decoder_hidden: Vec<f32> = vec![
            0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
        ];
        let test1_encoder_hidden: Vec<f32> = vec![
            1.000000, 1.100000, 1.200000, 1.300000, 1.400000, 1.500000, 1.600000, 1.700000,
            1.800000, 1.900000, 2.000000, 2.100000,
        ];

        let decoder_hidden = Array3::from_shape_vec((1, 2, 4), test1_decoder_hidden)?;
        let encoder_hidden = Array3::from_shape_vec((1, 3, 4), test1_encoder_hidden)?;

        // === TEST 2: Cross-Attention with mask ===
        // Mask: last encoder position is padding (0 = mask out)
        let test2_cross_mask: Vec<f32> = vec![1.000000, 1.000000, 0.000000];
        let cross_mask = Array2::from_shape_vec((1, 3), test2_cross_mask)?;

        let (k_cache, v_cache) = attn.precompute_encoder_kv(&encoder_hidden)?;
        let output = attn.forward(&decoder_hidden, &k_cache, &v_cache, Some(&cross_mask))?;

        // Golden output with mask
        let test2_output: Vec<f32> = vec![
            4.482478, 4.835866, 5.189254, 5.542642, 4.511338, 4.866982, 5.222627, 5.578271,
        ];
        let golden_output = Array3::from_shape_vec((1, 2, 4), test2_output)?;

        let diff = (&output - &golden_output).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Test 2 (With mask) Max Diff: {:.6}", max_diff);
        assert!(
            max_diff < 1e-4,
            "Cross-attention with mask mismatch. Max diff: {}",
            max_diff
        );

        Ok(())
    }

    #[test]
    fn test_cross_attention_precomputed_kv() -> Result<()> {
        let attn = create_test_cross_attention();

        // Inputs
        let test1_decoder_hidden: Vec<f32> = vec![
            0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
        ];
        let test1_encoder_hidden: Vec<f32> = vec![
            1.000000, 1.100000, 1.200000, 1.300000, 1.400000, 1.500000, 1.600000, 1.700000,
            1.800000, 1.900000, 2.000000, 2.100000,
        ];

        let decoder_hidden = Array3::from_shape_vec((1, 2, 4), test1_decoder_hidden)?;
        let encoder_hidden = Array3::from_shape_vec((1, 3, 4), test1_encoder_hidden)?;

        // === TEST 3: Precomputed K/V path ===
        let (k_cache, v_cache) = attn.precompute_encoder_kv(&encoder_hidden)?;

        // Verify K cache shape and values
        // Note: Rust stores K transposed as [B, H, D, S] for efficient matmul
        // Python golden is [B, H, S, D] = [1, 2, 3, 2]
        // test3_k_cache - Shape: [1, 2, 3, 2]
        let test3_k_cache: Vec<f32> = vec![
            0.866000, 1.050000, 1.162000, 1.410000, 1.458000, 1.770000, 1.234000, 1.418000,
            1.658000, 1.906000, 2.082000, 2.394000,
        ];

        // test3_v_cache - Shape: [1, 2, 3, 2]
        let test3_v_cache: Vec<f32> = vec![
            1.602000, 1.786000, 2.154000, 2.402000, 2.706000, 3.018000, 1.970000, 2.154000,
            2.650000, 2.898000, 3.330000, 3.642000,
        ];

        // V cache should match directly (shape [B, H, S, D])
        let golden_v = Array4::from_shape_vec((1, 2, 3, 2), test3_v_cache)?;
        let v_diff = (&v_cache - &golden_v).mapv(|x| x.abs());
        let max_v_diff = v_diff.fold(0.0f32, |a, &b| a.max(b));
        println!("Test 3 V cache Max Diff: {:.6}", max_v_diff);
        assert!(
            max_v_diff < 1e-4,
            "V cache mismatch. Max diff: {}",
            max_v_diff
        );

        // K cache is transposed [B, H, D, S] in Rust vs [B, H, S, D] in Python
        // Need to transpose for comparison or compare element-wise
        let golden_k_bhsd = Array4::from_shape_vec((1, 2, 3, 2), test3_k_cache)?;
        let golden_k_bhds = golden_k_bhsd.permuted_axes([0, 1, 3, 2]);
        let k_diff = (&k_cache - &golden_k_bhds).mapv(|x| x.abs());
        let max_k_diff = k_diff.fold(0.0f32, |a, &b| a.max(b));
        println!("Test 3 K cache Max Diff: {:.6}", max_k_diff);
        assert!(
            max_k_diff < 1e-4,
            "K cache mismatch. Max diff: {}",
            max_k_diff
        );

        // Verify output matches non-cached path
        let output_cached = attn.forward(&decoder_hidden, &k_cache, &v_cache, None)?;

        let test3_output: Vec<f32> = vec![
            5.161280, 5.568286, 5.975293, 6.382300, 5.237971, 5.650974, 6.063977, 6.476981,
        ];
        let golden_output = Array3::from_shape_vec((1, 2, 4), test3_output)?;

        let diff = (&output_cached - &golden_output).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));
        println!("Test 3 (Precomputed K/V) Output Max Diff: {:.6}", max_diff);
        assert!(
            max_diff < 1e-4,
            "Precomputed K/V output mismatch. Max diff: {}",
            max_diff
        );

        Ok(())
    }

    #[test]
    fn test_cross_attention_batched_with_masks() -> Result<()> {
        let attn = create_test_cross_attention();

        // === TEST 4: Batched (batch=2) with different masks ===
        let test4_decoder_hidden: Vec<f32> = vec![
            0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
            0.800000, 0.900000, 1.000000, 1.100000, 1.200000, 1.300000, 1.400000, 1.500000,
        ];
        let test4_encoder_hidden: Vec<f32> = vec![
            1.000000, 1.100000, 1.200000, 1.300000, 1.400000, 1.500000, 1.600000, 1.700000,
            1.800000, 1.900000, 2.000000, 2.100000, 2.200000, 2.300000, 2.400000, 2.500000,
            2.600000, 2.700000, 2.800000, 2.900000, 3.000000, 3.100000, 3.200000, 3.300000,
        ];
        let test4_cross_mask: Vec<f32> =
            vec![1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.000000];

        let decoder_hidden = Array3::from_shape_vec((2, 2, 4), test4_decoder_hidden)?;
        let encoder_hidden = Array3::from_shape_vec((2, 3, 4), test4_encoder_hidden)?;
        let cross_mask = Array2::from_shape_vec((2, 3), test4_cross_mask)?;

        let (k_cache, v_cache) = attn.precompute_encoder_kv(&encoder_hidden)?;
        let output = attn.forward(&decoder_hidden, &k_cache, &v_cache, Some(&cross_mask))?;

        let test4_output: Vec<f32> = vec![
            5.161280, 5.568287, 5.975293, 6.382300, 5.237972, 5.650974, 6.063977, 6.476981,
            8.476384, 9.145302, 9.814219, 10.483137, 8.504461, 9.175573, 9.846687, 10.517800,
        ];
        let golden_output = Array3::from_shape_vec((2, 2, 4), test4_output)?;

        let diff = (&output - &golden_output).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Test 4 (Batched with masks) Max Diff: {:.6}", max_diff);
        assert!(
            max_diff < 1e-4,
            "Batched cross-attention mismatch. Max diff: {}",
            max_diff
        );

        Ok(())
    }

    #[test]
    fn test_cross_attention_single_token_decode() -> Result<()> {
        let attn = create_test_cross_attention();

        // === TEST 5: Single decoder token (decode mode) ===
        let test5_decoder_hidden: Vec<f32> = vec![0.500000, 0.600000, 0.700000, 0.800000];
        let test5_encoder_hidden: Vec<f32> = vec![
            0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
            0.800000, 0.900000, 1.000000, 1.100000,
        ];

        let decoder_hidden = Array3::from_shape_vec((1, 1, 4), test5_decoder_hidden)?;
        let encoder_hidden = Array3::from_shape_vec((1, 3, 4), test5_encoder_hidden)?;

        let (k_cache, v_cache) = attn.precompute_encoder_kv(&encoder_hidden)?;
        let output = attn.forward(&decoder_hidden, &k_cache, &v_cache, None)?;

        let test5_output: Vec<f32> = vec![1.976542, 2.131828, 2.287114, 2.442401];
        let golden_output = Array3::from_shape_vec((1, 1, 4), test5_output)?;

        let diff = (&output - &golden_output).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Test 5 (Single token decode) Max Diff: {:.6}", max_diff);
        assert!(
            max_diff < 1e-4,
            "Single token decode mismatch. Max diff: {}",
            max_diff
        );

        Ok(())
    }

    #[test]
    fn test_cross_attention_mask_effect() -> Result<()> {
        // Verify that masking actually changes the output
        let attn = create_test_cross_attention();

        let decoder_hidden: Vec<f32> = vec![
            0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000,
        ];
        let encoder_hidden: Vec<f32> = vec![
            1.000000, 1.100000, 1.200000, 1.300000, 1.400000, 1.500000, 1.600000, 1.700000,
            1.800000, 1.900000, 2.000000, 2.100000,
        ];

        let decoder_hidden = Array3::from_shape_vec((1, 2, 4), decoder_hidden)?;
        let encoder_hidden = Array3::from_shape_vec((1, 3, 4), encoder_hidden)?;

        let (k_cache, v_cache) = attn.precompute_encoder_kv(&encoder_hidden)?;

        // Output without mask
        let output_no_mask = attn.forward(&decoder_hidden, &k_cache, &v_cache, None)?;

        // Output with mask (mask last position)
        let mask = Array2::from_shape_vec((1, 3), vec![1.0, 1.0, 0.0])?;
        let output_with_mask = attn.forward(&decoder_hidden, &k_cache, &v_cache, Some(&mask))?;

        // They should be different
        let diff = (&output_no_mask - &output_with_mask).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Mask effect - Output diff: {:.6}", max_diff);
        assert!(
            max_diff > 0.1,
            "Mask should change output! Diff was only {}",
            max_diff
        );

        Ok(())
    }
}
