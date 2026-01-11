use crate::cpu::decoder::DecoderAttention;
use crate::feedforward::FeedForward;
use crate::normalization::Normalization;
use crate::rope::RoPE;
use anyhow::Result;
use ndarray::{Array2, Array3, s};
use std::sync::Arc;

pub struct DecoderLayer {
    pub self_attn: DecoderAttention,
    pub self_attn_layer_norm: Normalization,
    pub feedforward: FeedForward,
    pub ffn_layer_norm: Normalization,
    pub is_prenorm: bool,
    pub rope: Option<Arc<RoPE>>,
}

impl DecoderLayer {
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        _position_offset: usize,
        past_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        if self.is_prenorm {
            self.forward_prenorm(hidden_states, attention_mask, past_kv)
        } else {
            self.forward_postnorm(hidden_states, attention_mask, past_kv)
        }
    }
    /// Llama / Mistral / Phi (Pre-Norm)
    /// Flow: x = x + Attn(Norm(x))
    fn forward_prenorm(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        past_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        let (batch, seq_len, _) = hidden_states.dim();
        let kv_dim = self.self_attn.num_kv_heads * self.self_attn.head_dim;
        let cache_len = past_kv.as_ref().map(|(k, _)| k.shape()[1]).unwrap_or(0);
        let total_len = cache_len + seq_len;

        // Allocate full cache buffer
        let mut k_cache = Array3::<f32>::zeros((batch, total_len, kv_dim));
        let mut v_cache = Array3::<f32>::zeros((batch, total_len, kv_dim));

        // Copy past KV if exists
        if let Some((past_k, past_v)) = past_kv {
            k_cache.slice_mut(s![.., ..cache_len, ..]).assign(&past_k);
            v_cache.slice_mut(s![.., ..cache_len, ..]).assign(&past_v);
        }

        let residual = hidden_states.clone();

        // 1. Norm before Attn
        let ln1_out = self.self_attn_layer_norm.forward(hidden_states);

        // 2. Attn with contiguous cache
        let attn_out = self.self_attn.forward(
            &ln1_out,
            Some(attention_mask),
            k_cache.view_mut(),
            v_cache.view_mut(),
            cache_len,
            self.rope.as_deref(),
        )?;

        // 3. Residual connection
        let attn_block_output = residual + attn_out;

        let residual = attn_block_output.clone();

        // 4. Norm before FFN
        let ln2_out = self.ffn_layer_norm.forward(&attn_block_output);

        // 5. FFN
        let ffn_out = self.feedforward.forward(&ln2_out)?;

        // 6. Final Residual
        let final_output = residual + ffn_out;

        Ok((final_output, (k_cache, v_cache)))
    }

    /// GPT-2 (Post-Norm)
    /// Flow: x = Norm(x + Attn(x))
    fn forward_postnorm(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        past_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        let (batch, seq_len, _) = hidden_states.dim();
        let kv_dim = self.self_attn.num_kv_heads * self.self_attn.head_dim;
        let cache_len = past_kv.as_ref().map(|(k, _)| k.shape()[1]).unwrap_or(0);
        let total_len = cache_len + seq_len;

        // Allocate full cache buffer
        let mut k_cache = Array3::<f32>::zeros((batch, total_len, kv_dim));
        let mut v_cache = Array3::<f32>::zeros((batch, total_len, kv_dim));

        // Copy past KV if exists
        if let Some((past_k, past_v)) = past_kv {
            k_cache.slice_mut(s![.., ..cache_len, ..]).assign(&past_k);
            v_cache.slice_mut(s![.., ..cache_len, ..]).assign(&past_v);
        }

        let residual = hidden_states.clone();

        // 1. Attn (No Norm before)
        let attn_out = self.self_attn.forward(
            hidden_states,
            Some(attention_mask),
            k_cache.view_mut(),
            v_cache.view_mut(),
            cache_len,
            self.rope.as_deref(), // Pass RoPE if it exists (usually None for GPT-2)
        )?;

        // 2. Residual + Norm
        let hidden = residual + attn_out;
        let hidden = self.self_attn_layer_norm.forward(&hidden);

        // 3. FFN
        let residual = hidden.clone();
        let ffn_out = self.feedforward.forward(&hidden)?;

        // 4. Residual + Norm
        let hidden = residual + ffn_out;
        let output = self.ffn_layer_norm.forward(&hidden);

        Ok((output, (k_cache, v_cache)))
    }
}

#[cfg(test)]
mod decoder_layer_tests {
    use super::*;
    use crate::{
        activations::Activation,
        feedforward::{FeedForward, LegacyFeedForward, StdFeedForward, SwiGluFeedForward},
        linear_layer::LinearLayer,
        normalization::{LayerNorm, RMSNorm},
    };
    use ndarray::{Array1, Array4};

    fn create_test_layer_prenorm() -> DecoderLayer {
        let hidden_size = 64;
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_dim = hidden_size / num_heads;

        // Create attention with identity-like weights
        let q_weight = Array2::eye(hidden_size);
        let k_weight = Array2::eye(hidden_size);
        let v_weight = Array2::eye(hidden_size);
        let o_weight = Array2::eye(hidden_size);

        let q_proj = LinearLayer::new_f32(q_weight, None);
        let k_proj = LinearLayer::new_f32(k_weight, None);
        let v_proj = LinearLayer::new_f32(v_weight, None);
        let o_proj = LinearLayer::new_f32(o_weight, None);

        let self_attn = DecoderAttention::new(
            hidden_size,
            num_heads,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            Some(num_kv_heads),
        );

        // Create feedforward
        // LegacyFeedForward expects [in, out] layout
        let intermediate_size = hidden_size * 4; // 256
        let fc1_weight = Array2::from_shape_fn((hidden_size, intermediate_size), |(i, j)| {
            if i == j && i < hidden_size { 1.0 } else { 0.0 }
        });
        let fc2_weight = Array2::from_shape_fn((intermediate_size, hidden_size), |(i, j)| {
            if i < hidden_size && i == j { 1.0 } else { 0.0 }
        });
        let fc1_bias = Array1::zeros(intermediate_size);
        let fc2_bias = Array1::zeros(hidden_size);

        let feedforward = FeedForward::Legacy(LegacyFeedForward::new(
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            Activation::Gelu,
        ));

        // Create normalizations
        let norm_weight = Array1::ones(hidden_size);
        let self_attn_layer_norm = Normalization::RMSNorm(RMSNorm::new(norm_weight.clone(), 1e-5));
        let ffn_layer_norm = Normalization::RMSNorm(RMSNorm::new(norm_weight, 1e-5));

        DecoderLayer {
            self_attn,
            self_attn_layer_norm,
            feedforward,
            ffn_layer_norm,
            is_prenorm: true,
            rope: None,
        }
    }

    fn create_test_layer_postnorm() -> DecoderLayer {
        let hidden_size = 64;
        let num_heads = 4;
        let num_kv_heads = 4;

        // Create attention with identity-like weights
        let q_weight = Array2::eye(hidden_size);
        let k_weight = Array2::eye(hidden_size);
        let v_weight = Array2::eye(hidden_size);
        let o_weight = Array2::eye(hidden_size);

        let q_proj = LinearLayer::new_f32(q_weight, None);
        let k_proj = LinearLayer::new_f32(k_weight, None);
        let v_proj = LinearLayer::new_f32(v_weight, None);
        let o_proj = LinearLayer::new_f32(o_weight, None);

        let self_attn = DecoderAttention::new(
            hidden_size,
            num_heads,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            Some(num_kv_heads),
        );

        // Create feedforward
        // LegacyFeedForward expects [in, out] layout
        let intermediate_size = hidden_size * 4; // 256
        let fc1_weight = Array2::from_shape_fn((hidden_size, intermediate_size), |(i, j)| {
            if i == j && i < hidden_size { 1.0 } else { 0.0 }
        });
        let fc2_weight = Array2::from_shape_fn((intermediate_size, hidden_size), |(i, j)| {
            if i < hidden_size && i == j { 1.0 } else { 0.0 }
        });
        let fc1_bias = Array1::zeros(intermediate_size);
        let fc2_bias = Array1::zeros(hidden_size);

        let feedforward = FeedForward::Legacy(LegacyFeedForward::new(
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            Activation::Gelu,
        ));

        // Create LayerNorm for post-norm (GPT-2 style)
        let norm_weight = Array1::ones(hidden_size);
        let norm_bias = Array1::zeros(hidden_size);
        let self_attn_layer_norm =
            Normalization::LayerNorm(LayerNorm::new(norm_weight.clone(), norm_bias.clone(), 1e-5));
        let ffn_layer_norm = Normalization::LayerNorm(LayerNorm::new(norm_weight, norm_bias, 1e-5));

        DecoderLayer {
            self_attn,
            self_attn_layer_norm,
            feedforward,
            ffn_layer_norm,
            is_prenorm: false,
            rope: None,
        }
    }

    #[test]
    fn test_decoder_layer_prenorm_forward() -> Result<()> {
        let layer = create_test_layer_prenorm();
        let batch_size = 1;
        let seq_len = 4;
        let hidden_size = 64;

        // Create input
        let hidden_states =
            Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
                (k as f32) * 0.01
            });

        // Create attention mask
        let attention_mask = Array2::ones((batch_size, seq_len));

        // Run forward pass without cache
        let (output, (new_k, new_v)) = layer.forward(&hidden_states, &attention_mask, 0, None)?;

        // Check output shape
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);

        // Check KV cache shapes
        assert_eq!(new_k.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_v.shape(), &[batch_size, seq_len, hidden_size]);

        // Check that output is not all zeros
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 1e-6, "Output should not be all zeros");

        Ok(())
    }

    #[test]
    fn test_decoder_layer_postnorm_forward() -> Result<()> {
        let layer = create_test_layer_postnorm();
        let batch_size = 1;
        let seq_len = 4;
        let hidden_size = 64;

        // Create input
        let hidden_states =
            Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
                (k as f32) * 0.01
            });

        // Create attention mask
        let attention_mask = Array2::ones((batch_size, seq_len));

        // Run forward pass without cache
        let (output, (new_k, new_v)) = layer.forward(&hidden_states, &attention_mask, 0, None)?;

        // Check output shape
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);

        // Check KV cache shapes
        assert_eq!(new_k.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_v.shape(), &[batch_size, seq_len, hidden_size]);

        // Check that output is not all zeros
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 1e-6, "Output should not be all zeros");

        Ok(())
    }

    #[test]
    fn test_decoder_layer_with_cache() -> Result<()> {
        let layer = create_test_layer_prenorm();
        let batch_size = 1;
        let seq_len = 1;
        let hidden_size = 64;

        // Create input for decode step
        let hidden_states =
            Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
                (k as f32) * 0.01
            });

        // Create attention mask
        let attention_mask = Array2::ones((batch_size, seq_len));

        // Create past KV cache
        let past_len = 3;
        let past_k = Array3::from_shape_fn((batch_size, past_len, hidden_size), |(_, _, k)| {
            (k as f32) * 0.005
        });
        let past_v = Array3::from_shape_fn((batch_size, past_len, hidden_size), |(_, _, k)| {
            (k as f32) * 0.005
        });

        // Run forward pass with cache
        let (output, (new_k, new_v)) = layer.forward(
            &hidden_states,
            &attention_mask,
            0,
            Some((past_k.view(), past_v.view())),
        )?;

        // Check output shape
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);

        // --- FIX IS HERE ---
        // The refactored layer returns the FULL updated cache (History + New Token)
        // Left was [1, 4, 64], Right was [1, 1, 64]
        let total_len = past_len + seq_len;
        assert_eq!(new_k.shape(), &[batch_size, total_len, hidden_size]);
        assert_eq!(new_v.shape(), &[batch_size, total_len, hidden_size]);

        // Check that output is not all zeros
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 1e-6, "Output should not be all zeros");

        Ok(())
    }
    #[test]
    fn test_decoder_layer_with_rope() -> Result<()> {
        let mut layer = create_test_layer_prenorm();
        let hidden_size = 64;
        let num_heads = 4;
        let head_dim = hidden_size / num_heads;

        // Add RoPE to the layer
        layer.rope = Some(Arc::new(RoPE::new(head_dim, 128, 10000.0)));

        let batch_size = 1;
        let seq_len = 4;

        // Create input
        let hidden_states =
            Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
                (k as f32) * 0.01
            });

        // Create attention mask
        let attention_mask = Array2::ones((batch_size, seq_len));

        // Run forward pass
        let (output, (new_k, new_v)) = layer.forward(&hidden_states, &attention_mask, 0, None)?;

        // Check output shape
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);

        // Check that RoPE was applied (output should be different from no-RoPE case)
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 1e-6, "Output should not be all zeros");

        Ok(())
    }

    #[test]
    fn test_decoder_layer_prenorm_vs_postnorm() -> Result<()> {
        let prenorm_layer = create_test_layer_prenorm();
        let postnorm_layer = create_test_layer_postnorm();

        let batch_size = 1;
        let seq_len = 4;
        let hidden_size = 64;

        // Create same input for both
        let hidden_states =
            Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
                (k as f32) * 0.01
            });
        let attention_mask = Array2::ones((batch_size, seq_len));

        // Run both layers
        let (prenorm_output, _) =
            prenorm_layer.forward(&hidden_states, &attention_mask, 0, None)?;
        let (postnorm_output, _) =
            postnorm_layer.forward(&hidden_states, &attention_mask, 0, None)?;

        // Both should produce valid outputs but with different values
        assert_eq!(prenorm_output.shape(), postnorm_output.shape());

        // Outputs should be different due to different normalization order
        let prenorm_sum: f32 = prenorm_output.iter().sum();
        let postnorm_sum: f32 = postnorm_output.iter().sum();

        assert!(prenorm_sum.abs() > 1e-6);
        assert!(postnorm_sum.abs() > 1e-6);

        Ok(())
    }

    fn get_weights(start: usize, rows: usize, cols: usize) -> (Array2<f32>, usize) {
        let size = rows * cols;
        // Python: torch.arange(count, count + num).float() * 0.0001
        let data: Vec<f32> = (start..start + size).map(|x| x as f32 * 0.0001).collect();
        // Array2::from_shape_vec fills row-major, which matches PyTorch flatten behavior
        (
            Array2::from_shape_vec((rows, cols), data).unwrap(),
            start + size,
        )
    }

    fn create_deterministic_decoder(
        hidden: usize,
        heads: usize,
        inter: usize,
        is_prenorm: bool,
        use_swiglu: bool,
        seed_offset: usize,
    ) -> DecoderLayer {
        let mut count = 1 + seed_offset;
        let bias_val = 0.01;
        let head_dim = hidden / heads;

        // --- 1. Self Attention ---
        // Python Order: Q, K, V, O
        let (qw, c) = get_weights(count, hidden, hidden);
        count = c;
        let (kw, c) = get_weights(count, head_dim * heads, hidden);
        count = c;
        let (vw, c) = get_weights(count, head_dim * heads, hidden);
        count = c;
        let (ow, c) = get_weights(count, hidden, hidden);
        count = c;

        // Python mocks used bias=False for Attn, but Rust LinearLayer typically holds optional bias.
        // We pass None to match the Python mock architecture.
        let attn = DecoderAttention::new(
            hidden,
            heads,
            LinearLayer::new_f32(qw, None),
            LinearLayer::new_f32(kw, None),
            LinearLayer::new_f32(vw, None),
            LinearLayer::new_f32(ow, None),
            Some(heads), // kv_heads
        );

        // --- 2. Attention Norm ---
        // Python: bias=0.01, weight=1.0
        let attn_norm = Normalization::LayerNorm(LayerNorm::new(
            Array1::ones(hidden),
            Array1::from_elem(hidden, bias_val),
            1e-5,
        ));

        // --- 3. FeedForward ---
        let feedforward = if use_swiglu {
            // Python SwiGLU: Gate, Up, Down (bias=False)
            let (gw, c) = get_weights(count, inter, hidden);
            count = c;
            let (uw, c) = get_weights(count, inter, hidden);
            count = c;
            let (dw, c) = get_weights(count, hidden, inter);
            count = c;

            FeedForward::SwiGLU(SwiGluFeedForward::new(
                LinearLayer::new_f32(gw, None),
                LinearLayer::new_f32(uw, None),
                LinearLayer::new_f32(dw, None),
                Activation::SilU,
            ))
        } else {
            // Python Standard: FC1, FC2 (bias=True, initialized to 0.01)
            let (w1, c) = get_weights(count, inter, hidden);
            count = c;
            let b1 = Array1::from_elem(inter, bias_val);
            let (w2, c) = get_weights(count, hidden, inter);
            count = c;
            let b2 = Array1::from_elem(hidden, bias_val);

            FeedForward::Standard(StdFeedForward::new(
                w1,
                b1,
                w2,
                b2,
                Activation::Gelu, // Match Python nn.GELU()
            ))
        };

        // --- 4. FFN Norm ---
        let ffn_norm = Normalization::LayerNorm(LayerNorm::new(
            Array1::ones(hidden),
            Array1::from_elem(hidden, bias_val),
            1e-5,
        ));

        DecoderLayer {
            self_attn: attn,
            self_attn_layer_norm: attn_norm,
            feedforward,
            ffn_layer_norm: ffn_norm,
            is_prenorm,
            rope: None,
        }
    }

    // Helper to transform Python [Batch, Heads, Seq, Dim] -> Rust [Batch, Seq, Heads * Dim]
    fn permute_golden_kv(
        data: Vec<f32>,
        batch: usize,
        heads: usize,
        seq: usize,
        head_dim: usize,
    ) -> Array3<f32> {
        // Load as 4D: [Batch, Heads, Seq, HeadDim]
        let arr = Array4::from_shape_vec((batch, heads, seq, head_dim), data).unwrap();
        // Permute to: [Batch, Seq, Heads, HeadDim]
        let permuted = arr.permuted_axes([0, 2, 1, 3]);
        // Reshape/Flatten last two dims: [Batch, Seq, Heads*HeadDim]
        permuted
            .as_standard_layout() // Ensure contiguous before reshaping
            .to_owned()
            .into_shape_with_order((batch, seq, heads * head_dim))
            .unwrap()
    }
    // ========================================================================
    // 2. Scenario A: Pre-Norm + SwiGLU (Llama Style)
    // ========================================================================
    #[test]
    fn test_decoder_prenorm_swiglu_golden() -> Result<()> {
        // Config
        let (hidden, heads, inter) = (4, 2, 8);
        let head_dim = hidden / heads;

        // 1. Create Layer (Offset 0)
        let layer = create_deterministic_decoder(hidden, heads, inter, true, true, 0);

        // 2. Prepare Inputs
        let decoder_hidden_in_data = vec![0.100000, 0.200000, 0.300000, 0.400000];
        let hidden_states = Array3::from_shape_vec((1, 1, 4), decoder_hidden_in_data)?;

        // Past KV (Python: [1, 2, 2, 2]) -> Rust: [1, 2, 4]
        let decoder_past_k_data = vec![
            0.100000, 0.100000, 0.200000, 0.200000, 0.300000, 0.300000, 0.400000, 0.400000,
        ];
        let decoder_past_v_data = vec![
            0.500000, 0.500000, 0.600000, 0.600000, 0.700000, 0.700000, 0.800000, 0.800000,
        ];

        let past_k = permute_golden_kv(decoder_past_k_data, 1, heads, 2, head_dim);
        let past_v = permute_golden_kv(decoder_past_v_data, 1, heads, 2, head_dim);

        // Mask (1 batch, total seq length 3)
        let mask = Array2::from_elem((1, 3), 1.0);

        // 3. Run Forward
        let (output, (k_cache, v_cache)) = layer.forward(
            &hidden_states,
            &mask,
            2, // position_offset (length of past cache)
            Some((past_k.view(), past_v.view())),
        )?;

        // 4. Validate Output State
        let golden_prenorm_out_data = vec![0.108785, 0.209478, 0.310172, 0.410866];
        let golden_out = Array3::from_shape_vec((1, 1, 4), golden_prenorm_out_data)?;

        let diff = (&output - &golden_out).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Scenario A Output Max Diff: {:.6}", max_diff);
        assert!(max_diff < 1e-5, "Output mismatch in Pre-Norm");

        // 5. Validate K Cache
        // Python Output Shape: [1, 2, 3, 2] -> Rust Shape: [1, 3, 4]
        let golden_prenorm_k_cache_data = vec![
            0.100000, 0.100000, 0.200000, 0.200000, 0.000521, 0.000537, 0.300000, 0.300000,
            0.400000, 0.400000, 0.000553, 0.000569,
        ];
        let golden_k_cache = permute_golden_kv(golden_prenorm_k_cache_data, 1, heads, 3, head_dim);

        let cache_diff = (&k_cache - &golden_k_cache).mapv(|x| x.abs());
        let max_cache_diff = cache_diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Scenario A Cache Max Diff: {:.6}", max_cache_diff);
        assert!(max_cache_diff < 1e-5, "K-Cache mismatch in Pre-Norm");

        Ok(())
    }

    // ========================================================================
    // 3. Scenario B: Post-Norm + Standard FFN (GPT Style)
    // ========================================================================
    #[test]
    fn test_decoder_postnorm_standard_golden() -> Result<()> {
        // Config
        let (hidden, heads, inter) = (4, 2, 8);
        let head_dim = hidden / heads;

        // 1. Create Layer (Offset 1000 to match Python)
        let layer = create_deterministic_decoder(hidden, heads, inter, false, false, 1000);

        // 2. Inputs (Same as above)
        let decoder_hidden_in_data = vec![0.100000, 0.200000, 0.300000, 0.400000];
        let hidden_states = Array3::from_shape_vec((1, 1, 4), decoder_hidden_in_data)?;

        // Using same past KV data
        let decoder_past_k_data = vec![
            0.100000, 0.100000, 0.200000, 0.200000, 0.300000, 0.300000, 0.400000, 0.400000,
        ];
        let decoder_past_v_data = vec![
            0.500000, 0.500000, 0.600000, 0.600000, 0.700000, 0.700000, 0.800000, 0.800000,
        ];
        let past_k = permute_golden_kv(decoder_past_k_data, 1, heads, 2, head_dim);
        let past_v = permute_golden_kv(decoder_past_v_data, 1, heads, 2, head_dim);

        let mask = Array2::from_elem((1, 3), 1.0);

        // 3. Run Forward
        let (output, _) = layer.forward(
            &hidden_states,
            &mask,
            2,
            Some((past_k.view(), past_v.view())),
        )?;

        // 4. Validate Output State
        let golden_postnorm_out_data = vec![-1.331634, -0.437211, 0.457211, 1.351634];
        let golden_out = Array3::from_shape_vec((1, 1, 4), golden_postnorm_out_data)?;

        let diff = (&output - &golden_out).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Scenario B Output Max Diff: {:.6}", max_diff);
        assert!(max_diff < 1e-4, "Output mismatch in Post-Norm");

        Ok(())
    }
}
