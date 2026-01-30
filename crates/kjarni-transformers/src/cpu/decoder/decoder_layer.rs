use std::sync::Arc;

use anyhow::Result;
use ndarray::{s, Array2, Array3};

use crate::cpu::decoder::DecoderAttention;
use crate::cpu::normalization::Normalization;
use crate::feedforward::FeedForward;
use crate::rope::RoPE;

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

        let mut k_cache = Array3::<f32>::zeros((batch, total_len, kv_dim));
        let mut v_cache = Array3::<f32>::zeros((batch, total_len, kv_dim));

        if let Some((past_k, past_v)) = past_kv {
            k_cache.slice_mut(s![.., ..cache_len, ..]).assign(&past_k);
            v_cache.slice_mut(s![.., ..cache_len, ..]).assign(&past_v);
        }

        let residual = hidden_states.clone();

        let ln1_out = self.self_attn_layer_norm.forward(hidden_states);

        let attn_out = self.self_attn.forward(
            &ln1_out,
            Some(attention_mask),
            k_cache.view_mut(),
            v_cache.view_mut(),
            cache_len,
            self.rope.as_deref(),
        )?;

        let attn_block_output = residual + attn_out;

        let residual = attn_block_output.clone();

        let ln2_out = self.ffn_layer_norm.forward(&attn_block_output);

        let ffn_out = self.feedforward.forward(&ln2_out)?;

        let final_output = residual + ffn_out;

        Ok((final_output, (k_cache, v_cache)))
    }

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

        let mut k_cache = Array3::<f32>::zeros((batch, total_len, kv_dim));
        let mut v_cache = Array3::<f32>::zeros((batch, total_len, kv_dim));

        if let Some((past_k, past_v)) = past_kv {
            k_cache.slice_mut(s![.., ..cache_len, ..]).assign(&past_k);
            v_cache.slice_mut(s![.., ..cache_len, ..]).assign(&past_v);
        }

        let residual = hidden_states.clone();

        let attn_out = self.self_attn.forward(
            hidden_states,
            Some(attention_mask),
            k_cache.view_mut(),
            v_cache.view_mut(),
            cache_len,
            self.rope.as_deref(),
        )?;

        let hidden = residual + attn_out;
        let hidden = self.self_attn_layer_norm.forward(&hidden);

        let residual = hidden.clone();
        let ffn_out = self.feedforward.forward(&hidden)?;

        let hidden = residual + ffn_out;
        let output = self.ffn_layer_norm.forward(&hidden);

        Ok((output, (k_cache, v_cache)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        activations::Activation,
        cpu::normalization::{LayerNorm, RMSNorm},
        feedforward::{FeedForward, LegacyFeedForward, StdFeedForward, SwiGluFeedForward},
        linear_layer::LinearLayer,
    };
    use ndarray::{Array1, Array4};

    fn create_test_layer_prenorm() -> DecoderLayer {
        let hidden_size = 64;
        let num_heads = 4;
        let num_kv_heads = 4;

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

        let intermediate_size = hidden_size * 4;
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

        let intermediate_size = hidden_size * 4;
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

        let hidden_states =
            Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
                (k as f32) * 0.01
            });

        let attention_mask = Array2::ones((batch_size, seq_len));

        let (output, (new_k, new_v)) = layer.forward(&hidden_states, &attention_mask, 0, None)?;

        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_k.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_v.shape(), &[batch_size, seq_len, hidden_size]);

        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 1e-6, "output should not be all zeros");

        Ok(())
    }

    #[test]
    fn test_decoder_layer_postnorm_forward() -> Result<()> {
        let layer = create_test_layer_postnorm();
        let batch_size = 1;
        let seq_len = 4;
        let hidden_size = 64;

        let hidden_states =
            Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
                (k as f32) * 0.01
            });

        let attention_mask = Array2::ones((batch_size, seq_len));

        let (output, (new_k, new_v)) = layer.forward(&hidden_states, &attention_mask, 0, None)?;

        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_k.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_v.shape(), &[batch_size, seq_len, hidden_size]);

        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 1e-6, "output should not be all zeros");

        Ok(())
    }

    #[test]
    fn test_decoder_layer_with_cache() -> Result<()> {
        let layer = create_test_layer_prenorm();
        let batch_size = 1;
        let seq_len = 1;
        let hidden_size = 64;

        let hidden_states =
            Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
                (k as f32) * 0.01
            });

        let attention_mask = Array2::ones((batch_size, seq_len));

        let past_len = 3;
        let past_k = Array3::from_shape_fn((batch_size, past_len, hidden_size), |(_, _, k)| {
            (k as f32) * 0.005
        });
        let past_v = Array3::from_shape_fn((batch_size, past_len, hidden_size), |(_, _, k)| {
            (k as f32) * 0.005
        });

        let (output, (new_k, new_v)) = layer.forward(
            &hidden_states,
            &attention_mask,
            0,
            Some((past_k.view(), past_v.view())),
        )?;

        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);

        let total_len = past_len + seq_len;
        assert_eq!(new_k.shape(), &[batch_size, total_len, hidden_size]);
        assert_eq!(new_v.shape(), &[batch_size, total_len, hidden_size]);

        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 1e-6, "output should not be all zeros");

        Ok(())
    }

    #[test]
    fn test_decoder_layer_with_rope() -> Result<()> {
        let mut layer = create_test_layer_prenorm();
        let hidden_size = 64;
        let num_heads = 4;
        let head_dim = hidden_size / num_heads;

        layer.rope = Some(Arc::new(RoPE::new(head_dim, 128, 10000.0)));

        let batch_size = 1;
        let seq_len = 4;

        let hidden_states =
            Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
                (k as f32) * 0.01
            });

        let attention_mask = Array2::ones((batch_size, seq_len));

        let (output, (_new_k, _new_v)) = layer.forward(&hidden_states, &attention_mask, 0, None)?;

        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);

        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 1e-6, "output should not be all zeros");

        Ok(())
    }

    #[test]
    fn test_decoder_layer_prenorm_vs_postnorm() -> Result<()> {
        let prenorm_layer = create_test_layer_prenorm();
        let postnorm_layer = create_test_layer_postnorm();

        let batch_size = 1;
        let seq_len = 4;
        let hidden_size = 64;

        let hidden_states =
            Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
                (k as f32) * 0.01
            });
        let attention_mask = Array2::ones((batch_size, seq_len));

        let (prenorm_output, _) =
            prenorm_layer.forward(&hidden_states, &attention_mask, 0, None)?;
        let (postnorm_output, _) =
            postnorm_layer.forward(&hidden_states, &attention_mask, 0, None)?;

        assert_eq!(prenorm_output.shape(), postnorm_output.shape());

        let prenorm_sum: f32 = prenorm_output.iter().sum();
        let postnorm_sum: f32 = postnorm_output.iter().sum();

        assert!(prenorm_sum.abs() > 1e-6);
        assert!(postnorm_sum.abs() > 1e-6);

        Ok(())
    }

    fn get_weights(start: usize, rows: usize, cols: usize) -> (Array2<f32>, usize) {
        let size = rows * cols;
        let data: Vec<f32> = (start..start + size).map(|x| x as f32 * 0.0001).collect();
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

        let (qw, c) = get_weights(count, hidden, hidden);
        count = c;
        let (kw, c) = get_weights(count, head_dim * heads, hidden);
        count = c;
        let (vw, c) = get_weights(count, head_dim * heads, hidden);
        count = c;
        let (ow, c) = get_weights(count, hidden, hidden);
        count = c;

        let attn = DecoderAttention::new(
            hidden,
            heads,
            LinearLayer::new_f32(qw, None),
            LinearLayer::new_f32(kw, None),
            LinearLayer::new_f32(vw, None),
            LinearLayer::new_f32(ow, None),
            Some(heads),
        );

        let attn_norm = Normalization::LayerNorm(LayerNorm::new(
            Array1::ones(hidden),
            Array1::from_elem(hidden, bias_val),
            1e-5,
        ));

        let feedforward = if use_swiglu {
            let (gw, c) = get_weights(count, inter, hidden);
            count = c;
            let (uw, c) = get_weights(count, inter, hidden);
            count = c;
            let (dw, _c) = get_weights(count, hidden, inter);

            FeedForward::SwiGLU(SwiGluFeedForward::new(
                LinearLayer::new_f32(gw, None),
                LinearLayer::new_f32(uw, None),
                LinearLayer::new_f32(dw, None),
                Activation::SilU,
            ))
        } else {
            let (w1, c) = get_weights(count, inter, hidden);
            count = c;
            let b1 = Array1::from_elem(inter, bias_val);
            let (w2, _c) = get_weights(count, hidden, inter);
            let b2 = Array1::from_elem(hidden, bias_val);

            FeedForward::Standard(StdFeedForward::new(
                w1,
                b1,
                w2,
                b2,
                Activation::Gelu,
            ))
        };

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

    fn permute_golden_kv(
        data: Vec<f32>,
        batch: usize,
        heads: usize,
        seq: usize,
        head_dim: usize,
    ) -> Array3<f32> {
        let arr = Array4::from_shape_vec((batch, heads, seq, head_dim), data).unwrap();
        let permuted = arr.permuted_axes([0, 2, 1, 3]);
        permuted
            .as_standard_layout()
            .to_owned()
            .into_shape_with_order((batch, seq, heads * head_dim))
            .unwrap()
    }

    #[test]
    fn test_decoder_prenorm_swiglu_golden() -> Result<()> {
        let (hidden, heads, inter) = (4, 2, 8);
        let head_dim = hidden / heads;

        let layer = create_deterministic_decoder(hidden, heads, inter, true, true, 0);

        let decoder_hidden_in_data = vec![0.100000, 0.200000, 0.300000, 0.400000];
        let hidden_states = Array3::from_shape_vec((1, 1, 4), decoder_hidden_in_data)?;

        let decoder_past_k_data = vec![
            0.100000, 0.100000, 0.200000, 0.200000, 0.300000, 0.300000, 0.400000, 0.400000,
        ];
        let decoder_past_v_data = vec![
            0.500000, 0.500000, 0.600000, 0.600000, 0.700000, 0.700000, 0.800000, 0.800000,
        ];

        let past_k = permute_golden_kv(decoder_past_k_data, 1, heads, 2, head_dim);
        let past_v = permute_golden_kv(decoder_past_v_data, 1, heads, 2, head_dim);

        let mask = Array2::from_elem((1, 3), 1.0);

        let (output, (k_cache, _v_cache)) = layer.forward(
            &hidden_states,
            &mask,
            2,
            Some((past_k.view(), past_v.view())),
        )?;

        let golden_prenorm_out_data = vec![0.108785, 0.209478, 0.310172, 0.410866];
        let golden_out = Array3::from_shape_vec((1, 1, 4), golden_prenorm_out_data)?;

        let diff = (&output - &golden_out).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        assert!(max_diff < 1e-5, "output mismatch in pre-norm: {:.6}", max_diff);

        let golden_prenorm_k_cache_data = vec![
            0.100000, 0.100000, 0.200000, 0.200000, 0.000521, 0.000537, 0.300000, 0.300000,
            0.400000, 0.400000, 0.000553, 0.000569,
        ];
        let golden_k_cache = permute_golden_kv(golden_prenorm_k_cache_data, 1, heads, 3, head_dim);

        let cache_diff = (&k_cache - &golden_k_cache).mapv(|x| x.abs());
        let max_cache_diff = cache_diff.fold(0.0f32, |a, &b| a.max(b));

        assert!(max_cache_diff < 1e-5, "k-cache mismatch in pre-norm: {:.6}", max_cache_diff);

        Ok(())
    }

    #[test]
    fn test_decoder_postnorm_standard_golden() -> Result<()> {
        let (hidden, heads, inter) = (4, 2, 8);
        let head_dim = hidden / heads;

        let layer = create_deterministic_decoder(hidden, heads, inter, false, false, 1000);

        let decoder_hidden_in_data = vec![0.100000, 0.200000, 0.300000, 0.400000];
        let hidden_states = Array3::from_shape_vec((1, 1, 4), decoder_hidden_in_data)?;

        let decoder_past_k_data = vec![
            0.100000, 0.100000, 0.200000, 0.200000, 0.300000, 0.300000, 0.400000, 0.400000,
        ];
        let decoder_past_v_data = vec![
            0.500000, 0.500000, 0.600000, 0.600000, 0.700000, 0.700000, 0.800000, 0.800000,
        ];
        let past_k = permute_golden_kv(decoder_past_k_data, 1, heads, 2, head_dim);
        let past_v = permute_golden_kv(decoder_past_v_data, 1, heads, 2, head_dim);

        let mask = Array2::from_elem((1, 3), 1.0);

        let (output, _) = layer.forward(
            &hidden_states,
            &mask,
            2,
            Some((past_k.view(), past_v.view())),
        )?;

        let golden_postnorm_out_data = vec![-1.331634, -0.437211, 0.457211, 1.351634];
        let golden_out = Array3::from_shape_vec((1, 1, 4), golden_postnorm_out_data)?;

        let diff = (&output - &golden_out).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        assert!(max_diff < 1e-4, "output mismatch in post-norm: {:.6}", max_diff);

        Ok(())
    }
}