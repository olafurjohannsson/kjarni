//! CPU-based decoder layer for modern LLMs with RoPE and pre-normalization.

use anyhow::Result;
use ndarray::{Array2, Array3};
use std::sync::Arc;

use crate::{
    Normalization, cpu::decoder::DecoderAttention, feedforward::SwiGluFeedForward, rope::RoPE
};

/// A complete transformer decoder layer with RoPE, pre-normalization, and SwiGLU.
pub struct CpuRoPEDecoderLayer {
    pub attention: DecoderAttention,
    pub feed_forward: SwiGluFeedForward,
    pub attention_norm: Normalization,
    pub ffn_norm: Normalization,
    pub rope: Arc<RoPE>,
}

impl CpuRoPEDecoderLayer {
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        k_cache: ndarray::ArrayViewMut3<f32>,
        v_cache: ndarray::ArrayViewMut3<f32>,
    ) -> Result<Array3<f32>> {
        let norm_1 = self.attention_norm.forward(hidden_states);
        let attn_out = self.attention.forward(
            &norm_1,
            Some(attention_mask),
            k_cache,
            v_cache,
            position_offset,
            Some(&self.rope),
        )?;
        let residual_1 = hidden_states + &attn_out;
        let norm_2 = self.ffn_norm.forward(&residual_1);
        let ffn_out = self.feed_forward.forward(&norm_2)?;
        let output = residual_1 + ffn_out;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::{linear_layer::LinearLayer, cpu::normalization::RMSNorm};
    use ndarray::{Array1, Array2};

    fn create_test_layer() -> CpuRoPEDecoderLayer {
        let hidden_size = 64;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = hidden_size / num_heads; // 16
        let intermediate_size = 128;
        let kv_dim = num_kv_heads * head_dim;

        let q_weight =
            Array2::from_shape_fn(
                (hidden_size, hidden_size),
                |(i, j)| {
                    if i == j { 1.0 } else { 0.0 }
                },
            );
        let k_weight = Array2::from_shape_fn((kv_dim, hidden_size), |(i, j)| {
            if i < hidden_size && i == j { 1.0 } else { 0.0 }
        });
        let v_weight = Array2::from_shape_fn((kv_dim, hidden_size), |(i, j)| {
            if i < hidden_size && i == j { 1.0 } else { 0.0 }
        });
        let o_weight =
            Array2::from_shape_fn(
                (hidden_size, hidden_size),
                |(i, j)| {
                    if i == j { 1.0 } else { 0.0 }
                },
            );

        let q_proj = LinearLayer::new_f32(q_weight, None);
        let k_proj = LinearLayer::new_f32(k_weight, None);
        let v_proj = LinearLayer::new_f32(v_weight, None);
        let o_proj = LinearLayer::new_f32(o_weight, None);

        let attention = DecoderAttention::new(
            hidden_size,
            num_heads,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            Some(num_kv_heads),
        );

        // Create SwiGLU feedforward
        let gate_weight = Array2::from_shape_fn((intermediate_size, hidden_size), |(i, j)| {
            if i == j { 1.0 } else { 0.0 }
        });
        let up_weight = gate_weight.clone();
        let down_weight = Array2::from_shape_fn((hidden_size, intermediate_size), |(i, j)| {
            if i == j { 1.0 } else { 0.0 }
        });

        let gate_proj = LinearLayer::new_f32(gate_weight, None);
        let up_proj = LinearLayer::new_f32(up_weight, None);
        let down_proj = LinearLayer::new_f32(down_weight, None);
        let feed_forward = SwiGluFeedForward::new(gate_proj, up_proj, down_proj, Activation::SilU);
        let norm_weight = Array1::ones(hidden_size);
        let attention_norm = Normalization::RMSNorm(RMSNorm::new(norm_weight.clone(), 1e-5));
        let ffn_norm = Normalization::RMSNorm(RMSNorm::new(norm_weight, 1e-5));
        let rope = Arc::new(RoPE::new(head_dim, 128, 10000.0));
        CpuRoPEDecoderLayer {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
            rope,
        }
    }

    #[test]
    fn test_rope_decoder_layer_forward_new() -> Result<()> {
        let layer = create_test_layer();
        let batch_size = 1;
        let seq_len = 4;
        let hidden_size = 64;
        let hidden_states =
            Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
                (k as f32) * 0.01
            });
        let attention_mask = Array2::ones((batch_size, seq_len));
        let kv_dim = layer.attention.num_kv_heads * layer.attention.head_dim;
        let mut k_cache = Array3::<f32>::zeros((batch_size, seq_len, kv_dim));
        let mut v_cache = Array3::<f32>::zeros((batch_size, seq_len, kv_dim));
        let output = layer.forward(
            &hidden_states,
            &attention_mask,
            0,
            k_cache.view_mut(),
            v_cache.view_mut(),
        )?;
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 1e-6, "Output should not be all zeros");

        Ok(())
    }
}
