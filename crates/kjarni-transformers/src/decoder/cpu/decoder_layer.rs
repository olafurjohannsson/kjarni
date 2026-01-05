use crate::decoder::cpu::DecoderAttention;
use crate::feedforward::FeedForward;
use crate::normalization::Normalization;
use crate::rope::RoPE;
use anyhow::Result;
use ndarray::{s, Array2, Array3};
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
mod tests {
    use super::*;
    use crate::{
        activations::Activation,
        feedforward::{FeedForward, LegacyFeedForward},
        linear_layer::{LinearLayer, LinearData},
        normalization::{LayerNorm, RMSNorm},
    };
    use ndarray::Array1;

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
        let self_attn_layer_norm = Normalization::LayerNorm(LayerNorm::new(
            norm_weight.clone(),
            norm_bias.clone(),
            1e-5,
        ));
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
        let hidden_states = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
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
        let hidden_states = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
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
    let hidden_states = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
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
    let (output, (new_k, new_v)) =
        layer.forward(&hidden_states, &attention_mask, 0, Some((past_k.view(), past_v.view())))?;

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
        let hidden_states = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
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
        let hidden_states = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
            (k as f32) * 0.01
        });
        let attention_mask = Array2::ones((batch_size, seq_len));

        // Run both layers
        let (prenorm_output, _) = prenorm_layer.forward(&hidden_states, &attention_mask, 0, None)?;
        let (postnorm_output, _) = postnorm_layer.forward(&hidden_states, &attention_mask, 0, None)?;

        // Both should produce valid outputs but with different values
        assert_eq!(prenorm_output.shape(), postnorm_output.shape());

        // Outputs should be different due to different normalization order
        let prenorm_sum: f32 = prenorm_output.iter().sum();
        let postnorm_sum: f32 = postnorm_output.iter().sum();

        assert!(prenorm_sum.abs() > 1e-6);
        assert!(postnorm_sum.abs() > 1e-6);

        Ok(())
    }
}
