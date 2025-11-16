use crate::attention::MultiHeadAttention;

use crate::feedforward::FeedForward;
use crate::normalization::Normalization;
use crate::utils::linear_algebra::matmul_3d_2d;
use crate::rope::RoPE;
use anyhow::Result;
use ndarray::{Array2, Array3, Axis, s};
use std::sync::Arc;

/// Represents a single layer for a decoder-only transformer model (e.g., GPT-2).
///
/// This layer is specialized for causal self-attention and follows a specific
/// sub-layer order (e.g., pre-LayerNorm). It is designed to be stateless
/// with respect to the KV cache, making the data flow explicit and suitable for
/// high-performance backends like GPUs.

pub struct DecoderLayer {
    pub self_attn: MultiHeadAttention,
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
        position_offset: usize,
        past_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        if self.is_prenorm {
            self.forward_prenorm(hidden_states, attention_mask, position_offset, past_kv)
        } else {
            self.forward_postnorm(hidden_states, attention_mask, position_offset, past_kv)
        }
    }
    fn forward_prenorm(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        past_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        let residual = hidden_states.clone();

        // 1. Normalize the input. This is the input to the Q, K, and V projections.
        let ln1_out = self.self_attn_layer_norm.forward(hidden_states);

        let (attn_out, new_k, new_v) = self.self_attn.forward_with_cache(
            &ln1_out, // The query source is the normalized hidden state
            None,     // For self-attention, key_value source is the same
            Some(attention_mask),
            true, // is_causal is always true for a DecoderLayer
            past_kv,
            self.rope.as_deref(),
        )?;

        // 3. First residual connection.
        let attn_block_output = residual + attn_out;

        // 4. FFN block (this part is correct)
        let residual = attn_block_output.clone();
        let ln2_out = self.ffn_layer_norm.forward(&attn_block_output);
        let ffn_out = self.feedforward.forward(&ln2_out)?;
        let final_output = residual + ffn_out;

        // 5. Return the final output and the new K/V pair to be cached by the caller.
        Ok((final_output, (new_k, new_v)))
    }
    fn forward_postnorm(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        past_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        let residual = hidden_states.clone();
        let batch_size = hidden_states.shape()[0];
        let cache_len = position_offset; // Use the provided offset

        // === 1. Manually perform ALL preparation steps ===

        // 1a. Project Q, K, and V from the raw hidden_states
        let mut q_proj = matmul_3d_2d(hidden_states, &self.self_attn.q_weight);
        if !self.self_attn.q_bias.is_empty() {
            q_proj += &self.self_attn.q_bias;
        }
        let (new_k, new_v) = self.self_attn.project_kv(hidden_states);

        // 1b. Apply RoPE to the new Q and K
        let (rotated_q, rotated_k) = if let Some(r) = self.rope.as_deref() {
            r.apply_3d(
                &q_proj,
                &new_k,
                self.self_attn.num_heads,
                self.self_attn.num_kv_heads,
                cache_len,
            )?
        } else {
            (q_proj, new_k.clone())
        };

        // 1c. Concatenate with cache (YOUR IMPLEMENTATION)
        let (full_k, full_v) = if let Some((cached_k, cached_v)) = past_kv {
            let new_len = rotated_k.shape()[1];
            let full_len = cache_len + new_len;
            let hidden_size = rotated_k.shape()[2];

            let mut temp_full_k = Array3::zeros((batch_size, full_len, hidden_size));
            temp_full_k
                .slice_mut(s![.., 0..cache_len, ..])
                .assign(&cached_k);
            temp_full_k
                .slice_mut(s![.., cache_len..full_len, ..])
                .assign(&rotated_k);

            let mut temp_full_v = Array3::zeros((batch_size, full_len, hidden_size));
            temp_full_v
                .slice_mut(s![.., 0..cache_len, ..])
                .assign(&cached_v);
            temp_full_v
                .slice_mut(s![.., cache_len..full_len, ..])
                .assign(&new_v);

            (temp_full_k, temp_full_v)
        } else {
            (rotated_k.clone(), new_v.clone())
        };

        // === 2. Call the core attend function ===
        let context = self.self_attn.attend(
            &rotated_q,
            &full_k,
            &full_v,
            Some(attention_mask),
            true,
            cache_len,
        )?;

        // === 3. Perform the final output projection ===
        let attn_out = if !self.self_attn.output_bias.is_empty() {
            matmul_3d_2d(&context, &self.self_attn.output_weight) + &self.self_attn.output_bias
        } else {
            matmul_3d_2d(&context, &self.self_attn.output_weight)
        };

        // === 4. Add & Norm ===
        let hidden = residual + attn_out;
        let hidden = self.self_attn_layer_norm.forward(&hidden);

        // FFN block
        let residual = hidden.clone();
        let ffn_out = self.feedforward.forward(&hidden)?;
        let hidden = residual + ffn_out;
        let output = self.ffn_layer_norm.forward(&hidden);

        Ok((output, (rotated_k, new_v)))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::feedforward::{FeedForward, StdFeedForward, SwiGluFeedForward};
    use crate::normalization::{LayerNorm, RMSNorm};
    use crate::rope::RoPE;
    use ndarray::{Array1, Array2, Array3, s};
    use std::sync::Arc;

    #[test]
    fn test_decoder_layer_with_rope_and_gqa() {
        let hidden_size = 2048;
        let num_heads = 32;
        let num_kv_heads = 8;
        let head_dim = hidden_size / num_heads;
        let intermediate_size = 8192;

        // Create layer with GQA
        let q_weight = Array2::eye(hidden_size);
        let k_weight = Array2::eye(hidden_size).slice(s![.., 0..512]).to_owned();
        let v_weight = Array2::eye(hidden_size).slice(s![.., 0..512]).to_owned();
        let o_weight = Array2::eye(hidden_size);

        let attention = MultiHeadAttention::new(
            hidden_size,
            num_heads,
            q_weight,
            Array1::zeros(0),
            k_weight,
            Array1::zeros(0),
            v_weight,
            Array1::zeros(0),
            o_weight,
            Array1::zeros(0),
            Some(num_kv_heads),
        );

        let rope = Arc::new(RoPE::new(head_dim, 128, 10000.0));

        // Create feed-forward
        let gate_weight = Array2::from_shape_fn((hidden_size, intermediate_size), |(i, j)| {
            if i == j { 1.0 } else { 0.0 }
        });
        let up_weight = Array2::from_shape_fn((hidden_size, intermediate_size), |(i, j)| {
            if i == j { 1.0 } else { 0.0 }
        });
        let down_weight = Array2::from_shape_fn((intermediate_size, hidden_size), |(i, j)| {
            if i == j { 1.0 } else { 0.0 }
        });

        let feedforward =
            FeedForward::SwiGLU(SwiGluFeedForward::new(gate_weight, up_weight, down_weight));

        let norm1 = Normalization::RMSNorm(RMSNorm::new(Array1::ones(hidden_size), 1e-5));
        let norm2 = Normalization::RMSNorm(RMSNorm::new(Array1::ones(hidden_size), 1e-5));

        let layer = DecoderLayer {
            self_attn: attention,
            self_attn_layer_norm: norm1,
            feedforward,
            ffn_layer_norm: norm2,
            is_prenorm: true,
            rope: Some(rope),
        };

        // Test 1: Prefill (no cache)
        let input = Array3::ones((1, 10, hidden_size));
        let mask = Array2::ones((1, 10));

        let result = layer.forward(&input, &mask, 0, None);
        assert!(result.is_ok(), "Prefill should succeed");

        let (output, (k, v)) = result.unwrap();
        assert_eq!(output.shape(), &[1, 10, hidden_size]);
        assert_eq!(k.shape(), &[1, 10, 512]); // GQA: 512 not 2048
        assert_eq!(v.shape(), &[1, 10, 512]);

        // Test 2: Generate (with cache)
        let input2 = Array3::ones((1, 1, hidden_size));
        let mask2 = Array2::ones((1, 11));

        let result2 = layer.forward(&input2, &mask2, 10, Some((k.view(), v.view())));
        assert!(result2.is_ok(), "Generation should succeed");

        let (output2, (k2, v2)) = result2.unwrap();
        assert_eq!(output2.shape(), &[1, 1, hidden_size]);
        assert_eq!(k2.shape(), &[1, 1, 512]);
        assert_eq!(v2.shape(), &[1, 1, 512]);

        println!("✓ Decoder layer integration test passed");
    }
}
