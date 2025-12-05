use crate::attention::MultiHeadAttention;
use crate::decoder_attention::DecoderAttention;
use crate::feedforward::FeedForward;
use crate::normalization::Normalization;
use crate::rope::RoPE;
use crate::utils::linear_algebra::matmul_3d_2d;
use anyhow::Result;
use ndarray::{Array2, Array3, Axis, s};
use std::sync::Arc;

/// Represents a single layer for a decoder-only transformer model (e.g., GPT-2).
///
/// This layer is specialized for causal self-attention and follows a specific
/// sub-layer order (e.g., pre-LayerNorm). It is designed to be stateless
/// with respect to the KV cache, making the data flow explicit and suitable for
/// high-performance backends like GPUs.

pub enum CpuAttention {
    Legacy(MultiHeadAttention), // GPT-2
    Modern(DecoderAttention),   // Llama/Phi
}

impl CpuAttention {
    /// Unified forward method that dispatches to the correct implementation.
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: Option<&Array2<f32>>,
        cached_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
        rope: Option<&RoPE>,
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        match self {
            CpuAttention::Legacy(legacy) => {
                // Map the legacy signature to the unified signature
                // Legacy often doesn't take RoPE explicitly (applied inside or manually),
                // but if your updated MultiHeadAttention takes it, pass it.
                // Assuming legacy.forward_with_cache returns (out, k, v)

                // Note: Legacy implementation might expect query/key_value/mask args slightly differently.
                // Adapt as needed to match MultiHeadAttention's signature.
                legacy.forward_with_cache(
                    hidden_states, // query
                    None,          // key_value (None = self attn)
                    attention_mask,
                    true, // is_causal
                    cached_kv,
                    rope, // Pass RoPE if legacy supports it now
                )
            }
            CpuAttention::Modern(modern) => {
                // Modern signature matches directly
                modern.forward(hidden_states, attention_mask, cached_kv, rope)
            }
        }
    }
}

pub struct DecoderLayer {
    pub self_attn: CpuAttention,
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
        let t0 = std::time::Instant::now();

        let residual = hidden_states.clone();
        let ln1_out = self.self_attn_layer_norm.forward(hidden_states);
        log::info!("  LayerNorm1: {:?}", t0.elapsed());

        let t1 = std::time::Instant::now();
        // --- DISPATCH LOGIC ---
        let (attn_out, new_k, new_v) = match &self.self_attn {
            CpuAttention::Legacy(legacy_attn) => {
                // Keep your old logic for MultiHeadAttention
                legacy_attn.forward_with_cache(
                    &ln1_out,
                    None,
                    Some(attention_mask),
                    true,
                    past_kv,
                    self.rope.as_deref(),
                )?
            }
            CpuAttention::Modern(modern_attn) => {
                // New logic for DecoderAttention (LinearLayer / BF16)
                // Note: Modern attention usually handles RoPE internally or accepts it
                modern_attn.forward(
                    &ln1_out,
                    Some(attention_mask),
                    past_kv,
                    self.rope.as_deref(),
                )?
            }
        };
        // --- END DISPATCH ---
        log::info!("  Attention: {:?}", t1.elapsed());

        let t2 = std::time::Instant::now();
        let attn_block_output = residual + attn_out;
        let residual = attn_block_output.clone();
        let ln2_out = self.ffn_layer_norm.forward(&attn_block_output);
        log::info!("  LayerNorm2: {:?}", t2.elapsed());

        let t3 = std::time::Instant::now();
        let ffn_out = self.feedforward.forward(&ln2_out)?;
        log::info!("  FFN: {:?}", t3.elapsed());

        let final_output = residual + ffn_out;

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
        let legacy_attn = match &self.self_attn {
            CpuAttention::Legacy(a) => a,
            CpuAttention::Modern(_) => {
                return Err(anyhow::anyhow!(
                    "Modern BF16 Attention not supported in Post-Norm (Legacy) path yet."
                ));
            }
        };

        // 1a. Project Q, K, and V from the raw hidden_states
        let mut q_proj = matmul_3d_2d(hidden_states, &legacy_attn.q_weight);
        if !legacy_attn.q_bias.is_empty() {
            q_proj += &legacy_attn.q_bias;
        }
        let (new_k, new_v) = legacy_attn.project_kv(hidden_states);

        // 1b. Apply RoPE to the new Q and K
        let (rotated_q, rotated_k) = if let Some(r) = self.rope.as_deref() {
            r.apply_3d(
                &q_proj,
                &new_k,
                legacy_attn.num_heads,
                legacy_attn.num_kv_heads,
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
        let context = legacy_attn.attend(
            &rotated_q,
            &full_k,
            &full_v,
            Some(attention_mask),
            true,
            cache_len,
        )?;

        // === 3. Perform the final output projection ===
        let attn_out = if !legacy_attn.output_bias.is_empty() {
            matmul_3d_2d(&context, &legacy_attn.output_weight) + &legacy_attn.output_bias
        } else {
            matmul_3d_2d(&context, &legacy_attn.output_weight)
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
    use crate::linear_layer::LinearLayer;
    use crate::normalization::{LayerNorm, RMSNorm};
    use crate::rope::RoPE;
    use ndarray::{Array1, Array2, Array3, s};
    use std::sync::Arc;

    #[test]
    fn test_decoder_layer_with_rope_and_gqa() {
        let hidden_size = 2048;
        let num_heads = 32;
        let num_kv_heads = 8;
        let head_dim = hidden_size / num_heads; // 64
        let kv_dim = num_kv_heads * head_dim; // 512
        let intermediate_size = 8192;

        // Attention weights: [out_features, in_features]
        let q_weight = LinearLayer::from(Array2::<f32>::zeros((hidden_size, hidden_size)));
        let k_weight = LinearLayer::from(Array2::<f32>::zeros((kv_dim, hidden_size)));
        let v_weight = LinearLayer::from(Array2::<f32>::zeros((kv_dim, hidden_size)));
        let o_weight = LinearLayer::from(Array2::<f32>::zeros((hidden_size, hidden_size)));

        let attention = DecoderAttention::new(
            hidden_size,
            num_heads,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            Some(num_kv_heads),
        );

        let rope = Arc::new(RoPE::new(head_dim, 128, 10000.0));

        // FFN weights: [out_features, in_features]
        let gate_weight = LinearLayer::from(Array2::<f32>::zeros((intermediate_size, hidden_size)));
        let up_weight = LinearLayer::from(Array2::<f32>::zeros((intermediate_size, hidden_size)));
        let down_weight = LinearLayer::from(Array2::<f32>::zeros((hidden_size, intermediate_size)));

        let feedforward =
            FeedForward::SwiGLU(SwiGluFeedForward::new(gate_weight, up_weight, down_weight));

        let norm1 = Normalization::RMSNorm(RMSNorm::new(Array1::ones(hidden_size), 1e-5));
        let norm2 = Normalization::RMSNorm(RMSNorm::new(Array1::ones(hidden_size), 1e-5));

        let layer = DecoderLayer {
            self_attn: CpuAttention::Modern(attention),
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
        assert_eq!(k.shape(), &[1, 10, kv_dim]);
        assert_eq!(v.shape(), &[1, 10, kv_dim]);

        // Test 2: Generate (with cache)
        let input2 = Array3::ones((1, 1, hidden_size));
        let mask2 = Array2::ones((1, 11));

        let result2 = layer.forward(&input2, &mask2, 10, Some((k.view(), v.view())));
        assert!(result2.is_ok(), "Generation should succeed");

        let (output2, (k2, v2)) = result2.unwrap();
        assert_eq!(output2.shape(), &[1, 1, hidden_size]);
        assert_eq!(k2.shape(), &[1, 1, kv_dim]);
        assert_eq!(v2.shape(), &[1, 1, kv_dim]);

        println!("✓ Decoder layer integration test passed");
    }
}


