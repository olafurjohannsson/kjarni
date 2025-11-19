//! Multi-head attention implementation with KV caching support

use crate::activations::softmax;
use crate::rope::RoPE;
use crate::utils::MASK_VALUE;
use crate::utils::linear_algebra::{matmul_3d_2d, matmul_4d};
use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, Array3, Array4, Axis, Zip, s};

/// Multi-head attention mechanism
pub struct MultiHeadAttention {
    // Weights are stored as [in_features, out_features] for efficient matmul
    pub q_weight: Array2<f32>,
    pub q_bias: Array1<f32>,
    pub k_weight: Array2<f32>,
    pub k_bias: Array1<f32>,
    pub v_weight: Array2<f32>,
    pub v_bias: Array1<f32>,
    pub output_weight: Array2<f32>,
    pub output_bias: Array1<f32>,

    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub scale_factor: f32,
}

impl MultiHeadAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        query_weight: Array2<f32>,
        query_bias: Array1<f32>,
        key_weight: Array2<f32>,
        key_bias: Array1<f32>,
        value_weight: Array2<f32>,
        value_bias: Array1<f32>,
        output_weight: Array2<f32>,
        output_bias: Array1<f32>,
        num_kv_heads: Option<usize>,
    ) -> Self {
        let num_kv_heads = num_kv_heads.unwrap_or(num_heads);
        let head_dim = hidden_size / num_heads;
        let scale_factor = 1.0 / (head_dim as f32).sqrt();

        Self {
            q_weight: query_weight,
            q_bias: query_bias,
            k_weight: key_weight,
            k_bias: key_bias,
            v_weight: value_weight,
            v_bias: value_bias,
            output_weight: output_weight,
            output_bias: output_bias,
            num_heads,
            head_dim,
            scale_factor,
            num_kv_heads,
        }
    }

    pub fn project_kv(&self, key_value_source: &Array3<f32>) -> (Array3<f32>, Array3<f32>) {
        // K projection
        let new_k = if self.k_bias.len() > 0 {
            matmul_3d_2d(key_value_source, &self.k_weight) + &self.k_bias
        } else {
            matmul_3d_2d(key_value_source, &self.k_weight)
        };

        // V projection
        let new_v = if self.v_bias.len() > 0 {
            matmul_3d_2d(key_value_source, &self.v_weight) + &self.v_bias
        } else {
            matmul_3d_2d(key_value_source, &self.v_weight)
        };
        (new_k, new_v)
    }

    pub fn attend(
        &self,
        q: &Array3<f32>,
        full_k_cache: &Array3<f32>,
        full_v_cache: &Array3<f32>,
        attention_mask: Option<&Array2<f32>>,
        is_causal: bool,
        position_offset: usize,
    ) -> Result<Array3<f32>> {
        let batch_size = q.shape()[0];
        let seq_len = q.shape()[1];
        let full_kv_len = full_k_cache.shape()[1];

        assert_eq!(
            full_k_cache.shape(),
            full_v_cache.shape(),
            "K and V cache shapes must match."
        );
        assert_eq!(
            q.shape()[0],
            full_k_cache.shape()[0],
            "Batch sizes of query and cache must match."
        );
        if let Some(mask) = attention_mask {
            assert_eq!(
                mask.shape()[0],
                batch_size,
                "Mask batch size does not match input."
            );
            assert_eq!(
                mask.shape()[1],
                full_kv_len,
                "Mask length does not match total key/value sequence length."
            );
        }

        // 2. Reshape Q, K, V to [batch, num_heads, seq, head_dim]
        let mut q_reshaped = q
            .to_owned()
            .into_shape_with_order((batch_size, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let mut k_reshaped = full_k_cache
            .to_owned()
            .into_shape_with_order((batch_size, full_kv_len, self.num_kv_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let mut v_reshaped = full_v_cache
            .to_owned()
            .into_shape_with_order((batch_size, full_kv_len, self.num_kv_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        if self.num_kv_heads != self.num_heads {
            let num_groups = self.num_heads / self.num_kv_heads;

            k_reshaped = repeat_kv(&k_reshaped, num_groups)?;
            v_reshaped = repeat_kv(&v_reshaped, num_groups)?;
        }

        // 4. Compute attention scores
        let q_contiguous = q_reshaped.as_standard_layout().to_owned();
        let k_transposed = k_reshaped.permuted_axes([0, 1, 3, 2]);
        let k_transposed_contiguous = k_transposed.as_standard_layout().to_owned();

        let mut scores = matmul_4d(&q_contiguous, &k_transposed_contiguous);
        scores *= self.scale_factor;

        // 5. Apply masks
        if let Some(mask) = attention_mask {
            scores = apply_padding_mask(scores, mask)?;
        }
        if is_causal {
            scores = apply_causal_mask(scores, position_offset)?;
        }

        // 6. Compute attention output
        let weights = softmax(&scores);
        let v_contiguous = v_reshaped.as_standard_layout().to_owned();
        let context = matmul_4d(&weights, &v_contiguous);

        // 7. Reshape back to [batch, seq, hidden]
        let context_reshaped = context
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .into_shape_with_order((batch_size, seq_len, self.num_heads * self.head_dim))?
            .to_owned();

        Ok(context_reshaped)
    }

    pub fn forward_with_cache(
        &self,
        query: &Array3<f32>,
        key_value: Option<&Array3<f32>>,
        attention_mask: Option<&Array2<f32>>,
        is_causal: bool,
        cached_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
        rope: Option<&RoPE>, // New parameter
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        let batch_size = query.shape()[0];
        let seq_len = query.shape()[1];

        // 1. Project new K and V
        let kv_source = key_value.unwrap_or(query);

        // project Q
        let mut q_proj = matmul_3d_2d(query, &self.q_weight);
        if self.q_bias.len() > 0 {
            q_proj = q_proj + &self.q_bias
        }

        let mut new_k = matmul_3d_2d(kv_source, &self.k_weight);
        if self.k_bias.len() > 0 {
            new_k = new_k + &self.k_bias
        }

        let mut new_v = matmul_3d_2d(kv_source, &self.v_weight);
        if self.v_bias.len() > 0 {
            new_v = new_v + &self.v_bias;
        }
        let cache_len = cached_kv.map_or(0, |(k, _)| k.shape()[1]);

        // Apply RoPE to the *new* Q and K projections ONLY, before caching.
        let (rotated_q, rotated_k) = if let Some(r) = rope {
            // We use apply_3d because we are still working with [batch, seq, hidden] tensors.
            // The apply_3d helper will handle the necessary reshaping.
            r.apply_3d(
                &q_proj,
                &new_k,
                self.num_heads,
                self.num_kv_heads,
                cache_len,
            )?
        } else {
            (q_proj, new_k)
        };
        let (full_k, full_v) = if let Some((cached_k, cached_v)) = cached_kv {
            let new_len = rotated_k.shape()[1];
            let full_len = cache_len + new_len;
            let hidden_size = rotated_k.shape()[2];

            let mut temp_full_k = Array3::zeros((batch_size, full_len, hidden_size));
            let mut temp_full_v = Array3::zeros((batch_size, full_len, hidden_size));

            temp_full_k
                .slice_mut(s![.., 0..cache_len, ..])
                .assign(&cached_k);
            temp_full_k
                .slice_mut(s![.., cache_len..full_len, ..])
                .assign(&rotated_k);

            temp_full_v
                .slice_mut(s![.., 0..cache_len, ..])
                .assign(&cached_v);
            temp_full_v
                .slice_mut(s![.., cache_len..full_len, ..])
                .assign(&new_v); // Use original new_v

            (temp_full_k, temp_full_v)
        } else {
            (rotated_k.clone(), new_v.clone())
        };

        // 3. Compute attention with proper position offset for RoPE
        let context_reshaped = self.attend(
            &rotated_q,
            &full_k,
            &full_v,
            attention_mask,
            is_causal,
            cache_len, // This is the position_offset for RoPE
        )?;

        // 8. Output projection
        let output = if self.output_bias.len() > 0 {
            matmul_3d_2d(&context_reshaped, &self.output_weight) + &self.output_bias
        } else {
            matmul_3d_2d(&context_reshaped, &self.output_weight)
        };

        Ok((output, rotated_k, new_v))
    }
}

/// Repeat KV heads for Grouped Query Attention
/// [batch, num_kv_heads, seq, head_dim] → [batch, num_heads, seq, head_dim]
fn repeat_kv(kv: &Array4<f32>, num_groups: usize) -> Result<Array4<f32>> {
    let (batch, num_kv_heads, seq_len, head_dim) = kv.dim();
    let num_heads = num_kv_heads * num_groups;

    let mut repeated = Array4::zeros((batch, num_heads, seq_len, head_dim));

    for b in 0..batch {
        for kv_head in 0..num_kv_heads {
            // Repeat this KV head num_groups times
            for g in 0..num_groups {
                let q_head = kv_head * num_groups + g;
                repeated
                    .slice_mut(s![b, q_head, .., ..])
                    .assign(&kv.slice(s![b, kv_head, .., ..]));
            }
        }
    }
    // let mut repeated = Array4::zeros((batch, num_heads, seq_len, head_dim));
    //
    // for b in 0..batch {
    //     for kv_head in 0..num_kv_heads {
    //         // Repeat this KV head num_groups times
    //         for g in 0..num_groups {
    //             let q_head = kv_head * num_groups + g;
    //             repeated
    //                 .slice_mut(s![b, q_head, .., ..])
    //                 .assign(&kv.slice(s![b, kv_head, .., ..]));
    //         }
    //     }
    // }

    Ok(repeated)
}
/// Apply padding mask to attention scores
///
/// Masks positions where mask[batch, key_pos] == 0
pub fn apply_padding_mask(mut scores: Array4<f32>, mask: &Array2<f32>) -> Result<Array4<f32>> {
    let (batch_size, num_heads, seq_q, seq_k) = scores.dim();

    if mask.shape()[0] != batch_size {
        return Err(anyhow!(
            "Mask batch size {} doesn't match scores batch size {}",
            mask.shape()[0],
            batch_size
        ));
    }

    if mask.shape()[1] != seq_k {
        return Err(anyhow!(
            "Mask sequence length {} doesn't match key sequence length {}",
            mask.shape()[1],
            seq_k
        ));
    }

    // Expand mask: [batch, seq_k] → [batch, 1, 1, seq_k]
    let mask_expanded = mask.view().insert_axis(Axis(1)).insert_axis(Axis(1));

    // Broadcast and apply
    if let Some(broadcast_mask) = mask_expanded.broadcast((batch_size, num_heads, seq_q, seq_k)) {
        Zip::from(&mut scores)
            .and(&broadcast_mask)
            .for_each(|s, &m| {
                if m == 0.0 {
                    *s = MASK_VALUE;
                }
            });
    }

    Ok(scores)
}
/// Apply causal mask to attention scores
///
/// Ensures position `i` can only attend to positions `0..=i` in the full sequence.
/// Takes into account the cache length for proper positioning.
pub fn apply_causal_mask(mut scores: Array4<f32>, cache_len: usize) -> Result<Array4<f32>> {
    let (_batch_size, _num_heads, seq_q, seq_k) = scores.dim();

    // For each query position...
    for i in 0..seq_q {
        // This is the absolute position of the query token in the full sequence.
        let query_pos = cache_len + i;

        // ...iterate through all key positions.
        for j in 0..seq_k {
            // A key is "in the future" if its absolute position `j` is
            // greater than the query's absolute position `query_pos`.
            if j > query_pos {
                // Mask this position for all batches and heads.
                scores.slice_mut(s![.., .., i, j]).fill(MASK_VALUE);
            }
        }
    }

    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    fn create_mock_attention(
        hidden: usize,
        heads: usize,
        kv_heads: Option<usize>,
    ) -> MultiHeadAttention {
        let kv_heads = kv_heads.unwrap_or(heads);
        let k_v_dim = (hidden / heads) * kv_heads;

        MultiHeadAttention::new(
            hidden,
            heads,
            Array2::eye(hidden),
            Array1::zeros(0),
            Array2::eye(hidden).slice(s![.., 0..k_v_dim]).to_owned(),
            Array1::zeros(0),
            Array2::eye(hidden).slice(s![.., 0..k_v_dim]).to_owned(),
            Array1::zeros(0),
            Array2::eye(hidden),
            Array1::zeros(0),
            Some(kv_heads),
        )
    }

    #[test]
    fn test_mha_forward_no_cache() -> Result<()> {
        let attn = create_mock_attention(64, 4, None);
        let input = Array3::ones((1, 10, 64));
        let mask = Array2::ones((1, 10));

        let (output, k, v) =
            attn.forward_with_cache(&input, None, Some(&mask), false, None, None)?;

        assert_eq!(output.shape(), &[1, 10, 64]);
        assert_eq!(k.shape(), &[1, 10, 64]);
        assert_eq!(v.shape(), &[1, 10, 64]);
        Ok(())
    }

    #[test]
    fn test_mha_with_cache() -> Result<()> {
        let attn = create_mock_attention(64, 4, None);
        let past_k = Array3::zeros((1, 5, 64));
        let past_v = Array3::zeros((1, 5, 64));
        let input = Array3::ones((1, 1, 64));
        let mask = Array2::ones((1, 6)); // Mask for full sequence length

        let (output, k, v) = attn.forward_with_cache(
            &input,
            None,
            Some(&mask),
            true,
            Some((past_k.view(), past_v.view())),
            None,
        )?;

        assert_eq!(output.shape(), &[1, 1, 64]);
        assert_eq!(k.shape(), &[1, 1, 64]); // Returns only the NEW key
        assert_eq!(v.shape(), &[1, 1, 64]); // Returns only the NEW value
        Ok(())
    }

    #[test]
    fn test_gqa_with_rope() -> Result<()> {
        let attn = create_mock_attention(64, 4, Some(2)); // 4 Q heads, 2 KV heads
        let rope = RoPE::new(16, 128, 10000.0);
        let input = Array3::ones((1, 10, 64));
        let mask = Array2::ones((1, 10));

        let (output, k, v) =
            attn.forward_with_cache(&input, None, Some(&mask), true, None, Some(&rope))?;

        assert_eq!(output.shape(), &[1, 10, 64]);
        // K and V have reduced dimension due to GQA
        assert_eq!(k.shape(), &[1, 10, 32]);
        assert_eq!(v.shape(), &[1, 10, 32]);
        Ok(())
    }

    #[test]
    fn test_attention_with_cache() {
        let hidden_size = 64;
        let num_heads = 4;
        let batch_size = 1;
        let seq_len = 1; // Single token (incremental decoding)
        let cache_len = 5; // 5 tokens already cached

        let q_weight = Array2::zeros((hidden_size, hidden_size));
        let q_bias = Array1::zeros(hidden_size);
        let k_weight = Array2::zeros((hidden_size, hidden_size));
        let k_bias = Array1::zeros(hidden_size);
        let v_weight = Array2::zeros((hidden_size, hidden_size));
        let v_bias = Array1::zeros(hidden_size);
        let output_weight = Array2::zeros((hidden_size, hidden_size));
        let output_bias = Array1::zeros(hidden_size);

        let attention = MultiHeadAttention::new(
            hidden_size,
            num_heads,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            output_weight,
            output_bias,
            None,
        );

        let input = Array3::zeros((batch_size, seq_len, hidden_size));
        let cached_k = Array3::zeros((batch_size, cache_len, hidden_size));
        let cached_v = Array3::zeros((batch_size, cache_len, hidden_size));
        let mask = Array2::ones((batch_size, cache_len + seq_len));

        let result = attention.forward_with_cache(
            &input,
            None,
            Some(&mask),
            true, // Causal
            Some((cached_k.view(), cached_v.view())),
            None,
        );

        assert!(result.is_ok());

        let (output, new_k, new_v) = result.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_k.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_v.shape(), &[batch_size, seq_len, hidden_size]);
    }
    #[test]
    fn test_attention_without_bias() {
        // Test LLaMA-style attention (no biases)
        let hidden_size = 64;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 10;

        // Create weights without biases (zero-length arrays)
        let q_weight = Array2::zeros((hidden_size, hidden_size));
        let q_bias = Array1::zeros(0); // No bias
        let k_weight = Array2::zeros((hidden_size, hidden_size));
        let k_bias = Array1::zeros(0);
        let v_weight = Array2::zeros((hidden_size, hidden_size));
        let v_bias = Array1::zeros(0);
        let output_weight = Array2::zeros((hidden_size, hidden_size));
        let output_bias = Array1::zeros(0);

        let attention = MultiHeadAttention::new(
            hidden_size,
            num_heads,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            output_weight,
            output_bias,
            None,
        );

        let input = Array3::zeros((batch_size, seq_len, hidden_size));
        let mask = Array2::ones((batch_size, seq_len));

        let q_proj = matmul_3d_2d(&input, &attention.q_weight); // Add bias if exists
        let (k_states, v_states) = attention.project_kv(&input);

        let result = attention.attend(&q_proj, &k_states, &v_states, Some(&mask), false, 0);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_attention_with_bias() {
        // Test GPT-2-style attention (with biases)
        let hidden_size = 64;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 10;

        // Create weights WITH biases
        let q_weight = Array2::zeros((hidden_size, hidden_size));
        let q_bias = Array1::zeros(hidden_size); // Has bias
        let k_weight = Array2::zeros((hidden_size, hidden_size));
        let k_bias = Array1::zeros(hidden_size);
        let v_weight = Array2::zeros((hidden_size, hidden_size));
        let v_bias = Array1::zeros(hidden_size);
        let output_weight = Array2::zeros((hidden_size, hidden_size));
        let output_bias = Array1::zeros(hidden_size);

        let attention = MultiHeadAttention::new(
            hidden_size,
            num_heads,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            output_weight,
            output_bias,
            None,
        );

        let input = Array3::zeros((batch_size, seq_len, hidden_size));
        let mask = Array2::ones((batch_size, seq_len));

        let q_proj = matmul_3d_2d(&input, &attention.q_weight); // Add bias if exists
        let (k_states, v_states) = attention.project_kv(&input);

        let result = attention.attend(&q_proj, &k_states, &v_states, Some(&mask), false, 0);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_attention_with_cache_and_rope() {
        let hidden_size = 64;
        let num_heads = 4;
        let head_dim = hidden_size / num_heads;
        let batch_size = 1;
        let seq_len = 1;
        let cache_len = 5;
        let max_seq_len = 128;

        let q_weight = Array2::eye(hidden_size);
        let q_bias = Array1::zeros(0);
        let k_weight = Array2::eye(hidden_size);
        let k_bias = Array1::zeros(0);
        let v_weight = Array2::eye(hidden_size);
        let v_bias = Array1::zeros(0);
        let output_weight = Array2::eye(hidden_size);
        let output_bias = Array1::zeros(0);

        let attention = MultiHeadAttention::new(
            hidden_size,
            num_heads,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            output_weight,
            output_bias,
            None,
        );

        let rope = RoPE::new(head_dim, max_seq_len, 10000.0);
        let input = Array3::ones((batch_size, seq_len, hidden_size));
        let cached_k = Array3::ones((batch_size, cache_len, hidden_size));
        let cached_v = Array3::ones((batch_size, cache_len, hidden_size));
        let mask = Array2::ones((batch_size, cache_len + seq_len));

        let result = attention.forward_with_cache(
            &input,
            None,
            Some(&mask),
            true,
            Some((cached_k.view(), cached_v.view())),
            Some(&rope),
        );

        assert!(result.is_ok());
        let (output, new_k, new_v) = result.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_k.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_v.shape(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_repeat_kv_gqa() {
        use ndarray::{Array4, s};

        // 8 KV heads, need to repeat 4x to get 32 Q heads
        let kv = Array4::from_shape_fn((2, 8, 10, 64), |(b, h, s, d)| {
            (b * 1000 + h * 100 + s * 10 + d) as f32
        });

        let repeated = repeat_kv(&kv, 4).unwrap();

        // Check shape
        assert_eq!(repeated.shape(), &[2, 32, 10, 64]);

        // Check that each KV head is repeated 4 times
        for kv_head in 0..8 {
            for group in 0..4 {
                let q_head = kv_head * 4 + group;

                // All values in this Q head should match the original KV head
                for b in 0..2 {
                    for s in 0..10 {
                        for d in 0..64 {
                            let original = kv[[b, kv_head, s, d]];
                            let repeated_val = repeated[[b, q_head, s, d]];
                            assert_eq!(
                                original, repeated_val,
                                "Mismatch at batch={}, kv_head={}, q_head={}, seq={}, dim={}",
                                b, kv_head, q_head, s, d
                            );
                        }
                    }
                }
            }
        }

        println!("✓ GQA repeat_kv test passed");
    }

    #[test]
    fn test_gqa_attention_shapes() {
        let hidden_size = 2048;
        let num_heads = 32;
        let num_kv_heads = 8;
        let batch_size = 1;
        let seq_len = 10;

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

        let input = Array3::ones((batch_size, seq_len, hidden_size));
        let mask = Array2::ones((batch_size, seq_len));

        // ✅ Use forward() which handles projection
        let result = attention.forward_with_cache(&input, None, Some(&mask), false, None, None);

        assert!(result.is_ok(), "GQA attention should not fail");
        let (output, _, _) = result.unwrap();

        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);

        println!("✓ GQA attention shapes test passed");
    }
    #[test]
    fn test_causal_mask_with_cache() {
        use ndarray::Array4;

        // Simulate: 5 tokens in cache, generating token 6
        let scores = Array4::ones((1, 4, 1, 6)); // [batch, heads, query_seq=1, key_seq=6]
        let cache_len = 5;

        let masked = apply_causal_mask(scores, cache_len).unwrap();

        // Query position is 5 (cache_len + 0)
        // Should be able to attend to positions 0-5 (all 6 keys)
        for key_pos in 0..6 {
            assert_ne!(
                masked[[0, 0, 0, key_pos]],
                MASK_VALUE,
                "Position {} should NOT be masked when query at position 5",
                key_pos
            );
        }

        println!("✓ Causal mask with cache test passed");
    }

    #[test]
    fn test_causal_mask_blocks_future() {
        use ndarray::Array4;

        // No cache, processing 3 tokens
        let scores = Array4::ones((1, 4, 3, 3));
        let cache_len = 0;

        let masked = apply_causal_mask(scores, cache_len).unwrap();

        // Query 0 can see: [0]
        // Query 1 can see: [0, 1]
        // Query 2 can see: [0, 1, 2]

        assert_ne!(masked[[0, 0, 0, 0]], MASK_VALUE); // Q0 sees K0
        assert_eq!(masked[[0, 0, 0, 1]], MASK_VALUE); // Q0 doesn't see K1
        assert_eq!(masked[[0, 0, 0, 2]], MASK_VALUE); // Q0 doesn't see K2

        assert_ne!(masked[[0, 0, 1, 0]], MASK_VALUE); // Q1 sees K0
        assert_ne!(masked[[0, 0, 1, 1]], MASK_VALUE); // Q1 sees K1
        assert_eq!(masked[[0, 0, 1, 2]], MASK_VALUE); // Q1 doesn't see K2

        println!("✓ Causal mask blocks future test passed");
    }
    #[test]
    fn test_rope_position_offset_correctness() {
        let hidden_size = 64;
        let num_heads = 4;
        let head_dim = hidden_size / num_heads;
        let batch_size = 1;
        let seq_len = 1;
        let max_seq_len = 128;

        // ✅ Use identity weights for clearer signal
        let q_weight = Array2::eye(hidden_size);
        let q_bias = Array1::zeros(0);
        let k_weight = Array2::eye(hidden_size);
        let k_bias = Array1::zeros(0);
        let v_weight = Array2::eye(hidden_size);
        let v_bias = Array1::zeros(0);
        let output_weight = Array2::eye(hidden_size);
        let output_bias = Array1::zeros(0);

        let attention = MultiHeadAttention::new(
            hidden_size,
            num_heads,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            output_weight,
            output_bias,
            None,
        );

        let rope = RoPE::new(head_dim, max_seq_len, 10000.0);

        // ✅ Use more varied input
        let input1 = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, i)| {
            if i % 2 == 0 { 1.0 } else { 0.5 }
        });

        let input2 = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, i)| {
            if i % 2 == 0 { 0.8 } else { 0.6 }
        });

        println!("\n=== RoPE Position Test ===");

        // First call: position 0
        let result1 = attention.forward_with_cache(&input1, None, None, true, None, Some(&rope));
        assert!(result1.is_ok());
        let (output1, k1, v1) = result1.unwrap();
        println!("Output1 mean: {}", output1.mean().unwrap());

        // Second call: position 1 (cached position is 1)
        let result2 = attention.forward_with_cache(
            &input2,
            None,
            None,
            true,
            Some((k1.view(), v1.view())),
            Some(&rope),
        );
        assert!(result2.is_ok());
        let (output2, _k2, _v2) = result2.unwrap();
        println!("Output2 mean: {}", output2.mean().unwrap());

        // Outputs should differ
        let diff: f32 = (&output1 - &output2).mapv(|x| x.abs()).sum();
        println!("Difference: {}", diff);

        // Also check that RoPE actually modified the keys
        let k1_mean = k1.mean().unwrap();
        let input1_mean = input1.mean().unwrap();
        println!("K1 mean: {}, Input1 mean: {}", k1_mean, input1_mean);

        assert!(
            diff > 1e-6,
            "Outputs should differ due to RoPE position encoding, diff={}",
            diff
        );
    }
    #[test]
    fn test_apply_causal_mask_no_offset() {
        let mut scores = Array4::<f32>::zeros((1, 1, 4, 4));
        let masked = apply_causal_mask(scores, 0).unwrap();

        assert_eq!(masked[[0, 0, 0, 1]], MASK_VALUE);
        assert_eq!(masked[[0, 0, 1, 2]], MASK_VALUE);
        assert_eq!(masked[[0, 0, 2, 3]], MASK_VALUE);
        assert_eq!(masked[[0, 0, 1, 1]], 0.0);
    }

    #[test]
    fn test_apply_causal_mask_with_offset() {
        // Simulates generating the 3rd token (index 2) when 2 tokens are in cache
        let mut scores = Array4::<f32>::zeros((1, 1, 1, 3)); // Query len=1, Key len=3
        let masked = apply_causal_mask(scores, 2).unwrap(); // cache_len = 2

        // Query at pos 2 can attend to keys at pos 0, 1, 2.
        assert_eq!(masked[[0, 0, 0, 0]], 0.0);
        assert_eq!(masked[[0, 0, 0, 1]], 0.0);
        assert_eq!(masked[[0, 0, 0, 2]], 0.0);
    }

    #[test]
    fn test_apply_padding_mask() {
        let mut scores = Array4::<f32>::zeros((1, 2, 2, 4)); // b, h, q, k
        let mask = Array2::from_shape_vec((1, 4), vec![1.0, 1.0, 0.0, 0.0]).unwrap();
        let masked = apply_padding_mask(scores, &mask).unwrap();

        // The last two key positions should be masked for all queries and heads
        assert_eq!(masked[[0, 0, 0, 2]], MASK_VALUE);
        assert_eq!(masked[[0, 0, 1, 3]], MASK_VALUE);
        assert_eq!(masked[[0, 1, 0, 2]], MASK_VALUE);
        assert_eq!(masked[[0, 0, 0, 1]], 0.0);
    }
}
