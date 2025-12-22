//! Multi-head attention implementation with KV caching support

use crate::activations::softmax;
use crate::rope::RoPE;
use crate::utils::linear_algebra::{matmul_3d_2d, matmul_4d};
use crate::utils::masks::{apply_causal_mask, apply_padding_mask};
use anyhow::Result;
use ndarray::{s, Array1, Array2, Array3, Array4, Axis};

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
            apply_causal_mask(&mut scores, position_offset);
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
    /// Performs self-attention for a decoder layer, handling the KV cache.
    ///
    /// This is a specialized version of `forward` for decoder self-attention blocks.
    /// It projects Q, K, and V from the same `hidden_states` input, concatenates the
    /// new K/V with the cached K/V, performs causal attention, and returns the
    /// attention output along with the new K/V states to be appended to the cache.
    ///
    /// # Arguments
    /// * `hidden_states`: The input from the previous layer. Shape: `[batch, seq_len, hidden_size]`.
    /// * `attention_mask`: Padding mask. Shape: `[batch, total_seq_len]`.
    /// * `cached_kv`: A tuple of `(key_cache, value_cache)` from previous steps.
    ///
    /// # Returns
    /// A tuple of `(attention_output, new_key_state, new_value_state)`.
    pub fn forward_self_attn(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: Option<&Array2<f32>>,
        cached_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        // 1. Project Q, K, V from the input hidden_states
        let q_proj = matmul_3d_2d(hidden_states, &self.q_weight) + &self.q_bias;
        let new_k = matmul_3d_2d(hidden_states, &self.k_weight) + &self.k_bias;
        let new_v = matmul_3d_2d(hidden_states, &self.v_weight) + &self.v_bias;

        // 2. Combine new K/V with the cached K/V from previous steps
        let (full_k, full_v) = if let Some((cached_k, cached_v)) = cached_kv {
            // --- THE FIX IS HERE ---
            // After concatenating, we force the result into a standard, contiguous
            // memory layout. This resolves the `IncompatibleLayout` error.
            let full_k = ndarray::concatenate![Axis(1), cached_k, new_k.view()]
                .as_standard_layout()
                .to_owned();
            let full_v = ndarray::concatenate![Axis(1), cached_v, new_v.view()]
                .as_standard_layout()
                .to_owned();
            // --- END FIX ---
            (full_k, full_v)
        } else {
            // If no cache, the full K/V is just the new K/V (which is already contiguous)
            (new_k.clone(), new_v.clone())
        };

        // 3. Determine the position offset for the causal mask
        let position_offset = cached_kv.map_or(0, |(k, _)| k.shape()[1]);

        // 4. Compute attention using the now-contiguous full K/V history
        let context_reshaped = self.attend(
            &q_proj,
            &full_k,
            &full_v,
            attention_mask,
            true, // Self-attention in a decoder is always causal
            position_offset,
        )?;

        // 5. Final output projection
        let output = matmul_3d_2d(&context_reshaped, &self.output_weight) + &self.output_bias;

        // 6. Return the output and the *new* K/V states
        Ok((output, new_k, new_v))
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

#[cfg(test)]
mod tests;