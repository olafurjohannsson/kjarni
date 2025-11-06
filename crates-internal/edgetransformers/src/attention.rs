//! Multi-head attention implementation with KV caching support

use crate::activations::softmax;
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
    ) -> Self {
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
        }
    }

    // /// Forward pass without cache (for encoders or testing)
    // pub fn forward(
    //     &self,
    //     hidden_states: &Array3<f32>,
    //     encoder_hidden_states: Option<&Array3<f32>>,
    //     attention_mask: Option<&Array2<f32>>,
    // ) -> Result<Array3<f32>> {
    //     let (output, _, _) = self.forward_with_cache(
    //         hidden_states,
    //         encoder_hidden_states,
    //         attention_mask,
    //         false, // Not causal for encoder
    //         None,  // No cache
    //     )?;
    //     Ok(output)
    // }

    /// Projects the input hidden states into new Key and Value states.
    /// This should be called by the orchestrator (e.g., the decoder loop)
    /// to generate the K/V for the current token(s).
    ///
    /// # Returns
    /// A tuple of `(new_k, new_v)`.
    pub fn project_kv(&self, key_value_source: &Array3<f32>) -> (Array3<f32>, Array3<f32>) {
        let new_k = matmul_3d_2d(key_value_source, &self.k_weight) + &self.k_bias;
        let new_v = matmul_3d_2d(key_value_source, &self.v_weight) + &self.v_bias;
        (new_k, new_v)
    }

    /// Performs the core attention calculation using a query and the full, final
    /// Key and Value caches for this step.
    ///
    /// The caller is responsible for cache management (i.e., updating the cache
    /// with the latest projections before calling this function).
    ///
    /// # Arguments
    /// * `query` - Query hidden states for the current step [B, S_q, H]
    /// * `full_k_cache` - The complete Key cache for this step [B, S_total, H]
    /// * `full_v_cache` - The complete Value cache for this step [B, S_total, H]
    /// * `attention_mask` - Optional mask [B, S_total]
    /// * `is_causal` - Whether to apply causal masking.
    /// * `position_offset` - The starting position for causal masking (i.e., the length of the cache before this step).
    pub fn attend(
        &self,
        query: &Array3<f32>,
        full_k_cache: &Array3<f32>,
        full_v_cache: &Array3<f32>,
        attention_mask: Option<&Array2<f32>>,
        is_causal: bool,
        position_offset: usize,
    ) -> Result<Array3<f32>> {
        let batch_size = query.shape()[0];
        let seq_len = query.shape()[1];
        let full_kv_len = full_k_cache.shape()[1];

        // --- Assertions for Robustness ---
        assert_eq!(
            full_k_cache.shape(),
            full_v_cache.shape(),
            "K and V cache shapes must match."
        );
        assert_eq!(
            query.shape()[0],
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
        // --- End Assertions ---

        // Project Query
        let q = matmul_3d_2d(query, &self.q_weight) + &self.q_bias;

        // Reshape, Permute, Matmul, etc. (The rest of your existing logic)
        // IMPORTANT: Use `full_k_cache` and `full_v_cache` directly.
        // DO NOT perform any concatenation here.

        let q_reshaped = q
            .into_shape_with_order((batch_size, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let k_reshaped = full_k_cache
            .to_owned() // Use to_owned() or clone() as we don't own the cache
            .into_shape_with_order((batch_size, full_kv_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let v_reshaped = full_v_cache
            .to_owned()
            .into_shape_with_order((batch_size, full_kv_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let q_contiguous = q_reshaped.as_standard_layout().to_owned();
        let k_transposed = k_reshaped.permuted_axes([0, 1, 3, 2]);
        let k_transposed_contiguous = k_transposed.as_standard_layout().to_owned();

        let mut scores = matmul_4d(&q_contiguous, &k_transposed_contiguous);
        scores *= self.scale_factor;

        if let Some(mask) = attention_mask {
            scores = apply_padding_mask(scores, mask)?;
        }
        if is_causal {
            // Use the position_offset here instead of calculating cache_len
            scores = apply_causal_mask(scores, position_offset)?;
        }

        // ... rest of the function (softmax, context, output projection)
        let weights = softmax(&scores);
        let v_contiguous = v_reshaped.as_standard_layout().to_owned();
        let context = matmul_4d(&weights, &v_contiguous);

        let context_reshaped = context
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .into_shape_with_order((batch_size, seq_len, self.num_heads * self.head_dim))?
            .to_owned();

        Ok(matmul_3d_2d(&context_reshaped, &self.output_weight) + &self.output_bias)
    }
    /// A simple forward pass for non-causal contexts (like an encoder).
    /// This can now be implemented as a simple wrapper around the new methods.
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        // cross-attention source
        key_value_source: Option<&Array3<f32>>,
        attention_mask: Option<&Array2<f32>>,
    ) -> Result<Array3<f32>> {
        let kv_source = key_value_source.unwrap_or(hidden_states);

        // In this simple case, the "cache" is just the projections of the input itself.
        let (k_states, v_states) = self.project_kv(kv_source);

        self.attend(
            hidden_states, // query
            &k_states,     // full_k_cache
            &v_states,     // full_v_cache
            attention_mask,
            false, // is_causal = false for encoders
            0,     // position_offset = 0
        )
    }

    /// Forward pass with KV caching support
    ///
    /// # Arguments
    /// * `query` - Query hidden states [batch, seq_len, hidden]
    /// * `key_value` - Optional encoder hidden states for cross-attention [batch, enc_len, hidden]
    /// * `attention_mask` - Optional attention mask [batch, total_seq_len]
    /// * `is_causal` - Whether to apply causal masking (for decoder self-attention)
    /// * `cached_kv` - Optional cached (K, V) from previous steps
    ///
    /// # Returns
    /// Tuple of (output, new_k, new_v) where:
    /// - output: [batch, seq_len, hidden]
    /// - new_k: [batch, seq_len, hidden] - NEW keys only (not concatenated)
    /// - new_v: [batch, seq_len, hidden] - NEW values only (not concatenated)
    pub fn forward_with_cache(
        &self,
        query: &Array3<f32>,
        key_value: Option<&Array3<f32>>, // For cross-attention
        attention_mask: Option<&Array2<f32>>,
        is_causal: bool,
        cached_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        // Step 1: Compute the new K and V states. These are the owned values we will return.
        let kv_source = key_value.unwrap_or(query);
        let new_k = matmul_3d_2d(kv_source, &self.k_weight) + &self.k_bias;
        let new_v = matmul_3d_2d(kv_source, &self.v_weight) + &self.v_bias;

        // This block calculates the final `output` tensor.
        let output = {
            let batch_size = query.shape()[0];
            let seq_len = query.shape()[1];
            let q = matmul_3d_2d(query, &self.q_weight) + &self.q_bias;

            // Step 2: Manually and performantly construct the temporary calculation tensors.
            let (full_k, full_v) = if let Some((cached_k, cached_v)) = cached_kv {
                let cache_len = cached_k.shape()[1];
                let new_len = new_k.shape()[1];
                let full_len = cache_len + new_len;
                let hidden_size = new_k.shape()[2];

                // Pre-allocate the temporary arrays. This is what `concatenate` does internally.
                let mut full_k = Array3::zeros((batch_size, full_len, hidden_size));
                let mut full_v = Array3::zeros((batch_size, full_len, hidden_size));

                // Perform two fast, explicit copies. This has no complex lifetimes.
                full_k.slice_mut(s![.., 0..cache_len, ..]).assign(&cached_k);
                full_k
                    .slice_mut(s![.., cache_len..full_len, ..])
                    .assign(&new_k);

                full_v.slice_mut(s![.., 0..cache_len, ..]).assign(&cached_v);
                full_v
                    .slice_mut(s![.., cache_len..full_len, ..])
                    .assign(&new_v);

                (full_k, full_v)
            } else {
                // No cache exists. To allow `new_k` and `new_v` to be returned, `full_k`
                // and `full_v` must take ownership of clones.
                (new_k.clone(), new_v.clone())
            };

            // --- The rest of the calculation is now guaranteed to be safe ---
            let cache_len = cached_kv.map_or(0, |(k, _)| k.shape()[1]);
            let full_kv_len = full_k.shape()[1];

            // (All your existing, correct logic for reshape, matmul, mask, etc. goes here)
            let q_reshaped = q
                .into_shape_with_order((batch_size, seq_len, self.num_heads, self.head_dim))?
                .permuted_axes([0, 2, 1, 3]);

            let k_reshaped = full_k
                .into_shape_with_order((batch_size, full_kv_len, self.num_heads, self.head_dim))?
                .permuted_axes([0, 2, 1, 3]);

            let v_reshaped = full_v
                .into_shape_with_order((batch_size, full_kv_len, self.num_heads, self.head_dim))?
                .permuted_axes([0, 2, 1, 3]);

            let q_contiguous = q_reshaped.as_standard_layout().to_owned();
            let k_transposed = k_reshaped.permuted_axes([0, 1, 3, 2]);
            let k_transposed_contiguous = k_transposed.as_standard_layout().to_owned();

            let mut scores = matmul_4d(&q_contiguous, &k_transposed_contiguous);
            scores *= self.scale_factor;

            if let Some(mask) = attention_mask {
                scores = apply_padding_mask(scores, mask)?;
            }
            if is_causal {
                scores = apply_causal_mask(scores, cache_len)?;
            }

            let weights = softmax(&scores);
            let v_contiguous = v_reshaped.as_standard_layout().to_owned();
            let context = matmul_4d(&weights, &v_contiguous);

            let context_reshaped = context
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .into_shape_with_order((batch_size, seq_len, self.num_heads * self.head_dim))?
                .to_owned();

            matmul_3d_2d(&context_reshaped, &self.output_weight) + &self.output_bias
        }; // All temporaries, including `full_k` and `full_v`, are dropped here.

        // Step 3: Return the original, un-borrowed values.
        Ok((output, new_k, new_v))
    }
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

    #[test]
    fn test_attention_without_cache() {
        let hidden_size = 64;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 10;

        // Create dummy weights
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
        );

        let input = Array3::zeros((batch_size, seq_len, hidden_size));
        let mask = Array2::ones((batch_size, seq_len));

        let result = attention.forward(&input, None, Some(&mask));
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
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
        );

        assert!(result.is_ok());

        let (output, new_k, new_v) = result.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_k.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_v.shape(), &[batch_size, seq_len, hidden_size]);
    }
}
