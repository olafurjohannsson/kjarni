//! Multi-head attention implementation with KV caching support

use crate::activations::softmax;
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

    /// Forward pass without cache (for encoders or testing)
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: Option<&Array3<f32>>,
        attention_mask: Option<&Array2<f32>>,
    ) -> Result<Array3<f32>> {
        let (output, _, _) = self.forward_with_cache(
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            false, // Not causal for encoder
            None,  // No cache
        )?;
        Ok(output)
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
        cached_kv: Option<(&Array3<f32>, &Array3<f32>)>,
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        let batch_size = query.shape()[0];
        let seq_len = query.shape()[1];

        // === 1. Linear Projections & Attention Scaling ===

        // Q projection (always from query input)
        let mut q = matmul_3d_2d(query, &self.q_weight) + &self.q_bias;

        // FIX #1: Scale the 3D Query tensor *before* reshaping and matrix multiplication.
        // This matches the standard implementation in both BART and GPT-2.
        // q *= self.scale_factor;

        // K, V projections
        let kv_source = key_value.unwrap_or(query);
        let k = matmul_3d_2d(kv_source, &self.k_weight) + &self.k_bias;
        let v = matmul_3d_2d(kv_source, &self.v_weight) + &self.v_bias;

        // === 2. Concatenate with Cached K, V ===

        let (full_k, full_v) = if let Some((cached_k, cached_v)) = cached_kv {
            // Concatenate: [batch, cache_len, hidden] + [batch, seq_len, hidden]
            let full_k = ndarray::concatenate(Axis(1), &[cached_k.view(), k.view()])?;
            let full_v = ndarray::concatenate(Axis(1), &[cached_v.view(), v.view()])?;
            (full_k, full_v)
        } else {
            (k.clone(), v.clone())
        };

        let full_kv_len = full_k.shape()[1];

        // === 3. Reshape for Multi-Head Attention ===

        // Q: [batch, seq_len, hidden] → [batch, num_heads, seq_len, head_dim]
        let q = q
            .into_shape_with_order((batch_size, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        // K: [batch, full_kv_len, hidden] → [batch, num_heads, full_kv_len, head_dim]
        let k_reshaped = full_k
            .into_shape_with_order((batch_size, full_kv_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        // V: [batch, full_kv_len, hidden] → [batch, num_heads, full_kv_len, head_dim]
        let v_reshaped = full_v
            .into_shape_with_order((batch_size, full_kv_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        // === 4. Compute Attention Scores: Q @ K^T ===

        // FIX #2: Ensure all array "views" are converted to a standard, contiguous
        // memory layout before being passed to matmul to prevent incorrect results.
        let q_contiguous = q.as_standard_layout().to_owned();
        let k_transposed = k_reshaped.permuted_axes([0, 1, 3, 2]);
        let k_transposed_contiguous = k_transposed.as_standard_layout().to_owned();

        let mut scores = matmul_4d(&q_contiguous, &k_transposed_contiguous);
        scores *= self.scale_factor;
        // scores *= self.scale_factor;
        // The incorrect scaling (`scores *= self.scale_factor;`) has been REMOVED from here.

        // === 5. Apply Masks ===

        if let Some(mask) = attention_mask {
            scores = self.apply_padding_mask(scores, mask)?;
        }

        if is_causal {
            let cache_len = full_kv_len - seq_len;
            scores = self.apply_causal_mask(scores, cache_len)?;
        }

        // === 6. Softmax ===

        let weights = softmax(&scores);

        // === 7. Apply Attention to Values ===

        // FIX #2 (continued): Ensure 'v' is also contiguous for the final matmul.
        let v_contiguous = v_reshaped.as_standard_layout().to_owned();
        let context = matmul_4d(&weights, &v_contiguous);

        // === 8. Reshape Back ===

        let context = context
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .into_shape_with_order((batch_size, seq_len, self.num_heads * self.head_dim))?
            .to_owned();

        // === 9. Output Projection ===

        let output = matmul_3d_2d(&context, &self.output_weight) + &self.output_bias;

        // === 10. Return (output, new_k, new_v) ===

        Ok((output, k, v))
    }

    /// Apply padding mask to attention scores
    ///
    /// Masks positions where mask[batch, key_pos] == 0
    fn apply_padding_mask(
        &self,
        mut scores: Array4<f32>,
        mask: &Array2<f32>,
    ) -> Result<Array4<f32>> {
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
        if let Some(broadcast_mask) = mask_expanded.broadcast((batch_size, num_heads, seq_q, seq_k))
        {
            Zip::from(&mut scores)
                .and(&broadcast_mask)
                .for_each(|s, &m| {
                    if m == 0.0 {
                        *s = f32::NEG_INFINITY;
                    }
                });
        }

        Ok(scores)
    }
    /// Apply causal mask to attention scores
    ///
    /// Ensures position `i` can only attend to positions `0..=i` in the full sequence.
    /// Takes into account the cache length for proper positioning.
    fn apply_causal_mask(&self, mut scores: Array4<f32>, cache_len: usize) -> Result<Array4<f32>> {
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
                    scores.slice_mut(s![.., .., i, j]).fill(f32::NEG_INFINITY);
                }
            }
        }

        Ok(scores)
    }
    // /// Apply causal mask to attention scores
    // ///
    // /// Ensures position i can only attend to positions 0..=i in the full sequence
    // /// Takes into account the cache length for proper positioning
    // fn apply_causal_mask(
    //     &self,
    //     mut scores: Array4<f32>,
    //     cache_len: usize,
    // ) -> Result<Array4<f32>> {
    //     let (batch_size, num_heads, seq_q, seq_k) = scores.dim();

    //     // For each query position, mask out positions that are "in the future"
    //     for b in 0..batch_size {
    //         for h in 0..num_heads {
    //             for i in 0..seq_q {
    //                 // Current position in the full sequence
    //                 let current_pos = cache_len + i;

    //                 // Mask all key positions > current_pos
    //                 for j in 0..seq_k {
    //                     if j > current_pos {
    //                         scores[[b, h, i, j]] = f32::NEG_INFINITY;
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     Ok(scores)
    // }
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
            Some((&cached_k, &cached_v)),
        );

        assert!(result.is_ok());

        let (output, new_k, new_v) = result.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_k.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_v.shape(), &[batch_size, seq_len, hidden_size]);
    }
}

// //! Multi-head attention implementation

// use crate::activations::softmax;
// use crate::utils::linear_algebra::{
//     apply_attention_mask, matmul_3d_2d, matmul_4d,
// };
// use anyhow::Result;
// use ndarray::{Array1, Array2, Array3, Array4, Axis, Zip, s};

// /// Multi-head attention mechanism
// pub struct MultiHeadAttention {
//     pub query_weight_t: Array2<f32>,
//     pub query_bias: Array1<f32>,
//     pub key_weight_t: Array2<f32>,
//     pub key_bias: Array1<f32>,
//     pub value_weight_t: Array2<f32>,
//     pub value_bias: Array1<f32>,
//     pub output_weight_t: Array2<f32>,
//     pub output_bias: Array1<f32>,
//     pub num_heads: usize,
//     pub head_dim: usize,
//     pub scale_factor: f32,
// }

// impl MultiHeadAttention {
//     pub fn new(
//         hidden_size: usize,
//         num_heads: usize,
//         query_weight: Array2<f32>,
//         query_bias: Array1<f32>,
//         key_weight: Array2<f32>,
//         key_bias: Array1<f32>,
//         value_weight: Array2<f32>,
//         value_bias: Array1<f32>,
//         output_weight: Array2<f32>,
//         output_bias: Array1<f32>,
//     ) -> Self {
//         let head_dim = hidden_size / num_heads;
//         let scale_factor = 1.0 / (head_dim as f32).sqrt();

//         Self {
//             query_weight_t: query_weight,
//             query_bias,
//             key_weight_t: key_weight,
//             key_bias,
//             value_weight_t: value_weight,
//             value_bias,
//             output_weight_t: output_weight,
//             output_bias,
//             num_heads,
//             head_dim,
//             scale_factor,
//         }
//     }

//     pub fn forward(
//         &self,
//         hidden_states: &Array3<f32>,
//         // If this is Some, we perform cross-attention (encoder-decoder), else we perform self-attention (decoder)
//         encoder_hidden_states: Option<&Array3<f32>>,
//         attention_mask: Option<&Array2<f32>>,
//     ) -> Result<Array3<f32>> {
//         let batch_size = hidden_states.shape()[0];
//         let seq_len = hidden_states.shape()[1];

//         // Linear projections with bias

//         // Project Q from decoders hidden states
//         let mut q = matmul_3d_2d(hidden_states, &self.query_weight_t);
//         q += &self.query_bias;

//         // Determine source
//         let kv_source = encoder_hidden_states.unwrap_or(hidden_states);
//         let mut k = matmul_3d_2d(kv_source, &self.key_weight_t);
//         k += &self.key_bias;
//         let mut v = matmul_3d_2d(kv_source, &self.value_weight_t);
//         v += &self.value_bias;

//         // Reshape for multi-head attention
//         let q = q
//             .into_shape_with_order((batch_size, seq_len, self.num_heads, self.head_dim))?
//             .permuted_axes([0, 2, 1, 3]);

//         let k_seq_len = kv_source.shape()[1];
//         let k = k
//             .into_shape_with_order((batch_size, k_seq_len, self.num_heads, self.head_dim))?
//             .permuted_axes([0, 2, 1, 3]);

//         let v = v
//             .into_shape_with_order((batch_size, k_seq_len, self.num_heads, self.head_dim))?
//             .permuted_axes([0, 2, 1, 3]);

//         // Compute attention scores
//         let mut scores = matmul_4d(&q, &k.permuted_axes([0, 1, 3, 2]));
//         scores *= self.scale_factor;

//         // Apply mask

//         if let Some(mask) = attention_mask {
//             scores = apply_attention_mask(scores, mask);
//         }

//         // Softmax
//         let weights = softmax(&scores);

//         // Apply attention to values
//         let context = matmul_4d(&weights, &v);

//         // Reshape back
//         let context = context.permuted_axes([0, 2, 1, 3]);
//         let context = context
//             .as_standard_layout()
//             .into_shape_with_order((batch_size, seq_len, self.num_heads * self.head_dim))?
//             .to_owned();

//         // Output projection
//         let mut output = matmul_3d_2d(&context, &self.output_weight_t);
//         output += &self.output_bias;

//         Ok(output)
//     }

//     pub fn forward_bart(
//         &self,
//         hidden_states: &Array3<f32>,
//         encoder_hidden_states: Option<&Array3<f32>>,
//         attention_mask: Option<&Array2<f32>>,
//         is_causal: bool,
//         past_kv: Option<&(Array4<f32>, Array4<f32>)>,
//     ) -> Result<(Array3<f32>, (Array4<f32>, Array4<f32>))> {
//         let batch_size = hidden_states.shape()[0];
//         let query_len = hidden_states.shape()[1];

//         let q_proj = matmul_3d_2d(hidden_states, &self.query_weight_t) + &self.query_bias;
//         let kv_source = encoder_hidden_states.unwrap_or(hidden_states);
//         let k_proj = matmul_3d_2d(kv_source, &self.key_weight_t) + &self.key_bias;
//         let v_proj = matmul_3d_2d(kv_source, &self.value_weight_t) + &self.value_bias;
//         let q_proj_scaled = &q_proj * self.scale_factor;

//         let q = q_proj_scaled
//             .into_shape_with_order((batch_size, query_len, self.num_heads, self.head_dim))?
//             .permuted_axes([0, 2, 1, 3]);
//         let k_current_seq_len = kv_source.shape()[1];
//         let mut k = k_proj
//             .into_shape_with_order((batch_size, k_current_seq_len, self.num_heads, self.head_dim))?
//             .permuted_axes([0, 2, 1, 3]);
//         let mut v = v_proj
//             .into_shape_with_order((batch_size, k_current_seq_len, self.num_heads, self.head_dim))?
//             .permuted_axes([0, 2, 1, 3]);

//         if let Some((past_k, past_v)) = past_kv {
//             k = ndarray::concatenate(Axis(2), &[past_k.view(), k.view()])?;
//             v = ndarray::concatenate(Axis(2), &[past_v.view(), v.view()])?;
//         }
//         let present_kv = (k.clone(), v.clone());
//         let k_permuted = k.clone().permuted_axes([0, 1, 3, 2]);
//         let q_contiguous = q.as_standard_layout().to_owned();
//         let k_contiguous = k_permuted.as_standard_layout().to_owned();

//         let mut scores = matmul_4d(&q_contiguous, &k_contiguous);

//         if let Some(mask) = attention_mask {
//             scores = if is_causal {
//                 let sliced_mask = mask.slice(s![..query_len, ..k.shape()[2]]);
//                 self.apply_causal_mask(scores, &sliced_mask.to_owned())
//             } else {
//                 self.apply_padding_mask(scores, mask)
//             };
//         }
//         let weights = softmax(&scores);
//         let context = matmul_4d(&weights, &v.as_standard_layout().to_owned()); // Make v contiguous too, just to be safe
//         let context = context.permuted_axes([0, 2, 1, 3]);
//         let context = context
//             .as_standard_layout()
//             .into_shape_with_order((batch_size, query_len, self.num_heads * self.head_dim))?
//             .to_owned();
//         let output = matmul_3d_2d(&context, &self.output_weight_t) + &self.output_bias;

//         Ok((output, present_kv))
//     }
//     pub fn apply_bart_attention_mask(
//         &self,
//         mut scores: Array4<f32>,
//         mask: &Array2<f32>,
//     ) -> Array4<f32> {
//         // mask shape: [batch, seq_k]
//         // scores shape: [batch, heads, seq_q, seq_k]
//         // Need to broadcast mask to [batch, 1, 1, seq_k]

//         let (batch_size, num_heads, seq_q, seq_k) = scores.dim();
//         assert_eq!(mask.shape()[0], batch_size);
//         assert_eq!(mask.shape()[1], seq_k);

//         // Expand mask dimensions correctly for BART
//         let mask_expanded = mask
//             .view()
//             .insert_axis(Axis(1)) // [batch, 1, seq_k]
//             .insert_axis(Axis(1)); // [batch, 1, 1, seq_k]

//         if let Some(broadcast_mask) = mask_expanded.broadcast((batch_size, num_heads, seq_q, seq_k))
//         {
//             Zip::from(&mut scores)
//                 .and(&broadcast_mask)
//                 .for_each(|s, &m| {
//                     if m == 0.0 {
//                         *s = f32::NEG_INFINITY;
//                     }
//                 });
//         }

//         scores
//     }
//     /// Apply causal attention mask for decoder self-attention
//     pub fn apply_causal_mask(&self, mut scores: Array4<f32>, mask: &Array2<f32>) -> Array4<f32> {
//         // mask shape: [seq_q, seq_k]
//         // scores shape: [batch, heads, seq_q, seq_k]

//         let (batch_size, num_heads, seq_q, seq_k) = scores.dim();
//         assert_eq!(mask.shape()[0], seq_q);
//         assert_eq!(mask.shape()[1], seq_k);

//         // Expand: [seq_q, seq_k] -> [1, 1, seq_q, seq_k]
//         let mask_expanded = mask
//             .view()
//             .insert_axis(Axis(0)) // [1, seq_q, seq_k]
//             .insert_axis(Axis(0)); // [1, 1, seq_q, seq_k]

//         if let Some(broadcast_mask) = mask_expanded.broadcast((batch_size, num_heads, seq_q, seq_k))
//         {
//             Zip::from(&mut scores)
//                 .and(&broadcast_mask)
//                 .for_each(|s, &m| {
//                     if m == 0.0 {
//                         *s = f32::NEG_INFINITY;
//                     }
//                 });
//         }

//         scores
//     }

//     /// Apply padding attention mask (for encoder or cross-attention)
//     pub fn apply_padding_mask(&self, mut scores: Array4<f32>, mask: &Array2<f32>) -> Array4<f32> {
//         // mask shape: [batch, seq_k]
//         // scores shape: [batch, heads, seq_q, seq_k]

//         let (batch_size, num_heads, seq_q, seq_k) = scores.dim();
//         assert_eq!(mask.shape()[0], batch_size);
//         assert_eq!(mask.shape()[1], seq_k);

//         // Expand: [batch, seq_k] -> [batch, 1, 1, seq_k]
//         let mask_expanded = mask
//             .view()
//             .insert_axis(Axis(1)) // [batch, 1, seq_k]
//             .insert_axis(Axis(1)); // [batch, 1, 1, seq_k]

//         if let Some(broadcast_mask) = mask_expanded.broadcast((batch_size, num_heads, seq_q, seq_k))
//         {
//             Zip::from(&mut scores)
//                 .and(&broadcast_mask)
//                 .for_each(|s, &m| {
//                     if m == 0.0 {
//                         *s = f32::NEG_INFINITY;
//                     }
//                 });
//         }

//         scores
//     }
// }
