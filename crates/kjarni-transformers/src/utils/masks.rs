use anyhow::anyhow;
use ndarray::{Array2, Array3, Array4, Axis, Zip, s};

pub const MASK_VALUE: f32 = -1e9; // SAME as GPU

/// Apply padding mask to attention scores
///
/// Masks positions where mask[batch, key_pos] == 0
pub fn apply_padding_mask(
    mut scores: Array4<f32>,
    mask: &Array2<f32>,
) -> anyhow::Result<Array4<f32>> {
    let (batch_size, num_heads, seq_q, seq_k) = scores.dim();

    if mask.shape() != [batch_size, seq_k] {
        return Err(anyhow!(
            "Padding mask shape {:?} does not match expected [{}, {}]",
            mask.shape(),
            batch_size,
            seq_k
        ));
    }

    // Expand mask: [batch, seq_k] → [batch, 1, 1, seq_k]
    // Broadcasts across heads and query positions
    let mask_expanded = mask.view().insert_axis(Axis(1)).insert_axis(Axis(1));

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

/// Apply a general bias mask (e.g. Alibi or Causal mask provided as tensor)
/// Assumes mask shape is [Q_Len, K_Len] or [1, 1, Q_Len, K_Len]
pub fn apply_bias_mask(mut scores: Array4<f32>, mask: &Array2<f32>) -> anyhow::Result<Array4<f32>> {
    let (batch, heads, q, k) = scores.dim();

    // Check if dimensions match the attention window
    if mask.shape() != [q, k] {
        return Err(anyhow!(
            "Bias mask shape {:?} does not match attention window [{}, {}]",
            mask.shape(),
            q,
            k
        ));
    }

    // Expand: [Q, K] -> [1, 1, Q, K]
    // Broadcasts across Batch and Heads
    let mask_view = mask.view().insert_axis(Axis(0)).insert_axis(Axis(0));

    if let Some(m) = mask_view.broadcast((batch, heads, q, k)) {
        Zip::from(&mut scores).and(&m).for_each(|s, &mask_val| {
            // Assuming mask contains 1.0 to keep, 0.0 to mask, or additive bias?
            // If it's a binary mask (1.0/0.0):
            if mask_val == 0.0 {
                *s = MASK_VALUE;
            }
            // If it's an additive bias (like Alibi), you should use += instead.
        });
    }
    Ok(scores)
}

/// Pools the hidden states by taking the hidden state of the last non-padding token.
///
/// # Arguments
/// * `hidden_states` - `[batch_size, seq_len, hidden_size]`
/// * `attention_mask` - `[batch_size, seq_len]` where `1`s are real tokens and `0`s are padding.
///
/// # Returns
/// Pooled output `[batch_size, hidden_size]`
pub fn last_token_pool(hidden_states: &Array3<f32>, attention_mask: &Array2<f32>) -> Array2<f32> {
    let (batch_size, _, hidden_size) = hidden_states.dim();
    let mut pooled = Array2::zeros((batch_size, hidden_size));

    for i in 0..batch_size {
        // For each item in the batch, find the index of the last '1' in its attention mask.
        let last_token_index = attention_mask
            .row(i)
            .iter()
            .rposition(|&x| x == 1.0) // Find the last position of a 1.0
            .unwrap_or(0); // Default to the first token if all are masked (edge case)

        // Select the hidden state at that specific index for this batch item.
        let last_token_hidden_state = hidden_states.slice(s![i, last_token_index, ..]);

        // Assign it to the corresponding row in the output.
        pooled.row_mut(i).assign(&last_token_hidden_state);
    }

    pooled
}

pub fn apply_attention_mask(mut scores: Array4<f32>, mask: &Array2<f32>) -> Array4<f32> {
    let batch = scores.dim().0;

    // The mask can be [batch, k_len] or [q_len, k_len]
    // We expand to [batch, 1, q_len, k_len] for broadcasting
    let mask_view = if mask.nrows() == batch {
        // Padding mask style: [Batch, Keys]
        mask.view().insert_axis(Axis(1)).insert_axis(Axis(2))
    } else {
        // Causal mask style: [Queries, Keys]
        mask.view().insert_axis(Axis(0)).insert_axis(Axis(1))
    };

    if let Some(m) = mask_view.broadcast(scores.dim()) {
        Zip::from(&mut scores).and(&m).for_each(|s, &mask_val| {
            if mask_val == 0.0 {
                *s = MASK_VALUE;
            }
        });
    }
    scores
}

/// Apply causal mask to attention scores
///
/// Ensures position `i` can only attend to positions `0..=i` in the full sequence.
/// Takes into account the cache length for proper positioning.
pub fn apply_causal_mask(scores: &mut Array4<f32>, cache_len: usize) {
    let (_, _, seq_q, _) = scores.dim();
    for i in 0..seq_q {
        let query_pos = cache_len + i;
        for j in 0..scores.shape()[3] {
            if j > query_pos {
                scores.slice_mut(s![.., .., i, j]).fill(MASK_VALUE);
            }
        }
    }
}

/// Create a full attention mask (all positions visible)
///
/// Used for testing or when no masking is needed.
/// Returns [batch_size, seq_len] with all 1.0
pub fn create_full_attention_mask(batch_size: usize, seq_len: usize) -> Array2<f32> {
    Array2::ones((batch_size, seq_len))
}

/// Create a causal attention mask for any shape.
///
/// # Arguments
/// * `q_len` - Number of query tokens (usually 1 during decode)
/// * `total_len` - Total tokens in sequence (past_cache + current)
pub fn create_causal_mask(q_len: usize, total_len: usize) -> Array2<f32> {
    if total_len < q_len {
        // Graceful handling or panic, but helpful message
        panic!(
            "create_causal_mask: total_len ({}) cannot be less than q_len ({})",
            total_len, q_len
        );
    }

    let mut mask = Array2::zeros((q_len, total_len));
    let past_len = total_len - q_len;

    for i in 0..q_len {
        let current_abs_pos = past_len + i;
        for j in 0..total_len {
            if j <= current_abs_pos {
                mask[[i, j]] = 1.0;
            }
        }
    }
    mask
}

/// Create a causal mask for a batch
///
/// Returns [batch_size, seq_len, seq_len]
pub fn create_batched_causal_mask(batch_size: usize, seq_len: usize) -> Array3<f32> {
    let single_mask = create_causal_mask(seq_len, seq_len);
    let mut batched = Array3::zeros((batch_size, seq_len, seq_len));

    for b in 0..batch_size {
        batched
            .slice_mut(ndarray::s![b, .., ..])
            .assign(&single_mask);
    }

    batched
}
/// Create a padding mask from token IDs
///
/// Marks positions with pad_token_id as 0.0, others as 1.0
/// Returns [batch_size, seq_len]
pub fn create_padding_mask_from_tokens(token_ids: &Array2<f32>, pad_token_id: f32) -> Array2<f32> {
    token_ids.mapv(|id| if id == pad_token_id { 0.0 } else { 1.0 })
}

/// Expand padding mask for multi-head attention
///
/// Takes [batch, seq_len] and expands to [batch, num_heads, seq_len, seq_len]
/// for use in batched multi-head attention
pub fn expand_mask_for_attention(mask: &Array2<f32>, num_heads: usize) -> Array3<f32> {
    let (batch_size, seq_len) = mask.dim();
    let mut expanded = Array3::zeros((batch_size, num_heads, seq_len));

    for b in 0..batch_size {
        for h in 0..num_heads {
            expanded
                .slice_mut(ndarray::s![b, h, ..])
                .assign(&mask.row(b));
        }
    }

    expanded
}

#[cfg(test)]
mod masking_tests {
    use super::*;
    use ndarray::{Array4, arr2, arr3};

    // ========================================================================
    //  Helper for float comparison
    // ========================================================================
    fn assert_array4_close(a: &Array4<f32>, b: &Array4<f32>) {
        assert_eq!(a.shape(), b.shape());
        for (v1, v2) in a.iter().zip(b.iter()) {
            if v1.is_finite() && v2.is_finite() {
                assert!((v1 - v2).abs() < 1e-5, "Mismatch: {} vs {}", v1, v2);
            } else {
                // Handle MASK_VALUE (large negative) equality
                assert_eq!(v1, v2, "Mismatch on non-finite/mask values");
            }
        }
    }

    // ========================================================================
    //  1. Padding Mask Tests
    // ========================================================================

    #[test]
    fn test_apply_padding_mask_basic() {
        // Batch=1, Heads=1, Q=2, K=3
        let mut scores = Array4::zeros((1, 1, 2, 3));

        // Mask: Keep indices 0 and 1, mask index 2
        // Shape [1, 3]
        let mask = arr2(&[[1.0, 1.0, 0.0]]);

        let result = apply_padding_mask(scores.clone(), &mask).unwrap();

        // Check Row 0 (Query 0)
        assert_eq!(result[[0, 0, 0, 0]], 0.0); // Kept
        assert_eq!(result[[0, 0, 0, 1]], 0.0); // Kept
        assert_eq!(result[[0, 0, 0, 2]], MASK_VALUE); // Masked

        // Check Row 1 (Query 1) - Should broadcast same mask
        assert_eq!(result[[0, 0, 1, 0]], 0.0);
        assert_eq!(result[[0, 0, 1, 2]], MASK_VALUE);
    }

    #[test]
    fn test_apply_padding_mask_shape_mismatch() {
        let scores = Array4::zeros((1, 1, 2, 3));
        let mask = arr2(&[[1.0, 1.0]]); // Shape [1, 2] instead of [1, 3]

        let result = apply_padding_mask(scores, &mask);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("does not match expected")
        );
    }

    #[test]
    fn test_create_padding_mask_from_tokens() {
        // Tokens: [1, 2, 0, 0] (0 is pad)
        let tokens = arr2(&[[1.0, 2.0, 0.0, 0.0]]);
        let pad_id = 0.0;

        let mask = create_padding_mask_from_tokens(&tokens, pad_id);

        assert_eq!(mask, arr2(&[[1.0, 1.0, 0.0, 0.0]]));
    }

    // ========================================================================
    //  2. Bias / General Mask Tests
    // ========================================================================

    #[test]
    fn test_apply_bias_mask_broadcasting() {
        // Batch=2, Heads=1, Q=2, K=2
        let mut scores = Array4::zeros((2, 1, 2, 2));

        // Bias Mask [2, 2] - e.g. Causal or ALiBi
        // 1 0
        // 1 1
        let mask = arr2(&[[1.0, 0.0], [1.0, 1.0]]);

        let result = apply_bias_mask(scores.clone(), &mask).unwrap();

        // Batch 0
        assert_eq!(result[[0, 0, 0, 1]], MASK_VALUE); // Top right masked
        assert_eq!(result[[0, 0, 1, 0]], 0.0); // Bottom left kept

        // Batch 1 (Should match)
        assert_eq!(result[[1, 0, 0, 1]], MASK_VALUE);
    }

    #[test]
    fn test_apply_bias_mask_shape_error() {
        let scores = Array4::zeros((1, 1, 2, 2));
        let mask = arr2(&[[1.0]]); // Wrong shape
        assert!(apply_bias_mask(scores, &mask).is_err());
    }

    // ========================================================================
    //  3. Unified Attention Mask Dispatch
    // ========================================================================

    #[test]
    fn test_apply_attention_mask_dispatch() {
        let scores = Array4::zeros((1, 1, 2, 2));

        // Case A: Padding mask [Batch=1, K=2]
        let pad_mask = arr2(&[[1.0, 0.0]]);
        let res_pad = apply_attention_mask(scores.clone(), &pad_mask);
        // Should mask column 1 for all queries
        assert_eq!(res_pad[[0, 0, 0, 1]], MASK_VALUE);
        assert_eq!(res_pad[[0, 0, 1, 1]], MASK_VALUE);

        // Case B: Causal/Bias mask [Q=2, K=2]
        let causal_mask = arr2(&[[1.0, 0.0], [1.0, 1.0]]);
        let res_causal = apply_attention_mask(scores.clone(), &causal_mask);
        // Should mask top right [0, 1]
        assert_eq!(res_causal[[0, 0, 0, 1]], MASK_VALUE);
        // Should keep bottom left [1, 0]
        assert_eq!(res_causal[[0, 0, 1, 0]], 0.0);
    }

    // ========================================================================
    //  4. Causal Masking Logic
    // ========================================================================

    #[test]
    fn test_create_causal_mask_square() {
        let mask = create_causal_mask(3, 3);
        let expected = arr2(&[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]);
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_create_causal_mask_rectangular_decoding() {
        // Scenario: Decoding step.
        // We have 5 tokens total (4 past + 1 current).
        // We only compute Q for the 1 current token.
        // It should attend to all 5 tokens.

        let q_len = 1;
        let total_len = 5;
        let mask = create_causal_mask(q_len, total_len);

        assert_eq!(mask.shape(), &[1, 5]);
        // The single query should see everything before it (indices 0..=4)
        // Since it's the last token, it sees everything.
        assert_eq!(mask, arr2(&[[1.0, 1.0, 1.0, 1.0, 1.0]]));
    }

    #[test]
    fn test_create_causal_mask_window() {
        // Scenario: Processing a chunk of 2 tokens in a sequence of 4
        // Q=2, Total=4. Past=2.
        // Q[0] is abs pos 2. Should see 0,1,2.
        // Q[1] is abs pos 3. Should see 0,1,2,3.
        let mask = create_causal_mask(2, 4);
        let expected = arr2(&[[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]);
        assert_eq!(mask, expected);
    }

    #[test]
    #[should_panic(expected = "cannot be less than")]
    fn test_create_causal_mask_invalid() {
        create_causal_mask(5, 2);
    }

    #[test]
    fn test_apply_causal_mask_inplace() {
        // Batch=1, Heads=1, Q=3, K=3
        let mut scores = Array4::zeros((1, 1, 3, 3));

        // No cache (start from 0)
        apply_causal_mask(&mut scores, 0);

        assert_eq!(scores[[0, 0, 0, 1]], MASK_VALUE); // Top right
        assert_eq!(scores[[0, 0, 1, 2]], MASK_VALUE); // Middle right
        assert_eq!(scores[[0, 0, 2, 0]], 0.0); // Bottom left OK
    }

    #[test]
    fn test_apply_causal_mask_with_cache() {
        // Q=1, K=5. Cache=4.
        // This is the decoding step for the 5th token.
        let mut scores = Array4::zeros((1, 1, 1, 5));

        apply_causal_mask(&mut scores, 4);

        // The query is at absolute pos 4.
        // It can see indices 0, 1, 2, 3, 4.
        // So nothing should be masked.
        assert_eq!(scores[[0, 0, 0, 4]], 0.0);
    }

    // ========================================================================
    //  5. Pooling Tests
    // ========================================================================

    #[test]
    fn test_last_token_pool() {
        // Batch=2, Seq=3, Hidden=2
        // [[A, B, Pad],
        //  [C, Pad, Pad]]
        let mut hidden = Array3::zeros((2, 3, 2));

        // B0, S0 = [1, 1]
        // B0, S1 = [2, 2] <-- Last Valid
        // B0, S2 = [0, 0]
        hidden.slice_mut(s![0, 0, ..]).fill(1.0);
        hidden.slice_mut(s![0, 1, ..]).fill(2.0);

        // B1, S0 = [3, 3] <-- Last Valid
        hidden.slice_mut(s![1, 0, ..]).fill(3.0);

        let mask = arr2(&[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]);

        let pooled = last_token_pool(&hidden, &mask);

        // Expected output shape [2, 2]
        assert_eq!(pooled.shape(), &[2, 2]);

        // Batch 0 should grab index 1 ([2.0, 2.0])
        assert_eq!(pooled.row(0), ndarray::arr1(&[2.0, 2.0]));

        // Batch 1 should grab index 0 ([3.0, 3.0])
        assert_eq!(pooled.row(1), ndarray::arr1(&[3.0, 3.0]));
    }

    #[test]
    fn test_last_token_pool_all_masked_edge_case() {
        let hidden = Array3::zeros((1, 2, 2));
        // Mask is all zeros
        let mask = arr2(&[[0.0, 0.0]]);

        // Should default to first token
        let pooled = last_token_pool(&hidden, &mask);
        assert_eq!(pooled.shape(), &[1, 2]);
    }

    // ========================================================================
    //  6. Bug Reproduction / Verification
    // ========================================================================

    #[test]
    fn test_create_batched_causal_mask() {
        let mask = create_batched_causal_mask(2, 4);

        assert_eq!(mask.dim(), (2, 4, 4));

        // Verify causal pattern: lower triangle (including diagonal) is 1.0, upper triangle is 0.0 or -inf
        for b in 0..2 {
            for i in 0..4 {
                for j in 0..4 {
                    if j <= i {
                        // Can attend to current and previous positions
                        assert_eq!(
                            mask[[b, i, j]],
                            1.0,
                            "Expected 1.0 at [{}, {}, {}]",
                            b,
                            i,
                            j
                        );
                    } else {
                        // Cannot attend to future positions (either 0.0 or -inf depending on implementation)
                        assert!(
                            mask[[b, i, j]] == 0.0
                                || (mask[[b, i, j]].is_infinite() && mask[[b, i, j]] < 0.0),
                            "Expected 0.0 or -inf at [{}, {}, {}], got {}",
                            b,
                            i,
                            j,
                            mask[[b, i, j]]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_expand_mask_for_attention() {
        // Input: [Batch=1, Seq=2]
        let mask = arr2(&[[1.0, 0.0]]);
        let num_heads = 2;

        let expanded = expand_mask_for_attention(&mask, num_heads);

        // Expected: [1, 2, 2]
        assert_eq!(expanded.shape(), &[1, 2, 2]);

        // Head 0
        assert_eq!(expanded[[0, 0, 0]], 1.0);
        assert_eq!(expanded[[0, 0, 1]], 0.0);

        // Head 1 (Copy)
        assert_eq!(expanded[[0, 1, 0]], 1.0);
        assert_eq!(expanded[[0, 1, 1]], 0.0);
    }

    #[test]
    fn test_causal_mask() {
        let mask = create_causal_mask(3, 3);

        // Expected:
        // [[1, 0, 0],
        //  [1, 1, 0],
        //  [1, 1, 1]]

        assert_eq!(mask[[0, 0]], 1.0);
        assert_eq!(mask[[0, 1]], 0.0);
        assert_eq!(mask[[0, 2]], 0.0);

        assert_eq!(mask[[1, 0]], 1.0);
        assert_eq!(mask[[1, 1]], 1.0);
        assert_eq!(mask[[1, 2]], 0.0);

        assert_eq!(mask[[2, 0]], 1.0);
        assert_eq!(mask[[2, 1]], 1.0);
        assert_eq!(mask[[2, 2]], 1.0);
    }

    #[test]
    fn test_padding_mask() {
        let tokens = Array2::from_shape_vec((1, 5), vec![1.0, 2.0, 3.0, 0.0, 0.0]).unwrap();
        let mask = create_padding_mask_from_tokens(&tokens, 0.0);

        assert_eq!(mask[[0, 0]], 1.0);
        assert_eq!(mask[[0, 1]], 1.0);
        assert_eq!(mask[[0, 2]], 1.0);
        assert_eq!(mask[[0, 3]], 0.0);
        assert_eq!(mask[[0, 4]], 0.0);
    }
}
