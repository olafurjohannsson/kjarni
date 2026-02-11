use anyhow::anyhow;
use ndarray::{Array2, Array3, Array4, Axis, Zip, s};

pub const MASK_VALUE: f32 = -1e9;

/// Apply padding mask to attention scores
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

    let mask_view = mask.view().insert_axis(Axis(0)).insert_axis(Axis(0));

    if let Some(m) = mask_view.broadcast((batch, heads, q, k)) {
        Zip::from(&mut scores).and(&m).for_each(|s, &mask_val| {
            if mask_val == 0.0 {
                *s = MASK_VALUE;
            }
        });
    }
    Ok(scores)
}

/// Pools the hidden states by taking the hidden state of the last non-padding token.
pub fn last_token_pool(hidden_states: &Array3<f32>, attention_mask: &Array2<f32>) -> Array2<f32> {
    let (batch_size, _, hidden_size) = hidden_states.dim();
    let mut pooled = Array2::zeros((batch_size, hidden_size));

    for i in 0..batch_size {
        let last_token_index = attention_mask
            .row(i)
            .iter()
            .rposition(|&x| x == 1.0)
            .unwrap_or(0); 

        let last_token_hidden_state = hidden_states.slice(s![i, last_token_index, ..]);

        pooled.row_mut(i).assign(&last_token_hidden_state);
    }

    pooled
}

pub fn apply_attention_mask(mut scores: Array4<f32>, mask: &Array2<f32>) -> Array4<f32> {
    let batch = scores.dim().0;

    let mask_view = if mask.nrows() == batch {
        mask.view().insert_axis(Axis(1)).insert_axis(Axis(2))
    } else {
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
pub fn create_full_attention_mask(batch_size: usize, seq_len: usize) -> Array2<f32> {
    Array2::ones((batch_size, seq_len))
}

/// Create a causal attention mask for any shape.
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
pub fn create_padding_mask_from_tokens(token_ids: &Array2<f32>, pad_token_id: f32) -> Array2<f32> {
    token_ids.mapv(|id| if id == pad_token_id { 0.0 } else { 1.0 })
}

/// Expand padding mask for multi-head attention
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
    use ndarray::{Array4, arr2};

    #[test]
    fn test_apply_padding_mask_basic() {
        // Batch=1, Heads=1, Q=2, K=3
        let scores = Array4::zeros((1, 1, 2, 3));

        let mask = arr2(&[[1.0, 1.0, 0.0]]);

        let result = apply_padding_mask(scores.clone(), &mask).unwrap();

        assert_eq!(result[[0, 0, 0, 0]], 0.0);
        assert_eq!(result[[0, 0, 0, 1]], 0.0);
        assert_eq!(result[[0, 0, 0, 2]], MASK_VALUE);

        assert_eq!(result[[0, 0, 1, 0]], 0.0);
        assert_eq!(result[[0, 0, 1, 2]], MASK_VALUE);
    }

    #[test]
    fn test_apply_padding_mask_shape_mismatch() {
        let scores = Array4::zeros((1, 1, 2, 3));
        let mask = arr2(&[[1.0, 1.0]]); 

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
        let tokens = arr2(&[[1.0, 2.0, 0.0, 0.0]]);
        let pad_id = 0.0;

        let mask = create_padding_mask_from_tokens(&tokens, pad_id);

        assert_eq!(mask, arr2(&[[1.0, 1.0, 0.0, 0.0]]));
    }

    #[test]
    fn test_apply_bias_mask_broadcasting() {
        let scores = Array4::zeros((2, 1, 2, 2));

        let mask = arr2(&[[1.0, 0.0], [1.0, 1.0]]);

        let result = apply_bias_mask(scores.clone(), &mask).unwrap();

        assert_eq!(result[[0, 0, 0, 1]], MASK_VALUE);
        assert_eq!(result[[0, 0, 1, 0]], 0.0); 

        assert_eq!(result[[1, 0, 0, 1]], MASK_VALUE);
    }

    #[test]
    fn test_apply_bias_mask_shape_error() {
        let scores = Array4::zeros((1, 1, 2, 2));
        let mask = arr2(&[[1.0]]); // Wrong shape
        assert!(apply_bias_mask(scores, &mask).is_err());
    }

    #[test]
    fn test_apply_attention_mask_dispatch() {
        let scores = Array4::zeros((1, 1, 2, 2));

        let pad_mask = arr2(&[[1.0, 0.0]]);
        let res_pad = apply_attention_mask(scores.clone(), &pad_mask);
        assert_eq!(res_pad[[0, 0, 0, 1]], MASK_VALUE);
        assert_eq!(res_pad[[0, 0, 1, 1]], MASK_VALUE);
        let causal_mask = arr2(&[[1.0, 0.0], [1.0, 1.0]]);
        let res_causal = apply_attention_mask(scores.clone(), &causal_mask);
        assert_eq!(res_causal[[0, 0, 0, 1]], MASK_VALUE);
    
        assert_eq!(res_causal[[0, 0, 1, 0]], 0.0);
    }

    #[test]
    fn test_create_causal_mask_square() {
        let mask = create_causal_mask(3, 3);
        let expected = arr2(&[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]);
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_create_causal_mask_rectangular_decoding() {
        let q_len = 1;
        let total_len = 5;
        let mask = create_causal_mask(q_len, total_len);

        assert_eq!(mask.shape(), &[1, 5]);
        assert_eq!(mask, arr2(&[[1.0, 1.0, 1.0, 1.0, 1.0]]));
    }

    #[test]
    fn test_create_causal_mask_window() {
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
        let mut scores = Array4::zeros((1, 1, 3, 3));

        apply_causal_mask(&mut scores, 0);

        assert_eq!(scores[[0, 0, 0, 1]], MASK_VALUE); 
        assert_eq!(scores[[0, 0, 1, 2]], MASK_VALUE); 
        assert_eq!(scores[[0, 0, 2, 0]], 0.0); 
    }

    #[test]
    fn test_apply_causal_mask_with_cache() {
        let mut scores = Array4::zeros((1, 1, 1, 5));

        apply_causal_mask(&mut scores, 4);

        assert_eq!(scores[[0, 0, 0, 4]], 0.0);
    }
    #[test]
    fn test_last_token_pool() {
        let mut hidden = Array3::zeros((2, 3, 2));

        hidden.slice_mut(s![0, 0, ..]).fill(1.0);
        hidden.slice_mut(s![0, 1, ..]).fill(2.0);

        hidden.slice_mut(s![1, 0, ..]).fill(3.0);

        let mask = arr2(&[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]);

        let pooled = last_token_pool(&hidden, &mask);

        assert_eq!(pooled.shape(), &[2, 2]);

        assert_eq!(pooled.row(0), ndarray::arr1(&[2.0, 2.0]));

        assert_eq!(pooled.row(1), ndarray::arr1(&[3.0, 3.0]));
    }

    #[test]
    fn test_last_token_pool_all_masked_edge_case() {
        let hidden = Array3::zeros((1, 2, 2));
        let mask = arr2(&[[0.0, 0.0]]);

        let pooled = last_token_pool(&hidden, &mask);
        assert_eq!(pooled.shape(), &[1, 2]);
    }

    #[test]
    fn test_create_batched_causal_mask() {
        let mask = create_batched_causal_mask(2, 4);

        assert_eq!(mask.dim(), (2, 4, 4));
            for b in 0..2 {
            for i in 0..4 {
                for j in 0..4 {
                    if j <= i {
                        assert_eq!(
                            mask[[b, i, j]],
                            1.0,
                            "Expected 1.0 at [{}, {}, {}]",
                            b,
                            i,
                            j
                        );
                    } else {
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
        let mask = arr2(&[[1.0, 0.0]]);
        let num_heads = 2;

        let expanded = expand_mask_for_attention(&mask, num_heads);

        assert_eq!(expanded.shape(), &[1, 2, 2]);

        assert_eq!(expanded[[0, 0, 0]], 1.0);
        assert_eq!(expanded[[0, 0, 1]], 0.0);

        assert_eq!(expanded[[0, 1, 0]], 1.0);
        assert_eq!(expanded[[0, 1, 1]], 0.0);
    }

    #[test]
    fn test_causal_mask() {
        let mask = create_causal_mask(3, 3);

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
