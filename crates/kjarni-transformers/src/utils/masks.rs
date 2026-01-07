use anyhow::anyhow;
use ndarray::{s, Array2, Array3, Array4, Axis, Zip};

pub const MASK_VALUE: f32 = -1e9; // SAME as GPU

/// Apply padding mask to attention scores
///
/// Masks positions where mask[batch, key_pos] == 0
pub fn apply_padding_mask(mut scores: Array4<f32>, mask: &Array2<f32>) -> anyhow::Result<Array4<f32>> {
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
    let mut mask = Array2::zeros((q_len, total_len));
    let past_len = total_len - q_len;

    for i in 0..q_len {
        // The query at index 'i' is actually at absolute position 'past_len + i'
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
    let single_mask = create_causal_mask(seq_len, 0);
    let mut batched = Array3::zeros((batch_size, seq_len, seq_len));

    for b in 0..batch_size {
        batched.slice_mut(ndarray::s![b, .., ..])
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
pub fn expand_mask_for_attention(
    mask: &Array2<f32>,
    num_heads: usize,
) -> Array3<f32> {
    let (batch_size, seq_len) = mask.dim();
    let mut expanded = Array3::zeros((batch_size, num_heads, seq_len));

    for b in 0..batch_size {
        for h in 0..num_heads {
            expanded.slice_mut(ndarray::s![b, h, ..])
                .assign(&mask.row(b));
        }
    }

    expanded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask() {
        let mask = create_causal_mask(3, 0);

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