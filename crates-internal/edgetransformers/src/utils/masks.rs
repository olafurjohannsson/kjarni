use ndarray::{Array2, Array3};

pub const MASK_VALUE: f32 = -1e9; // SAME as GPU

/// Create a full attention mask (all positions visible)
///
/// Used for testing or when no masking is needed.
/// Returns [batch_size, seq_len] with all 1.0
pub fn create_full_attention_mask(batch_size: usize, seq_len: usize) -> Array2<f32> {
    Array2::ones((batch_size, seq_len))
}

/// Create a causal attention mask where position i can only attend to positions 0..=i
///
/// Used for autoregressive generation (GPT-2, GPT-3, etc.)
/// Returns [seq_len, seq_len] with 1.0 for allowed positions, 0.0 for masked
pub fn create_causal_mask(seq_len: usize) -> Array2<f32> {
    let mut mask = Array2::zeros((seq_len, seq_len));
    for i in 0..seq_len {
        for j in 0..=i {
            mask[[i, j]] = 1.0;
        }
    }
    mask
}

/// Create a causal mask for a batch
///
/// Returns [batch_size, seq_len, seq_len]
pub fn create_batched_causal_mask(batch_size: usize, seq_len: usize) -> Array3<f32> {
    let single_mask = create_causal_mask(seq_len);
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
        let mask = create_causal_mask(3);
        
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