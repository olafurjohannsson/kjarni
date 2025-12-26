//! Pooling strategies for transformer outputs

use anyhow::Result;
use ndarray::{Array2, Array3, Axis};
use crate::utils::MASK_VALUE;

pub use crate::encoder::config::{EncodingConfig, PoolingStrategy};

/// Performs mean pooling over sequence dimension to convert token embeddings to sentence embeddings.
///
/// Takes a 3D tensor of shape [batch, sequence_length, hidden_size] containing token embeddings
/// and produces a 2D tensor of shape [batch, hidden_size] containing sentence embeddings.
///
/// The attention_mask indicates which tokens are real (1.0) vs padding (0.0).
/// Only real tokens contribute to the mean.
pub fn mean_pool(hidden: &Array3<f32>, attention_mask: &Array2<f32>) -> Result<Array2<f32>> {
    // Expand attention mask from [batch, seq] to [batch, seq, 1] for broadcasting
    let mask_expanded = attention_mask.view().insert_axis(ndarray::Axis(2));
    
    // Element-wise multiply hidden states with the mask. This zeros out padding tokens.
    let masked_hidden = hidden * &mask_expanded;
    
    // Sum along the sequence axis. Shape becomes [batch, hidden_size]
    let sum = masked_hidden.sum_axis(ndarray::Axis(1));
    
    // Count the number of non-padding tokens for each sentence in the batch.
    // Add a small epsilon (or clamp to 1.0) to avoid division by zero for empty sequences.
    let count = attention_mask
        .sum_axis(ndarray::Axis(1))
        .mapv(|x| x.max(1e-9)) // Use max to avoid division by zero
        .insert_axis(ndarray::Axis(1));

    // Divide the sum by the count to get the mean
    Ok(sum / &count)
}

/// Extract CLS token embedding (first token)
pub fn cls_pool(hidden: &Array3<f32>) -> Result<Array2<f32>> {
    Ok(hidden.slice(ndarray::s![.., 0, ..]).to_owned())
}

/// Max pooling over sequence dimension
pub fn max_pool(hidden: &Array3<f32>, attention_mask: &Array2<f32>) -> Result<Array2<f32>> {
    let mask_expanded = attention_mask.view().insert_axis(Axis(2));
    // Set masked positions to -inf before max pooling
    let mut masked_hidden = hidden.clone();
    masked_hidden.zip_mut_with(&mask_expanded, |h, &m| {
        if m == 0.0 {
            *h = MASK_VALUE;
        }
    });

    Ok(masked_hidden.fold_axis(Axis(1), MASK_VALUE, |&acc, &x| acc.max(x)))
}

/// Extract last token embedding
pub fn last_token_pool(hidden: &Array3<f32>, attention_mask: &Array2<f32>) -> Result<Array2<f32>> {
    let batch_size = hidden.shape()[0];
    let hidden_size = hidden.shape()[2];
    let mut output = Array2::<f32>::zeros((batch_size, hidden_size));

    for i in 0..batch_size {
        // Find last non-padding position
        let seq_mask = attention_mask.row(i);
        let last_pos = seq_mask.iter().rposition(|&x| x > 0.0).unwrap_or(0);

        output
            .row_mut(i)
            .assign(&hidden.slice(ndarray::s![i, last_pos, ..]));
    }

    Ok(output)
}
