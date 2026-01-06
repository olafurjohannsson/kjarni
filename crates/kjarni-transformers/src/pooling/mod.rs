//! Pooling strategies for transformer outputs

use crate::utils::MASK_VALUE;
use anyhow::Result;
use ndarray::{Array2, Array3, Axis};

pub use crate::cpu::encoder::config::{EncodingConfig, PoolingStrategy};

pub fn mean_pool(hidden: &Array3<f32>, attention_mask: &Array2<f32>) -> Result<Array2<f32>> {
    let mask_expanded = attention_mask.view().insert_axis(ndarray::Axis(2));
    let masked_hidden = hidden * &mask_expanded;
    let sum = masked_hidden.sum_axis(ndarray::Axis(1));

    // Count of valid tokens
    let count = attention_mask.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1));

    // Avoid division by zero: if count == 0, use 1.0 to keep original values
    let count_safe = count.mapv(|x| if x == 0.0 { 1.0 } else { x });

    let pooled = &sum / &count_safe;

    // For fully masked sequences, fallback to first token
    let batch_size = hidden.shape()[0];
    let hidden_size = hidden.shape()[2];
    let mut output = Array2::<f32>::zeros((batch_size, hidden_size));
    for i in 0..batch_size {
        if attention_mask.row(i).sum() == 0.0 {
            output.row_mut(i).assign(&hidden.slice(ndarray::s![i, 0, ..]));
        } else {
            output.row_mut(i).assign(&pooled.row(i));
        }
    }

    Ok(output)
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mean_pool_basic() {
        let hidden = array![
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ]; // shape [2,2,2]

        let attention_mask = array![
            [1.0, 1.0],
            [1.0, 0.0]
        ]; // second batch has padding in second token

        let pooled = mean_pool(&hidden, &attention_mask).unwrap();

        // Batch 0: mean of [[1,2],[3,4]] = [2,3]
        // Batch 1: mean of [[5,6]] = [5,6] (second token ignored)
        assert_eq!(pooled.shape(), &[2, 2]);
        assert!((pooled[[0, 0]] - 2.0).abs() < 1e-6);
        assert!((pooled[[0, 1]] - 3.0).abs() < 1e-6);
        assert!((pooled[[1, 0]] - 5.0).abs() < 1e-6);
        assert!((pooled[[1, 1]] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_cls_pool() {
        let hidden = array![
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ]; // shape [2,2,2]

        let pooled = cls_pool(&hidden).unwrap();

        // CLS is first token
        assert_eq!(pooled.shape(), &[2, 2]);
        assert_eq!(pooled[[0, 0]], 1.0);
        assert_eq!(pooled[[0, 1]], 2.0);
        assert_eq!(pooled[[1, 0]], 5.0);
        assert_eq!(pooled[[1, 1]], 6.0);
    }

    #[test]
    fn test_max_pool_with_padding() {
        let hidden = array![
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ]; // shape [2,2,2]

        let attention_mask = array![
            [1.0, 1.0],
            [1.0, 0.0]
        ]; // second batch has padding

        let pooled = max_pool(&hidden, &attention_mask).unwrap();

        // Batch 0: max of [[1,2],[3,4]] = [3,4]
        // Batch 1: max of [[5,6]] = [5,6] (second token ignored)
        assert_eq!(pooled.shape(), &[2, 2]);
        assert!((pooled[[0, 0]] - 3.0).abs() < 1e-6);
        assert!((pooled[[0, 1]] - 4.0).abs() < 1e-6);
        assert!((pooled[[1, 0]] - 5.0).abs() < 1e-6);
        assert!((pooled[[1, 1]] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_last_token_pool_with_padding() {
        let hidden = array![
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ]; // shape [2,3,2]

        let attention_mask = array![
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ]; // first batch last token is padding

        let pooled = last_token_pool(&hidden, &attention_mask).unwrap();

        // Batch 0: last valid token = second token [3,4]
        // Batch 1: last valid token = third token [11,12]
        assert_eq!(pooled.shape(), &[2, 2]);
        assert_eq!(pooled[[0, 0]], 3.0);
        assert_eq!(pooled[[0, 1]], 4.0);
        assert_eq!(pooled[[1, 0]], 11.0);
        assert_eq!(pooled[[1, 1]], 12.0);
    }

    #[test]
    fn test_mean_pool_empty_sequence() {
        let hidden = array![[[1.0, 2.0]]]; // shape [1,1,2]
        let attention_mask = array![[0.0]]; // all padding

        let pooled = mean_pool(&hidden, &attention_mask).unwrap();

        // Should not divide by zero
        assert_eq!(pooled[[0, 0]], 1.0);
        assert_eq!(pooled[[0, 1]], 2.0);
    }

    #[test]
    fn test_max_pool_all_masked() {
        let hidden = array![[[1.0, 2.0]]]; // shape [1,1,2]
        let attention_mask = array![[0.0]]; // all masked

        let pooled = max_pool(&hidden, &attention_mask).unwrap();

        // All masked -> returns MASK_VALUE
        assert_eq!(pooled[[0, 0]], MASK_VALUE);
        assert_eq!(pooled[[0, 1]], MASK_VALUE);
    }
}
