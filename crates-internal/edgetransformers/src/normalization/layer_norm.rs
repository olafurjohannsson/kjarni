//! Layer normalization implementation

use ndarray::{Array1, Array3, Axis};

/// Layer normalization
pub struct LayerNorm {
    pub weight: Array1<f32>,
    pub bias: Array1<f32>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Array1<f32>, bias: Array1<f32>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// Apply layer norm to a 3D tensor
    pub fn forward_3d(&self, hidden: &Array3<f32>) -> Array3<f32> {
        // 1. Calculate the mean and variance along the last axis (the feature dimension).
        //    `keep_dims` is false, so these will be 2D arrays.
        let mean = hidden.mean_axis(Axis(2)).unwrap();
        let variance = hidden.var_axis(Axis(2), 0.0);

        // 2. Expand the dimensions of the mean and variance so they can be broadcast
        //    for subtraction and division. Shape goes from [batch, seq] to [batch, seq, 1].
        let mean_expanded = mean.insert_axis(Axis(2));
        let var_expanded = variance.insert_axis(Axis(2));

        // 3. Normalize the hidden state: (x - mean) / sqrt(var + epsilon)
        let inv_std = (&var_expanded + self.eps).mapv(|x| 1.0 / x.sqrt());
        let normalized_hidden = (hidden - &mean_expanded) * &inv_std;

        // 4. Apply the learnable parameters: y = normalized * weight + bias
        //    This correctly broadcasts the 1D weight and bias vectors.
        let scaled = normalized_hidden * &self.weight;
        let shifted = scaled + &self.bias;

        shifted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_layer_norm_basic() {
        // Simple test with known values
        let weight = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let bias = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let eps = 1e-6;
        let layer_norm = LayerNorm::new(weight, bias, eps);

        // Input: [1, 1, 3] tensor with values [1.0, 2.0, 3.0]
        // Mean = 2.0, Variance = 2/3
        let hidden = Array3::from_shape_vec((1, 1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let output = layer_norm.forward_3d(&hidden);

        // After normalization: mean=0, variance=1
        let output_mean = (output[[0, 0, 0]] + output[[0, 0, 1]] + output[[0, 0, 2]]) / 3.0;
        assert!(output_mean.abs() < 1e-5); // Mean should be ~0

        // Verify normalized values
        // (1-2)/sqrt(2/3) ≈ -1.2247, (2-2)/sqrt(2/3) = 0, (3-2)/sqrt(2/3) ≈ 1.2247
        assert!((output[[0, 0, 0]] - (-1.2247)).abs() < 1e-3);
        assert!((output[[0, 0, 1]] - 0.0).abs() < 1e-5);
        assert!((output[[0, 0, 2]] - 1.2247).abs() < 1e-3);
    }

    #[test]
    fn test_layer_norm_with_scale_and_bias() {
        let weight = Array1::from_vec(vec![2.0, 0.5, 1.5]);
        let bias = Array1::from_vec(vec![1.0, -1.0, 0.5]);
        let eps = 1e-6;
        let layer_norm = LayerNorm::new(weight, bias, eps);

        let hidden = Array3::from_shape_vec((1, 1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let output = layer_norm.forward_3d(&hidden);

        // First normalize, then scale by weight and add bias
        let mean = 2.0;
        let var = 2.0 / 3.0;
        let std = (var + eps).sqrt();
        
        let normalized_0 = (1.0 - mean) / std;
        let normalized_1 = (2.0 - mean) / std;
        let normalized_2 = (3.0 - mean) / std;

        let expected_0 = normalized_0 * 2.0 + 1.0;
        let expected_1 = normalized_1 * 0.5 + (-1.0);
        let expected_2 = normalized_2 * 1.5 + 0.5;

        assert!((output[[0, 0, 0]] - expected_0).abs() < 1e-4);
        assert!((output[[0, 0, 1]] - expected_1).abs() < 1e-4);
        assert!((output[[0, 0, 2]] - expected_2).abs() < 1e-4);
    }

    #[test]
    fn test_layer_norm_batch() {
        let weight = Array1::from_vec(vec![1.0, 1.0]);
        let bias = Array1::from_vec(vec![0.0, 0.0]);
        let eps = 1e-5;
        let layer_norm = LayerNorm::new(weight, bias, eps);

        // [2, 2, 2] - batch=2, seq=2, hidden=2
        let hidden = Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                1.0, 3.0,  // batch 0, pos 0: mean=2, var=1
                2.0, 4.0,  // batch 0, pos 1: mean=3, var=1
                5.0, 7.0,  // batch 1, pos 0: mean=6, var=1
                6.0, 8.0,  // batch 1, pos 1: mean=7, var=1
            ],
        )
        .unwrap();

        let output = layer_norm.forward_3d(&hidden);

        // Each position should be normalized independently
        // Position [0, 0]: [1.0, 3.0] -> normalized to [-1, 1]
        assert!((output[[0, 0, 0]] - (-1.0)).abs() < 1e-3);
        assert!((output[[0, 0, 1]] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_layer_norm_pytorch_parity() {
        // Test against known PyTorch output
        // PyTorch code:
        // ```python
        // import torch
        // x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        // layer_norm = torch.nn.LayerNorm(4)
        // layer_norm.weight.data = torch.tensor([1.0, 1.0, 1.0, 1.0])
        // layer_norm.bias.data = torch.tensor([0.0, 0.0, 0.0, 0.0])
        // output = layer_norm(x)
        // print(output)
        // ```
        // Output: tensor([[[-1.3416, -0.4472,  0.4472,  1.3416]]])

        let weight = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let bias = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let eps = 1e-5;
        let layer_norm = LayerNorm::new(weight, bias, eps);

        let hidden = Array3::from_shape_vec((1, 1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = layer_norm.forward_3d(&hidden);

        assert!((output[[0, 0, 0]] - (-1.3416)).abs() < 1e-3);
        assert!((output[[0, 0, 1]] - (-0.4472)).abs() < 1e-3);
        assert!((output[[0, 0, 2]] - 0.4472).abs() < 1e-3);
        assert!((output[[0, 0, 3]] - 1.3416).abs() < 1e-3);
    }

    #[test]
    fn test_layer_norm_constant_input() {
        // When all values are the same, variance is 0
        // Should handle gracefully (eps prevents division by zero)
        let weight = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let bias = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let eps = 1e-5;
        let layer_norm = LayerNorm::new(weight, bias, eps);

        let hidden = Array3::from_shape_vec((1, 1, 3), vec![5.0, 5.0, 5.0]).unwrap();
        let output = layer_norm.forward_3d(&hidden);

        // All outputs should be 0 (mean-centered with no variance)
        assert!(output[[0, 0, 0]].abs() < 1e-3);
        assert!(output[[0, 0, 1]].abs() < 1e-3);
        assert!(output[[0, 0, 2]].abs() < 1e-3);
    }
}