//! SwiGLU Feed-Forward Network
//!
//! Used in LLaMA and other modern architectures.
//! SwiGLU: SiLU(gate(x)) ⊙ up(x), then down()
//!
//! Unlike standard FFN with 2 projections, SwiGLU has 3:
//! - gate_proj: [hidden_size, intermediate_size]
//! - up_proj:   [hidden_size, intermediate_size]  
//! - down_proj: [intermediate_size, hidden_size]

use crate::utils::linear_algebra::matmul_3d_2d;
use anyhow::Result;
use ndarray::{Array2, Array3};

/// SwiGLU activation-based Feed-Forward Network
///
/// Formula: FFN(x) = down(SiLU(gate(x)) ⊙ up(x))
/// where ⊙ is element-wise multiplication
pub struct SwiGluFeedForward {
    /// Gate projection: [hidden_size, intermediate_size]
    pub gate_weight: Array2<f32>,

    /// Up projection: [hidden_size, intermediate_size]
    pub up_weight: Array2<f32>,

    /// Down projection: [intermediate_size, hidden_size]
    pub down_weight: Array2<f32>,
}

impl SwiGluFeedForward {
    /// Create a new SwiGLU FFN layer
    ///
    /// # Arguments
    /// * `gate_weight` - Gate projection weights [hidden_size, intermediate_size]
    /// * `up_weight` - Up projection weights [hidden_size, intermediate_size]
    /// * `down_weight` - Down projection weights [intermediate_size, hidden_size]
    ///
    /// # Note
    /// LLaMA models do NOT have bias terms in FFN layers
    pub fn new(gate_weight: Array2<f32>, up_weight: Array2<f32>, down_weight: Array2<f32>) -> Self {
        // Validate shapes
        let hidden_size = gate_weight.shape()[0];
        let intermediate_size = gate_weight.shape()[1];

        assert_eq!(
            up_weight.shape(),
            &[hidden_size, intermediate_size],
            "up_weight shape must match gate_weight"
        );
        assert_eq!(
            down_weight.shape(),
            &[intermediate_size, hidden_size],
            "down_weight shape must be [intermediate_size, hidden_size]"
        );

        Self {
            gate_weight,
            up_weight,
            down_weight,
        }
    }

    /// Forward pass for 3D input [batch, seq_len, hidden_size]
    pub fn forward(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        // 1. Gate and Up projections
        let gate_out = matmul_3d_2d(hidden, &self.gate_weight);
        let up_out = matmul_3d_2d(hidden, &self.up_weight);

        // 2. ✅ CORRECTED: Apply SwiGLU activation in a vectorized way
        let activated = silu_3d(&gate_out) * up_out;

        // 3. Down projection
        Ok(matmul_3d_2d(&activated, &self.down_weight))
    }

    /// Forward pass for 2D input [batch, hidden_size]
    pub fn forward_2d(&self, hidden: &Array2<f32>) -> Array2<f32> {
        // 1. Gate and up projections
        let gate_out = hidden.dot(&self.gate_weight);
        let up_out = hidden.dot(&self.up_weight);

        // 2. ✅ CORRECTED: Apply SwiGLU activation in a vectorized way
        let activated = silu_2d(&gate_out) * up_out;

        // 3. Down projection
        activated.dot(&self.down_weight)
    }
}

/// SiLU (Swish) activation function: x * sigmoid(x)
///
/// Also known as Swish, this is used in the gate projection of SwiGLU.
/// Formula: SiLU(x) = x / (1 + exp(-x))
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Vectorized SiLU for Array3
pub fn silu_3d(x: &Array3<f32>) -> Array3<f32> {
    x.mapv(silu)
}

/// Vectorized SiLU for Array2
pub fn silu_2d(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(silu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    #[test]
    fn test_silu_activation() {
        // Test SiLU function
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!((silu(1.0) - 0.7311).abs() < 1e-3); // 1 * sigmoid(1) ≈ 0.7311
        assert!((silu(-1.0) - (-0.2689)).abs() < 1e-3); // -1 * sigmoid(-1) ≈ -0.2689

        // SiLU should be close to x for large positive x
        assert!((silu(10.0) - 10.0).abs() < 0.01);

        // SiLU should be close to 0 for large negative x
        assert!(silu(-10.0).abs() < 0.01);
    }

    #[test]
    fn test_swiglu_ffn_shapes() -> Result<()> {
        let hidden_size = 64;
        let intermediate_size = 256;
        let batch_size = 2;
        let seq_len = 10;

        let gate_weight = Array2::zeros((hidden_size, intermediate_size));
        let up_weight = Array2::zeros((hidden_size, intermediate_size));
        let down_weight = Array2::zeros((intermediate_size, hidden_size));

        let ffn = SwiGluFeedForward::new(gate_weight, up_weight, down_weight);

        let input = Array3::zeros((batch_size, seq_len, hidden_size));
        let output = ffn.forward(&input)?;

        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
        Ok(())
    }

    #[test]
    fn test_swiglu_ffn_basic() -> Result<()> {
        // Simple test with small dimensions
        let hidden_size = 4;
        let intermediate_size = 8;

        // Identity-like weights for testing
        let mut gate_weight = Array2::zeros((hidden_size, intermediate_size));
        let mut up_weight = Array2::zeros((hidden_size, intermediate_size));
        let mut down_weight = Array2::zeros((intermediate_size, hidden_size));

        // Set some non-zero values
        for i in 0..hidden_size.min(intermediate_size) {
            gate_weight[[i, i]] = 1.0;
            up_weight[[i, i]] = 1.0;
            down_weight[[i, i]] = 1.0;
        }

        let ffn = SwiGluFeedForward::new(gate_weight, up_weight, down_weight);

        let input = Array3::ones((1, 1, hidden_size));
        let output = ffn.forward(&input)?;

        // Output should be finite and reasonable
        assert!(output.iter().all(|&x| x.is_finite()));
        assert_eq!(output.shape(), &[1, 1, hidden_size]);
        Ok(())
    }

    #[test]
    fn test_swiglu_vs_standard_ffn() -> Result<()> {
        // Compare SwiGLU behavior with known values
        let hidden_size = 2;
        let intermediate_size = 4;

        // Simple weights
        let gate_weight =
            Array2::from_shape_vec((2, 4), vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();

        let up_weight =
            Array2::from_shape_vec((2, 4), vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();

        let down_weight =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]).unwrap();

        let ffn = SwiGluFeedForward::new(gate_weight, up_weight, down_weight);

        // Input: [1, 2]
        let input = Array3::from_shape_vec((1, 1, 2), vec![1.0, 2.0]).unwrap();
        let output = ffn.forward(&input)?;

        // Gate output: [1, 2, 0, 0] (from first two cols of gate_weight)
        // Up output: [0, 0, 1, 2] (from last two cols of up_weight)
        // SwiGLU: [SiLU(1)*0, SiLU(2)*0, SiLU(0)*1, SiLU(0)*2]
        //       = [0, 0, 0, 0] (because up values are multiplied by 0 from gate)
        // This is expected - gate acts as a gating mechanism

        assert_eq!(output.shape(), &[1, 1, 2]);
        assert!(output.iter().all(|&x| x.is_finite()));
        Ok(())
    }

    #[test]
    fn test_swiglu_pytorch_parity() -> Result<()> {
        // Test against known PyTorch output
        // PyTorch code:
        // ```python
        // import torch
        // import torch.nn.functional as F
        //
        // x = torch.tensor([[[1.0, 2.0]]])  # [1, 1, 2]
        // gate_weight = torch.tensor([[1.0, 0.5], [0.5, 1.0]])  # [2, 2]
        // up_weight = torch.tensor([[0.5, 1.0], [1.0, 0.5]])    # [2, 2]
        // down_weight = torch.tensor([[1.0, 0.5], [0.5, 1.0]])  # [2, 2]
        //
        // gate_out = x @ gate_weight  # [1, 1, 2]
        // up_out = x @ up_weight      # [1, 1, 2]
        // activated = F.silu(gate_out) * up_out
        // output = activated @ down_weight
        // print(output)
        // ```
        // Output: tensor([[[6.7143, 6.8227]]])

        let gate_weight = Array2::from_shape_vec((2, 2), vec![1.0, 0.5, 0.5, 1.0]).unwrap();

        let up_weight = Array2::from_shape_vec((2, 2), vec![0.5, 1.0, 1.0, 0.5]).unwrap();

        let down_weight = Array2::from_shape_vec((2, 2), vec![1.0, 0.5, 0.5, 1.0]).unwrap();

        let ffn = SwiGluFeedForward::new(gate_weight, up_weight, down_weight);

        let input = Array3::from_shape_vec((1, 1, 2), vec![1.0, 2.0]).unwrap();
        let output = ffn.forward(&input)?;

    assert!((output[[0, 0, 0]] - 6.7142).abs() < 1e-3, 
        "Expected ~6.7142, got {}", output[[0, 0, 0]]);
    assert!((output[[0, 0, 1]] - 6.8224).abs() < 1e-3,
        "Expected ~6.8224, got {}", output[[0, 0, 1]]);
        Ok(())
    }

    #[test]
    fn test_swiglu_2d() {
        let hidden_size = 4;
        let intermediate_size = 8;
        let batch_size = 3;

        let gate_weight = Array2::ones((hidden_size, intermediate_size));
        let up_weight = Array2::ones((hidden_size, intermediate_size));
        let down_weight = Array2::ones((intermediate_size, hidden_size));

        let ffn = SwiGluFeedForward::new(gate_weight, up_weight, down_weight);

        let input = Array2::ones((batch_size, hidden_size));
        let output = ffn.forward_2d(&input);

        assert_eq!(output.shape(), &[batch_size, hidden_size]);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_swiglu_nonlinearity() -> std::result::Result<(), anyhow::Error> {
        // Test that SwiGLU is actually non-linear
        let hidden_size = 4;
        let intermediate_size = 4;

        let gate_weight = Array2::eye(hidden_size);
        let up_weight = Array2::eye(hidden_size);
        let down_weight = Array2::eye(hidden_size);

        let ffn = SwiGluFeedForward::new(gate_weight, up_weight, down_weight);

        // Input 1: all ones
        let input1 = Array3::ones((1, 1, hidden_size));
        let output1 = ffn.forward(&input1)?;

        // Input 2: all twos (2x input1)
        let input2 = Array3::from_elem((1, 1, hidden_size), 2.0);
        let output2 = ffn.forward(&input2)?;

        // Due to SiLU non-linearity, output2 should NOT be 2*output1
        let ratio = output2[[0, 0, 0]] / output1[[0, 0, 0]];
        assert!((ratio - 2.0).abs() > 0.1, "SwiGLU should be non-linear");

        Ok(())
    }
}
