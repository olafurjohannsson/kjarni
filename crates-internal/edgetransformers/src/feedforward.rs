//! Feed-forward network implementation for the CPU backend.

use crate::activations::{apply_activation, Activation};
use crate::utils::linear_algebra::matmul_3d_2d;
use anyhow::Result;
use ndarray::{Array1, Array2, Array3};

/// A standard two-layer feed-forward network (FFN) for CPU computation.
///
pub struct FeedForward {
    // These fields are now public again, as they represent the clean, expected layout.
    /// The weight matrix for the first linear layer (up-projection).
    /// Shape: `[hidden_size, intermediate_size]`.
    pub dense1_weight: Array2<f32>,
    /// The bias vector for the first linear layer.
    /// Shape: `[intermediate_size]`.
    pub dense1_bias: Array1<f32>,
    /// The weight matrix for the second linear layer (down-projection).
    /// Shape: `[intermediate_size, hidden_size]`.
    pub dense2_weight: Array2<f32>,
    /// The bias vector for the second linear layer.
    /// Shape: `[hidden_size]`.
    pub dense2_bias: Array1<f32>,
    /// The activation function to use between the two linear layers.
    pub activation: Activation,
}

impl FeedForward {
    /// Creates a new `FeedForward` layer from pre-prepared weights.
    ///
    /// # Arguments
    /// * `dense1_weight` - Weight matrix for FC1, must have shape `[hidden_size, intermediate_size]`.
    /// * `dense1_bias` - Bias vector for FC1, shape `[intermediate_size]`.
    /// * `dense2_weight` - Weight matrix for FC2, must have shape `[intermediate_size, hidden_size]`.
    /// * `dense2_bias` - Bias vector for FC2, shape `[hidden_size]`.
    /// * `activation` - The activation function to use.
    pub fn new(
        dense1_weight: Array2<f32>,
        dense1_bias: Array1<f32>,
        dense2_weight: Array2<f32>,
        dense2_bias: Array1<f32>,
        activation: Activation,
    ) -> Self {
        // Assertions to enforce the [in, out] layout invariant.
        assert_eq!(dense1_weight.shape()[1], dense1_bias.shape()[0], "FC1 weight and bias dimensions do not match (intermediate_size)");
        assert_eq!(dense2_weight.shape()[1], dense2_bias.shape()[0], "FC2 weight and bias dimensions do not match (hidden_size)");
        assert_eq!(dense1_weight.shape()[1], dense2_weight.shape()[0], "Inner dimensions of FC1 and FC2 do not match (intermediate_size)");

        Self {
            dense1_weight,
            dense1_bias,
            dense2_weight,
            dense2_bias,
            activation,
        }
    }

    /// Performs the first linear transformation (FC1).
    pub fn fc1(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        // This works because matmul_3d_2d expects the weight in [in, out] format.
        let mut output = matmul_3d_2d(hidden, &self.dense1_weight);
        output += &self.dense1_bias;
        Ok(output)
    }

    /// Applies the configured activation function in-place.
    pub fn apply_activation(&self, hidden: &mut Array3<f32>) {
        apply_activation(hidden, self.activation);
    }

    /// Performs the second linear transformation (FC2).
    pub fn fc2(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let mut output = matmul_3d_2d(hidden, &self.dense2_weight);
        output += &self.dense2_bias;
        Ok(output)
    }

    /// Performs the complete forward pass for the FeedForward network.
    pub fn forward(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let mut intermediate = self.fc1(hidden)?;
        self.apply_activation(&mut intermediate);
        let output = self.fc2(&intermediate)?;
        Ok(output)
    }
}