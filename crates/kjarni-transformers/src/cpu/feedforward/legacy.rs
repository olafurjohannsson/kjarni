//! Feed-forward network implementation for the CPU backend.

use crate::activations::{Activation, apply_activation};
use crate::utils::linear_algebra::matmul_3d_2d;
use anyhow::Result;
use ndarray::{Array1, Array2, Array3};

/// A standard two-layer feed-forward network (FFN) for CPU computation.
///
pub struct LegacyFeedForward {
    pub dense1_weight: Array2<f32>,
    pub dense1_bias: Array1<f32>,
    pub dense2_weight: Array2<f32>,
    pub dense2_bias: Array1<f32>,
    pub activation: Activation,
}

impl LegacyFeedForward {
    pub fn new(
        dense1_weight: Array2<f32>,
        dense1_bias: Array1<f32>,
        dense2_weight: Array2<f32>,
        dense2_bias: Array1<f32>,
        activation: Activation,
    ) -> Self {
        assert_eq!(
            dense1_weight.shape()[1],
            dense1_bias.shape()[0],
            "FC1 weight and bias dimensions do not match (intermediate_size)"
        );
        assert_eq!(
            dense2_weight.shape()[1],
            dense2_bias.shape()[0],
            "FC2 weight and bias dimensions do not match (hidden_size)"
        );
        assert_eq!(
            dense1_weight.shape()[1],
            dense2_weight.shape()[0],
            "Inner dimensions of FC1 and FC2 do not match (intermediate_size)"
        );

        Self {
            dense1_weight,
            dense1_bias,
            dense2_weight,
            dense2_bias,
            activation,
        }
    }

    pub fn forward(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let mut intermediate = self.fc1(hidden)?;
        self.apply_activation(&mut intermediate);
        let output = self.fc2(&intermediate)?;
        Ok(output)
    }
    pub fn fc1(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let mut hidden = matmul_3d_2d(hidden, &self.dense1_weight);
        if self.dense1_bias.len() > 0 {
            hidden += &self.dense1_bias;
        }
        Ok(hidden)
    }

    pub fn apply_activation(&self, hidden: &mut Array3<f32>) {
        apply_activation(hidden, self.activation);
    }

    pub fn fc2(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let mut hidden = matmul_3d_2d(hidden, &self.dense2_weight);
        if self.dense2_bias.len() > 0 {
            hidden += &self.dense2_bias;
        }
        Ok(hidden)
    }
}
