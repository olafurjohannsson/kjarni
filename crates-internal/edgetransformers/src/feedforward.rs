//! Feed-forward network implementation

use crate::activations::gelu;
use crate::utils::linear_algebra::{matmul_3d_2d};
use anyhow::Result;
use ndarray::{Array1, Array2, Array3};

/// Feed-forward network with two linear layers
pub struct FeedForward {
    pub dense1_weight_t: Array2<f32>,
    pub dense1_bias: Array1<f32>,
    pub dense2_weight_t: Array2<f32>,
    pub dense2_bias: Array1<f32>,
}

impl FeedForward {
    pub fn new(
        dense1_weight: Array2<f32>,
        dense1_bias: Array1<f32>,
        dense2_weight: Array2<f32>,
        dense2_bias: Array1<f32>,
    ) -> Self {
        Self {
            dense1_weight_t: dense1_weight,
            dense1_bias,
            dense2_weight_t: dense2_weight,
            dense2_bias,
        }
    }

    pub fn fc1(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let mut output = matmul_3d_2d(hidden, &self.dense1_weight_t);
        output += &self.dense1_bias;
        Ok(output)
    }

    pub fn apply_activation(&self, hidden: &mut Array3<f32>) {
        gelu(hidden);
    }

    pub fn fc2(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let mut output = matmul_3d_2d(hidden, &self.dense2_weight_t);
        output += &self.dense2_bias;
        Ok(output)
    }

    pub fn forward(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let mut intermediate = self.fc1(hidden)?;
        self.apply_activation(&mut intermediate);
        let output = self.fc2(&intermediate)?;
        Ok(output)
    }
}
