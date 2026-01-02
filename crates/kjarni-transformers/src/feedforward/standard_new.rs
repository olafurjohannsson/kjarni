use crate::activations::{self, Activation, apply_activation, apply_activation_2d};
use crate::linear_layer::LinearLayer;
use anyhow::Result;
use ndarray::{Array1, Array2, Array3};

pub struct StdFeedForwardNew {
    fc1: LinearLayer,
    fc2: LinearLayer,
    activation: Activation,
}

impl StdFeedForwardNew {
  
    pub fn new(
        fc1: impl Into<LinearLayer>,
        fc2: impl Into<LinearLayer>,
        activation: Activation,
    ) -> Self {
        Self {
            fc1: fc1.into(),
            fc2: fc2.into(),
            activation,
        }
    }


    pub fn forward(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, _) = hidden.dim();

        // Ensure contiguous layout before reshape
        let hidden_contig = hidden.as_standard_layout();
        let hidden_2d = hidden_contig
            .view()
            .into_shape_with_order((batch * seq, hidden.shape()[2]))?;

        // FC1 + Activation
        let mut intermediate = self.fc1.matmul(&hidden_2d);
        apply_activation_2d(&mut intermediate, self.activation);

        // FC2
        let output = self.fc2.matmul(&intermediate.view());

        Ok(output.into_shape_with_order((batch, seq, self.fc2.out_features()))?)
    }

}
