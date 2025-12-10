use crate::activations::{Activation, apply_activation_2d, apply_activation};
use crate::linear_layer::LinearLayer;
use anyhow::Result;
use ndarray::{Array1, Array2, Array3};

pub struct StdFeedForward {
    fc1: LinearLayer,
    fc2: LinearLayer,
    activation: Activation,
    pub dense1_weight: Array2<f32>,
    pub dense1_bias: Array1<f32>,
    pub dense2_weight: Array2<f32>,
    pub dense2_bias: Array1<f32>,
}

impl StdFeedForward {
    /// Creates a new `FeedForward` layer.
    ///
    /// # Arguments
    /// * `dense1_weight` - Shape `[intermediate_size, hidden_size]` ([Out, In])
    /// * `dense1_bias` - Shape `[intermediate_size]`
    /// * `dense2_weight` - Shape `[hidden_size, intermediate_size]` ([Out, In])
    /// * `dense2_bias` - Shape `[hidden_size]`
    pub fn new(
        dense1_weight: Array2<f32>,
        dense1_bias: Array1<f32>,
        dense2_weight: Array2<f32>,
        dense2_bias: Array1<f32>,
        activation: Activation,
    ) -> Self {
        assert_eq!(dense1_weight.shape()[0], dense1_bias.len());
        assert_eq!(dense2_weight.shape()[0], dense2_bias.len());
        assert_eq!(dense1_weight.shape()[0], dense2_weight.shape()[1]);

        Self {
            fc1: LinearLayer::new_f32(dense1_weight.clone(), Some(dense1_bias.clone())),
            fc2: LinearLayer::new_f32(dense2_weight.clone(), Some(dense2_bias.clone())),
            activation,
            dense1_weight,
            dense1_bias,
            dense2_weight,
            dense2_bias,
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

    /// First linear transformation (for testing)
    pub fn fc1(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, _) = hidden.dim();
        let hidden_contig = hidden.as_standard_layout();
        let hidden_2d = hidden_contig
            .view()
            .into_shape_with_order((batch * seq, hidden.shape()[2]))?;

        let result = self.fc1.matmul(&hidden_2d);
        Ok(result.into_shape_with_order((batch, seq, self.fc1.out_features()))?)
    }

    /// Second linear transformation (for testing)
    pub fn fc2(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, _) = hidden.dim();
        let hidden_contig = hidden.as_standard_layout();
        let hidden_2d = hidden_contig
            .view()
            .into_shape_with_order((batch * seq, hidden.shape()[2]))?;

        let result = self.fc2.matmul(&hidden_2d);
        Ok(result.into_shape_with_order((batch, seq, self.fc2.out_features()))?)
    }

    /// Apply activation in-place (for testing)
    pub fn apply_activation(&self, hidden: &mut Array3<f32>) {
        apply_activation(hidden, self.activation);
    }
}