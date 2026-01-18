use crate::activations::{
    Activation, apply_activation, apply_activation_2d, apply_activation_2d_mut,
};
use crate::cpu::encoder::buffers::EncoderBuffers;
use crate::linear_layer::LinearLayer;
use anyhow::Result;
use ndarray::{Array1, Array2, Array3, ArrayView2, s};

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

    /// Forward pass writing to pre-allocated EncoderBuffers (no allocation).
    ///
    /// Uses `buffers.ffn_intermediate` for hidden activations.
    /// Writes output to `buffers.ffn_output`.
    ///
    /// # Arguments
    ///
    /// * `hidden` - Input tensor [tokens, hidden]
    /// * `buffers` - Pre-allocated encoder buffers
    ///
    /// # Panics
    ///
    /// In debug mode, panics if buffer dimensions don't match.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut buffers = EncoderBuffers::new_auto(1024, 768, 3072);
    /// ffn.forward_noalloc(&hidden_2d, &mut buffers);
    /// // Result in buffers.ffn_output
    /// ```
    pub fn forward_noalloc(&self, hidden: &ArrayView2<f32>, buffers: &mut EncoderBuffers) {
        let tokens = hidden.shape()[0];
        let intermediate_dim = self.fc1.out_features();
        let hidden_dim = self.fc2.out_features();

        #[cfg(debug_assertions)]
        {
            buffers.ensure_capacity_tokens(tokens);
            if buffers.intermediate_dim() != intermediate_dim {
                panic!(
                    "FFN intermediate dimension mismatch: layer has {}, buffer has {}",
                    intermediate_dim,
                    buffers.intermediate_dim()
                );
            }
        }

        // FC1 into ffn_intermediate (full buffer, we'll use slice for tokens)
        self.fc1
            .matmul_noalloc(hidden, &mut buffers.ffn_intermediate);

        // Apply activation in-place to actual tokens only
        {
            let mut intermediate_slice = buffers.ffn_intermediate.slice_mut(s![..tokens, ..]);
            apply_activation_2d_mut(&mut intermediate_slice, self.activation);
        }

        // FC2: need contiguous input
        // Take slice and make it contiguous
        let intermediate_slice = buffers.ffn_intermediate.slice(s![..tokens, ..]);
        let intermediate_owned = intermediate_slice.as_standard_layout().to_owned();

        // Create a mutable slice of output for just the tokens we need
        let mut output_slice = buffers.ffn_output.slice_mut(s![..tokens, ..]);

        // Manual matmul since we need to write to a slice
        let result = self.fc2.matmul(&intermediate_owned.view());
        output_slice.assign(&result);
    }
}
