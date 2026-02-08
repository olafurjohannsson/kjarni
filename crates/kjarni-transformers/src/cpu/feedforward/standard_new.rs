use crate::activations::{Activation, apply_activation_2d, apply_activation_2d_mut};
use crate::linear_layer::LinearLayer;
use crate::cpu::encoder::buffers::EncoderBuffers;
use anyhow::Result;
use ndarray::{Array3, ArrayView2, s};

pub struct StdFeedForwardNew {
    pub fc1: LinearLayer,
    pub fc2: LinearLayer,
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

        // FC1 into ffn_intermediate
        self.fc1.matmul_noalloc(hidden, &mut buffers.ffn_intermediate);

        // Apply activation in-place
        // Only apply to the actual tokens, not the full buffer capacity
        {
            let mut intermediate_slice = buffers
                .ffn_intermediate
                .slice_mut(s![..tokens, ..intermediate_dim]);
            apply_activation_2d_mut(&mut intermediate_slice, self.activation);
        }

        // FC2 into ffn_output
        let intermediate_view = buffers
            .ffn_intermediate
            .slice(s![..tokens, ..intermediate_dim]);
        self.fc2.matmul_noalloc(&intermediate_view, &mut buffers.ffn_output);
    }

}


#[cfg(test)]
mod feedforward_new_tests {
    use super::*;
    use ndarray::{Array1, Array2, Array3, arr3};
    use crate::activations::Activation;
    use crate::feedforward::StdFeedForward;
    use crate::linear_layer::LinearLayer; 

    /// Helper for approximate float comparison
    fn assert_all_close(a: &Array3<f32>, b: &Array3<f32>, tol: f32) {
        assert_eq!(a.dim(), b.dim(), "Dimensions mismatch");
        let diff = a - b;
        let max_diff = diff.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_diff <= tol,
            "Max difference {} exceeds tolerance {}.\nGot:\n{:?}\nExpected:\n{:?}",
            max_diff,
            tol,
            a,
            b
        );
    }

    fn mock_linear(weights_data: Vec<f32>, shape: (usize, usize)) -> LinearLayer {
        let weights = Array2::from_shape_vec(shape, weights_data).unwrap();
        LinearLayer::new_f32(weights, None) 
    }

    #[test]
    fn test_ffn_shapes() -> Result<()> {
        let batch = 2;
        let seq = 5;
        let d_in = 4;
        let d_hidden = 16;
        let d_out = 8; 

        // FC1: [d_hidden, d_in]
        let fc1 = mock_linear(vec![0.0; d_hidden * d_in], (d_hidden, d_in));
        // FC2: [d_out, d_hidden]
        let fc2 = mock_linear(vec![0.0; d_out * d_hidden], (d_out, d_hidden));

        let ffn = StdFeedForwardNew::new(fc1, fc2, Activation::Gelu);

        let input = Array3::<f32>::zeros((batch, seq, d_in));
        let output = ffn.forward(&input)?;

        assert_eq!(output.dim(), (batch, seq, d_out));
        Ok(())
    }

    #[test]
    fn test_ffn_logic_relu() -> Result<()> {
        // Simple manual check
        // Input: [1.0, -1.0]
        let input = arr3(&[[[1.0, -1.0]]]);

        // FC1 (Identity): [[1, 0], [0, 1]]
        let fc1 = mock_linear(vec![1.0, 0.0, 0.0, 1.0], (2, 2));

        // FC2 (Scale by 2): [[2, 0], [0, 2]]
        let fc2 = mock_linear(vec![2.0, 0.0, 0.0, 2.0], (2, 2));

        let ffn = StdFeedForwardNew::new(fc1, fc2, Activation::Relu);
        let output = ffn.forward(&input)?;

        // Relu clips -1.0 to 0.0, then FC2 scales
        let expected = arr3(&[[[2.0, 0.0]]]);
        assert_all_close(&output, &expected, 1e-6);

        Ok(())
    }

    #[test]
    fn test_ffn_golden_values_gelu() -> Result<()> {
        // Golden values from Python script
        // Config: Batch=1, Seq=2, In=3, Hidden=4, Out=3
        
        let input_data = vec![
            0.5, -0.2000, 0.1000, 
            -0.5, 0.0, 0.8000
        ];
        let input = Array3::from_shape_vec((1, 2, 3), input_data)?;

        // FC1 Weights [4, 3]
        let w1_data = vec![
            0.4414, 0.4792, -0.1353, 
            0.5304, -0.1265, 0.1165, 
            -0.2811, 0.3391, 0.509, 
            -0.4236, 0.5018, 0.1081
        ];
        let fc1 = mock_linear(w1_data, (4, 3));

        // FC2 Weights [3, 4]
        let w2_data = vec![
            0.3694, 0.0677, 0.2411, -0.0706, 
            0.3854, 0.0739, -0.2334, 0.1274, 
            -0.2304, -0.0586, -0.2031, 0.3317
        ];
        let fc2 = mock_linear(w2_data, (3, 4));

        let ffn = StdFeedForwardNew::new(fc1, fc2, Activation::Gelu);
        let output = ffn.forward(&input)?;

        // Expected Output
        let expected_data = vec![
            0.0266, 0.0386, -0.0491, 
            0.0304, -0.1196, 0.0148
        ];
        let expected = Array3::from_shape_vec((1, 2, 3), expected_data)?;

        // Tolerance 1e-4 is standard for Cross-Language Float comparisons (especially involving Gelu approx)
        assert_all_close(&output, &expected, 1e-4);

        Ok(())
    }

        #[test]
    fn test_ffn_golden_values_gelu2() -> Result<()> {
        // === 1. Setup Input ===
        // Batch=1, Seq=2, In=3
        let input_data = vec![
            0.5, -0.2000, 0.1000, 
            -0.5, 0.0, 0.8000
        ];
        let input = Array3::from_shape_vec((1, 2, 3), input_data)?;

        // === 2. Setup FC1 (Dense 1) ===
        // Shape: [Out=4, In=3]
        let w1_data = vec![
            0.4414, 0.4792, -0.1353, 
            0.5304, -0.1265, 0.1165, 
            -0.2811, 0.3391, 0.509, 
            -0.4236, 0.5018, 0.1081
        ];
        let dense1_weight = Array2::from_shape_vec((4, 3), w1_data)?;
        // Previous mock used no bias, so we provide explicit zeros here
        let dense1_bias = Array1::zeros(4);

        // === 3. Setup FC2 (Dense 2) ===
        // Shape: [Out=3, In=4]
        let w2_data = vec![
            0.3694, 0.0677, 0.2411, -0.0706, 
            0.3854, 0.0739, -0.2334, 0.1274, 
            -0.2304, -0.0586, -0.2031, 0.3317
        ];
        let dense2_weight = Array2::from_shape_vec((3, 4), w2_data)?;
        let dense2_bias = Array1::zeros(3);

        // === 4. Initialize Legacy StdFeedForward ===
        let ffn = StdFeedForward::new(
            dense1_weight,
            dense1_bias,
            dense2_weight,
            dense2_bias,
            Activation::Gelu, // Standard Gelu used in previous context
        );

        // === 5. Forward Pass ===
        let output = ffn.forward(&input)?;

        // === 6. Verify against Golden Values ===
        // These values match the output of (x @ W1.T).gelu() @ W2.T
        let expected_data = vec![
            0.0266, 0.0386, -0.0491, 
            0.0304, -0.1196, 0.0148
        ];
        let expected = Array3::from_shape_vec((1, 2, 3), expected_data)?;

        // Tolerance for GELU approximation differences
        assert_all_close(&output, &expected, 1e-4);

        Ok(())
    }
}