//! SwiGLU Feed-Forward Network.
//!
//! Implements the gated linear unit with configurable activation:
//! ```text
//! output = down_proj(activation(gate_proj(x)) * up_proj(x))
//! ```

use crate::activations::{apply_activation_2d, Activation};
use crate::linear_layer::LinearLayer;
use anyhow::Result;
use ndarray::{Array2, Array3};

/// SwiGLU Feed-Forward Network.
///
/// Uses parallel execution for gate and up projections via rayon::join.
pub struct SwiGluFeedForward {
    pub gate: LinearLayer,
    pub up: LinearLayer,
    pub down: LinearLayer,
    pub activation: Activation,
}

impl SwiGluFeedForward {
    /// Creates a new SwiGLU feed-forward layer.
    pub fn new(
        gate: impl Into<LinearLayer>,
        up: impl Into<LinearLayer>,
        down: impl Into<LinearLayer>,
        activation: Activation,
    ) -> Self {
        Self {
            gate: gate.into(),
            up: up.into(),
            down: down.into(),
            activation,
        }
    }

    /// Forward pass with automatic dispatch based on sequence length.
    #[inline]
    pub fn forward(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, hidden_dim) = hidden.dim();
        let hidden_2d = hidden
            .view()
            .into_shape_with_order((batch * seq, hidden_dim))?;

        // Gate and Up projections in parallel
        let (mut gate_out, up_out) = rayon::join(
            || self.gate.matmul(&hidden_2d),
            || self.up.matmul(&hidden_2d),
        );

        // Apply activation and multiply
        apply_activation_2d(&mut gate_out, self.activation);
        gate_out.zip_mut_with(&up_out, |g, &u| *g *= u);

        // Down projection
        let output_2d = self.down.matmul(&gate_out.view());

        Ok(output_2d.into_shape_with_order((batch, seq, self.down.out_features()))?)
    }

    /// 2D forward pass (no reshape overhead).
    pub fn forward_2d(&self, hidden: &Array2<f32>) -> Result<Array2<f32>> {
        let (mut gate_out, up_out) = rayon::join(
            || self.gate.matmul(&hidden.view()),
            || self.up.matmul(&hidden.view()),
        );

        apply_activation_2d(&mut gate_out, self.activation);
        gate_out.zip_mut_with(&up_out, |g, &u| *g *= u);

        Ok(self.down.matmul(&gate_out.view()))
    }

    /// Returns (hidden_size, intermediate_size).
    pub fn dimensions(&self) -> (usize, usize) {
        (self.gate.in_features(), self.gate.out_features())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_test_ffn() -> SwiGluFeedForward {
        let gate = Array2::from_shape_fn((512, 256), |(i, j)| {
            ((i * 17 + j * 13) % 100) as f32 * 0.01 - 0.5
        });
        let up = Array2::from_shape_fn((512, 256), |(i, j)| {
            ((i * 19 + j * 11) % 100) as f32 * 0.01 - 0.5
        });
        let down = Array2::from_shape_fn((256, 512), |(i, j)| {
            ((i * 23 + j * 7) % 100) as f32 * 0.01 - 0.5
        });

        SwiGluFeedForward::new(
            crate::linear_layer::LinearLayer::new_f32(gate, None),
            crate::linear_layer::LinearLayer::new_f32(up, None),
            crate::linear_layer::LinearLayer::new_f32(down, None),
            Activation::SilU,
        )
    }

    #[test]
    fn test_forward_decode() {
        let ffn = create_test_ffn();
        let input = Array3::from_shape_fn((1, 1, 256), |(_, _, i)| i as f32 * 0.01);
        let output = ffn.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 256]);
    }

    #[test]
    fn test_forward_prefill() {
        let ffn = create_test_ffn();
        let input = Array3::from_shape_fn((1, 16, 256), |(_, s, i)| (s * 256 + i) as f32 * 0.001);
        let output = ffn.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 16, 256]);
    }

    #[test]
    fn test_forward_2d() {
        let ffn = create_test_ffn();
        let input = Array2::from_shape_fn((8, 256), |(s, i)| (s * 256 + i) as f32 * 0.001);
        let output = ffn.forward_2d(&input).unwrap();
        assert_eq!(output.shape(), &[8, 256]);
    }
}
