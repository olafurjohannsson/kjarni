//! SwiGLU Feed-Forward Network.

use crate::activations::{Activation, apply_activation_2d, apply_activation_2d_mut};
use crate::cpu::encoder::buffers::EncoderBuffers;
use crate::linear_layer::LinearLayer;
use anyhow::Result;
use ndarray::{Array2, Array3, ArrayView2, s};

/// SwiGLU Feed-Forward Network.
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

    /// Writes into caller-owned buffers instead of allocating.
    ///
    /// Without this, `FeedForward::forward_noalloc` fell through to a panic for
    /// SwiGLU. That was unreachable while a token-count threshold sent most
    /// requests down the allocating path; once every encode takes the buffered
    /// path it means any SwiGLU encoder, Nomic among them, panics on its first
    /// layer.
    ///
    /// The gate result lands in `ffn_intermediate`, which is sized for the
    /// intermediate dimension, and the up projection needs a second buffer of the
    /// same width, so that one is still allocated here. Removing it needs another
    /// scratch buffer on `EncoderBuffers`.
    pub fn forward_noalloc(&self, hidden: &ArrayView2<f32>, buffers: &mut EncoderBuffers) {
        let tokens = hidden.shape()[0];
        let intermediate = self.gate.out_features();

        let up_out = self.up.matmul(hidden);
        self.gate
            .matmul_noalloc(hidden, &mut buffers.ffn_intermediate);

        {
            let mut gate_slice = buffers
                .ffn_intermediate
                .slice_mut(s![..tokens, ..intermediate]);
            apply_activation_2d_mut(&mut gate_slice, self.activation);
            gate_slice.zip_mut_with(&up_out.slice(s![..tokens, ..intermediate]), |g, &u| *g *= u);
        }

        let gated = buffers.ffn_intermediate.slice(s![..tokens, ..intermediate]);
        self.down.matmul_noalloc(&gated, &mut buffers.ffn_output);
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
