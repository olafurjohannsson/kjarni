//! SwiGLU Feed-Forward Network
//!
//! Used in LLaMA and other modern architectures.
//! SwiGLU: SiLU(gate(x)) ⊙ up(x), then down()
//!
//! Unlike standard FFN with 2 projections, SwiGLU has 3:
//! - gate_proj: [hidden_size, intermediate_size]
//! - up_proj:   [hidden_size, intermediate_size]  
//! - down_proj: [intermediate_size, hidden_size]

use crate::linear_layer::LinearLayer;
use anyhow::Result;
use ndarray::{Array2, Array3};

// If 'silu_parallel' is in activations.rs, import it.
// Otherwise we use the inline implementation below.
use crate::activations::silu_parallel;

pub struct SwiGluFeedForward {
    pub gate: LinearLayer,
    pub up: LinearLayer,
    pub down: LinearLayer,
}

impl SwiGluFeedForward {
    pub fn new(
        gate: impl Into<LinearLayer>,
        up: impl Into<LinearLayer>,
        down: impl Into<LinearLayer>,
    ) -> Self {
        Self {
            gate: gate.into(),
            up: up.into(),
            down: down.into(),
        }
    }
    pub fn forward_2d(&self, hidden: &Array2<f32>) -> Result<Array2<f32>> {
        let (rows, dim) = hidden.dim();

        // 1. Gate & Up
        let mut gate_out = self.gate.matmul(&hidden.view());
        let up_out = self.up.matmul(&hidden.view());

        // 2. Activation
        silu_parallel(&mut gate_out);
        let activated = gate_out * up_out;

        // 3. Down
        let output_2d = self.down.matmul(&activated.view());

        Ok(output_2d)
    }
    pub fn forward(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, hidden_dim) = hidden.dim();
        let hidden_2d = hidden
            .view()
            .into_shape_with_order((batch * seq, hidden_dim))
            .unwrap();

        // Gate and Up can run simultaneously!
        let (mut gate_out, up_out) = rayon::join(
            || self.gate.matmul(&hidden_2d),
            || self.up.matmul(&hidden_2d),
        );

        // Activation + multiply
        silu_parallel(&mut gate_out);
        let activated = gate_out * up_out;

        // Down
        let output_2d = self.down.matmul(&activated.view());

        Ok(output_2d
            .into_shape_with_order((batch, seq, self.down.out_features()))
            .unwrap())
    }

    // You likely don't need forward_2d explicitly anymore since forward handles the flattening,
    // but here it is just in case:
    // pub fn forward_2d(&self, hidden: &Array2<f32>) -> Array2<f32> {
    //     let mut gate_out = matmul_2d_transposed(&hidden.view(), &self.gate_weight_t.view());
    //     let up_out = matmul_2d_transposed(&hidden.view(), &self.up_weight_t.view());

    //     silu_parallel(&mut gate_out);

    //     let activated = gate_out * up_out;
    //     matmul_2d_transposed(&activated.view(), &self.down_weight_t.view())
    // }
}
