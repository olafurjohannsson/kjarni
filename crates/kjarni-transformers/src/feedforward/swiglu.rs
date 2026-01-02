use crate::linear_layer::{LinearLayer};
use anyhow::Result;
use ndarray::{Array2, Array3};
use rayon::prelude::*;
use std::time::Instant;

/// A high-performance SwiGLU Feed-Forward Network.
///
/// This struct orchestrates the SwiGLU operation, a key component in modern
/// transformer architectures. It dispatches to specialized computation paths
/// based on the input sequence length for maximum performance.
pub struct SwiGluFeedForward {
    pub gate: LinearLayer,
    pub up: LinearLayer,
    pub down: LinearLayer,
}

impl SwiGluFeedForward {
    /// Creates a new `SwiGluFeedForward` layer.
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

    /// Performs the forward pass, dispatching to the optimal kernel based on sequence length.
    #[inline]
    pub fn forward(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        if hidden.shape()[1] == 1 {
            self.forward_decode(hidden)
        } else {
            self.forward_prefill(hidden)
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
    /// Hyper-optimized path for the decode step (batch=1, seq=1).
    /// Uses a specialized, fused SIMD kernel for the gate/up projections.
    fn forward_decode(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, hidden_dim) = hidden.dim();
        let hidden_2d = hidden.view().into_shape_with_order((batch, hidden_dim))?;

        let t_total = Instant::now();

        // --- Step 1: Call the clean, safe, fused kernel dispatcher from the `ops` module ---
        let t_gate_up = Instant::now();

        let (mut gate_out, up_out) = rayon::join(
            || self.gate.matmul(&hidden_2d),
            || self.up.matmul(&hidden_2d),
        );

        let d_gate_up = t_gate_up.elapsed();

        // --- Step 2: Activation & Element-wise Ops ---
        let t_act = Instant::now();
        silu_parallel(&mut gate_out);
        let activated = gate_out * up_out;
        let d_act = t_act.elapsed();

        // --- Step 3: Down Projection ---
        let t_down = Instant::now();
        let output_2d = self.down.matmul(&activated.view());
        let d_down = t_down.elapsed();

        let d_total = t_total.elapsed();
        if d_total.as_millis() > 1 {
            log::info!(
                "[FFN Perf DECODE] Total: {:?}, Gate+Up Fused: {:?}, Activation: {:?}, Down Matmul: {:?}",
                d_total,
                d_gate_up,
                d_act,
                d_down
            );
        }

        Ok(output_2d.into_shape_with_order((batch, seq, self.down.out_features()))?)
    }

    /// Parallel path for the prefill step (seq > 1).
    fn forward_prefill(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, hidden_dim) = hidden.dim();
        let hidden_2d = hidden.view().into_shape_with_order((batch * seq, hidden_dim))?;

        let t_total = Instant::now();

        let t_gate_up = Instant::now();
        let (mut gate_out, up_out) = rayon::join(
            || self.gate.matmul(&hidden_2d.view()),
            || self.up.matmul(&hidden_2d.view()),
        );
        let d_gate_up = t_gate_up.elapsed();

        let t_act = Instant::now();
        silu_parallel(&mut gate_out);
        let activated = gate_out * up_out;
        let d_act = t_act.elapsed();

        let t_down = Instant::now();
        let output_2d = self.down.matmul(&activated.view());
        let d_down = t_down.elapsed();

        let d_total = t_total.elapsed();
        if d_total.as_millis() > 5 {
            log::info!(
                "[FFN Perf PREFILL] Total: {:?}, Gate+Up Parallel: {:?}, Activation: {:?}, Down Matmul: {:?}",
                d_total,
                d_gate_up,
                d_act,
                d_down
            );
        }

        Ok(output_2d.into_shape_with_order((batch, seq, self.down.out_features()))?)
    }
}

/// A parallel, in-place SiLU (Swish) activation function.
#[inline]
fn silu_parallel(x: &mut Array2<f32>) {
    if let Some(slice) = x.as_slice_mut() {
        slice.par_iter_mut().for_each(|v| *v /= 1.0 + (-*v).exp());
    } else {
        x.par_mapv_inplace(|v| v / (1.0 + (-v).exp()));
    }
}
