//! SwiGLU Feed-Forward Network with fused kernel support.
//!
//! This module provides the SwiGLU FFN layer used in modern LLMs like Llama,
//! with automatic dispatch to fused kernels for optimal performance.

use crate::activations::{apply_activation_2d, Activation};
use crate::cpu::ops::fused;
use crate::linear_layer::LinearLayer;
use anyhow::Result;
use ndarray::{Array2, Array3};
use std::time::Instant;

/// SwiGLU Feed-Forward Network.
///
/// Implements the gated linear unit with SiLU activation:
/// ```text
/// output = down_proj(silu(gate_proj(x)) * up_proj(x))
/// ```
///
/// # Performance
///
/// For decode (seq_len=1), uses a fused kernel that combines gate_proj,
/// up_proj, silu, and element-wise multiply into a single pass, reducing
/// memory bandwidth by ~50%.
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

    /// Forward pass with automatic kernel selection.
    ///
    /// Dispatches to:
    /// - Fused decode kernel for seq_len=1
    /// - Fused prefill kernel for small seq_len (< 64)
    /// - Separate matmuls for large seq_len
    #[inline]
    pub fn forward(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let seq_len = hidden.shape()[1];

        if seq_len == 1 {
            self.forward_decode_fused(hidden)
        } else if seq_len < 64 {
            // Fused also helps for small prefill
            self.forward_prefill_fused(hidden)
        } else {
            // Large prefill: separate matmuls have better cache utilization
            self.forward_prefill_separate(hidden)
        }
    }

    /// 2D forward pass (for direct integration without reshape overhead).
    pub fn forward_2d(&self, hidden: &Array2<f32>) -> Result<Array2<f32>> {
        let tokens = hidden.shape()[0];

        if tokens == 1 {
            self.forward_2d_fused(hidden)
        } else {
            self.forward_2d_separate(hidden)
        }
    }

    // =========================================================================
    // Fused Paths
    // =========================================================================

    /// Fused decode path: gate+up+silu in one kernel.
    fn forward_decode_fused(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, hidden_dim) = hidden.dim();
        let intermediate_dim = self.gate.out_features();
        let tokens = batch * seq;

        let input = hidden
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Input must be contiguous"))?;

        let mut activated = vec![0.0f32; tokens * intermediate_dim];

        // Try fused kernel
        match fused::fused_gate_up_silu(&self.gate, &self.up, input, &mut activated, tokens) {
            Ok(()) => {
                // Fused succeeded - just do down projection
                let activated_arr =
                    Array2::from_shape_vec((tokens, intermediate_dim), activated)?;
                let output_2d = self.down.matmul(&activated_arr.view());
                Ok(output_2d.into_shape_with_order((batch, seq, self.down.out_features()))?)
            }
            Err(e) => {
                // Fused not supported for this dtype, fallback
                log::debug!("Fused kernel unavailable ({}), using fallback", e);
                self.forward_decode_separate(hidden)
            }
        }
    }

    /// Fused prefill path.
    fn forward_prefill_fused(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, hidden_dim) = hidden.dim();
        let intermediate_dim = self.gate.out_features();
        let tokens = batch * seq;

        let input = hidden
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Input must be contiguous"))?;

        let mut activated = vec![0.0f32; tokens * intermediate_dim];

        match fused::fused_gate_up_silu(&self.gate, &self.up, input, &mut activated, tokens) {
            Ok(()) => {
                let activated_arr =
                    Array2::from_shape_vec((tokens, intermediate_dim), activated)?;
                let output_2d = self.down.matmul(&activated_arr.view());
                Ok(output_2d.into_shape_with_order((batch, seq, self.down.out_features()))?)
            }
            Err(e) => {
                log::debug!("Fused kernel unavailable ({}), using fallback", e);
                self.forward_prefill_separate(hidden)
            }
        }
    }

    /// 2D fused path.
    fn forward_2d_fused(&self, hidden: &Array2<f32>) -> Result<Array2<f32>> {
        let (tokens, hidden_dim) = hidden.dim();
        let intermediate_dim = self.gate.out_features();

        let input = hidden
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Input must be contiguous"))?;

        let mut activated = vec![0.0f32; tokens * intermediate_dim];

        match fused::fused_gate_up_silu(&self.gate, &self.up, input, &mut activated, tokens) {
            Ok(()) => {
                let activated_arr =
                    Array2::from_shape_vec((tokens, intermediate_dim), activated)?;
                Ok(self.down.matmul(&activated_arr.view()))
            }
            Err(e) => {
                log::debug!("Fused kernel unavailable ({}), using fallback", e);
                self.forward_2d_separate(hidden)
            }
        }
    }

    // =========================================================================
    // Separate Paths (fallback)
    // =========================================================================

    /// Separate decode path: two matmuls + activation + multiply.
    fn forward_decode_separate(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, hidden_dim) = hidden.dim();
        let hidden_2d = hidden
            .view()
            .into_shape_with_order((batch * seq, hidden_dim))?;

        let t_total = Instant::now();

        // Gate + Up projections
        let t_gate_up = Instant::now();
        let mut gate_out = self.gate.matmul(&hidden_2d);
        let up_out = self.up.matmul(&hidden_2d);
        let d_gate_up = t_gate_up.elapsed();

        // Activation + multiply (in-place)
        let t_act = Instant::now();
        apply_activation_2d(&mut gate_out, self.activation);
        gate_out.zip_mut_with(&up_out, |g, &u| *g *= u);
        let d_act = t_act.elapsed();

        // Down projection
        let t_down = Instant::now();
        let output_2d = self.down.matmul(&gate_out.view());
        let d_down = t_down.elapsed();

        let d_total = t_total.elapsed();
        if d_total.as_millis() > 1 {
            log::debug!(
                "[FFN DECODE] Total: {:?}, Gate+Up: {:?}, Act: {:?}, Down: {:?}",
                d_total,
                d_gate_up,
                d_act,
                d_down
            );
        }

        Ok(output_2d.into_shape_with_order((batch, seq, self.down.out_features()))?)
    }

    /// Separate prefill path.
    fn forward_prefill_separate(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, hidden_dim) = hidden.dim();
        let hidden_2d = hidden
            .view()
            .into_shape_with_order((batch * seq, hidden_dim))?;

        let t_total = Instant::now();

        let t_gate_up = Instant::now();
        let mut gate_out = self.gate.matmul(&hidden_2d);
        let up_out = self.up.matmul(&hidden_2d);
        let d_gate_up = t_gate_up.elapsed();

        let t_act = Instant::now();
        apply_activation_2d(&mut gate_out, self.activation);
        gate_out.zip_mut_with(&up_out, |g, &u| *g *= u);
        let d_act = t_act.elapsed();

        let t_down = Instant::now();
        let output_2d = self.down.matmul(&gate_out.view());
        let d_down = t_down.elapsed();

        let d_total = t_total.elapsed();
        if d_total.as_millis() > 5 {
            log::debug!(
                "[FFN PREFILL] Total: {:?}, Gate+Up: {:?}, Act: {:?}, Down: {:?}",
                d_total,
                d_gate_up,
                d_act,
                d_down
            );
        }

        Ok(output_2d.into_shape_with_order((batch, seq, self.down.out_features()))?)
    }

    /// 2D separate path.
    fn forward_2d_separate(&self, hidden: &Array2<f32>) -> Result<Array2<f32>> {
        let mut gate_out = self.gate.matmul(&hidden.view());
        let up_out = self.up.matmul(&hidden.view());

        apply_activation_2d(&mut gate_out, self.activation);
        gate_out.zip_mut_with(&up_out, |g, &u| *g *= u);

        Ok(self.down.matmul(&gate_out.view()))
    }
}

// =============================================================================
// Parallel SiLU (for separate path)
// =============================================================================

/// In-place SiLU activation with rayon parallelization.
#[inline]
pub fn silu_parallel(x: &mut Array2<f32>) {
    use rayon::prelude::*;

    if let Some(slice) = x.as_slice_mut() {
        slice.par_iter_mut().for_each(|v| *v /= 1.0 + (-*v).exp());
    } else {
        x.par_mapv_inplace(|v| v / (1.0 + (-v).exp()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_test_ffn(hidden: usize, intermediate: usize) -> SwiGluFeedForward {
        let gate = Array2::from_shape_fn((intermediate, hidden), |(i, j)| {
            ((i * 17 + j * 13) % 100) as f32 * 0.01 - 0.5
        });
        let up = Array2::from_shape_fn((intermediate, hidden), |(i, j)| {
            ((i * 19 + j * 11) % 100) as f32 * 0.01 - 0.5
        });
        let down = Array2::from_shape_fn((hidden, intermediate), |(i, j)| {
            ((i * 23 + j * 7) % 100) as f32 * 0.01 - 0.5
        });

        SwiGluFeedForward::new(
            LinearLayer::new_f32(gate, None),
            LinearLayer::new_f32(up, None),
            LinearLayer::new_f32(down, None),
            crate::activations::Activation::SilU,
        )
    }

    #[test]
    fn test_forward_decode() {
        let ffn = create_test_ffn(256, 512);
        let input = Array3::from_shape_fn((1, 1, 256), |(_, _, i)| i as f32 * 0.01);

        let output = ffn.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 256]);
    }

    #[test]
    fn test_forward_prefill() {
        let ffn = create_test_ffn(256, 512);
        let input = Array3::from_shape_fn((1, 16, 256), |(_, s, i)| (s * 256 + i) as f32 * 0.001);

        let output = ffn.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 16, 256]);
    }

    #[test]
    fn test_fused_matches_separate_decode() {
        let ffn = create_test_ffn(256, 512);
        let input = Array3::from_shape_fn((1, 1, 256), |(_, _, i)| i as f32 * 0.01 - 0.5);

        // Force fused
        let fused_output = ffn.forward_decode_fused(&input).unwrap();

        // Force separate
        let separate_output = ffn.forward_decode_separate(&input).unwrap();

        // Compare
        for (f, s) in fused_output.iter().zip(separate_output.iter()) {
            let diff = (f - s).abs();
            assert!(diff < 1e-5, "Mismatch: fused={}, separate={}", f, s);
        }
    }

    #[test]
    fn test_fused_matches_separate_prefill() {
        let ffn = create_test_ffn(256, 512);
        let input = Array3::from_shape_fn((1, 8, 256), |(_, s, i)| (s * 256 + i) as f32 * 0.001 - 0.5);

        let fused_output = ffn.forward_prefill_fused(&input).unwrap();
        let separate_output = ffn.forward_prefill_separate(&input).unwrap();

        for (f, s) in fused_output.iter().zip(separate_output.iter()) {
            let diff = (f - s).abs();
            assert!(diff < 1e-4, "Mismatch: fused={}, separate={}", f, s);
        }
    }
}