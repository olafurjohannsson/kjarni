
//! Public, safe entry points for the fused SwiGLU operation.
//!
//! This module provides a safe wrapper around the specialized, unsafe SIMD kernels
//! that compute the gate and up projections of a SwiGLU FFN simultaneously.

use crate::kernels;
use anyhow::{anyhow, Result};
use half::bf16;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

/// Computes the fused gate and up projections for BF16 weights.
///
/// Dispatches to the best available SIMD kernel at runtime.
pub fn gate_up_fused_bf16(
    a: &ArrayView2<f32>,
    gate_w: &ArrayView2<bf16>,
    up_w: &ArrayView2<bf16>,
) -> Result<(Array2<f32>, Array2<f32>)> {
    let (m, k) = a.dim();
    let (n, k2) = gate_w.dim();
    if m != 1 || k != k2 || gate_w.dim() != up_w.dim() {
        return Err(anyhow!("Dimension mismatch for fused BF16 kernel"));
    }

    let mut gate_out = Array2::zeros((m, n));
    let mut up_out = Array2::zeros((m, n));

    let a_slice = a.as_slice().ok_or_else(|| anyhow!("Input 'a' not contiguous"))?;
    let gate_w_slice = gate_w.as_slice().ok_or_else(|| anyhow!("Gate weights not contiguous"))?;
    let up_w_slice = up_w.as_slice().ok_or_else(|| anyhow!("Up weights not contiguous"))?;

    // The fused kernel is most effective when parallelized over its output dimension (n).
    gate_out.as_slice_mut().unwrap()
        .par_iter_mut()
        .zip(up_out.as_slice_mut().unwrap().par_iter_mut())
        .enumerate()
        .for_each(|(i, (gate_out_i, up_out_i))| {
            let gate_w_row = &gate_w_slice[i * k..(i + 1) * k];
            let up_w_row = &up_w_slice[i * k..(i + 1) * k];

            unsafe {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    return kernels::x86::bf16::swiglu_fused_gate_up_bf16(
                        std::slice::from_mut(gate_out_i),
                        std::slice::from_mut(up_out_i),
                        a_slice,
                        std::mem::transmute(gate_w_row),
                        std::mem::transmute(up_w_row),
                        k,
                        1,
                    );
                }
                // Add NEON and other architectures here in the future.
            }
            // Scalar fallback if no SIMD is available (though this path is unlikely to be hit
            // if the matmul fallback is used instead).
            let (g, u) = scalar_fused_op(a_slice, gate_w_row, up_w_row);
            *gate_out_i = g;
            *up_out_i = u;
        });

    Ok((gate_out, up_out))
}

/// Scalar implementation for the fused operation, used as a fallback.
fn scalar_fused_op(a: &[f32], gate_w_row: &[bf16], up_w_row: &[bf16]) -> (f32, f32) {
    let gate_sum = a.iter().zip(gate_w_row.iter()).map(|(&av, &wv)| av * wv.to_f32()).sum();
    let up_sum = a.iter().zip(up_w_row.iter()).map(|(&av, &wv)| av * wv.to_f32()).sum();
    (gate_sum, up_sum)
}