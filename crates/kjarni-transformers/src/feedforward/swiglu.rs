// use crate::linear_layer::{LinearData, LinearLayer};
// use anyhow::{anyhow, Result};
// use ndarray::{s, Array2, Array3, ArrayView2};
// use rayon::prelude::*;
// use std::time::Instant;

// /// A high-performance SwiGLU Feed-Forward Network.
// ///
// /// This implementation contains a hyper-optimized path for the `decode_one` step (sequence length = 1)
// /// that uses a specialized SIMD kernel to fuse the gate and up projection computations.
// /// For the `prefill` step (sequence length > 1), it falls back to a parallel `rayon::join`.
// pub struct SwiGluFeedForward {
//     pub gate: LinearLayer,
//     pub up: LinearLayer,
//     pub down: LinearLayer,
// }

// impl SwiGluFeedForward {
//     /// The constructor remains unchanged. No fusion happens at load time.
//     pub fn new(
//         gate: impl Into<LinearLayer>,
//         up: impl Into<LinearLayer>,
//         down: impl Into<LinearLayer>,
//     ) -> Self {
//         Self {
//             gate: gate.into(),
//             up: up.into(),
//             down: down.into(),
//         }
//     }

//     /// Performs the forward pass, dispatching to the optimal kernel based on sequence length.
//     #[inline]
//     pub fn forward(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
//         let seq_len = hidden.shape()[1];

//         // Dispatch to the hyper-optimized kernel for the single-token decode step.
//         if seq_len == 1 {
//             self.forward_decode(hidden)
//         } else {
//             // Use the parallel join approach for the multi-token prefill step.
//             self.forward_prefill(hidden)
//         }
//     }

//     /// Hyper-optimized path for the decode step (batch=1, seq=1).
//     /// Uses a specialized, fused SIMD kernel for the gate/up projections.
//     fn forward_decode(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
//         let (batch, seq, hidden_dim) = hidden.dim();
//         let hidden_2d = hidden.view().into_shape((batch, hidden_dim))?;

//         let t_total = Instant::now();

//         // --- Step 1: Call the single, fused, hyper-optimized kernel ---
//         let t_gate_up = Instant::now();
//         let (mut gate_out, up_out) = match (&self.gate.data, &self.up.data) {
//             (LinearData::BF16(gate_w), LinearData::BF16(up_w)) => {
//                 kernels::gate_up_fused_bf16(&hidden_2d, &gate_w.view(), &up_w.view())?
//             }
//             (LinearData::F32(gate_w), LinearData::F32(up_w)) => {
//                  kernels::gate_up_fused_f32(&hidden_2d, &gate_w.view(), &up_w.view())?
//             }
//             // Fallback for mismatched types or other dtypes
//             _ => {
//                 log::warn!("Falling back to rayon::join in SwiGLU decode path due to mismatched dtypes.");
//                 rayon::join(
//                     || self.gate.matmul(&hidden_2d),
//                     || self.up.matmul(&hidden_2d),
//                 )
//             }
//         };
//         let d_gate_up = t_gate_up.elapsed();

//         // --- Step 2: Activation & Element-wise Ops ---
//         let t_act = Instant::now();
//         silu_parallel(&mut gate_out);
//         let activated = gate_out * up_out;
//         let d_act = t_act.elapsed();

//         // --- Step 3: Down Projection ---
//         let t_down = Instant::now();
//         let output_2d = self.down.matmul(&activated.view());
//         let d_down = t_down.elapsed();

//         let d_total = t_total.elapsed();
//         if d_total.as_millis() > 1 {
//             log::info!(
//                 "[FFN Perf DECODE] Total: {:?}, Gate+Up Fused: {:?}, Activation: {:?}, Down Matmul: {:?}",
//                 d_total, d_gate_up, d_act, d_down
//             );
//         }

//         Ok(output_2d.into_shape((batch, seq, self.down.out_features()))?)
//     }

//     /// Parallel path for the prefill step (seq > 1).
//     fn forward_prefill(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
//         let (batch, seq, hidden_dim) = hidden.dim();
//         let hidden_2d = hidden.view().into_shape((batch * seq, hidden_dim))?;

//         let t_total = Instant::now();

//         let t_gate_up = Instant::now();
//         let (mut gate_out, up_out) = rayon::join(
//             || self.gate.matmul(&hidden_2d.view()),
//             || self.up.matmul(&hidden_2d.view()),
//         );
//         let d_gate_up = t_gate_up.elapsed();

//         let t_act = Instant::now();
//         silu_parallel(&mut gate_out);
//         let activated = gate_out * up_out;
//         let d_act = t_act.elapsed();

//         let t_down = Instant::now();
//         let output_2d = self.down.matmul(&activated.view());
//         let d_down = t_down.elapsed();

//         let d_total = t_total.elapsed();
//         if d_total.as_millis() > 5 {
//              log::info!(
//                 "[FFN Perf PREFILL] Total: {:?}, Gate+Up Parallel: {:?}, Activation: {:?}, Down Matmul: {:?}",
//                 d_total, d_gate_up, d_act, d_down
//             );
//         }

//         Ok(output_2d.into_shape((batch, seq, self.down.out_features()))?)
//     }
// }

// /// A parallel, in-place SiLU activation function.
// #[inline]
// fn silu_parallel(x: &mut Array2<f32>) {
//     if let Some(slice) = x.as_slice_mut() {
//         slice.par_iter_mut().for_each(|v| *v = *v / (1.0 + (-*v).exp()));
//     } else {
//         x.par_mapv_inplace(|v| v / (1.0 + (-v).exp()));
//     }
// }

// /// Private module for unsafe, specialized SIMD kernels.
// mod kernels {
//     use anyhow::{anyhow, Result};
//     use ndarray::{Array2, ArrayView2};
//     use rayon::prelude::*;

//     #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
//     use std::arch::x86_64::*;

//     /// Safe wrapper for the fused BF16 gate/up kernel.
//     /// Safe wrapper for the fused BF16 gate/up kernel.
//     pub fn gate_up_fused_bf16(
//         a: &ArrayView2<f32>,
//         gate_w: &ArrayView2<half::bf16>,
//         up_w: &ArrayView2<half::bf16>,
//     ) -> Result<(Array2<f32>, Array2<f32>)> {
//         let (m, k) = a.dim();
//         let (n, k2) = gate_w.dim();
//         if m != 1 || k != k2 || gate_w.dim() != up_w.dim() {
//             return Err(anyhow!("Dimension mismatch for fused BF16 kernel"));
//         }

//         let mut gate_out = Array2::zeros((m, n));
//         let mut up_out = Array2::zeros((m, n));

//         let a_slice = a.as_slice().ok_or_else(|| anyhow!("Input not contiguous"))?;
//         let gate_w_slice = gate_w.as_slice().ok_or_else(|| anyhow!("Gate weights not contiguous"))?;
//         let up_w_slice = up_w.as_slice().ok_or_else(|| anyhow!("Up weights not contiguous"))?;
//         let gate_out_slice = gate_out.as_slice_mut().ok_or_else(|| anyhow!("Gate output not contiguous"))?;
//         let up_out_slice = up_out.as_slice_mut().ok_or_else(|| anyhow!("Up output not contiguous"))?;

//         #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
//         if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
//             unsafe {
//                 swiglu_gate_up_kernel_avx2_bf16(
//                     gate_out_slice,
//                     up_out_slice,
//                     a_slice,
//                     gate_w_slice,
//                     up_w_slice,
//                     k,
//                     n,
//                 );
//             }
//         } else {
//             return Err(anyhow!("AVX2/FMA required"));
//         }

//         Ok((gate_out, up_out))
//     }

//     /// Safe wrapper stub for the fused F32 gate/up kernel.
//     pub fn gate_up_fused_f32(
//         _a: &ArrayView2<f32>,
//         _gate_w: &ArrayView2<f32>,
//         _up_w: &ArrayView2<f32>,
//     ) -> Result<(Array2<f32>, Array2<f32>)> {
//          Err(anyhow!("Fused F32 kernel not implemented yet"))
//     }

//     /// A hyper-optimized AVX2/FMA kernel for the fused Gate/Up projection (BF16 weights).
//     #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
//     #[target_feature(enable = "avx2", enable = "fma")]
//     unsafe fn swiglu_gate_up_kernel_avx2_bf16(
//         gate_out: &mut [f32],
//         up_out: &mut [f32],
//         a: &[f32],
//         gate_w: &[half::bf16],
//         up_w: &[half::bf16],
//         k: usize,
//         n: usize,
//     ) {
//         // Use indices instead of pointers - rayon is happy with this
//         gate_out.par_iter_mut()
//             .zip(up_out.par_iter_mut())
//             .enumerate()
//             .for_each(|(i, (gate_out_i, up_out_i))| {
//                 let gate_w_row = &gate_w[i * k..(i + 1) * k];
//                 let up_w_row = &up_w[i * k..(i + 1) * k];

//                 let mut gate_sum_vec = _mm256_setzero_ps();
//                 let mut up_sum_vec = _mm256_setzero_ps();

//                 let mut j = 0;
//                 while j + 8 <= k {
//                     let a_vec = _mm256_loadu_ps(a.as_ptr().add(j));

//                     // Load BF16 and convert to F32
//                     let gate_w_u16 = _mm_loadu_si128(gate_w_row.as_ptr().add(j) as *const __m128i);
//                     let gate_w_u32 = _mm256_cvtepu16_epi32(gate_w_u16);
//                     let gate_w_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(gate_w_u32, 16));

//                     let up_w_u16 = _mm_loadu_si128(up_w_row.as_ptr().add(j) as *const __m128i);
//                     let up_w_u32 = _mm256_cvtepu16_epi32(up_w_u16);
//                     let up_w_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(up_w_u32, 16));

//                     gate_sum_vec = _mm256_fmadd_ps(a_vec, gate_w_f32, gate_sum_vec);
//                     up_sum_vec = _mm256_fmadd_ps(a_vec, up_w_f32, up_sum_vec);

//                     j += 8;
//                 }

//                 let mut gate_sum = hsum_ps_avx(gate_sum_vec);
//                 let mut up_sum = hsum_ps_avx(up_sum_vec);

//                 // Remainder
//                 while j < k {
//                     let a_val = a[j];
//                     gate_sum += a_val * gate_w_row[j].to_f32();
//                     up_sum += a_val * up_w_row[j].to_f32();
//                     j += 1;
//                 }

//                 *gate_out_i = gate_sum;
//                 *up_out_i = up_sum;
//             });
//     }

//     /// Helper function to horizontally sum a `__m256` vector.
//     #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
//     #[inline]
//     unsafe fn hsum_ps_avx(v: __m256) -> f32 {
//         let vlow = _mm256_castps256_ps128(v);
//         let vhigh = _mm256_extractf128_ps(v, 1);
//         let vsum = _mm_add_ps(vlow, vhigh);
//         let vsum = _mm_hadd_ps(vsum, vsum);
//         let vsum = _mm_hadd_ps(vsum, vsum);
//         _mm_cvtss_f32(vsum)
//     }
// }

use crate::linear_layer::{LinearData, LinearLayer};
use crate::ops; // Use the new, clean 'ops' module for computations
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
        let (mut gate_out, up_out) = match (&self.gate.data, &self.up.data) {
            (LinearData::BF16(gate_w), LinearData::BF16(up_w)) => {
                ops::swiglu_fused::gate_up_fused_bf16(&hidden_2d, &gate_w.view(), &up_w.view())?
            }
            (LinearData::F32(_gate_w), LinearData::F32(_up_w)) => {
                // This path will be enabled once the F32 fused kernel is implemented in ops/kernels
                log::warn!(
                    "Falling back to rayon::join in SwiGLU F32 decode path (fused kernel not yet implemented)."
                );
                rayon::join(
                    || self.gate.matmul(&hidden_2d),
                    || self.up.matmul(&hidden_2d),
                )
            }
            _ => {
                log::warn!(
                    "Falling back to rayon::join in SwiGLU decode path due to mismatched dtypes."
                );
                rayon::join(
                    || self.gate.matmul(&hidden_2d),
                    || self.up.matmul(&hidden_2d),
                )
            }
        };
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

        Ok(output_2d.into_shape((batch, seq, self.down.out_features()))?)
    }

    /// Parallel path for the prefill step (seq > 1).
    fn forward_prefill(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, hidden_dim) = hidden.dim();
        let hidden_2d = hidden.view().into_shape((batch * seq, hidden_dim))?;

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

        Ok(output_2d.into_shape((batch, seq, self.down.out_features()))?)
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
