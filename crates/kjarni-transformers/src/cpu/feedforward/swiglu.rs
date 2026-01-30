use crate::activations::{Activation, apply_activation_2d};
use crate::linear_layer::LinearLayer;
use anyhow::Result;
use ndarray::{Array2, Array3};
use rayon::prelude::*;
use std::time::Instant;

/// Fused gate+up+SiLU for decode with BF16 weights.
/// Input/output are F32, weights are BF16, accumulation in F32.
#[inline]
pub fn fused_gate_up_silu_bf16(
    input: &[f32],              // [hidden_dim]
    gate_weight: &[half::bf16], // [intermediate_dim, hidden_dim]
    up_weight: &[half::bf16],   // [intermediate_dim, hidden_dim]
    output: &mut [f32],         // [intermediate_dim]
    hidden_dim: usize,
) {
    use rayon::prelude::*;

    output.par_iter_mut().enumerate().for_each(|(n, out)| {
        let weight_offset = n * hidden_dim;

        let mut gate_sum = 0.0f32;
        let mut up_sum = 0.0f32;

        for k in 0..hidden_dim {
            let val = unsafe { *input.get_unchecked(k) };
            let g = unsafe { gate_weight.get_unchecked(weight_offset + k).to_f32() };
            let u = unsafe { up_weight.get_unchecked(weight_offset + k).to_f32() };
            gate_sum += val * g;
            up_sum += val * u;
        }

        // SiLU(gate) * up
        *out = (gate_sum / (1.0 + (-gate_sum).exp())) * up_sum;
    });
}

/// Fused gate+up+SiLU for decode (seq_len=1).
///
/// Computes: output = SiLU(input @ gate.T) * (input @ up.T)
///
/// Reads input once, streams both weight matrices together.
/// ~50% bandwidth reduction vs separate matmuls.
#[inline]
pub fn fused_gate_up_silu_decode(
    input: &[f32],       // [hidden_dim]
    gate_weight: &[f32], // [intermediate_dim, hidden_dim] row-major
    up_weight: &[f32],   // [intermediate_dim, hidden_dim] row-major
    output: &mut [f32],  // [intermediate_dim]
    hidden_dim: usize,
) {
    use rayon::prelude::*;

    output.par_iter_mut().enumerate().for_each(|(n, out)| {
        let weight_offset = n * hidden_dim;

        let mut gate_sum = 0.0f32;
        let mut up_sum = 0.0f32;

        // Fused loop: read input once, accumulate both projections
        for k in 0..hidden_dim {
            let val = unsafe { *input.get_unchecked(k) };
            gate_sum += val * unsafe { *gate_weight.get_unchecked(weight_offset + k) };
            up_sum += val * unsafe { *up_weight.get_unchecked(weight_offset + k) };
        }

        // SiLU(gate) * up
        *out = (gate_sum / (1.0 + (-gate_sum).exp())) * up_sum;
    });
}

/// Fused gate+up+SiLU for prefill (seq_len > 1).
#[inline]
pub fn fused_gate_up_silu_prefill(
    input: &[f32],       // [tokens, hidden_dim]
    gate_weight: &[f32], // [intermediate_dim, hidden_dim]
    up_weight: &[f32],   // [intermediate_dim, hidden_dim]
    output: &mut [f32],  // [tokens, intermediate_dim]
    tokens: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
) {
    use rayon::prelude::*;

    // Parallel over tokens (better cache locality for weight reuse)
    output
        .par_chunks_mut(intermediate_dim)
        .enumerate()
        .for_each(|(m, out_row)| {
            let input_offset = m * hidden_dim;

            for n in 0..intermediate_dim {
                let weight_offset = n * hidden_dim;

                let mut gate_sum = 0.0f32;
                let mut up_sum = 0.0f32;

                for k in 0..hidden_dim {
                    let val = unsafe { *input.get_unchecked(input_offset + k) };
                    gate_sum += val * unsafe { *gate_weight.get_unchecked(weight_offset + k) };
                    up_sum += val * unsafe { *up_weight.get_unchecked(weight_offset + k) };
                }

                out_row[n] = (gate_sum / (1.0 + (-gate_sum).exp())) * up_sum;
            }
        });
}

/// A high-performance SwiGLU Feed-Forward Network.
///
/// This struct orchestrates the SwiGLU operation, a key component in modern
/// transformer architectures. It dispatches to specialized computation paths
/// based on the input sequence length for maximum performance.
pub struct SwiGluFeedForward {
    pub gate: LinearLayer,
    pub up: LinearLayer,
    pub down: LinearLayer,
    pub activation: Activation,
}

impl SwiGluFeedForward {
    /// Creates a new `SwiGluFeedForward` layer.
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
            activation: activation,
        }
    }

    fn apply_activation(&self, x: &mut Array2<f32>) {
        match self.activation {
            Activation::Gelu => {
                // Use your existing GELU implementation
                x.par_mapv_inplace(|v| {
                    0.5 * v * (1.0 + f32::tanh(0.79788456 * (v + 0.044715 * v.powi(3))))
                });
            }
            Activation::GeluNew => {
                // Flan-T5 specifically requests this
                // 0.5 * x * (1 + tanh(...))
                x.par_mapv_inplace(|v| {
                    0.5 * v * (1.0 + f32::tanh(0.79788456 * (v + 0.044715 * v.powi(3))))
                });
            }
            _ => silu_parallel(x),
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
    // pub fn forward_2d(&self, hidden: &Array2<f32>) -> Result<Array2<f32>> {
    //     let (rows, dim) = hidden.dim();

    //     // 1. Gate & Up
    //     let mut gate_out = self.gate.matmul(&hidden.view());
    //     let up_out = self.up.matmul(&hidden.view());

    //     // 2. Activation
    //     silu_parallel(&mut gate_out);
    //     // self.apply_activation(&mut gate_out);
    //     let activated = gate_out * up_out;

    //     // 3. Down
    //     let output_2d = self.down.matmul(&activated.view());

    //     Ok(output_2d)
    // }
    pub fn forward_2d(&self, hidden: &Array2<f32>) -> Result<Array2<f32>> {
        // 1. Gate & Up Projections
        let mut gate_out = self.gate.matmul(&hidden.view());
        let up_out = self.up.matmul(&hidden.view());

        // 2. Activation (Gate = Act(Gate))
        // We use the centralized activation logic here
        apply_activation_2d(&mut gate_out, self.activation);

        // 3. Element-wise Mult (Gate * Up)
        // Note: ndarray handles SIMD for element-wise ops automatically in many cases
        let activated = gate_out * up_out;

        // 4. Down Projection
        let output_2d = self.down.matmul(&activated.view());

        Ok(output_2d)
    }
    fn forward_decode(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, hidden_dim) = hidden.dim();
        let intermediate_dim = self.gate.out_features();

        let input = hidden
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Input must be contiguous"))?;

        let mut activated = vec![0.0f32; batch * seq * intermediate_dim];

        let up_bf16 = self.up.weights_slice_bf16().unwrap();
        let gate_bf16 = self.gate.weights_slice_bf16().unwrap();
        fused_gate_up_silu_bf16(input, gate_bf16, up_bf16, &mut activated, hidden_dim);

        let activated_arr = Array2::from_shape_vec((batch * seq, intermediate_dim), activated)?;
        let output_2d = self.down.matmul(&activated_arr.view());

        Ok(output_2d.into_shape_with_order((batch, seq, self.down.out_features()))?)
    }
    //  /// Hyper-optimized path for the decode step (batch=1, seq=1).
    // fn forward_decode(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
    //     let (batch, seq, hidden_dim) = hidden.dim();
    //     // Flatten [B, S, H] -> [B*S, H]
    //     let hidden_2d = hidden.view().into_shape_with_order((batch * seq, hidden_dim))?;

    //     let t_total = Instant::now();

    //     // --- Step 1: Gate & Up ---
    //     let t_gate_up = Instant::now();

    //     // In a real optimized scenario, you might use rayon::join here,
    //     // but for safety/simplicity we run sequential or rely on internal BLAS threading.
    //     let mut gate_out = self.gate.matmul(&hidden_2d);
    //     let up_out = self.up.matmul(&hidden_2d);

    //     let d_gate_up = t_gate_up.elapsed();

    //     // --- Step 2: Activation & Element-wise Ops ---
    //     let t_act = Instant::now();

    //     // Apply activation in-place to gate_out
    //     apply_activation_2d(&mut gate_out, self.activation);

    //     let activated = gate_out * up_out;
    //     let d_act = t_act.elapsed();

    //     // --- Step 3: Down Projection ---
    //     let t_down = Instant::now();
    //     let output_2d = self.down.matmul(&activated.view());
    //     let d_down = t_down.elapsed();

    //     let d_total = t_total.elapsed();
    //     // Only log if it's unusually slow (microsecond logging usually too verbose)
    //     if d_total.as_millis() > 1 {
    //         log::debug!(
    //             "[FFN DECODE] Total: {:?}, Gate+Up: {:?}, Act: {:?}, Down: {:?}",
    //             d_total, d_gate_up, d_act, d_down
    //         );
    //     }

    //     Ok(output_2d.into_shape_with_order((batch, seq, self.down.out_features()))?)
    // }

    // /// Parallel path for the prefill step (seq > 1).
    // fn forward_prefill(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
    //     let (batch, seq, hidden_dim) = hidden.dim();
    //     let hidden_2d = hidden.view().into_shape_with_order((batch * seq, hidden_dim))?;

    //     let t_total = Instant::now();

    //     let t_gate_up = Instant::now();
    //     // let (mut gate_out, up_out) = rayon::join(
    //     //     || self.gate.matmul(&hidden_2d.view()),
    //     //     || self.up.matmul(&hidden_2d.view()),
    //     // );
    //     let mut gate_out = self.gate.matmul(&hidden_2d);
    //     let mut up_out = self.up.matmul(&hidden_2d);
    //     let d_gate_up = t_gate_up.elapsed();

    //     let t_act = Instant::now();
    //     // silu_parallel(&mut gate_out);
    //     self.apply_activation(&mut gate_out);
    //     let activated = gate_out * up_out;
    //     let d_act = t_act.elapsed();

    //     let t_down = Instant::now();
    //     let output_2d = self.down.matmul(&activated.view());
    //     let d_down = t_down.elapsed();

    //     let d_total = t_total.elapsed();
    //     if d_total.as_millis() > 5 {
    //         log::info!(
    //             "[FFN Perf PREFILL] Total: {:?}, Gate+Up Parallel: {:?}, Activation: {:?}, Down Matmul: {:?}",
    //             d_total,
    //             d_gate_up,
    //             d_act,
    //             d_down
    //         );
    //     }

    //     Ok(output_2d.into_shape_with_order((batch, seq, self.down.out_features()))?)
    // }
    /// Parallel path for the prefill step (seq > 1).
    fn forward_prefill(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
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

        // Use centralized activation
        apply_activation_2d(&mut gate_out, self.activation);

        let activated = gate_out * up_out;
        let d_act = t_act.elapsed();

        let t_down = Instant::now();
        let output_2d = self.down.matmul(&activated.view());
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

// ========================================================================
//  TEST SUITE
// ========================================================================

#[cfg(test)]
mod swiglu_tests {
    use super::*;
    use crate::activations::Activation;
    use crate::linear_layer::LinearLayer;
    use ndarray::{Array2, Array3};

    // --- Helpers ---

    /// Custom assertion to print detailed differences if tolerance is exceeded.
    fn assert_all_close(a: &Array3<f32>, b: &Array3<f32>, tol: f32) {
        assert_eq!(a.dim(), b.dim(), "Dimensions mismatch");
        let diff = a - b;
        let max_diff = diff.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

        if max_diff > tol {
            panic!(
                "Mismatch exceeding tolerance {}.\nMax Diff: {}\nGot:      {:?}\nExpected: {:?}",
                tol, max_diff, a, b
            );
        }
    }

    /// Creates a LinearLayer from flat data.
    /// Shape expected: (out_features, in_features)
    fn mock_linear(weights_data: Vec<f32>, shape: (usize, usize)) -> LinearLayer {
        let weights = Array2::from_shape_vec(shape, weights_data).unwrap();
        LinearLayer::new_f32(weights, None)
    }

    /// Generic test runner that sets up the exact environment from the Python script.
    fn run_golden_test(activation: Activation, expected_flat: Vec<f32>) -> Result<()> {
        // === SETUP DATA (From Python) ===
        // Batch=1, Seq=2, In=2, Hidden=2, Out=2

        // Input: [0.5, -0.5, 0.1, 0.2]
        let input_data = vec![0.5, -0.5, 0.10000000149011612, 0.20000000298023224];
        let input = Array3::from_shape_vec((1, 2, 2), input_data)?;

        // Gate Weights (Out=2, In=2)
        // [0.2, -0.1,
        //  0.3,  0.4]
        let w_gate_data = vec![
            0.20000000298023224,
            -0.10000000149011612,
            0.30000001192092896,
            0.4000000059604645,
        ];
        let gate = mock_linear(w_gate_data, (2, 2));

        // Up Weights (Out=2, In=2)
        // [ 0.5, 0.1,
        //  -0.2, 0.3]
        let w_up_data = vec![
            0.5,
            0.10000000149011612,
            -0.20000000298023224,
            0.30000001192092896,
        ];
        let up = mock_linear(w_up_data, (2, 2));

        // Down Weights (Out=2, In=2)
        // [ 0.1, 0.2,
        //  -0.1, 0.1]
        let w_down_data = vec![
            0.10000000149011612,
            0.20000000298023224,
            -0.10000000149011612,
            0.10000000149011612,
        ];
        let down = mock_linear(w_down_data, (2, 2));

        // === EXECUTION ===
        let ffn = SwiGluFeedForward::new(gate, up, down, activation);
        let output = ffn.forward(&input)?;

        // === VERIFICATION ===
        let expected = Array3::from_shape_vec((1, 2, 2), expected_flat)?;

        // Tolerance: 1e-5 to handle the float string parsing differences
        // between Python print output and Rust f32.
        assert_all_close(&output, &expected, 1e-5);

        Ok(())
    }

    // ========================================================================
    //  Golden Value Tests
    // ========================================================================

    #[test]
    fn test_swiglu_golden_relu() -> Result<()> {
        let expected = vec![0.003, -0.003, 0.00088, 0.00044];
        run_golden_test(Activation::Relu, expected)
    }

    #[test]
    fn test_swiglu_golden_gelu() -> Result<()> {
        let expected = vec![0.002879, -0.001079, 0.000479, 0.000239];
        run_golden_test(Activation::Gelu, expected)
    }

    #[test]
    fn test_swiglu_golden_gelu_new() -> Result<()> {
        let expected = vec![0.002879, -0.001079, 0.000479, 0.000239];
        run_golden_test(Activation::GeluNew, expected)
    }

    #[test]
    fn test_swiglu_golden_silu() -> Result<()> {
        let expected = vec![0.002831, -0.001003, 0.000464, 0.000232];
        run_golden_test(Activation::SilU, expected)
    }

    #[test]
    fn test_swiglu_golden_tanh() -> Result<()> {
        let expected = vec![0.005476, -0.001729, 0.000876, 0.000438];
        run_golden_test(Activation::Tanh, expected)
    }
}
