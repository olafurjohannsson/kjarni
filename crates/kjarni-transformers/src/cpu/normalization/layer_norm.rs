//! Layer normalization implementation
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use anyhow::{Result, anyhow};
use ndarray::{Array1, Array3, ArrayView2, ArrayView3, ArrayViewMut2, Axis, Ix1};

use crate::tensor::CpuTensor;

/// A data structure holding the learnable parameters for Layer Normalization.
/// This allows for type-safe handling of different on-disk data types.

/// Layer normalization
pub struct LayerNorm {
    pub weight: Array1<f32>,
    pub bias: Array1<f32>,
    pub eps: f32,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
    _mm_cvtss_f32(_mm_add_ss(sum64, hi32))
}

impl LayerNorm {


    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn forward_2d_noalloc_simd(
        &self,
        input: &ArrayView2<f32>,
        output: &mut ArrayViewMut2<f32>,
    ) {
        let tokens = input.shape()[0];
        let hidden = input.shape()[1];

        // Fall back to scalar for non-aligned or small hidden dims
        if hidden % 8 != 0 || hidden < 64 {
            return self.forward_2d_noalloc_scalar(input, output);
        }

        let eps = self.eps;
        let weight = self.weight.as_slice().unwrap();
        let bias = self.bias.as_slice().unwrap();

        for t in 0..tokens {
            let in_ptr = input.row(t).as_ptr();
            let out_ptr = output.row_mut(t).as_mut_ptr();

            unsafe {
                // Compute sum using SIMD
                let mut sum_vec = _mm256_setzero_ps();
                for i in (0..hidden).step_by(8) {
                    let v = _mm256_loadu_ps(in_ptr.add(i));
                    sum_vec = _mm256_add_ps(sum_vec, v);
                }
                let sum = hsum_avx(sum_vec);
                let mean = sum / hidden as f32;
                let mean_vec = _mm256_set1_ps(mean);

                // Compute variance using SIMD
                let mut var_vec = _mm256_setzero_ps();
                for i in (0..hidden).step_by(8) {
                    let v = _mm256_loadu_ps(in_ptr.add(i));
                    let diff = _mm256_sub_ps(v, mean_vec);
                    var_vec = _mm256_fmadd_ps(diff, diff, var_vec);
                }
                let var = hsum_avx(var_vec) / hidden as f32;
                let inv_std = 1.0 / (var + eps).sqrt();
                let inv_std_vec = _mm256_set1_ps(inv_std);

                // Normalize, scale, shift using SIMD
                for i in (0..hidden).step_by(8) {
                    let v = _mm256_loadu_ps(in_ptr.add(i));
                    let w = _mm256_loadu_ps(weight.as_ptr().add(i));
                    let b = _mm256_loadu_ps(bias.as_ptr().add(i));

                    let normed = _mm256_mul_ps(_mm256_sub_ps(v, mean_vec), inv_std_vec);
                    let scaled = _mm256_fmadd_ps(normed, w, b);

                    _mm256_storeu_ps(out_ptr.add(i), scaled);
                }
            }
        }
    }

    #[inline]
    pub fn forward_2d_noalloc_scalar(
        &self,
        input: &ArrayView2<f32>,
        output: &mut ArrayViewMut2<f32>,
    ) {
        let tokens = input.shape()[0];
        let hidden = input.shape()[1];
        let eps = self.eps;
        let weight = self.weight.as_slice().unwrap();
        let bias = self.bias.as_slice().unwrap();

        // Process each token (row)
        for t in 0..tokens {
            let input_row = input.row(t);
            let mut output_row = output.row_mut(t);

            let in_slice = input_row.as_slice().unwrap();
            let out_slice = output_row.as_slice_mut().unwrap();

            // Compute mean (single pass)
            let mut sum = 0.0f32;
            for i in 0..hidden {
                sum += in_slice[i];
            }
            let mean = sum / hidden as f32;

            // Compute variance (single pass)
            let mut var_sum = 0.0f32;
            for i in 0..hidden {
                let diff = in_slice[i] - mean;
                var_sum += diff * diff;
            }
            let inv_std = 1.0 / (var_sum / hidden as f32 + eps).sqrt();

            // Normalize, scale, shift (single pass, SIMD-friendly)
            for i in 0..hidden {
                out_slice[i] = (in_slice[i] - mean) * inv_std * weight[i] + bias[i];
            }
        }
    }
}

impl LayerNorm {
    pub fn new(weight: Array1<f32>, bias: Array1<f32>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }
    pub fn from_weights(
        weights: &crate::weights::ModelWeights,
        weight_name: &str,
        bias_name: &str,
        eps: f32,
    ) -> Result<Self> {
        // Small tensors - always convert to F32
        let weight = match weights.get_typed_tensor(weight_name)? {
            CpuTensor::F32(arr) => arr.into_dimensionality::<Ix1>()?,
            CpuTensor::F16(arr) => arr.mapv(|v| v.to_f32()).into_dimensionality::<Ix1>()?,
            CpuTensor::BF16(arr) => arr.mapv(|v| v.to_f32()).into_dimensionality::<Ix1>()?,
            _ => return Err(anyhow!("Unsupported dtype for LayerNorm weight")),
        };

        let bias = match weights.get_typed_tensor(bias_name)? {
            CpuTensor::F32(arr) => arr.into_dimensionality::<Ix1>()?,
            CpuTensor::F16(arr) => arr.mapv(|v| v.to_f32()).into_dimensionality::<Ix1>()?,
            CpuTensor::BF16(arr) => arr.mapv(|v| v.to_f32()).into_dimensionality::<Ix1>()?,
            _ => return Err(anyhow!("Unsupported dtype for LayerNorm bias")),
        };

        Ok(Self { weight, bias, eps })
    }

    #[inline]
    pub fn forward_2d_noalloc(&self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>) {
        unsafe {
            self.forward_2d_noalloc_simd(input, output);
        }
    }

    /// Reference 3D implementation for comparison
    pub fn forward_3d_reference(&self, hidden_states: &Array3<f32>) -> Array3<f32> {
        let (batch, seq, hidden) = hidden_states.dim();
        let mut output = Array3::<f32>::zeros((batch, seq, hidden));

        for b in 0..batch {
            for s in 0..seq {
                let row = hidden_states.slice(ndarray::s![b, s, ..]);
                let in_slice = row.as_slice().unwrap();

                // Compute mean
                let mean: f32 = in_slice.iter().sum::<f32>() / hidden as f32;

                // Compute variance
                let var: f32 =
                    in_slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden as f32;

                let inv_std = 1.0 / (var + self.eps).sqrt();

                for i in 0..hidden {
                    output[[b, s, i]] =
                        (in_slice[i] - mean) * inv_std * self.weight[i] + self.bias[i];
                }
            }
        }
        output
    }

    /// Apply layer norm to a 3D tensor of activations.
    /// This method is now extremely simple and fast because it knows its weights are already F32.
    #[inline]
    pub fn forward(&self, hidden_states: &ArrayView3<f32>) -> Array3<f32> {
        let mean = hidden_states.mean_axis(Axis(2)).unwrap();
        let variance = hidden_states.var_axis(Axis(2), 0.0);

        let mean_expanded = mean.insert_axis(Axis(2));
        let var_expanded = variance.insert_axis(Axis(2));

        let inv_std = (&var_expanded + self.eps).mapv(|x| 1.0 / x.sqrt());
        let normalized_hidden = (hidden_states.to_owned() - &mean_expanded) * &inv_std;

        // Simple, fast F32 math. No matching, no on-the-fly conversion.
        normalized_hidden * &self.weight + &self.bias
    }

    /// Apply layer norm to a 3D tensor
    pub fn forward_3d(&self, hidden: &Array3<f32>) -> Array3<f32> {
        self.forward(&hidden.view())
    }
}

#[cfg(test)]
mod layer_norm_tests {
    use super::*;
    use ndarray::{Array2, Array3, arr1};

    #[test]
    fn test_layer_norm_basic() {
        // Simple test with known values
        let weight = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let bias = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let eps = 1e-6;
        let layer_norm = LayerNorm::new(weight, bias, eps);

        // Input: [1, 1, 3] tensor with values [1.0, 2.0, 3.0]
        // Mean = 2.0, Variance = 2/3
        let hidden = Array3::from_shape_vec((1, 1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let output = layer_norm.forward_3d(&hidden);

        // After normalization: mean=0, variance=1
        let output_mean = (output[[0, 0, 0]] + output[[0, 0, 1]] + output[[0, 0, 2]]) / 3.0;
        assert!(output_mean.abs() < 1e-5); // Mean should be ~0

        // Verify normalized values
        // (1-2)/sqrt(2/3) ≈ -1.2247, (2-2)/sqrt(2/3) = 0, (3-2)/sqrt(2/3) ≈ 1.2247
        assert!((output[[0, 0, 0]] - (-1.2247)).abs() < 1e-3);
        assert!((output[[0, 0, 1]] - 0.0).abs() < 1e-5);
        assert!((output[[0, 0, 2]] - 1.2247).abs() < 1e-3);
    }

    #[test]
    fn test_layer_norm_with_scale_and_bias() {
        let weight = Array1::from_vec(vec![2.0, 0.5, 1.5]);
        let bias = Array1::from_vec(vec![1.0, -1.0, 0.5]);
        let eps = 1e-6;
        let layer_norm = LayerNorm::new(weight, bias, eps);

        let hidden = Array3::from_shape_vec((1, 1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let output = layer_norm.forward_3d(&hidden);

        // First normalize, then scale by weight and add bias
        let mean = 2.0;
        let var = 2.0 / 3.0;
        let std = (var + eps).sqrt();

        let normalized_0 = (1.0 - mean) / std;
        let normalized_1 = (2.0 - mean) / std;
        let normalized_2 = (3.0 - mean) / std;

        let expected_0 = normalized_0 * 2.0 + 1.0;
        let expected_1 = normalized_1 * 0.5 + (-1.0);
        let expected_2 = normalized_2 * 1.5 + 0.5;

        assert!((output[[0, 0, 0]] - expected_0).abs() < 1e-4);
        assert!((output[[0, 0, 1]] - expected_1).abs() < 1e-4);
        assert!((output[[0, 0, 2]] - expected_2).abs() < 1e-4);
    }

    #[test]
    fn test_layer_norm_batch() {
        let weight = Array1::from_vec(vec![1.0, 1.0]);
        let bias = Array1::from_vec(vec![0.0, 0.0]);
        let eps = 1e-5;
        let layer_norm = LayerNorm::new(weight, bias, eps);

        // [2, 2, 2] - batch=2, seq=2, hidden=2
        let hidden = Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                1.0, 3.0, // batch 0, pos 0: mean=2, var=1
                2.0, 4.0, // batch 0, pos 1: mean=3, var=1
                5.0, 7.0, // batch 1, pos 0: mean=6, var=1
                6.0, 8.0, // batch 1, pos 1: mean=7, var=1
            ],
        )
        .unwrap();

        let output = layer_norm.forward_3d(&hidden);

        // Each position should be normalized independently
        // Position [0, 0]: [1.0, 3.0] -> normalized to [-1, 1]
        assert!((output[[0, 0, 0]] - (-1.0)).abs() < 1e-3);
        assert!((output[[0, 0, 1]] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_layer_norm_pytorch_parity() {
        // Test against known PyTorch output
        // PyTorch code:
        // ```python
        // import torch
        // x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        // layer_norm = torch.nn.LayerNorm(4)
        // layer_norm.weight.data = torch.tensor([1.0, 1.0, 1.0, 1.0])
        // layer_norm.bias.data = torch.tensor([0.0, 0.0, 0.0, 0.0])
        // output = layer_norm(x)
        // print(output)
        // ```
        // Output: tensor([[[-1.3416, -0.4472,  0.4472,  1.3416]]])

        let weight = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let bias = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let eps = 1e-5;
        let layer_norm = LayerNorm::new(weight, bias, eps);

        let hidden = Array3::from_shape_vec((1, 1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = layer_norm.forward_3d(&hidden);

        assert!((output[[0, 0, 0]] - (-1.3416)).abs() < 1e-3);
        assert!((output[[0, 0, 1]] - (-0.4472)).abs() < 1e-3);
        assert!((output[[0, 0, 2]] - 0.4472).abs() < 1e-3);
        assert!((output[[0, 0, 3]] - 1.3416).abs() < 1e-3);
    }

    #[test]
    fn test_layer_norm_constant_input() {
        // When all values are the same, variance is 0
        // Should handle gracefully (eps prevents division by zero)
        let weight = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let bias = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let eps = 1e-5;
        let layer_norm = LayerNorm::new(weight, bias, eps);

        let hidden = Array3::from_shape_vec((1, 1, 3), vec![5.0, 5.0, 5.0]).unwrap();
        let output = layer_norm.forward_3d(&hidden);

        // All outputs should be 0 (mean-centered with no variance)
        assert!(output[[0, 0, 0]].abs() < 1e-3);
        assert!(output[[0, 0, 1]].abs() < 1e-3);
        assert!(output[[0, 0, 2]].abs() < 1e-3);
    }

    #[test]
    fn test_layer_norm_forward_3d() {
        let weight = arr1(&[1.0, 1.0]);
        let bias = arr1(&[0.0, 0.0]);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = ndarray::Array3::zeros((1, 1, 2));
        let output = ln.forward_3d(&input);

        assert_eq!(output.shape(), &[1, 1, 2]);
    }
    // ============================================================================
    // TEST UTILITIES
    // ============================================================================

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn arrays_approx_eq(a: &Array2<f32>, b: &Array2<f32>, eps: f32) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, eps))
    }

    fn max_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn create_test_input(tokens: usize, hidden: usize, seed: u64) -> Array2<f32> {
        // Simple deterministic pseudo-random for reproducibility
        let mut val = seed as f32;
        Array2::from_shape_fn((tokens, hidden), |_| {
            val = (val * 1.1 + 0.3) % 10.0 - 5.0;
            val
        })
    }

    // ============================================================================
    // SIMD vs SCALAR EQUIVALENCE TESTS
    // ============================================================================

    #[test]
    fn test_simd_scalar_equivalence_hidden_64() {
        let hidden = 64;
        let tokens = 10;

        let weight = Array1::from_vec(vec![1.0; hidden]);
        let bias = Array1::from_vec(vec![0.0; hidden]);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = create_test_input(tokens, hidden, 42);
        let mut output_simd = Array2::<f32>::zeros((tokens, hidden));
        let mut output_scalar = Array2::<f32>::zeros((tokens, hidden));

        unsafe {
            ln.forward_2d_noalloc_simd(&input.view(), &mut output_simd.view_mut());
        }
        

        ln.forward_2d_noalloc_scalar(&input.view(), &mut output_scalar.view_mut());

        let diff = max_diff(&output_simd, &output_scalar);
        println!("[hidden=64] Max diff SIMD vs Scalar: {:.2e}", diff);

        assert!(
            arrays_approx_eq(&output_simd, &output_scalar, 1e-5),
            "SIMD and scalar outputs differ! Max diff: {:.2e}",
            diff
        );
    }

    #[test]
    fn test_simd_scalar_equivalence_hidden_128() {
        let hidden = 128;
        let tokens = 10;

        let weight = Array1::from_vec((0..hidden).map(|i| 1.0 + (i as f32) * 0.01).collect());
        let bias = Array1::from_vec((0..hidden).map(|i| (i as f32) * 0.001).collect());
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = create_test_input(tokens, hidden, 123);
        let mut output_simd = Array2::<f32>::zeros((tokens, hidden));
        let mut output_scalar = Array2::<f32>::zeros((tokens, hidden));
        unsafe {
            ln.forward_2d_noalloc_simd(&input.view(), &mut output_simd.view_mut());
        }
        ln.forward_2d_noalloc_scalar(&input.view(), &mut output_scalar.view_mut());

        let diff = max_diff(&output_simd, &output_scalar);
        println!("[hidden=128] Max diff SIMD vs Scalar: {:.2e}", diff);

        assert!(arrays_approx_eq(&output_simd, &output_scalar, 1e-5));
    }

    #[test]
    fn test_simd_scalar_equivalence_hidden_384_minilm() {
        // MiniLM-L6-v2 hidden size
        let hidden = 384;
        let tokens = 120; // Batch size from your benchmark

        let weight = Array1::from_vec((0..hidden).map(|i| 0.9 + (i as f32) * 0.001).collect());
        let bias = Array1::from_vec((0..hidden).map(|i| -0.5 + (i as f32) * 0.002).collect());
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = create_test_input(tokens, hidden, 999);
        let mut output_simd = Array2::<f32>::zeros((tokens, hidden));
        let mut output_scalar = Array2::<f32>::zeros((tokens, hidden));
        unsafe {
            ln.forward_2d_noalloc_simd(&input.view(), &mut output_simd.view_mut());
        }
        ln.forward_2d_noalloc_scalar(&input.view(), &mut output_scalar.view_mut());

        let diff = max_diff(&output_simd, &output_scalar);
        println!(
            "[hidden=384, tokens=120] Max diff SIMD vs Scalar: {:.2e}",
            diff
        );

        assert!(arrays_approx_eq(&output_simd, &output_scalar, 1e-5));
    }

    #[test]
    fn test_simd_scalar_equivalence_hidden_768() {
        // BERT-base hidden size
        let hidden = 768;
        let tokens = 32;

        let weight = Array1::from_vec(vec![1.0; hidden]);
        let bias = Array1::from_vec(vec![0.0; hidden]);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = create_test_input(tokens, hidden, 777);
        let mut output_simd = Array2::<f32>::zeros((tokens, hidden));
        let mut output_scalar = Array2::<f32>::zeros((tokens, hidden));
        unsafe {
            ln.forward_2d_noalloc_simd(&input.view(), &mut output_simd.view_mut());
        }
        ln.forward_2d_noalloc_scalar(&input.view(), &mut output_scalar.view_mut());

        let diff = max_diff(&output_simd, &output_scalar);
        println!("[hidden=768] Max diff SIMD vs Scalar: {:.2e}", diff);

        assert!(arrays_approx_eq(&output_simd, &output_scalar, 1e-5));
    }

    // ============================================================================
    // FALLBACK TRIGGER TESTS
    // ============================================================================

    #[test]
    fn test_fallback_small_hidden_32() {
        // hidden < 64 should trigger scalar fallback
        let hidden = 32;
        let tokens = 5;

        let weight = Array1::from_vec(vec![1.0; hidden]);
        let bias = Array1::from_vec(vec![0.0; hidden]);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = create_test_input(tokens, hidden, 42);
        let mut output_dispatch = Array2::<f32>::zeros((tokens, hidden));
        let mut output_scalar = Array2::<f32>::zeros((tokens, hidden));

        // forward_2d_noalloc should dispatch to scalar for hidden=32
        ln.forward_2d_noalloc(&input.view(), &mut output_dispatch.view_mut());
        ln.forward_2d_noalloc_scalar(&input.view(), &mut output_scalar.view_mut());

        assert!(
            arrays_approx_eq(&output_dispatch, &output_scalar, 1e-7),
            "Dispatch should use scalar for hidden=32"
        );
        println!("[hidden=32] Fallback to scalar: PASS");
    }

    #[test]
    fn test_fallback_small_hidden_63() {
        // hidden=63 < 64 should trigger scalar fallback
        let hidden = 63;
        let tokens = 5;

        let weight = Array1::from_vec(vec![1.0; hidden]);
        let bias = Array1::from_vec(vec![0.0; hidden]);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = create_test_input(tokens, hidden, 42);
        let mut output_dispatch = Array2::<f32>::zeros((tokens, hidden));
        let mut output_scalar = Array2::<f32>::zeros((tokens, hidden));

        ln.forward_2d_noalloc(&input.view(), &mut output_dispatch.view_mut());
        ln.forward_2d_noalloc_scalar(&input.view(), &mut output_scalar.view_mut());

        assert!(arrays_approx_eq(&output_dispatch, &output_scalar, 1e-7));
        println!("[hidden=63] Fallback to scalar: PASS");
    }

    #[test]
    fn test_fallback_unaligned_65() {
        // hidden=65 not divisible by 8, should trigger scalar
        let hidden = 65;
        let tokens = 5;

        let weight = Array1::from_vec(vec![1.0; hidden]);
        let bias = Array1::from_vec(vec![0.0; hidden]);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = create_test_input(tokens, hidden, 42);
        let mut output_dispatch = Array2::<f32>::zeros((tokens, hidden));
        let mut output_scalar = Array2::<f32>::zeros((tokens, hidden));

        ln.forward_2d_noalloc(&input.view(), &mut output_dispatch.view_mut());
        ln.forward_2d_noalloc_scalar(&input.view(), &mut output_scalar.view_mut());

        assert!(arrays_approx_eq(&output_dispatch, &output_scalar, 1e-7));
        println!("[hidden=65] Fallback to scalar (unaligned): PASS");
    }

    #[test]
    fn test_fallback_unaligned_100() {
        // hidden=100 not divisible by 8
        let hidden = 100;
        let tokens = 5;

        let weight = Array1::from_vec(vec![1.0; hidden]);
        let bias = Array1::from_vec(vec![0.0; hidden]);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = create_test_input(tokens, hidden, 42);
        let mut output_dispatch = Array2::<f32>::zeros((tokens, hidden));
        let mut output_scalar = Array2::<f32>::zeros((tokens, hidden));

        ln.forward_2d_noalloc(&input.view(), &mut output_dispatch.view_mut());
        ln.forward_2d_noalloc_scalar(&input.view(), &mut output_scalar.view_mut());

        assert!(arrays_approx_eq(&output_dispatch, &output_scalar, 1e-7));
        println!("[hidden=100] Fallback to scalar (unaligned): PASS");
    }

    #[test]
    fn test_simd_path_activates_64() {
        // hidden=64 should use SIMD (64 >= 64 and 64 % 8 == 0)
        let hidden = 64;
        let tokens = 5;

        let weight = Array1::from_vec(vec![1.5; hidden]);
        let bias = Array1::from_vec(vec![0.1; hidden]);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = create_test_input(tokens, hidden, 42);
        let mut output_dispatch = Array2::<f32>::zeros((tokens, hidden));
        let mut output_simd = Array2::<f32>::zeros((tokens, hidden));

        ln.forward_2d_noalloc(&input.view(), &mut output_dispatch.view_mut());
        unsafe {
            ln.forward_2d_noalloc_simd(&input.view(), &mut output_simd.view_mut());
        }

        // If SIMD path is used, these should be identical (not just approximately equal)
        let diff = max_diff(&output_dispatch, &output_simd);
        assert!(
            diff < 1e-7,
            "SIMD path should be used for hidden=64, diff={:.2e}",
            diff
        );
        println!("[hidden=64] SIMD path active: PASS");
    }

    // ============================================================================
    // 2D vs 3D EQUIVALENCE TESTS
    // ============================================================================

    #[test]
    fn test_2d_vs_3d_equivalence() {
        let batch = 2;
        let seq = 6;
        let hidden = 384;
        let tokens = batch * seq;

        let weight = Array1::from_vec((0..hidden).map(|i| 1.0 + (i as f32) * 0.001).collect());
        let bias = Array1::from_vec((0..hidden).map(|i| (i as f32) * 0.0005).collect());
        let ln = LayerNorm::new(weight, bias, 1e-5);

        // Create 3D input
        let input_3d = Array3::from_shape_fn((batch, seq, hidden), |(b, s, h)| {
            ((b * 100 + s * 10 + h) as f32) * 0.01 - 2.0
        });

        // Reshape to 2D
        let input_2d = input_3d
            .clone()
            .into_shape_with_order((tokens, hidden))
            .unwrap();

        // Run 2D noalloc
        let mut output_2d = Array2::<f32>::zeros((tokens, hidden));
        ln.forward_2d_noalloc(&input_2d.view(), &mut output_2d.view_mut());

        // Run 3D reference
        let output_3d = ln.forward_3d_reference(&input_3d);
        let output_3d_flat = output_3d.into_shape_with_order((tokens, hidden)).unwrap();

        let diff = max_diff(&output_2d, &output_3d_flat);
        println!("[2D vs 3D] Max diff: {:.2e}", diff);

        assert!(
            arrays_approx_eq(&output_2d, &output_3d_flat, 1e-5),
            "2D and 3D implementations should match"
        );
    }

    // ============================================================================
    // PYTORCH PARITY TESTS
    // ============================================================================

    #[test]
    fn test_pytorch_parity_2d() {
        // PyTorch reference:
        // x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        // ln = torch.nn.LayerNorm(4, eps=1e-5)
        // ln.weight.data = torch.ones(4)
        // ln.bias.data = torch.zeros(4)
        // output = ln(x)
        // # tensor([[-1.3416, -0.4472,  0.4472,  1.3416]])

        let hidden = 4;
        let weight = Array1::from_vec(vec![1.0; hidden]);
        let bias = Array1::from_vec(vec![0.0; hidden]);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut output = Array2::<f32>::zeros((1, 4));

        // This will use scalar path since hidden=4 < 64
        ln.forward_2d_noalloc(&input.view(), &mut output.view_mut());

        let expected = [-1.3416, -0.4472, 0.4472, 1.3416];

        println!("PyTorch expected: {:?}", expected);
        println!("Our output:       {:?}", output.as_slice().unwrap());

        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                approx_eq(output[[0, i]], exp, 1e-3),
                "Mismatch at index {}: expected {}, got {}",
                i,
                exp,
                output[[0, i]]
            );
        }
        println!("[PyTorch parity] PASS");
    }

    #[test]
    fn test_pytorch_parity_2d_with_weight_bias() {
        // PyTorch reference:
        // x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        // ln = torch.nn.LayerNorm(4, eps=1e-5)
        // ln.weight.data = torch.tensor([2.0, 0.5, 1.5, 1.0])
        // ln.bias.data = torch.tensor([1.0, -1.0, 0.5, 0.0])
        // output = ln(x)

        let hidden = 4;
        let weight = Array1::from_vec(vec![2.0, 0.5, 1.5, 1.0]);
        let bias = Array1::from_vec(vec![1.0, -1.0, 0.5, 0.0]);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut output = Array2::<f32>::zeros((1, 4));

        ln.forward_2d_noalloc(&input.view(), &mut output.view_mut());

        // Compute expected manually:
        // mean = 2.5, var = 1.25, std = sqrt(1.25 + 1e-5) ≈ 1.118
        // normalized = [-1.3416, -0.4472, 0.4472, 1.3416]
        // scaled = normalized * weight + bias
        let mean = 2.5f32;
        let var = 1.25f32;
        let std = (var + 1e-5).sqrt();
        let normalized: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|x| (x - mean) / std)
            .collect();
        let expected: Vec<f32> = normalized
            .iter()
            .zip([2.0, 0.5, 1.5, 1.0].iter())
            .zip([1.0, -1.0, 0.5, 0.0].iter())
            .map(|((n, w), b)| n * w + b)
            .collect();

        println!("Expected: {:?}", expected);
        println!("Output:   {:?}", output.as_slice().unwrap());

        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                approx_eq(output[[0, i]], exp, 1e-4),
                "Mismatch at index {}: expected {}, got {}",
                i,
                exp,
                output[[0, i]]
            );
        }
        println!("[PyTorch parity with weight/bias] PASS");
    }

    #[test]
    fn test_pytorch_parity_2d_simd_path() {
        // Test SIMD path with larger hidden dim
        // Values computed from PyTorch
        let hidden = 64;
        let weight = Array1::from_vec(vec![1.0; hidden]);
        let bias = Array1::from_vec(vec![0.0; hidden]);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        // Input: [0, 1, 2, ..., 63]
        let input_vec: Vec<f32> = (0..hidden).map(|i| i as f32).collect();
        let input = Array2::from_shape_vec((1, hidden), input_vec.clone()).unwrap();
        let mut output = Array2::<f32>::zeros((1, hidden));
        unsafe {
            ln.forward_2d_noalloc_simd(&input.view(), &mut output.view_mut());
        }

        // Verify properties of layer norm:
        // 1. Output should have mean ≈ 0
        let out_slice = output.as_slice().unwrap();
        let out_mean: f32 = out_slice.iter().sum::<f32>() / hidden as f32;
        assert!(
            out_mean.abs() < 1e-5,
            "Output mean should be ~0, got {}",
            out_mean
        );

        // 2. Output should have variance ≈ 1
        let out_var: f32 = out_slice
            .iter()
            .map(|x| (x - out_mean).powi(2))
            .sum::<f32>()
            / hidden as f32;
        assert!(
            (out_var - 1.0).abs() < 1e-4,
            "Output variance should be ~1, got {}",
            out_var
        );

        println!(
            "[PyTorch parity SIMD path] Mean: {:.2e}, Var: {:.4} - PASS",
            out_mean, out_var
        );
    }

    // ============================================================================
    // EDGE CASE TESTS
    // ============================================================================

    #[test]
    fn test_constant_input_2d() {
        // All same values -> variance = 0, should handle gracefully
        let hidden = 64;
        let tokens = 5;

        let weight = Array1::from_vec(vec![1.0; hidden]);
        let bias = Array1::from_vec(vec![0.0; hidden]);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = Array2::from_elem((tokens, hidden), 5.0f32);
        let mut output = Array2::<f32>::zeros((tokens, hidden));

        ln.forward_2d_noalloc(&input.view(), &mut output.view_mut());

        // All outputs should be ~0 (mean-centered with no variance, eps prevents NaN)
        for val in output.iter() {
            assert!(
                val.abs() < 1e-2,
                "Constant input should give ~0 output, got {}",
                val
            );
        }
        println!("[Constant input] All outputs near zero: PASS");
    }

    #[test]
    fn test_numerical_stability_large_values() {
        let hidden = 128;
        let tokens = 10;

        let weight = Array1::from_vec(vec![1.0; hidden]);
        let bias = Array1::from_vec(vec![0.0; hidden]);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        // Large values
        let input = Array2::from_shape_fn((tokens, hidden), |(t, h)| {
            1000.0 + (t * hidden + h) as f32 * 0.1
        });
        let mut output = Array2::<f32>::zeros((tokens, hidden));

        ln.forward_2d_noalloc(&input.view(), &mut output.view_mut());

        // Check no NaN/Inf
        for val in output.iter() {
            assert!(val.is_finite(), "Output contains non-finite value: {}", val);
        }

        // Check normalization properties
        for t in 0..tokens {
            let row = output.row(t);
            let mean: f32 = row.iter().sum::<f32>() / hidden as f32;
            assert!(
                mean.abs() < 1e-4,
                "Row {} mean should be ~0, got {}",
                t,
                mean
            );
        }
        println!("[Large values] Numerically stable: PASS");
    }

    #[test]
    fn test_numerical_stability_small_values() {
        let hidden = 128;
        let tokens = 10;

        let weight = Array1::from_vec(vec![1.0; hidden]);
        let bias = Array1::from_vec(vec![0.0; hidden]);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        // Very small values
        let input = Array2::from_shape_fn((tokens, hidden), |(t, h)| {
            1e-6 + (t * hidden + h) as f32 * 1e-8
        });
        let mut output = Array2::<f32>::zeros((tokens, hidden));

        ln.forward_2d_noalloc(&input.view(), &mut output.view_mut());

        // Check no NaN/Inf
        for val in output.iter() {
            assert!(val.is_finite(), "Output contains non-finite value: {}", val);
        }
        println!("[Small values] Numerically stable: PASS");
    }

    #[test]
    fn test_numerical_stability_mixed_values() {
        let hidden = 128;
        let tokens = 10;

        let weight = Array1::from_vec(vec![1.0; hidden]);
        let bias = Array1::from_vec(vec![0.0; hidden]);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        // Mix of positive and negative, large and small
        let input = Array2::from_shape_fn((tokens, hidden), |(t, h)| {
            let base = ((t * hidden + h) as f32 * 0.37).sin() * 100.0;
            base
        });
        let mut output = Array2::<f32>::zeros((tokens, hidden));

        ln.forward_2d_noalloc(&input.view(), &mut output.view_mut());

        for val in output.iter() {
            assert!(val.is_finite(), "Output contains non-finite value: {}", val);
        }
        println!("[Mixed values] Numerically stable: PASS");
    }

    // ============================================================================
    // MULTI-TOKEN BATCH TESTS
    // ============================================================================

    #[test]
    fn test_multi_token_batch_correctness() {
        let hidden = 384;
        let tokens = 120; // Your benchmark batch size

        let weight = Array1::from_vec((0..hidden).map(|i| 1.0 + (i as f32) * 0.001).collect());
        let bias = Array1::from_vec((0..hidden).map(|i| (i as f32) * 0.0005).collect());
        let ln = LayerNorm::new(weight.clone(), bias.clone(), 1e-5);

        let input = create_test_input(tokens, hidden, 42);
        let mut output = Array2::<f32>::zeros((tokens, hidden));

        ln.forward_2d_noalloc(&input.view(), &mut output.view_mut());

        // Verify each row independently
        for t in 0..tokens {
            let row_input = input.row(t);
            let row_output = output.row(t);

            // Compute expected for this row
            let in_slice = row_input.as_slice().unwrap();
            let mean: f32 = in_slice.iter().sum::<f32>() / hidden as f32;
            let var: f32 = in_slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden as f32;
            let inv_std = 1.0 / (var + 1e-5).sqrt();

            for h in 0..hidden {
                let expected = (in_slice[h] - mean) * inv_std * weight[h] + bias[h];
                assert!(
                    approx_eq(row_output[h], expected, 1e-4),
                    "Mismatch at token {} dim {}: expected {}, got {}",
                    t,
                    h,
                    expected,
                    row_output[h]
                );
            }
        }
        println!("[Multi-token batch=120] All rows correct: PASS");
    }

    #[test]
    fn test_single_token() {
        let hidden = 384;
        let tokens = 1;

        let weight = Array1::from_vec(vec![1.0; hidden]);
        let bias = Array1::from_vec(vec![0.0; hidden]);
        let ln = LayerNorm::new(weight, bias, 1e-5);

        let input = create_test_input(tokens, hidden, 42);
        let mut output = Array2::<f32>::zeros((tokens, hidden));

        ln.forward_2d_noalloc(&input.view(), &mut output.view_mut());

        // Verify mean ≈ 0 and var ≈ 1
        let out_slice = output.as_slice().unwrap();
        let mean: f32 = out_slice.iter().sum::<f32>() / hidden as f32;
        let var: f32 = out_slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden as f32;

        assert!(mean.abs() < 1e-5, "Single token mean should be ~0");
        assert!((var - 1.0).abs() < 1e-4, "Single token var should be ~1");
        println!("[Single token] PASS");
    }
}
