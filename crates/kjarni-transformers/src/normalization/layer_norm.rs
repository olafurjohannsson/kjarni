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

#[inline]
#[cfg(target_arch = "x86_64")]
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
    #[inline]
    pub fn forward_2d_noalloc_simd(
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
        let hidden = input.shape()[1];
        let tokens = input.shape()[0];

        self.forward_2d_noalloc_simd(input, output);

        // for t in 0..tokens {
        //     let row = input.row(t);
        //     let mut out_row = output.row_mut(t);

        //     // Compute mean
        //     let mean = row.sum() / hidden as f32;

        //     // Compute variance
        //     let var = row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / hidden as f32;

        //     // Normalize and scale
        //     let std_inv = 1.0 / (var + self.eps).sqrt();
        //     for i in 0..hidden {
        //         out_row[i] = (row[i] - mean) * std_inv * self.weight[i] + self.bias[i];
        //     }
        // }
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
mod tests {
    use super::*;
    use ndarray::{Array3, arr1};

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
}
