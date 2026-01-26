//! Root Mean Square Layer Normalization (RMSNorm)
//!
//! RMSNorm is a simpler and more efficient alternative to LayerNorm,
//! used in LLaMA and other modern architectures.
//!
//! Unlike LayerNorm, RMSNorm:
//! - Does not subtract the mean (no centering)
//! - Does not have a bias term
//! - Only normalizes by the RMS (root mean square)

use ndarray::{Array1, Array2, Array3, Axis};

use crate::cpu::kernels::x86::rms_norm::rms_norm_avx2;

/// Root Mean Square Layer Normalization
///
/// Formula: y = (x / RMS(x)) * weight
/// where RMS(x) = sqrt(mean(x^2) + eps)
#[derive(Clone)]
pub struct RMSNorm {
    pub weight: Array1<f32>,
    pub eps: f32,
}

impl RMSNorm {
    /// Create a new RMSNorm layer
    ///
    /// # Arguments
    /// * `weight` - Learnable scale parameter (shape: [hidden_size])
    /// * `eps` - Small constant for numerical stability (typically 1e-6)
    pub fn new(weight: Array1<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Apply RMS normalization to a 3D tensor
    ///
    /// # Arguments
    /// * `hidden` - Input tensor of shape [batch, seq_len, hidden_size]
    ///
    /// # Returns
    /// Normalized tensor of the same shape
    pub fn forward_3d(&self, hidden: &Array3<f32>) -> Array3<f32> {
        // 1. Compute the mean of squared values along the last axis (features)
        //    mean(x^2) for each position
        let squared = hidden.mapv(|x| x * x);
        let mean_squared = squared.mean_axis(Axis(2)).unwrap();

        // 2. Expand dimensions for broadcasting: [batch, seq] -> [batch, seq, 1]
        let mean_squared_expanded = mean_squared.insert_axis(Axis(2));

        // 3. Compute RMS: sqrt(mean(x^2) + eps)
        let rms = (&mean_squared_expanded + self.eps).mapv(|x| x.sqrt());

        // 4. Normalize: x / RMS(x)
        let normalized = hidden / &rms;

        // 5. Scale by learnable weight parameter
        normalized * &self.weight
    }

    /// Apply RMS normalization to a 2D tensor
    ///
    /// # Arguments
    /// * `hidden` - Input tensor of shape [seq_len, hidden_size] or [batch, hidden_size]
    ///
    /// # Returns
    /// Normalized tensor of the same shape
    pub fn forward_2d(&self, hidden: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        let squared = hidden.mapv(|x| x * x);
        let mean_squared = squared.mean_axis(Axis(1)).unwrap();
        let mean_squared_expanded = mean_squared.insert_axis(Axis(1));
        let rms = (&mean_squared_expanded + self.eps).mapv(|x| x.sqrt());
        let normalized = hidden / &rms;
        normalized * &self.weight
    }
}

#[derive(Clone)]
pub struct RMSNormSIMD {
    pub weight: Array1<f32>,
    pub eps: f32,
}

impl RMSNormSIMD {
    pub fn new(weight: Array1<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    #[inline]
    fn apply_row(&self, row: &mut [f32]) {
        let w = self.weight.as_slice().expect("weight must be contiguous");

        debug_assert_eq!(row.len(), w.len());

        if cfg!(target_arch = "x86_64")
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
        {
            unsafe {
                rms_norm_avx2(row, w, self.eps);
            }
        } else {
            // Scalar fallback
            let sum_sq: f32 = row.iter().map(|v| v * v).sum();
            let mean = sum_sq / row.len() as f32;
            let scale = 1.0 / (mean + self.eps).sqrt();

            for (x, w) in row.iter_mut().zip(w.iter()) {
                *x = *x * scale * *w;
            }
        }
    }

    pub fn forward_2d(&self, hidden: &Array2<f32>) -> Array2<f32> {
        let mut out = hidden.to_owned();

        for mut row in out.outer_iter_mut() {
            let row_slice = row.as_slice_mut().expect("row must be contiguous");
            self.apply_row(row_slice);
        }

        out
    }

    pub fn forward_3d(&self, hidden: &Array3<f32>) -> Array3<f32> {
        let mut out = hidden.to_owned();

        for mut batch in out.outer_iter_mut() {
            for mut row in batch.outer_iter_mut() {
                let row_slice = row.as_slice_mut().expect("row must be contiguous");
                self.apply_row(row_slice);
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array3};

    /// Helper function to compare two 3D tensors for approximate equality.
    fn assert_tensors_approx_equal(a: &Array3<f32>, b: &Array3<f32>, tolerance: f32) {
        assert_eq!(a.shape(), b.shape(), "Tensor shapes do not match");
        for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (val_a - val_b).abs() < tolerance,
                "Tensor values differ at index {}: a={}, b={}. Difference: {}",
                i,
                val_a,
                val_b,
                (val_a - val_b).abs()
            );
        }
        println!("✓ Tensors are approximately equal.");
    }

    fn make_test_weight(hidden: usize) -> Array1<f32> {
        Array1::from_shape_fn(hidden, |i| 1.0 + (i as f32) * 0.001)
    }

    fn make_test_input_2d(rows: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, hidden), |(i, j)| {
            ((i * hidden + j) as f32) * 0.01 - 1.5
        })
    }

    fn make_test_input_3d(batch: usize, seq: usize, hidden: usize) -> Array3<f32> {
        Array3::from_shape_fn((batch, seq, hidden), |(b, s, h)| {
            ((b * 1000 + s * hidden + h) as f32) * 0.005 - 2.0
        })
    }

    #[test]
    fn rms_norm_2d_simd_matches_scalar() {
        let hidden = 513; // deliberately not a multiple of 8
        let rows = 7;
        let eps = 1e-6;

        let weight = make_test_weight(hidden);
        let input = make_test_input_2d(rows, hidden);

        let scalar = RMSNorm::new(weight.clone(), eps);
        let simd = RMSNormSIMD::new(weight, eps);

        let y_scalar = scalar.forward_2d(&input);
        let y_simd = simd.forward_2d(&input);

        assert_eq!(y_scalar.shape(), y_simd.shape());

        for ((i, j), &v_ref) in y_scalar.indexed_iter() {
            let v_simd = y_simd[(i, j)];
            assert_abs_diff_eq!(v_ref, v_simd, epsilon = 1e-5);
        }
    }

    #[test]
    fn rms_norm_3d_simd_matches_scalar() {
        let batch = 3;
        let seq = 5;
        let hidden = 257; // remainder-heavy
        let eps = 1e-6;

        let weight = make_test_weight(hidden);
        let input = make_test_input_3d(batch, seq, hidden);

        let scalar = RMSNorm::new(weight.clone(), eps);
        let simd = RMSNormSIMD::new(weight, eps);

        let y_scalar = scalar.forward_3d(&input);
        let y_simd = simd.forward_3d(&input);

        assert_eq!(y_scalar.shape(), y_simd.shape());

        for ((b, s, h), &v_ref) in y_scalar.indexed_iter() {
            let v_simd = y_simd[(b, s, h)];
            assert_abs_diff_eq!(v_ref, v_simd, epsilon = 1e-5);
        }
    }

    #[test]
    fn rms_norm_preserves_zero_input() {
        let hidden = 128;
        let rows = 4;
        let eps = 1e-6;

        let weight = Array1::ones(hidden);
        let input = Array2::<f32>::zeros((rows, hidden));

        let scalar = RMSNorm::new(weight.clone(), eps);
        let simd = RMSNormSIMD::new(weight, eps);

        let y_scalar = scalar.forward_2d(&input);
        let y_simd = simd.forward_2d(&input);

        for ((i, j), &v_ref) in y_scalar.indexed_iter() {
            let v_simd = y_simd[(i, j)];
            assert_abs_diff_eq!(v_ref, v_simd, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_rmsnorm_pytorch_parity() {
        // --- 1. Arrange: Use the golden values from the Python script ---
        let eps = 1e-5;

        // --- RMSNorm Input ---
        // Shape: [1, 1, 8]
        let input_vec = vec![
            0.33669036626815796,
            0.12880940735340118,
            0.23446236550807953,
            0.23033303022384644,
            -1.1228563785552979,
            -0.18632829189300537,
            2.2082014083862305,
            -0.637997031211853,
        ];
        let input_cpu = Array3::from_shape_vec((1, 1, 8), input_vec).unwrap();

        // --- RMSNorm Gamma (Weight) ---
        // Shape: [8]
        let gamma_vec = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let gamma_cpu = Array1::from_vec(gamma_vec);

        // --- RMSNorm Golden Output ---
        // Shape: [1, 1, 8]
        let expected_output_vec = vec![
            0.18237116932868958,
            0.1395414024591446,
            0.3809955418109894,
            0.49904727935791016,
            -3.041023015975952,
            -0.6055577397346497,
            8.372635841369629,
            -2.7646117210388184,
        ];
        let expected_output_cpu = Array3::from_shape_vec((1, 1, 8), expected_output_vec).unwrap();

        // --- 2. Act: Run our Rust RMSNorm implementation ---
        let rms_norm = RMSNorm::new(gamma_cpu, eps);
        // Directly call the `forward_3d` method which matches the tensor shape.
        let actual_output_cpu = rms_norm.forward_3d(&input_cpu);

        // --- 3. Assert: Compare the results ---
        // A slightly higher tolerance might be needed if libm implementations differ,
        // but 1e-6 is a good starting point for CPU-to-CPU.
        let tolerance = 1e-6;
        assert_tensors_approx_equal(&expected_output_cpu, &actual_output_cpu, tolerance);

        println!("✓ CPU RMSNorm implementation passed PyTorch parity test.");
    }

    #[test]
    fn test_rms_norm_near_zero() {
        // Test stability with near-zero values
        let weight = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let eps = 1e-6;
        let rms_norm = RMSNorm::new(weight, eps);

        let hidden = Array3::from_shape_vec((1, 1, 3), vec![1e-8, 2e-8, 1e-8]).unwrap();
        // Call the method being tested
        let output = rms_norm.forward_3d(&hidden);

        // Should not panic or produce NaN/Inf
        assert!(output[[0, 0, 0]].is_finite());
        assert!(output[[0, 0, 1]].is_finite());
        assert!(output[[0, 0, 2]].is_finite());
        println!("✓ RMSNorm near-zero stability test passed.");
    }
    #[test]
    fn test_rms_norm_basic() {
        // Simple test with known values
        let weight = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let eps = 1e-6;
        let rms_norm = RMSNorm::new(weight, eps);

        // Input: [1, 1, 3] tensor with values [3.0, 4.0, 0.0]
        // Expected RMS = sqrt((9 + 16 + 0) / 3) = sqrt(25/3) = sqrt(8.333...) ≈ 2.8868
        let hidden = Array3::from_shape_vec((1, 1, 3), vec![3.0, 4.0, 0.0]).unwrap();

        let output = rms_norm.forward_3d(&hidden);

        // Expected output ≈ [1.0392, 1.3856, 0.0]
        let rms = ((3.0_f32.powi(2) + 4.0_f32.powi(2) + 0.0_f32.powi(2)) / 3.0).sqrt();
        let expected_0 = 3.0 / rms;
        let expected_1 = 4.0 / rms;
        let expected_2 = 0.0 / rms;

        assert!((output[[0, 0, 0]] - expected_0).abs() < 1e-4);
        assert!((output[[0, 0, 1]] - expected_1).abs() < 1e-4);
        assert!((output[[0, 0, 2]] - expected_2).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm_with_scale() {
        // Test with non-unit weight
        let weight = Array1::from_vec(vec![2.0, 0.5, 1.5]);
        let eps = 1e-6;
        let rms_norm = RMSNorm::new(weight, eps);

        let hidden = Array3::from_shape_vec((1, 1, 3), vec![3.0, 4.0, 0.0]).unwrap();
        let output = rms_norm.forward_3d(&hidden);

        let rms = ((3.0_f32.powi(2) + 4.0_f32.powi(2) + 0.0_f32.powi(2)) / 3.0).sqrt();
        let expected_0 = (3.0 / rms) * 2.0;
        let expected_1 = (4.0 / rms) * 0.5;
        let expected_2 = (0.0 / rms) * 1.5;

        assert!((output[[0, 0, 0]] - expected_0).abs() < 1e-4);
        assert!((output[[0, 0, 1]] - expected_1).abs() < 1e-4);
        assert!((output[[0, 0, 2]] - expected_2).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm_batch() {
        // Test with batch size > 1
        let weight = Array1::from_vec(vec![1.0, 1.0]);
        let eps = 1e-6;
        let rms_norm = RMSNorm::new(weight, eps);

        // [2, 2, 2] - batch=2, seq=2, hidden=2
        let hidden = Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                1.0, 2.0, // batch 0, pos 0
                3.0, 4.0, // batch 0, pos 1
                5.0, 6.0, // batch 1, pos 0
                7.0, 8.0, // batch 1, pos 1
            ],
        )
            .unwrap();

        let output = rms_norm.forward_3d(&hidden);

        // Verify each position independently
        // Position [0, 0]: [1.0, 2.0] -> RMS = sqrt((1+4)/2) = sqrt(2.5)
        let rms_00 = ((1.0_f32.powi(2) + 2.0_f32.powi(2)) / 2.0).sqrt();
        assert!((output[[0, 0, 0]] - (1.0 / rms_00)).abs() < 1e-4);
        assert!((output[[0, 0, 1]] - (2.0 / rms_00)).abs() < 1e-4);

        // Position [1, 1]: [7.0, 8.0] -> RMS = sqrt((49+64)/2) = sqrt(56.5)
        let rms_11 = ((7.0_f32.powi(2) + 8.0_f32.powi(2)) / 2.0).sqrt();
        assert!((output[[1, 1, 0]] - (7.0 / rms_11)).abs() < 1e-4);
        assert!((output[[1, 1, 1]] - (8.0 / rms_11)).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm_near_zero_2() {
        // Test stability with near-zero values
        let weight = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let eps = 1e-6;
        let rms_norm = RMSNorm::new(weight, eps);

        let hidden = Array3::from_shape_vec((1, 1, 3), vec![1e-8, 2e-8, 1e-8]).unwrap();
        let output = rms_norm.forward_3d(&hidden);

        // Should not panic or produce NaN/Inf
        assert!(output[[0, 0, 0]].is_finite());
        assert!(output[[0, 0, 1]].is_finite());
        assert!(output[[0, 0, 2]].is_finite());
    }
    #[test]
    fn test_rms_norm_pytorch_parity() {
        // Test against known PyTorch output
        // PyTorch code:
        // ```python
        // import torch
        // x = torch.tensor([[[1.0, 2.0, 3.0]]])
        // weight = torch.tensor([0.5, 1.0, 1.5])
        // eps = 1e-6
        //
        // # RMSNorm
        // rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        // normalized = x / rms
        // output = normalized * weight
        // print(output)
        // ```
        //
        // Manual calculation:
        // mean(x^2) = (1^2 + 2^2 + 3^2) / 3 = 14/3 = 4.6667
        // RMS = sqrt(4.6667 + 1e-6) ≈ 2.1602
        // normalized = [1.0/2.1602, 2.0/2.1602, 3.0/2.1602] = [0.4629, 0.9258, 1.3887]
        // output = [0.4629*0.5, 0.9258*1.0, 1.3887*1.5] = [0.2315, 0.9258, 2.0831]
        //
        // Output: tensor([[[0.2315, 0.9258, 2.0831]]])

        let weight = Array1::from_vec(vec![0.5, 1.0, 1.5]);
        let eps = 1e-6;
        let rms_norm = RMSNorm::new(weight, eps);

        let hidden = Array3::from_shape_vec((1, 1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let output = rms_norm.forward_3d(&hidden);

        // Corrected expected values
        assert!(
            (output[[0, 0, 0]] - 0.2315).abs() < 1e-3,
            "Expected ~0.2315, got {}",
            output[[0, 0, 0]]
        );
        assert!(
            (output[[0, 0, 1]] - 0.9258).abs() < 1e-3,
            "Expected ~0.9258, got {}",
            output[[0, 0, 1]]
        );
        assert!(
            (output[[0, 0, 2]] - 2.0831).abs() < 1e-3,
            "Expected ~2.0831, got {}",
            output[[0, 0, 2]]
        );
    }
}
