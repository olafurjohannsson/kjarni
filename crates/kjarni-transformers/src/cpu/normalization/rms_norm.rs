//! Root Mean Square Layer Normalization

#[cfg(target_arch = "x86_64")]
use crate::cpu::kernels::x86::rms_norm::rms_norm_avx2;
use ndarray::{Array1, Array2, Array3, Axis};

/// Root Mean Square Layer Normalization
#[derive(Clone)]
pub struct RMSNorm {
    pub weight: Array1<f32>,
    pub eps: f32,
}

impl RMSNorm {
    /// Create a new RMSNorm layer
    pub fn new(weight: Array1<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Apply RMS normalization to a 3D tensor
    pub fn forward_3d(&self, hidden: &Array3<f32>) -> Array3<f32> {
        let squared = hidden.mapv(|x| x * x);
        let mean_squared = squared.mean_axis(Axis(2)).unwrap();
        let mean_squared_expanded = mean_squared.insert_axis(Axis(2));
        let rms = (&mean_squared_expanded + self.eps).mapv(|x| x.sqrt());
        let normalized = hidden / &rms;
        normalized * &self.weight
    }

    /// Apply RMS normalization to a 2D tensor
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

        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                unsafe {
                    rms_norm_avx2(row, w, self.eps);
                }
                return;
            }
        }

        // Scalar fallback
        let sum_sq: f32 = row.iter().map(|v| v * v).sum();
        let mean = sum_sq / row.len() as f32;
        let scale = 1.0 / (mean + self.eps).sqrt();
        for (x, w) in row.iter_mut().zip(w.iter()) {
            *x = *x * scale * *w;
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
    use ndarray::Array3;

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
        let eps = 1e-5;

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

        let gamma_vec = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let gamma_cpu = Array1::from_vec(gamma_vec);

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

        let rms_norm = RMSNorm::new(gamma_cpu, eps);
        let actual_output_cpu = rms_norm.forward_3d(&input_cpu);

        let tolerance = 1e-6;
        assert_tensors_approx_equal(&expected_output_cpu, &actual_output_cpu, tolerance);

        println!("✓ CPU RMSNorm implementation passed PyTorch parity test.");
    }

    #[test]
    fn test_rms_norm_near_zero() {
        let weight = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let eps = 1e-6;
        let rms_norm = RMSNorm::new(weight, eps);

        let hidden = Array3::from_shape_vec((1, 1, 3), vec![1e-8, 2e-8, 1e-8]).unwrap();
        let output = rms_norm.forward_3d(&hidden);

        assert!(output[[0, 0, 0]].is_finite());
        assert!(output[[0, 0, 1]].is_finite());
        assert!(output[[0, 0, 2]].is_finite());
        println!("✓ RMSNorm near-zero stability test passed.");
    }
    #[test]
    fn test_rms_norm_basic() {
        let weight = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let eps = 1e-6;
        let rms_norm = RMSNorm::new(weight, eps);

        let hidden = Array3::from_shape_vec((1, 1, 3), vec![3.0, 4.0, 0.0]).unwrap();

        let output = rms_norm.forward_3d(&hidden);

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
        let weight = Array1::from_vec(vec![1.0, 1.0]);
        let eps = 1e-6;
        let rms_norm = RMSNorm::new(weight, eps);

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
        let rms_00 = ((1.0_f32.powi(2) + 2.0_f32.powi(2)) / 2.0).sqrt();
        assert!((output[[0, 0, 0]] - (1.0 / rms_00)).abs() < 1e-4);
        assert!((output[[0, 0, 1]] - (2.0 / rms_00)).abs() < 1e-4);
        let rms_11 = ((7.0_f32.powi(2) + 8.0_f32.powi(2)) / 2.0).sqrt();
        assert!((output[[1, 1, 0]] - (7.0 / rms_11)).abs() < 1e-4);
        assert!((output[[1, 1, 1]] - (8.0 / rms_11)).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm_near_zero_2() {
        let weight = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let eps = 1e-6;
        let rms_norm = RMSNorm::new(weight, eps);

        let hidden = Array3::from_shape_vec((1, 1, 3), vec![1e-8, 2e-8, 1e-8]).unwrap();
        let output = rms_norm.forward_3d(&hidden);

        assert!(output[[0, 0, 0]].is_finite());
        assert!(output[[0, 0, 1]].is_finite());
        assert!(output[[0, 0, 2]].is_finite());
    }
    #[test]
    fn test_rms_norm_pytorch_parity() {
        let weight = Array1::from_vec(vec![0.5, 1.0, 1.5]);
        let eps = 1e-6;
        let rms_norm = RMSNorm::new(weight, eps);

        let hidden = Array3::from_shape_vec((1, 1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let output = rms_norm.forward_3d(&hidden);

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
