use libm::{erff, expf, tanhf};
use ndarray::parallel::prelude::*;
use ndarray::{Array2, Array3, Array4, Axis};
use ndarray::{ArrayBase, DataMut};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

pub const PARALLEL_THRESHOLD: usize = 16_384;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[serde(alias = "gelu")]
    Gelu,
    #[serde(alias = "gelu_new")]
    GeluNew,
    #[serde(alias = "relu")]
    Relu,
    #[serde(alias = "silu", alias = "swish")]
    SilU,
    #[serde(alias = "tanh")]
    Tanh,
}

impl FromStr for Activation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "gelu" => Ok(Activation::GeluNew),
            "gelu_new" | "gelu_fast" => Ok(Activation::GeluNew),
            "relu" => Ok(Activation::Relu),
            "silu" | "swish" => Ok(Activation::SilU),
            "tanh" => Ok(Activation::Tanh),
            _ => Err(format!("Unknown activation function: {}", s)),
        }
    }
}

impl Default for Activation {
    fn default() -> Self {
        Activation::GeluNew
    }
}

// ==================== Scalar Functions ====================

const SQRT_2_INV: f32 = 0.7071067811865475;
const SQRT_2_OVER_PI: f32 = 0.7978845608;
const GELU_COEFF: f32 = 0.044715;

#[inline(always)]
pub fn gelu_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + erff(x * SQRT_2_INV))
}

#[inline(always)]
pub fn gelu_new_scalar(x: f32) -> f32 {
    let x_cubed = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    0.5 * x * (1.0 + tanhf(inner))
}

#[inline(always)]
pub fn relu_scalar(x: f32) -> f32 {
    x.max(0.0)
}

#[inline(always)]
pub fn silu_scalar(x: f32) -> f32 {
    if x <= -20.0 {
        0.0
    } else if x >= 20.0 {
        x
    } else {
        x / (1.0 + expf(-x))
    }
}

#[inline(always)]
pub fn silu_fast_scalar(x: f32) -> f32 {
    x / (1.0 + expf(-x))
}

#[inline(always)]
pub fn tanh_scalar(x: f32) -> f32 {
    tanhf(x)
}

// ==================== Unified Apply Functions ====================

fn apply_activation_slice(slice: &mut [f32], activation: Activation, use_parallel: bool) {
    match (activation, use_parallel) {
        (Activation::Gelu, true) => slice.par_iter_mut().for_each(|x| *x = gelu_scalar(*x)),
        (Activation::Gelu, false) => slice.iter_mut().for_each(|x| *x = gelu_scalar(*x)),

        (Activation::GeluNew, true) => slice.par_iter_mut().for_each(|x| *x = gelu_new_scalar(*x)),
        (Activation::GeluNew, false) => slice.iter_mut().for_each(|x| *x = gelu_new_scalar(*x)),

        (Activation::Relu, true) => slice.par_iter_mut().for_each(|x| *x = relu_scalar(*x)),
        (Activation::Relu, false) => slice.iter_mut().for_each(|x| *x = relu_scalar(*x)),

        (Activation::SilU, true) => slice.par_iter_mut().for_each(|x| *x = silu_scalar(*x)),
        (Activation::SilU, false) => slice.iter_mut().for_each(|x| *x = silu_scalar(*x)),

        (Activation::Tanh, true) => slice.par_iter_mut().for_each(|x| *x = tanh_scalar(*x)),
        (Activation::Tanh, false) => slice.iter_mut().for_each(|x| *x = tanh_scalar(*x)),
    }
}

pub fn apply_activation_2d(arr: &mut Array2<f32>, activation: Activation) {
    let use_parallel = arr.len() >= PARALLEL_THRESHOLD;
    if let Some(slice) = arr.as_slice_mut() {
        apply_activation_slice(slice, activation, use_parallel);
    } else {
        // Fallback for non-contiguous arrays
        let use_parallel = arr.len() >= PARALLEL_THRESHOLD;
        match (activation, use_parallel) {
            (Activation::Gelu, true) => arr.par_mapv_inplace(gelu_scalar),
            (Activation::Gelu, false) => arr.mapv_inplace(gelu_scalar),
            (Activation::GeluNew, true) => arr.par_mapv_inplace(gelu_new_scalar),
            (Activation::GeluNew, false) => arr.mapv_inplace(gelu_new_scalar),
            (Activation::Relu, true) => arr.par_mapv_inplace(relu_scalar),
            (Activation::Relu, false) => arr.mapv_inplace(relu_scalar),
            (Activation::SilU, true) => arr.par_mapv_inplace(silu_scalar),
            (Activation::SilU, false) => arr.mapv_inplace(silu_scalar),
            (Activation::Tanh, true) => arr.par_mapv_inplace(tanh_scalar),
            (Activation::Tanh, false) => arr.mapv_inplace(tanh_scalar),
        }
    }
}

pub fn apply_activation(arr: &mut Array3<f32>, activation: Activation) {
    let use_parallel = arr.len() >= PARALLEL_THRESHOLD;
    if let Some(slice) = arr.as_slice_mut() {
        apply_activation_slice(slice, activation, use_parallel);
    } else {
        match (activation, use_parallel) {
            (Activation::Gelu, true) => arr.par_mapv_inplace(gelu_scalar),
            (Activation::Gelu, false) => arr.mapv_inplace(gelu_scalar),
            (Activation::GeluNew, true) => arr.par_mapv_inplace(gelu_new_scalar),
            (Activation::GeluNew, false) => arr.mapv_inplace(gelu_new_scalar),
            (Activation::Relu, true) => arr.par_mapv_inplace(relu_scalar),
            (Activation::Relu, false) => arr.mapv_inplace(relu_scalar),
            (Activation::SilU, true) => arr.par_mapv_inplace(silu_scalar),
            (Activation::SilU, false) => arr.mapv_inplace(silu_scalar),
            (Activation::Tanh, true) => arr.par_mapv_inplace(tanh_scalar),
            (Activation::Tanh, false) => arr.mapv_inplace(tanh_scalar),
        }
    }
}

// ==================== Legacy Array Functions (for backwards compat) ====================

pub fn gelu(x: &mut Array3<f32>) {
    apply_activation(x, Activation::Gelu);
}

pub fn gelu_new(x: &mut Array3<f32>) {
    apply_activation(x, Activation::GeluNew);
}

pub fn gelu_parallel(x: &mut Array3<f32>) {
    x.par_mapv_inplace(gelu_scalar);
}

pub fn gelu_new_parallel(x: &mut Array3<f32>) {
    x.par_mapv_inplace(gelu_new_scalar);
}

pub fn relu(x: &mut Array3<f32>) {
    apply_activation(x, Activation::Relu);
}

pub fn relu_parallel(x: &mut Array3<f32>) {
    x.par_mapv_inplace(relu_scalar);
}

pub fn silu_generic<S, D>(x: &mut ArrayBase<S, D>)
where
    S: DataMut<Elem = f32>,
    D: ndarray::Dimension,
{
    x.mapv_inplace(silu_scalar);
}

pub fn silu_fast<S, D>(x: &mut ArrayBase<S, D>)
where
    S: DataMut<Elem = f32>,
    D: ndarray::Dimension,
{
    x.mapv_inplace(silu_fast_scalar);
}

pub fn silu_parallel_3d(x: &mut Array3<f32>) {
    x.par_mapv_inplace(silu_scalar);
}

pub fn silu_parallel(x: &mut Array2<f32>) {
    x.par_mapv_inplace(silu_scalar);
}

// ==================== Softmax ====================

pub fn softmax(scores: &Array4<f32>) -> Array4<f32> {
    let max_vals = scores.fold_axis(Axis(3), f32::NEG_INFINITY, |&acc, &x| acc.max(x));
    let max_expanded = max_vals.insert_axis(Axis(3));

    let mut result = scores - &max_expanded;
    result.mapv_inplace(f32::exp);

    let sum_exp = result.sum_axis(Axis(3)).insert_axis(Axis(3));
    result /= &sum_exp;

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use ndarray::{Array2, Array3, Array4};

    // ==================== Scalar Tests ====================
    #[test]
    fn test_scalars_correctness() {
        // ReLU
        assert_eq!(relu_scalar(1.0), 1.0);
        assert_eq!(relu_scalar(-1.0), 0.0);

        // SiLU (x * sigmoid(x))
        assert_relative_eq!(silu_scalar(0.0), 0.0);
        assert_relative_eq!(silu_scalar(1.0), 0.7310586, epsilon = 1e-5);
        assert_relative_eq!(silu_scalar(-100.0), 0.0); // Saturation check
        assert_relative_eq!(silu_scalar(100.0), 100.0); // Saturation check

        // SiLU Fast
        assert_relative_eq!(silu_fast_scalar(0.0), 0.0);

        // Tanh
        assert_relative_eq!(tanh_scalar(0.0), 0.0);
        assert_relative_eq!(tanh_scalar(10.0), 1.0, epsilon = 1e-4);

        // GELU
        assert_relative_eq!(gelu_scalar(0.0), 0.0);
        assert_relative_eq!(gelu_scalar(1.0), 0.8413447, epsilon = 1e-5);

        // GELU New
        assert_relative_eq!(gelu_new_scalar(0.0), 0.0);
        assert_relative_eq!(gelu_new_scalar(1.0), 0.841192, epsilon = 1e-5);
    }

    #[test]
    fn test_apply_activation_3d() {
        let variants = vec![
            Activation::Gelu,
            Activation::GeluNew,
            Activation::Relu,
            Activation::SilU,
            Activation::Tanh,
        ];

        for act in variants {
            // Test sequential path (small array)
            let mut arr = Array3::from_elem((1, 1, 5), -2.0f32);
            apply_activation(&mut arr, act);

            // Just verify mutation happened
            if act == Activation::Relu {
                assert_eq!(arr[[0, 0, 0]], 0.0);
            } else {
                assert_ne!(arr[[0, 0, 0]], -2.0);
            }
        }
    }
    #[test]
    fn test_legacy_wrappers_3d() {
        let mut a = Array3::from_elem((1, 1, 1), 1.0f32);

        // Standard wrappers
        gelu(&mut a);
        assert_relative_eq!(a[[0, 0, 0]], 0.8413447, epsilon = 1e-5);

        a.fill(1.0);
        gelu_new(&mut a);
        assert_relative_eq!(a[[0, 0, 0]], 0.841192, epsilon = 1e-5);

        a.fill(-1.0);
        relu(&mut a);
        assert_eq!(a[[0, 0, 0]], 0.0);
    }

    #[test]
    fn test_legacy_parallel_wrappers() {
        let mut a = Array3::from_elem((1, 1, 1), 1.0f32);

        gelu_parallel(&mut a);
        assert_relative_eq!(a[[0, 0, 0]], 0.8413447, epsilon = 1e-5);

        a.fill(1.0);
        gelu_new_parallel(&mut a);
        assert_relative_eq!(a[[0, 0, 0]], 0.841192, epsilon = 1e-5);

        a.fill(-1.0);
        relu_parallel(&mut a);
        assert_eq!(a[[0, 0, 0]], 0.0);

        a.fill(1.0);
        silu_parallel_3d(&mut a);
        assert_relative_eq!(a[[0, 0, 0]], 0.7310586, epsilon = 1e-5);
    }

    #[test]
    fn test_legacy_silu_variants() {
        let mut a = Array3::from_elem((1, 1, 1), 1.0f32);

        // Generic
        silu_generic(&mut a);
        assert_relative_eq!(a[[0, 0, 0]], 0.7310586, epsilon = 1e-5);

        // Fast
        a.fill(1.0);
        silu_fast(&mut a);
        assert_relative_eq!(a[[0, 0, 0]], 0.7310586, epsilon = 1e-5);

        // 2D Parallel
        let mut b = Array2::from_elem((1, 1), 1.0f32);
        silu_parallel(&mut b);
        assert_relative_eq!(b[[0, 0]], 0.7310586, epsilon = 1e-5);
    }

    #[test]
    fn test_activation_from_str() {
        assert_eq!(Activation::from_str("gelu").unwrap(), Activation::GeluNew);
        assert_eq!(
            Activation::from_str("gelu_new").unwrap(),
            Activation::GeluNew
        );
        assert_eq!(Activation::from_str("relu").unwrap(), Activation::Relu);
        assert_eq!(Activation::from_str("silu").unwrap(), Activation::SilU);
        assert_eq!(Activation::from_str("swish").unwrap(), Activation::SilU);
        assert_eq!(Activation::from_str("tanh").unwrap(), Activation::Tanh);

        assert!(Activation::from_str("invalid_name").is_err());
    }

    #[test]
    fn test_default() {
        assert_eq!(Activation::default(), Activation::GeluNew);
    }
    #[test]
    fn test_softmax() {
        let input = Array4::from_shape_vec((1, 1, 1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let output = softmax(&input);

        // Check sum to 1
        assert_relative_eq!(output.sum(), 1.0, epsilon = 1e-6);

        // Check stability with large numbers
        let large = Array4::from_shape_vec((1, 1, 1, 3), vec![1000.0, 1001.0, 1002.0]).unwrap();
        let output_large = softmax(&large);
        assert_relative_eq!(output_large.sum(), 1.0, epsilon = 1e-6);
        assert!(!output_large.iter().any(|x| x.is_nan()));
    }
    // #[test]
    // fn test_apply_activation_2d() {
    //     let mut arr = Array2::from_elem((2, 2), -1.0f32);
    //     apply_activation_2d(&mut arr, Activation::Relu);

    //     // Use approx macro to avoid PartialEq conflict with serde_json
    //     assert_abs_diff_eq!(arr, Array2::zeros((2, 2)), epsilon = 1e-6);
    // }
    #[test]
    fn test_parallel_execution_paths() {
        // Create array larger than PARALLEL_THRESHOLD (16_384) to trigger Rayon
        let size = PARALLEL_THRESHOLD + 100;
        let mut arr = Array3::from_elem((1, 1, size), 1.0f32);

        // Test 3D Parallel
        apply_activation(&mut arr, Activation::Relu);
        assert_eq!(arr[[0, 0, 0]], 1.0);

        // Test 2D Parallel
        let mut arr2 = Array2::from_elem((1, size), 1.0f32);
        apply_activation_2d(&mut arr2, Activation::Relu);
        assert_eq!(arr2[[0, 0]], 1.0);
    }

    #[test]
    fn test_relu_scalar() {
        assert_relative_eq!(relu_scalar(1.0), 1.0);
        assert_relative_eq!(relu_scalar(-1.0), 0.0);
        assert_relative_eq!(relu_scalar(0.0), 0.0);
    }

    #[test]
    fn test_silu_scalar() {
        // x * sigmoid(x)
        assert_relative_eq!(silu_scalar(0.0), 0.0);
        // sigmoid(1.0) approx 0.731058 -> 1.0 * 0.731058
        assert_relative_eq!(silu_scalar(1.0), 0.7310586, epsilon = 1e-6);
        // large negative -> 0
        assert_relative_eq!(silu_scalar(-25.0), 0.0);
        // large positive -> x
        assert_relative_eq!(silu_scalar(25.0), 25.0);
    }

    #[test]
    fn test_tanh_scalar() {
        assert_relative_eq!(tanh_scalar(0.0), 0.0);
        assert_relative_eq!(tanh_scalar(100.0), 1.0);
        assert_relative_eq!(tanh_scalar(-100.0), -1.0);
    }

    #[test]
    fn test_gelu_scalar() {
        // GELU(0) = 0
        assert_relative_eq!(gelu_scalar(0.0), 0.0);
        // GELU(1) approx 0.8413
        assert_relative_eq!(gelu_scalar(1.0), 0.8413447, epsilon = 1e-5);
    }

    #[test]
    fn test_gelu_new_scalar() {
        // Tanh approx is slightly different but close
        assert_relative_eq!(gelu_new_scalar(0.0), 0.0);
        assert_relative_eq!(gelu_new_scalar(1.0), 0.841192, epsilon = 1e-5);
    }

    // ==================== Array Application Tests ====================

    #[test]
    fn test_apply_activation_all_variants() {
        let input_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let shape = (1, 1, 5);

        let variants = vec![
            Activation::Gelu,
            Activation::GeluNew,
            Activation::Relu,
            Activation::SilU,
            Activation::Tanh,
        ];

        for act in variants {
            let mut arr = Array3::from_shape_vec(shape, input_data.clone()).unwrap();
            apply_activation(&mut arr, act);

            // Basic sanity check: output should not be input (unless ReLU positive side)
            // We rely on scalar tests for exact math correctness.
            // Here we check that the array was actually mutated.
            if act != Activation::Relu {
                // ReLU is identity for pos numbers, check 0 index (-2.0)
                assert_ne!(
                    arr[[0, 0, 0]],
                    -2.0,
                    "Activation {:?} failed to mutate array",
                    act
                );
            } else {
                assert_eq!(arr[[0, 0, 0]], 0.0, "ReLU failed to zero negative input");
            }
        }
    }

    #[test]
    fn test_apply_activation_2d_manual() {
        let mut arr = Array2::from_elem((2, 2), -1.0f32);
        apply_activation_2d(&mut arr, Activation::Relu);

        // Manual check that all elements are 0.0
        assert!(arr.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_parallel_execution() {
        // Force parallel path by creating a large array
        let size = PARALLEL_THRESHOLD + 100;
        let mut arr = Array3::from_elem((1, 1, size), 1.0f32);

        // Apply ReLU (1.0 -> 1.0)
        apply_activation(&mut arr, Activation::Relu);
        assert_eq!(arr[[0, 0, 0]], 1.0);

        // Apply with negative ( -1.0 -> 0.0)
        arr.fill(-1.0);
        apply_activation(&mut arr, Activation::Relu);
        assert_eq!(arr[[0, 0, 0]], 0.0);
    }

    // ==================== Softmax Tests ====================

    #[test]
    fn test_softmax_sum_to_one() {
        let input = Array4::from_shape_vec((1, 1, 1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let output = softmax(&input);

        let sum = output.sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // Check relative order is preserved (monotonicity)
        assert!(output[[0, 0, 0, 2]] > output[[0, 0, 0, 1]]);
        assert!(output[[0, 0, 0, 1]] > output[[0, 0, 0, 0]]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values should not produce NaN due to subtraction of max
        let input = Array4::from_shape_vec((1, 1, 1, 3), vec![1000.0, 1001.0, 1002.0]).unwrap();
        let output = softmax(&input);

        assert_relative_eq!(output.sum(), 1.0, epsilon = 1e-6);
        assert!(!output.iter().any(|x| x.is_nan()));
    }

    // ==================== Parsing Tests ====================

    #[test]
    fn test_activation_parsing() {
        assert_eq!(Activation::from_str("gelu").unwrap(), Activation::GeluNew);
        assert_eq!(Activation::from_str("relu").unwrap(), Activation::Relu);
        assert_eq!(Activation::from_str("swish").unwrap(), Activation::SilU);
        assert!(Activation::from_str("invalid").is_err());
    }

    // ==================== Legacy Wrapper Tests ====================
    // Ensure the old functions call the new logic correctly

    #[test]
    fn test_legacy_wrappers() {
        let mut a = Array3::from_elem((1, 1, 1), -5.0);
        relu(&mut a);
        assert_eq!(a[[0, 0, 0]], 0.0);

        let mut b = Array3::from_elem((1, 1, 1), 0.0);
        gelu(&mut b);
        assert_eq!(b[[0, 0, 0]], 0.0);
    }
}
