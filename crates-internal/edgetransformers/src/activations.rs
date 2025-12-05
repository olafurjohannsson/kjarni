//! Activation functions for transformers

use libm::{erff, expf, tanhf};
use ndarray::parallel::prelude::*;
use ndarray::{Array3, Array4, Axis};
use ndarray::{ArrayBase, DataMut};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

pub const PARALLEL_THRESHOLD: usize = 16_384; // 16K elements

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
        Activation::GeluNew // Most common in modern models
    }
}

pub fn apply_activation(hidden: &mut Array3<f32>, activation: Activation) {
    let num_elements = hidden.len();
    let use_parallel = num_elements >= PARALLEL_THRESHOLD;

    match (activation, use_parallel) {
        (Activation::Gelu, true) => gelu_parallel(hidden),
        (Activation::Gelu, false) => gelu(hidden),

        (Activation::GeluNew, true) => gelu_new_parallel(hidden),
        (Activation::GeluNew, false) => gelu_new(hidden),

        (Activation::Relu, true) => relu_parallel(hidden),
        (Activation::Relu, false) => relu(hidden),

        (Activation::SilU, true) => silu_parallel_3d(hidden),
        (Activation::SilU, false) => silu_generic(hidden),

        (Activation::Tanh, _) => {
            // Tanh is very fast, rarely worth parallelizing
            hidden.mapv_inplace(|x| x.tanh())
        }
    }
}

/// The standard GELU activation function, using the error function (erf).
/// This is the default implementation in PyTorch and is used by models like BART.
/// GELU (exact) - Uses error function
/// Formula: 0.5 * x * (1 + erf(x / sqrt(2)))
/// This is the mathematically exact GELU used in original Transformer papers
#[inline(always)]
pub fn gelu(x: &mut Array3<f32>) {
    const SQRT_2_INV: f32 = 0.7071067811865475; // 1.0 / sqrt(2.0)

    x.mapv_inplace(|val| 0.5 * val * (1.0 + erff(val * SQRT_2_INV)));
}

/// GELU_NEW (tanh approximation) - Used by BERT, GPT-2
/// Formula: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
/// Faster approximation with <0.1% error vs exact GELU
#[inline(always)]
pub fn gelu_new(x: &mut Array3<f32>) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    const GELU_COEFF: f32 = 0.044715;

    x.mapv_inplace(|val| {
        // Compute x^3 using multiplication (faster than powi)
        let val_cubed = val * val * val;
        let inner = SQRT_2_OVER_PI * (val + GELU_COEFF * val_cubed);
        0.5 * val * (1.0 + tanhf(inner))
    });
}

/// Parallel versions (use when array is large, e.g., >10k elements)
#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
pub fn gelu_parallel(x: &mut Array3<f32>) {
    const SQRT_2_INV: f32 = 0.7071067811865475;

    x.par_mapv_inplace(|val| 0.5 * val * (1.0 + erff(val * SQRT_2_INV)));
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
pub fn gelu_new_parallel(x: &mut Array3<f32>) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    const GELU_COEFF: f32 = 0.044715;

    x.par_mapv_inplace(|val| {
        let val_cubed = val * val * val;
        let inner = SQRT_2_OVER_PI * (val + GELU_COEFF * val_cubed);
        0.5 * val * (1.0 + tanhf(inner))
    });
}

/// Compute softmax over the last dimension of a 4D tensor
#[inline(always)]
pub fn softmax(scores: &Array4<f32>) -> Array4<f32> {
    // Find max for numerical stability
    let max_vals = scores.fold_axis(Axis(3), f32::NEG_INFINITY, |&acc, &x| acc.max(x));
    let max_expanded = max_vals.insert_axis(Axis(3));

    // Compute exp(x - max)
    let mut result = scores - &max_expanded;
    result.mapv_inplace(f32::exp);

    // Normalize
    let sum_exp = result.sum_axis(Axis(3)).insert_axis(Axis(3));
    result /= &sum_exp;

    result
}

/// Generic SiLU that works for any array dimension
#[inline(always)]
pub fn silu_generic<S, D>(x: &mut ArrayBase<S, D>)
where
    S: DataMut<Elem = f32>,
    D: ndarray::Dimension,
{
    x.mapv_inplace(|val| {
        if val <= -20.0 {
            0.0
        } else if val >= 20.0 {
            val
        } else {
            val / (1.0 + expf(-val))
        }
    });
}

/// Generic fast SiLU
#[inline(always)]
pub fn silu_fast_generic<S, D>(x: &mut ArrayBase<S, D>)
where
    S: DataMut<Elem = f32>,
    D: ndarray::Dimension,
{
    x.mapv_inplace(|val| val / (1.0 + expf(-val)));
}

/// Parallel version
#[inline(always)]
pub fn silu_parallel_generic<S, D>(x: &mut ArrayBase<S, D>)
where
    S: DataMut<Elem = f32>,
    D: ndarray::Dimension,
{
    x.par_mapv_inplace(|val| {
        if val <= -20.0 {
            0.0
        } else if val >= 20.0 {
            val
        } else {
            val / (1.0 + expf(-val))
        }
    });
}

/// Fast SiLU without stability checks (for well-normalized inputs)
/// Use this if your inputs are in [-10, 10] range
#[inline(always)]
pub fn silu_fast(x: &mut Array3<f32>) {
    x.mapv_inplace(|val| val / (1.0 + expf(-val)));
}

/// ReLU activation (in-place)
/// Formula: max(0, x)
#[inline(always)]
pub fn relu(x: &mut Array3<f32>) {
    x.mapv_inplace(|val| val.max(0.0));
}

/// Parallel versions (use for arrays >16K elements)
#[inline(always)]
pub fn relu_parallel(x: &mut Array3<f32>) {
    x.par_mapv_inplace(|val| val.max(0.0));
}

#[inline(always)]
pub fn silu_parallel_3d(x: &mut Array3<f32>) {
    x.par_mapv_inplace(|val| {
        if val <= -20.0 {
            0.0
        } else if val >= 20.0 {
            val
        } else {
            val / (1.0 + expf(-val))
        }
    });
}

#[inline(always)]
pub fn silu_parallel(x: &mut ndarray::Array2<f32>) {
    // Works for Array2 and Array3 via generics usually, but here strict typing
    // If you need it for Array3, just duplicate or make generic.
    // Since SwiGLU forward flattens to Array2, this signature is fine.
    
    x.par_mapv_inplace(|val| {
        // Fast approximation or standard sigmoid?
        // Standard SiLU: x / (1 + e^-x)
        let sigmoid = 1.0 / (1.0 + (-val).exp());
        val * sigmoid
    });
}

#[inline(always)]
pub fn silu_fast_parallel(x: &mut Array3<f32>) {
    x.par_mapv_inplace(|val| val / (1.0 + expf(-val)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2, Array3, Array4};

    // ============== Activation Enum ==============

    #[test]
    fn test_activation_from_str() {
        assert_eq!("gelu".parse::<Activation>().unwrap(), Activation::GeluNew);
        assert_eq!("gelu_new".parse::<Activation>().unwrap(), Activation::GeluNew);
        assert_eq!("gelu_fast".parse::<Activation>().unwrap(), Activation::GeluNew);
        assert_eq!("relu".parse::<Activation>().unwrap(), Activation::Relu);
        assert_eq!("silu".parse::<Activation>().unwrap(), Activation::SilU);
        assert_eq!("swish".parse::<Activation>().unwrap(), Activation::SilU);
        assert_eq!("tanh".parse::<Activation>().unwrap(), Activation::Tanh);
    }

    #[test]
    fn test_activation_from_str_case_insensitive() {
        assert_eq!("GELU".parse::<Activation>().unwrap(), Activation::GeluNew);
        assert_eq!("ReLU".parse::<Activation>().unwrap(), Activation::Relu);
        assert_eq!("SiLU".parse::<Activation>().unwrap(), Activation::SilU);
        assert_eq!("TANH".parse::<Activation>().unwrap(), Activation::Tanh);
    }

    #[test]
    fn test_activation_from_str_unknown() {
        let result = "unknown".parse::<Activation>();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown activation"));
    }

    #[test]
    fn test_activation_default() {
        assert_eq!(Activation::default(), Activation::GeluNew);
    }

    #[test]
    fn test_activation_serde() {
        let gelu = Activation::Gelu;
        let json = serde_json::to_string(&gelu).unwrap();
        assert_eq!(json, "\"gelu\"");
        
        let parsed: Activation = serde_json::from_str("\"relu\"").unwrap();
        assert_eq!(parsed, Activation::Relu);
        
        // Test alias
        let silu: Activation = serde_json::from_str("\"swish\"").unwrap();
        assert_eq!(silu, Activation::SilU);
    }

    // ============== GELU ==============

    #[test]
    fn test_gelu_basic() {
        let mut x = Array3::from_shape_vec((1, 1, 5), vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        gelu(&mut x);
        
        // GELU(0) = 0
        assert!((x[[0, 0, 2]] - 0.0).abs() < 1e-6);
        
        // GELU is approximately identity for large positive
        assert!(x[[0, 0, 4]] > 1.9);
        
        // GELU is approximately 0 for large negative
        assert!(x[[0, 0, 0]].abs() < 0.1);
        
        // GELU(-1) ≈ -0.158
        assert!((x[[0, 0, 1]] - (-0.158)).abs() < 0.01);
        
        // GELU(1) ≈ 0.841
        assert!((x[[0, 0, 3]] - 0.841).abs() < 0.01);
    }

    #[test]
    fn test_gelu_vs_gelu_parallel() {
        let data = vec![0.5; 1000];
        let mut x1 = Array3::from_shape_vec((10, 10, 10), data.clone()).unwrap();
        let mut x2 = Array3::from_shape_vec((10, 10, 10), data).unwrap();
        
        gelu(&mut x1);
        gelu_parallel(&mut x2);
        
        for (a, b) in x1.iter().zip(x2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ============== GELU_NEW ==============

    #[test]
    fn test_gelu_new_basic() {
        let mut x = Array3::from_shape_vec((1, 1, 5), vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        gelu_new(&mut x);
        
        // GELU_NEW(0) = 0
        assert!((x[[0, 0, 2]] - 0.0).abs() < 1e-6);
        
        // Should be close to exact GELU
        assert!(x[[0, 0, 4]] > 1.9);
        assert!(x[[0, 0, 0]].abs() < 0.1);
    }

    #[test]
    fn test_gelu_new_vs_gelu_new_parallel() {
        let data = vec![0.5; 1000];
        let mut x1 = Array3::from_shape_vec((10, 10, 10), data.clone()).unwrap();
        let mut x2 = Array3::from_shape_vec((10, 10, 10), data).unwrap();
        
        gelu_new(&mut x1);
        gelu_new_parallel(&mut x2);
        
        for (a, b) in x1.iter().zip(x2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gelu_vs_gelu_new_close() {
        // GELU and GELU_NEW should produce similar results
        let data = vec![-1.0, 0.0, 1.0, 2.0];
        let mut x1 = Array3::from_shape_vec((1, 1, 4), data.clone()).unwrap();
        let mut x2 = Array3::from_shape_vec((1, 1, 4), data).unwrap();
        
        gelu(&mut x1);
        gelu_new(&mut x2);
        
        for (a, b) in x1.iter().zip(x2.iter()) {
            assert!((a - b).abs() < 0.01, "GELU and GELU_NEW differ: {} vs {}", a, b);
        }
    }

    // ============== ReLU ==============

    #[test]
    fn test_relu_basic() {
        let mut x = Array3::from_shape_vec((1, 1, 5), vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        relu(&mut x);
        
        assert_eq!(x[[0, 0, 0]], 0.0);
        assert_eq!(x[[0, 0, 1]], 0.0);
        assert_eq!(x[[0, 0, 2]], 0.0);
        assert_eq!(x[[0, 0, 3]], 1.0);
        assert_eq!(x[[0, 0, 4]], 2.0);
    }

    #[test]
    fn test_relu_all_negative() {
        let mut x = Array3::from_shape_vec((1, 1, 3), vec![-5.0, -1.0, -0.001]).unwrap();
        relu(&mut x);
        
        assert!(x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_relu_all_positive() {
        let mut x = Array3::from_shape_vec((1, 1, 3), vec![0.001, 1.0, 100.0]).unwrap();
        let expected = x.clone();
        relu(&mut x);
        
        assert_eq!(x, expected);
    }

    #[test]
    fn test_relu_vs_relu_parallel() {
        let data: Vec<f32> = (-500..500).map(|x| x as f32 * 0.01).collect();
        let mut x1 = Array3::from_shape_vec((10, 10, 10), data.clone()).unwrap();
        let mut x2 = Array3::from_shape_vec((10, 10, 10), data).unwrap();
        
        relu(&mut x1);
        relu_parallel(&mut x2);
        
        assert_eq!(x1, x2);
    }

    // ============== SiLU ==============

    #[test]
    fn test_silu_basic() {
        let mut x = Array3::from_shape_vec((1, 1, 5), vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        silu_generic(&mut x);
        
        // SiLU(0) = 0
        assert!((x[[0, 0, 2]] - 0.0).abs() < 1e-6);
        
        // SiLU(x) ≈ x for large positive x
        assert!(x[[0, 0, 4]] > 1.7);
        
        // SiLU has a minimum around x ≈ -1.28
        // SiLU(-1) ≈ -0.269
        assert!((x[[0, 0, 1]] - (-0.269)).abs() < 0.01);
        
        // SiLU(1) ≈ 0.731
        assert!((x[[0, 0, 3]] - 0.731).abs() < 0.01);
    }

    #[test]
    fn test_silu_numerical_stability_large_negative() {
        let mut x = Array3::from_shape_vec((1, 1, 3), vec![-100.0, -50.0, -25.0]).unwrap();
        silu_generic(&mut x);
        
        // Should be approximately 0 for large negative values
        assert!(x.iter().all(|&v| v.abs() < 1e-6));
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_silu_numerical_stability_large_positive() {
        let mut x = Array3::from_shape_vec((1, 1, 3), vec![25.0, 50.0, 100.0]).unwrap();
        let expected = x.clone();
        silu_generic(&mut x);
        
        // Should be approximately identity for large positive values
        for (a, e) in x.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 0.01);
        }
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_silu_fast_vs_silu() {
        // For values in normal range, fast and standard should match
        let data = vec![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
        let mut x1 = Array3::from_shape_vec((1, 1, 7), data.clone()).unwrap();
        let mut x2 = Array3::from_shape_vec((1, 1, 7), data).unwrap();
        
        silu_generic(&mut x1);
        silu_fast(&mut x2);
        
        for (a, b) in x1.iter().zip(x2.iter()) {
            assert!((a - b).abs() < 0.01, "silu vs silu_fast: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_silu_parallel_3d_vs_sequential() {
        let data: Vec<f32> = (0..1000).map(|x| (x as f32 - 500.0) * 0.01).collect();
        let mut x1 = Array3::from_shape_vec((10, 10, 10), data.clone()).unwrap();
        let mut x2 = Array3::from_shape_vec((10, 10, 10), data).unwrap();
        
        silu_generic(&mut x1);
        silu_parallel_3d(&mut x2);
        
        for (a, b) in x1.iter().zip(x2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_silu_parallel_2d() {
        let mut x = Array2::from_shape_vec((2, 3), vec![-1.0, 0.0, 1.0, -2.0, 0.5, 2.0]).unwrap();
        silu_parallel(&mut x);
        
        // SiLU(0) = 0
        assert!((x[[0, 1]] - 0.0).abs() < 1e-6);
        // All values should be finite
        assert!(x.iter().all(|v| v.is_finite()));
    }

    // ============== Tanh ==============

    #[test]
    fn test_tanh_basic() {
        let mut x = Array3::from_shape_vec((1, 1, 5), vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        x.mapv_inplace(|v: f32| v.tanh());
        
        // tanh(0) = 0
        assert!((x[[0, 0, 2]] - 0.0).abs() < 1e-6);
        
        // tanh is odd function: tanh(-x) = -tanh(x)
        assert!((x[[0, 0, 0]] + x[[0, 0, 4]]).abs() < 1e-6);
        assert!((x[[0, 0, 1]] + x[[0, 0, 3]]).abs() < 1e-6);
        
        // tanh bounds: -1 < tanh(x) < 1
        assert!(x.iter().all(|&v| v > -1.0 && v < 1.0));
    }

    // ============== Softmax ==============

    #[test]
    fn test_softmax_basic() {
        let x = Array4::from_shape_vec((1, 1, 1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = softmax(&x);
        
        // Sum along last axis should be 1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // All values should be positive
        assert!(result.iter().all(|&v| v > 0.0));
        
        // Higher input = higher probability
        assert!(result[[0, 0, 0, 3]] > result[[0, 0, 0, 2]]);
        assert!(result[[0, 0, 0, 2]] > result[[0, 0, 0, 1]]);
        assert!(result[[0, 0, 0, 1]] > result[[0, 0, 0, 0]]);
    }

    #[test]
    fn test_softmax_uniform() {
        let x = Array4::from_shape_vec((1, 1, 1, 4), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let result = softmax(&x);
        
        // Equal inputs = equal probabilities
        for &v in result.iter() {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that could overflow without max subtraction
        let x = Array4::from_shape_vec((1, 1, 1, 3), vec![1000.0, 1001.0, 1002.0]).unwrap();
        let result = softmax(&x);
        
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_softmax_batched() {
        let x = Array4::from_shape_vec(
            (2, 2, 1, 3),
            vec![
                1.0, 2.0, 3.0,  // batch 0, head 0
                0.0, 0.0, 0.0,  // batch 0, head 1
                -1.0, 0.0, 1.0, // batch 1, head 0
                5.0, 5.0, 5.0,  // batch 1, head 1
            ],
        ).unwrap();
        let result = softmax(&x);
        
        // Each row along last axis should sum to 1
        for b in 0..2 {
            for h in 0..2 {
                let sum: f32 = (0..3).map(|i| result[[b, h, 0, i]]).sum();
                assert!((sum - 1.0).abs() < 1e-6, "batch={}, head={}, sum={}", b, h, sum);
            }
        }
    }

    // ============== apply_activation ==============

    #[test]
    fn test_apply_activation_dispatches_correctly() {
        // Small array - uses non-parallel
        let mut small = Array3::from_shape_vec((1, 1, 3), vec![0.0, 1.0, 2.0]).unwrap();
        apply_activation(&mut small, Activation::Relu);
        assert_eq!(small[[0, 0, 0]], 0.0);
        assert_eq!(small[[0, 0, 1]], 1.0);
        assert_eq!(small[[0, 0, 2]], 2.0);
    }

    #[test]
    fn test_apply_activation_large_array_parallel() {
        // Array larger than PARALLEL_THRESHOLD
        let size = PARALLEL_THRESHOLD + 1000;
        let mut large = Array3::from_shape_vec((1, 1, size), vec![-1.0; size]).unwrap();
        apply_activation(&mut large, Activation::Relu);
        
        // All should be 0 after ReLU on negative values
        assert!(large.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_apply_activation_all_types() {
        let activations = [
            Activation::Gelu,
            Activation::GeluNew,
            Activation::Relu,
            Activation::SilU,
            Activation::Tanh,
        ];
        
        for act in activations {
            let mut x = Array3::from_shape_vec((1, 1, 3), vec![-1.0, 0.0, 1.0]).unwrap();
            apply_activation(&mut x, act);
            
            // All results should be finite
            assert!(x.iter().all(|v| v.is_finite()), "Activation {:?} produced non-finite values", act);
        }
    }

    // ============== Edge Cases ==============

    #[test]
    fn test_activations_with_zeros() {
        let mut gelu_x = Array3::zeros((2, 2, 2));
        let mut relu_x = Array3::zeros((2, 2, 2));
        let mut silu_x = Array3::zeros((2, 2, 2));
        
        gelu(&mut gelu_x);
        relu(&mut relu_x);
        silu_generic(&mut silu_x);
        
        // All activations should map 0 -> 0
        assert!(gelu_x.iter().all(|&v| v.abs() < 1e-10));
        assert!(relu_x.iter().all(|&v| v.abs() < 1e-10));
        assert!(silu_x.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_activations_preserve_shape() {
        let shapes = [(1, 1, 10), (2, 3, 4), (5, 5, 5), (1, 100, 1)];
        
        for shape in shapes {
            let mut x = Array3::<f32>::ones(shape);
            let original_shape = x.shape().to_vec();
            
            apply_activation(&mut x, Activation::Gelu);
            
            assert_eq!(x.shape(), original_shape.as_slice());
        }
    }

    #[test]
    fn test_silu_gradient_at_zero() {
        // SiLU'(0) = 0.5, so small perturbations around 0 should show this
        let eps = 0.001;
        let mut x_pos = Array3::from_shape_vec((1, 1, 1), vec![eps]).unwrap();
        let mut x_neg = Array3::from_shape_vec((1, 1, 1), vec![-eps]).unwrap();
        
        silu_generic(&mut x_pos);
        silu_generic(&mut x_neg);
        
        // Numerical gradient ≈ (f(eps) - f(-eps)) / (2*eps) ≈ 0.5
        let numerical_grad = (x_pos[[0, 0, 0]] - x_neg[[0, 0, 0]]) / (2.0 * eps);
        assert!((numerical_grad - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_gelu_symmetry() {
        // GELU is not symmetric but has a specific relationship
        // GELU(-x) = -x * Φ(-x) where Φ is CDF of normal distribution
        let mut x_pos = Array3::from_shape_vec((1, 1, 1), vec![1.0]).unwrap();
        let mut x_neg = Array3::from_shape_vec((1, 1, 1), vec![-1.0]).unwrap();
        
        gelu(&mut x_pos);
        gelu(&mut x_neg);
        
        // GELU(1) + GELU(-1) should not equal 0 (unlike tanh)
        assert!((x_pos[[0, 0, 0]] + x_neg[[0, 0, 0]]).abs() > 0.1);
    }
}