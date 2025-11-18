//! Activation functions for transformers

use libm::{erff, expf, tanhf};
use ndarray::parallel::prelude::*;
use ndarray::{Array2, Array3, Array4, Axis};
use ndarray::{ArrayBase, Data, DataMut};
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

        (Activation::SilU, true) => silu_parallel(hidden),
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

// Now you can call:
// silu_generic(&mut array2);
// silu_generic(&mut array3);
// silu_generic(&mut array1); // etc.

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
pub fn silu_parallel(x: &mut Array3<f32>) {
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
pub fn silu_fast_parallel(x: &mut Array3<f32>) {
    x.par_mapv_inplace(|val| val / (1.0 + expf(-val)));
}
