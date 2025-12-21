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
    S: DataMut<Elem=f32>,
    D: ndarray::Dimension,
{
    x.mapv_inplace(silu_scalar);
}

pub fn silu_fast<S, D>(x: &mut ArrayBase<S, D>)
where
    S: DataMut<Elem=f32>,
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
