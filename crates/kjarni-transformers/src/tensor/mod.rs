
//! Defines the core tensor and data type structures for the engine.
//!
//! The central enum is `TypedCpuTensor`, which provides a type-safe container
//! for different numerical formats, including standard floats and quantized blocks.

// Re-export sub-modules for a clean public API.
pub mod dtype;
pub mod raw_tensor;

pub use dtype::DType;
pub use raw_tensor::RawTensor;

use crate::kernels::q_common::{BlockQ4_K, BlockQ8_0};
use anyhow::{anyhow, Result};
use half::{bf16, f16};
use ndarray::{Array1, Array2, ArrayD, Ix1, Ix2};

/// A wrapper for quantized matrix data, pairing the raw blocks with shape information.
///
/// This is crucial because the raw blocks themselves don't store the tensor's dimensions.
/// Storing the shape here allows for correct dimension checking and avoids panics.
#[derive(Debug, Clone)]
pub struct QuantizedMatrix<T> {
    pub blocks: Vec<T>,
    pub shape: [usize; 2], // [out_features, in_features]
}

/// A generic, typed tensor for CPU computation.
/// This enum holds the primary representation of tensor data for the engine.
pub enum TypedCpuTensor {
    F32(ArrayD<f32>),
    BF16(ArrayD<bf16>),
    F16(ArrayD<f16>),
    /// Quantized tensors are stored as a vector of their block structs for maximum performance.
    Q8_0(QuantizedMatrix<BlockQ8_0>),
    Q4_K(QuantizedMatrix<BlockQ4_K>),
}

impl TypedCpuTensor {
    /// Returns the `DType` of the tensor.
    pub fn dtype(&self) -> DType {
        match self {
            TypedCpuTensor::F32(_) => DType::F32,
            TypedCpuTensor::BF16(_) => DType::BF16,
            TypedCpuTensor::F16(_) => DType::F16,
            TypedCpuTensor::Q8_0(_) => DType::Q8_0,
            TypedCpuTensor::Q4_K(_) => DType::Q4_K,
        }
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        match self {
            TypedCpuTensor::F32(arr) => arr.shape(),
            TypedCpuTensor::F16(arr) => arr.shape(),
            TypedCpuTensor::BF16(arr) => arr.shape(),
            TypedCpuTensor::Q8_0(q) => &q.shape,
            TypedCpuTensor::Q4_K(q) => &q.shape,
        }
    }

    /// Converts the tensor to an `Array1<f32>`. Fails for matrix types.
    ///
    /// This is typically used for loading biases or layer normalization weights.
    pub fn to_array1_f32(&self) -> Result<Array1<f32>> {
        match self {
            TypedCpuTensor::F32(arr) => Ok(arr.clone().into_dimensionality::<Ix1>()?),
            TypedCpuTensor::F16(arr) => Ok(arr.mapv(|v| v.to_f32()).into_dimensionality::<Ix1>()?),
            TypedCpuTensor::BF16(arr) => Ok(arr.mapv(|v| v.to_f32()).into_dimensionality::<Ix1>()?),
            _ => Err(anyhow!("Cannot convert a matrix-quantized type to Array1")),
        }
    }

    /// Converts the tensor to an `Array2<f32>`.
    ///
    /// **Warning:** For quantized types, this performs a full, slow dequantization
    /// and should only be used for debugging or testing, not in performance-critical paths.
    pub fn to_array2_f32(&self) -> Result<Array2<f32>> {
        match self {
            TypedCpuTensor::F32(arr) => Ok(arr.clone().into_dimensionality::<Ix2>()?),
            TypedCpuTensor::F16(arr) => Ok(arr.mapv(|v| v.to_f32()).into_dimensionality::<Ix2>()?),
            TypedCpuTensor::BF16(arr) => Ok(arr.mapv(|v| v.to_f32()).into_dimensionality::<Ix2>()?),
            _ => Err(anyhow!(
                "Full dequantization to Array2 is a slow, debug-only operation and is not supported. Use block-wise operations instead."
            )),
        }
    }
}