//! Defines the core tensor and data type structures for the engine.
//!
//! The central enum is `CpuTensor`, which provides a type-safe container
//! for different numerical formats, including standard floats and quantized blocks.

// Re-export sub-modules for a clean public API.
pub mod dtype;
pub mod raw_tensor;

pub use dtype::DType;
pub use raw_tensor::TensorView;

use crate::kernels::{
    q_common::{BlockQ4_K, BlockQ6_K, BlockQ8_0},
    scalar::{dequantize_q4_k_block, dequantize_q6_k_block},
};
use anyhow::{Result, anyhow};
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
pub enum CpuTensor {
    F32(ArrayD<f32>),
    BF16(ArrayD<bf16>),
    F16(ArrayD<f16>),
    /// Quantized tensors are stored as a vector of their block structs for maximum performance.
    Q8_0(QuantizedMatrix<BlockQ8_0>),
    Q4_K(QuantizedMatrix<BlockQ4_K>),
    Q6_K(QuantizedMatrix<BlockQ6_K>),
}

impl CpuTensor {
    /// Returns the `DType` of the tensor.
    pub fn dtype(&self) -> DType {
        match self {
            CpuTensor::F32(_) => DType::F32,
            CpuTensor::BF16(_) => DType::BF16,
            CpuTensor::F16(_) => DType::F16,
            CpuTensor::Q8_0(_) => DType::Q8_0,
            CpuTensor::Q4_K(_) => DType::Q4_K,
            CpuTensor::Q6_K(_) => DType::Q6_K,
        }
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        match self {
            CpuTensor::F32(arr) => arr.shape(),
            CpuTensor::F16(arr) => arr.shape(),
            CpuTensor::BF16(arr) => arr.shape(),
            CpuTensor::Q8_0(q) => &q.shape,
            CpuTensor::Q4_K(q) => &q.shape,
            CpuTensor::Q6_K(q) => &q.shape,
        }
    }
    pub fn to_array2_f32(self) -> Result<Array2<f32>> {
        match self {
            CpuTensor::F32(arr) => {
                let shape = arr.shape();
                // Use only the first two dimensions, effectively squeezing [2048, 128256, 1] into [2048, 128256]
                Ok(arr
                    .clone()
                    .into_shape_with_order((shape[0], shape[1]))?
                    .into_dimensionality::<Ix2>()?)
            }
            CpuTensor::F16(arr) => {
                let shape = arr.shape();
                Ok(arr
                    .mapv(|v| v.to_f32())
                    .into_shape_with_order((shape[0], shape[1]))?
                    .into_dimensionality::<Ix2>()?)
            }
            CpuTensor::BF16(arr) => {
                let shape = arr.shape();
                Ok(arr
                    .mapv(|v| v.to_f32())
                    .into_shape_with_order((shape[0], shape[1]))?
                    .into_dimensionality::<Ix2>()?)
            }
            Self::Q4_K(matrix) => {
                let mut res = Array2::zeros((matrix.shape[0], matrix.shape[1]));
                for (row_idx, mut row) in res.axis_iter_mut(ndarray::Axis(0)).enumerate() {
                    let blocks_per_row = matrix.shape[1] / 256;
                    let row_blocks =
                        &matrix.blocks[row_idx * blocks_per_row..(row_idx + 1) * blocks_per_row];
                    let row_slice = row.as_slice_mut().unwrap();
                    for (b_idx, block) in row_blocks.iter().enumerate() {
                        dequantize_q4_k_block(
                            block,
                            &mut row_slice[b_idx * 256..(b_idx + 1) * 256],
                        );
                    }
                }
                Ok(res)
            }
            Self::Q6_K(matrix) => {
                let mut res = Array2::zeros((matrix.shape[0], matrix.shape[1]));
                for (row_idx, mut row) in res.axis_iter_mut(ndarray::Axis(0)).enumerate() {
                    let blocks_per_row = matrix.shape[1] / 256;
                    let row_blocks =
                        &matrix.blocks[row_idx * blocks_per_row..(row_idx + 1) * blocks_per_row];
                    let row_slice = row.as_slice_mut().unwrap();
                    for (b_idx, block) in row_blocks.iter().enumerate() {
                        dequantize_q6_k_block(
                            block,
                            &mut row_slice[b_idx * 256..(b_idx + 1) * 256],
                        );
                    }
                }
                Ok(res)
            }
            _ => Err(anyhow!(
                "Dequantization to F32 not implemented for this type"
            )),
        }
    }
    /// Converts the tensor to an `Array1<f32>`. Fails for matrix types.
    ///
    /// This is typically used for loading biases or layer normalization weights.
    pub fn to_array1_f32(&self) -> Result<Array1<f32>> {
        match self {
            CpuTensor::F32(arr) => {
                let len = arr.len(); // Total elements (e.g., 2048)
                // .into_shape_with_order(len) flattens [2048, 1] into [2048]
                Ok(arr
                    .clone()
                    .into_shape_with_order(len)?
                    .into_dimensionality::<Ix1>()?)
            }
            CpuTensor::F16(arr) => {
                let len = arr.len();
                Ok(arr
                    .mapv(|v| v.to_f32())
                    .into_shape_with_order(len)?
                    .into_dimensionality::<Ix1>()?)
            }
            CpuTensor::BF16(arr) => {
                let len = arr.len();
                Ok(arr
                    .mapv(|v| v.to_f32())
                    .into_shape_with_order(len)?
                    .into_dimensionality::<Ix1>()?)
            }
            _ => Err(anyhow!("Cannot convert a matrix-quantized type to Array1")),
        }
    }
}
