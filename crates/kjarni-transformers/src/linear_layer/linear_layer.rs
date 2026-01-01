//! A CPU-based linear layer that supports multiple data types and quantization.
//!
//! The core component is the `LinearLayer` struct. It is designed to be a "smart"
//! executor that consumes tensors loaded by the `weights` module.
//!
//! A key design principle here is the memory layout convention: all weight tensors
//! are stored in an `[OutFeatures, InFeatures]` layout. This is optimal for
//! custom SIMD/AVX kernels and matches the standard layout in formats like safetensors.
//! The `matmul` function is responsible for calling a compute kernel that correctly
//! handles this layout (e.g., by performing a transposed matrix multiplication).

use crate::gpu_ops::GpuTensor;
use crate::kernels::q_common::{BlockQ4_K, BlockQ6_K, BlockQ8_0};
use crate::linear_layer::LinearLayerBuilder;
use crate::tensor::{CpuTensor, DType, QuantizedMatrix, TensorView};
use crate::utils::tensor_ops;
use crate::weights::ModelWeights;
use crate::{WgpuContext, ops};
use anyhow::{Result, anyhow};
use half::bf16;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Ix1, Ix2};
use std::borrow::Cow;
use std::sync::Arc;

/// A strategy for F32 matrix multiplication, allowing for benchmarking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F32MatmulStrategy {
    /// Use the highly optimized `faer` library. Requires `[In, Out]` weight layout.
    Faer,
    /// Use custom-written SIMD kernels. Requires `[Out, In]` weight layout.
    CustomSimd,
}

/// A CPU-based linear transformation layer (y = xW^T + b).
pub struct LinearLayer {
    pub data: LinearData,
    pub bias: Option<Array1<f32>>,
    pub f32_strategy: F32MatmulStrategy,
}

/// An enum holding the actual weight data for a linear layer.
/// This allows for type-safe dispatch to different compute kernels.
pub enum LinearData {
    F32(Array2<f32>),
    BF16(Array2<bf16>),
    Q8_0(QuantizedMatrix<BlockQ8_0>),
    Q4_K(QuantizedMatrix<BlockQ4_K>),
    Q6_K(QuantizedMatrix<BlockQ6_K>),
}

impl LinearData {
    pub fn dtype(&self) -> DType {
        match self {
            LinearData::F32(_) => DType::F32,
            LinearData::BF16(_) => DType::BF16,
            LinearData::Q8_0(_) => DType::Q8_0,
            LinearData::Q6_K(_) => DType::Q6_K,
            LinearData::Q4_K(_) => DType::Q4_K,
        }
    }
}

impl LinearLayer {
    pub fn new(out_features: usize, in_features: usize, dtype: DType) -> Self {
        match dtype {
            DType::F32 => Self::new_f32(Array2::<f32>::zeros((out_features, in_features)), None),
            DType::BF16 => Self::new_bf16(Array2::<bf16>::zeros((out_features, in_features)), None),
            _ => {
                panic!("Unsupported dtype for LinearLayer::new: {:?}", dtype);
            }
        }
    }
    pub fn builder<'a>(weights: &'a ModelWeights, name: &str) -> LinearLayerBuilder<'a> {
        LinearLayerBuilder::new(weights, name)
    }
    /// Creates a new F32 `LinearLayer` from a weight matrix and an optional bias.
    ///
    /// This is a convenience constructor for testing or building models in code.
    /// It assumes the weight matrix is in the `[OutFeatures, InFeatures]` layout
    /// and defaults to the high-performance `CustomSimd` strategy.
    pub fn new_f32(weights: Array2<f32>, bias: impl Into<Option<Array1<f32>>>) -> Self {
        Self {
            data: LinearData::F32(weights),
            bias: bias.into(),
            f32_strategy: F32MatmulStrategy::CustomSimd,
        }
    }

    /// Creates a new BF16 `LinearLayer` from a weight matrix and an optional bias.
    pub fn new_bf16(weights: Array2<bf16>, bias: impl Into<Option<Array1<f32>>>) -> Self {
        Self {
            data: LinearData::BF16(weights),
            bias: bias.into(),
            // f32_strategy is not applicable here, but we set a default.
            f32_strategy: F32MatmulStrategy::CustomSimd,
        }
    }
    

    /// Computes `y = x @ W^T + b`, dispatching to the optimal backend kernel.
    #[inline]
    pub fn matmul(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        let mut result = match &self.data {
            LinearData::F32(w) => match self.f32_strategy {
                F32MatmulStrategy::Faer => tensor_ops::matmul_2d_faer(input, &w.view()),
                F32MatmulStrategy::CustomSimd => ops::matmul::matmul_2d_cpu_f32(input, &w.view()),
            },
            LinearData::BF16(w) => ops::matmul::matmul_2d_cpu_bf16(input, &w.view()),
            LinearData::Q8_0(w) => ops::matmul::matmul_2d_cpu_q8_0(input, &w.blocks),
            LinearData::Q6_K(w) => ops::matmul::matmul_2d_cpu_q6_k(input, &w.blocks),
            LinearData::Q4_K(w) => ops::matmul::matmul_2d_cpu_q4_k(input, &w.blocks),
        };

        if let Some(b) = &self.bias {
            result.outer_iter_mut().for_each(|mut row| row += b);
        }

        result
    }
/// Creates a new, quantized LinearLayer from an existing one.
    pub fn clone_as_quantized(&self, dtype: DType) -> Result<Self> {
        if self.bias.is_some() {
            log::warn!("Quantizing a layer that has a bias. Bias will be discarded.");
        }

        let quantized_data = match (&self.data, dtype) {
            (LinearData::BF16(w), DType::Q8_0) => {
                let w_f32 = w.mapv(|v| v.to_f32());
                let blocks = crate::kernels::quantize::quantize_matrix_q8_0(&w_f32)?;
                let shape = [w.shape()[0], w.shape()[1]];
                LinearData::Q8_0(QuantizedMatrix { blocks, shape })
            }
            (LinearData::F32(w), DType::Q8_0) => {
                let blocks = crate::kernels::quantize::quantize_matrix_q8_0(w)?;
                let shape = [w.shape()[0], w.shape()[1]];
                LinearData::Q8_0(QuantizedMatrix { blocks, shape })
            }
            (d, t) => {
                return Err(anyhow!(
                    "Unsupported quantization from {:?} to {:?}",
                    d.dtype(),
                    t
                ))
            }
        };

        Ok(Self {
            data: quantized_data,
            bias: None, // Bias is not used in quantized matmul for simplicity
            f32_strategy: self.f32_strategy,
        })
    }
    pub fn shape(&self) -> [usize; 2] {
        [self.out_features(), self.in_features()]
    }

    pub fn out_features(&self) -> usize {
        match &self.data {
            LinearData::F32(w) => match self.f32_strategy {
                F32MatmulStrategy::Faer => w.shape()[1],
                F32MatmulStrategy::CustomSimd => w.shape()[0],
            },
            LinearData::BF16(w) => w.shape()[0],
            LinearData::Q8_0(w) => w.shape[0],
            LinearData::Q4_K(w) => w.shape[0],
            LinearData::Q6_K(w) => w.shape[0],
        }
    }

    pub fn in_features(&self) -> usize {
        match &self.data {
            LinearData::F32(w) => match self.f32_strategy {
                F32MatmulStrategy::Faer => w.shape()[0],
                F32MatmulStrategy::CustomSimd => w.shape()[1],
            },
            LinearData::BF16(w) => w.shape()[1],
            LinearData::Q8_0(w) => w.shape[1],
            LinearData::Q4_K(w) => w.shape[1],
            LinearData::Q6_K(w) => w.shape[1],
        }
    }

    pub fn dtype(&self) -> DType {
        match &self.data {
            LinearData::F32(_) => DType::F32,
            LinearData::BF16(_) => DType::BF16,
            LinearData::Q8_0(_) => DType::Q8_0,
            LinearData::Q6_K(_) => DType::Q6_K,
            LinearData::Q4_K(_) => DType::Q4_K,
        }
    }

    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    // --- GPU Conversion Helpers ---

    /// A helper to upload the bias tensor (if it exists) to the GPU.
    pub fn bias_to_gpu(
        &self,
        ctx: &Arc<crate::WgpuContext>,
    ) -> Result<Option<crate::gpu_ops::GpuTensor>> {
        if let Some(bias) = &self.bias {
            let gpu_bias = crate::gpu_ops::GpuTensor::from_ndarray(ctx, bias)?;
            Ok(Some(gpu_bias))
        } else {
            Ok(None)
        }
    }

    pub fn to_gpu(&self, ctx: &Arc<WgpuContext>) -> Result<GpuTensor> {
        self.to_gpu_tensor(ctx, "to_gpu")
    }

    /// Converts the linear layer's weight matrix to a `GpuTensor`.
    ///
    /// This function efficiently handles all CPU data types (`F32`, `BF16`, quantized)
    /// and uploads them to the GPU's VRAM. It's designed to be as zero-copy as
    /// possible on the CPU side before the GPU transfer.
    ///
    /// # Arguments
    /// * `ctx` - A reference-counted pointer to the `WgpuContext`.
    /// * `label` - A descriptive label for the GPU buffer for debugging.
    ///
    /// # Returns
    /// A `Result` containing the new `GpuTensor`.
    pub fn to_gpu_tensor(&self, ctx: &Arc<WgpuContext>, label: &str) -> Result<GpuTensor> {
        match &self.data {
            LinearData::F32(arr) => {
                // For F32, we can get a direct slice of the data.
                let layout = arr.as_standard_layout();

                let slice = layout
                    .as_slice()
                    .ok_or_else(|| anyhow!("F32 tensor is not contiguous"))?;

                let raw_tensor = TensorView {
                    name: label.to_string(),
                    bytes: Cow::Borrowed(bytemuck::cast_slice(slice)),
                    shape: arr.shape().to_vec(),
                    dtype: DType::F32,
                };
                GpuTensor::from_raw(ctx, &raw_tensor, label)
            }
            LinearData::BF16(arr) => {
                let layout = arr.as_standard_layout();

                let slice = layout
                    .as_slice()
                    .ok_or_else(|| anyhow!("BF16 tensor is not contiguous"))?;

                let raw_tensor = TensorView {
                    name: label.to_string(),
                    bytes: Cow::Borrowed(bytemuck::cast_slice(slice)),
                    shape: arr.shape().to_vec(),
                    dtype: DType::BF16,
                };
                GpuTensor::from_raw(ctx, &raw_tensor, label)
            }
            LinearData::Q8_0(matrix) => {
                // For quantized blocks, the data is already a flat Vec of `#[repr(C)]` structs.
                // We can cast this directly to a byte slice for a true zero-copy upload.
                let raw_tensor = TensorView {
                    name: label.to_string(),
                    bytes: Cow::Borrowed(bytemuck::cast_slice(&matrix.blocks)),
                    shape: matrix.shape.to_vec(),
                    dtype: DType::Q8_0,
                };
                GpuTensor::from_raw(ctx, &raw_tensor, label)
            }
            LinearData::Q4_K(matrix) => {
                // Same logic as Q8_0.
                let raw_tensor = TensorView {
                    name: label.to_string(),
                    bytes: Cow::Borrowed(bytemuck::cast_slice(&matrix.blocks)),
                    shape: matrix.shape.to_vec(),
                    dtype: DType::Q4_K,
                };
                GpuTensor::from_raw(ctx, &raw_tensor, label)
            }
            LinearData::Q6_K(matrix) => {
                unimplemented!()
            }
        }
    }
}

/// Creates a `LinearLayer` from an F32 weight matrix with no bias.
impl From<Array2<f32>> for LinearLayer {
    fn from(weights: Array2<f32>) -> Self {
        LinearLayer::new_f32(weights, None)
    }
}

/// Creates a `LinearLayer` from an F32 weight matrix and a bias.
impl From<(Array2<f32>, Array1<f32>)> for LinearLayer {
    fn from((weights, bias): (Array2<f32>, Array1<f32>)) -> Self {
        LinearLayer::new_f32(weights, Some(bias))
    }
}

/// Creates a `LinearLayer` from a BF16 weight matrix with no bias.
impl From<Array2<bf16>> for LinearLayer {
    fn from(weights: Array2<bf16>) -> Self {
        LinearLayer::new_bf16(weights, None)
    }
}

/// Creates a `LinearLayer` from a BF16 weight matrix and a bias.
impl From<(Array2<bf16>, Array1<f32>)> for LinearLayer {
    fn from((weights, bias): (Array2<bf16>, Array1<f32>)) -> Self {
        LinearLayer::new_bf16(weights, Some(bias))
    }
}
