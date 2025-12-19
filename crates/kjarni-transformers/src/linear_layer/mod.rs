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

use crate::tensor::{DType, QuantizedTensor, TypedCpuTensor};
use crate::utils::linear_algebra::{
    matmul_2d, matmul_2d_f32_notranspose as matmul_2d_f32_custom_simd, matmul_2d_mixed_bf16, matmul_2d_mixed_bf16_new,
    matmul_2d_transposed, matmul_dequant_q4_k,
};
use crate::weights::ModelWeights;
use anyhow::{Result, anyhow};
use half::bf16;
use ndarray::{Array1, Array2, ArrayView2, Ix1, Ix2};
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
    f32_strategy: F32MatmulStrategy,
}

/// An enum holding the actual weight data for a linear layer.
/// This allows for type-safe dispatch to different compute kernels.
pub enum LinearData {
    F32(Array2<f32>),
    BF16(Array2<bf16>),
    Q4_K(Arc<QuantizedTensor>), // Use Arc for cheap cloning of layers
}

impl LinearLayer {
    /// The primary, simplified loader for creating a LinearLayer from model weights.
    ///
    /// It loads the tensor with its native data type and memory layout from the file,
    /// performing conversions only if explicitly requested by `dtype_override`.
    ///
    /// # Arguments
    /// * `weights`: The main `ModelWeights` loader.
    /// * `weight_name`: The full name of the weight tensor (e.g., `"model.layers.0.self_attn.q_proj.weight"`).
    /// * `bias_name`: An optional name for the bias tensor.
    /// * `dtype_override`: Optionally force the layer to load into a specific `DType`.
    pub fn from_weights(
        weights: &ModelWeights,
        weight_name: &str,
        bias_name: Option<&str>,
        dtype_override: Option<DType>,
        f32_strategy: Option<F32MatmulStrategy>,
    ) -> Result<Self> {
        let strategy = f32_strategy.unwrap_or(F32MatmulStrategy::Faer);
        let typed_tensor = weights.get_typed_tensor(weight_name)?;
        let target_dtype = dtype_override.unwrap_or(typed_tensor.dtype());

        let data = match (typed_tensor, target_dtype) {
            // Load F32 as F32
            // (TypedCpuTensor::F32(arr), DType::F32) => {
            //     LinearData::F32(arr.into_dimensionality::<Ix2>()?)
            // }
            // --- F32 PATH or UPCAST PATH (BF16 -> F32) ---
            (typed_tensor @ TypedCpuTensor::F32(_), DType::F32) |
            (typed_tensor @ TypedCpuTensor::BF16(_), DType::F32) => {
                // First, create the `weights_out_in` variable as a guaranteed `Array2<f32>`.
                let weights_out_in: Array2<f32> = match typed_tensor {
                    TypedCpuTensor::F32(arr) => arr.into_dimensionality::<Ix2>()?,
                    TypedCpuTensor::BF16(arr) => arr.mapv(|v| v.to_f32()).into_dimensionality::<Ix2>()?,
                    _ => unreachable!(), // The outer match arm ensures this
                };

                // Now, `match` on the strategy. Both branches now consume an `Array2<f32>`.
                match strategy {
                    F32MatmulStrategy::Faer => {
                        LinearData::F32(weights_out_in.t().as_standard_layout().to_owned())
                    }
                    F32MatmulStrategy::CustomSimd => {
                        LinearData::F32(weights_out_in)
                    }
                }
            }
            // Load BF16 as BF16
            (TypedCpuTensor::BF16(arr), DType::BF16) => {
                LinearData::BF16(arr.into_dimensionality::<Ix2>()?)
            }
            // Upcast BF16 to F32 if requested
            // (TypedCpuTensor::BF16(arr), DType::F32) => {
            //     LinearData::F32(arr.mapv(|val| val.to_f32()).into_dimensionality::<Ix2>()?)
            // }
            // STUB: This is where you would load a Q4 tensor from the weights file.
            // This path will be enabled once the GGUF loader is implemented.
            (_, DType::Q4_K) => {
                unimplemented!("Loading Q4_K weights is not yet implemented in ModelWeights.");
            }
            (t, d) => {
                return Err(anyhow!(
                    "Unsupported dtype conversion from {:?} to {:?}",
                    t.dtype(),
                    d
                ));
            }
        };

        // Load the bias if it exists. Biases are almost always F32.
        let bias = if let Some(b_name) = bias_name {
            if weights.contains(b_name) {
                match weights.get_typed_tensor(b_name)? {
                    TypedCpuTensor::F32(b) => Some(b.into_dimensionality::<Ix1>()?),
                    TypedCpuTensor::BF16(b) => {
                        Some(b.mapv(|val| val.to_f32()).into_dimensionality::<Ix1>()?)
                    }
                    _ => return Err(anyhow!("Bias tensor '{}' has an unsupported dtype", b_name)),
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            data,
            bias,
            f32_strategy: strategy,
        })
    }

    /// Computes y = x @ W^T + b.
    ///
    /// This is the "smart" executor. It knows its internal weight W is stored
    /// as `[OutFeatures, InFeatures]` and dispatches to a compute kernel
    /// that correctly handles this layout.
    #[inline]
    pub fn matmul(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        let mut result = match &self.data {
            LinearData::F32(w) => match self.f32_strategy {
                F32MatmulStrategy::Faer => matmul_2d(input, &w.view()),
                F32MatmulStrategy::CustomSimd => matmul_2d_f32_custom_simd(input, &w.view()),
            },
            LinearData::BF16(w) => matmul_2d_mixed_bf16_new(input, &w.view()),
            LinearData::Q4_K(w) => {
                // matmul_dequant_q4_k(input, w),
                unimplemented!("Q4_K matmul is not yet implemented.");
            }
        };
        // ... (bias logic) ...
        result
    }

    // --- Accessors ---

    pub fn out_features(&self) -> usize {
        match &self.data {
            LinearData::F32(w) => match self.f32_strategy {
                F32MatmulStrategy::Faer => w.shape()[1],       // [In, Out]
                F32MatmulStrategy::CustomSimd => w.shape()[0], // [Out, In]
            },
            LinearData::BF16(w) => w.shape()[0], // [Out, In]
            LinearData::Q4_K(w) => w.shape[0],   // [Out, In]
        }
    }

    pub fn in_features(&self) -> usize {
        match &self.data {
            LinearData::F32(w) => match self.f32_strategy {
                F32MatmulStrategy::Faer => w.shape()[0],       // [In, Out]
                F32MatmulStrategy::CustomSimd => w.shape()[1], // [Out, In]
            },
            LinearData::BF16(w) => w.shape()[1], // [Out, In]
            LinearData::Q4_K(w) => w.shape[1],   // [Out, In]
        }
    }

    pub fn dtype(&self) -> DType {
        match &self.data {
            LinearData::F32(_) => DType::F32,
            LinearData::BF16(_) => DType::BF16,
            LinearData::Q4_K(_) => DType::Q4_K,
        }
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
}
