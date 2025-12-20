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
    matmul_2d, matmul_2d_f32_notranspose as matmul_2d_f32_custom_simd, matmul_2d_mixed_bf16_new,
};
use crate::weights::ModelWeights;
use anyhow::{anyhow, Result};
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
    pub f32_strategy: F32MatmulStrategy,
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

        let weight_name = if weight_name.ends_with(".weight") {
            weight_name.to_string()
        } else {
            format!("{}.weight", weight_name)
        };

        let typed_tensor = weights.get_typed_tensor(&weight_name)?;
        let source_dtype = typed_tensor.dtype();
        let target_dtype = dtype_override.unwrap_or(typed_tensor.dtype());
        println!(
            "Loading layer: '{}'\n    - DType in file: {:?}\n    - User override: {:?}\n    - FINAL target dtype: {:?}",
            weight_name,
            source_dtype,
            dtype_override,
            target_dtype
        );
        let data = match (typed_tensor, target_dtype) {
            // PATH 1: Source is F32, Target is F32.
            (TypedCpuTensor::F32(arr), DType::F32) => {
                let weights_out_in = arr.into_dimensionality::<Ix2>()?;
                match strategy {
                    F32MatmulStrategy::Faer => {
                        println!("    - Storing as F32 with [In, Out] layout for Faer.");
                        LinearData::F32(weights_out_in.t().as_standard_layout().to_owned())
                    }
                    F32MatmulStrategy::CustomSimd => {
                        println!("    - Storing as F32 with [Out, In] layout for Custom SIMD.");
                        LinearData::F32(weights_out_in.as_standard_layout().to_owned())
                    }
                }
            }
            // PATH 2: Source is BF16, Target is F32 (Upcasting).
            (TypedCpuTensor::BF16(arr), DType::F32) => {
                let weights_out_in = arr.into_dimensionality::<Ix2>()?.mapv(|v| v.to_f32());
                match strategy {
                    F32MatmulStrategy::Faer => {
                        println!("    - Storing as F32 with [In, Out] layout for Faer.");
                        LinearData::F32(weights_out_in.t().as_standard_layout().to_owned())
                    }
                    F32MatmulStrategy::CustomSimd => {
                        println!("    - Storing as F32 with [Out, In] layout for Custom SIMD.");
                        LinearData::F32(weights_out_in.as_standard_layout().to_owned())
                    }
                }
            }
            // PATH 3: Source is BF16, Target is BF16 (No-op).
            (TypedCpuTensor::BF16(arr), DType::BF16) => {
                println!("    - Storing as BF16 with [Out, In] layout.");
                LinearData::BF16(arr.into_dimensionality::<Ix2>()?.as_standard_layout().to_owned())
            }
            // PATH 4: Source is F32, Target is BF16 (Downcasting).
            (TypedCpuTensor::F32(arr), DType::BF16) => {
                let weights_bf16 = arr
                    .into_dimensionality::<Ix2>()?
                    .mapv(|v| bf16::from_f32(v));
                LinearData::BF16(weights_bf16.as_standard_layout().to_owned())
            }
            // --- STUB for Q4 ---
            (_, DType::Q4_K) => {
                unimplemented!("Loading Q4_K weights is not yet implemented.");
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
        let bias_name_to_load = bias_name
            .map(|s| s.to_string())
            .unwrap_or_else(|| weight_name.replace(".weight", ".bias"));

        // 2. Load the bias if a tensor with that name exists.
        let bias = if weights.contains(&bias_name_to_load) {
            match weights.get_typed_tensor(&bias_name_to_load)? {
                TypedCpuTensor::F32(b) => Some(b.into_dimensionality::<Ix1>()?),
                TypedCpuTensor::BF16(b) => {
                    Some(b.mapv(|val| val.to_f32()).into_dimensionality::<Ix1>()?)
                }
                _ => {
                    return Err(anyhow!(
                        "Bias tensor '{}' has an unsupported dtype",
                        bias_name_to_load
                    ));
                }
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
