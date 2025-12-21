use crate::utils::linear_algebra::{matmul_2d_mixed_bf16, matmul_2d_transposed, matmul_2d};
use crate::weights_old::{ModelWeights};
use crate::tensor::{DType};
use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, ArrayView2, Ix1, Ix2};
use half::bf16;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightLayout {
    /// Shape: [InFeatures, OutFeatures]
    /// Math: y = x @ W
    /// Best for: CPU (Faer), Matrix-Vector multiplication where columns are contiguous
    InOut,

    /// Shape: [OutFeatures, InFeatures]
    /// Math: y = x @ W^T
    /// Best for: Custom AVX Kernels (Dot Product), GPU Coalescing
    OutIn,
}

pub struct LinearLayer {
    data: LinearData,
    pub bias: Option<Array1<f32>>,
    pub layout: WeightLayout,
}

pub enum LinearData {
    F32(Array2<f32>),
    BF16(Array2<u16>),
}

impl LinearLayer {
    /// Computes X @ W^T + b
    /// Weights stored as [out, in], computes input[batch, in] -> output[batch, out]
    #[inline]
    pub fn matmul(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        let mut result = match &self.data {
            // Use standard Faer transpose logic.
            // It might be slower than pre-transposing, but it is GUARANTEED correct.
            LinearData::F32(w) => matmul_2d_transposed(input, &w.view()),

            // Your custom kernel expects OutIn
            LinearData::BF16(w) => matmul_2d_mixed_bf16(input, &w.view()),
        };

        if let Some(bias) = &self.bias {
            result += bias;
        }
        result
    }

    // ==================== Constructors ====================

    pub fn new_f32(weight: Array2<f32>, bias: Option<Array1<f32>>) -> Self {
        Self {
            data: LinearData::F32(weight.as_standard_layout().to_owned()),
            bias,
            layout: WeightLayout::OutIn,
        }
    }


    // ==================== Loading ====================

    /// Load with explicit weight and bias names
    ///
    /// # Arguments
    /// * `weight_name` - Full name of weight tensor
    /// * `bias_name` - Full name of bias tensor, or None for no bias
    /// * `transpose` - If true, transpose weight from [in, out] to [out, in]
    /// * `dtype_override` - Force specific dtype (None = use file dtype)
    pub fn from_weight_and_bias(
        weights: &ModelWeights,
        weight_name: &str,
        bias_name: Option<&str>,
        transpose: bool,
        dtype_override: Option<DType>,
    ) -> Result<Self> {
        let raw_tensor = weights.get_raw(weight_name)?;
        let file_dtype = raw_tensor.dtype;
        let target_dtype = dtype_override.unwrap_or(file_dtype);

        let data = match (file_dtype, target_dtype) {
            (DType::BF16, DType::BF16) => {
                let mut w = weights.get_linear_weight_bf16(weight_name)?;
                if transpose {
                    w = w.t().as_standard_layout().to_owned();
                }
                LinearData::BF16(w.as_standard_layout().to_owned())
            }
            (DType::F32, DType::F32) | (DType::BF16, DType::F32) => {
                let mut w = weights.get_array2(weight_name)?;
                if transpose {
                    w = w.t().as_standard_layout().to_owned();
                }
                LinearData::F32(w.as_standard_layout().to_owned())
            }
            (src, tgt) => {
                return Err(anyhow!(
                    "Cannot load '{}': {:?} -> {:?} not supported",
                    weight_name,
                    src,
                    tgt
                ));
            }
        };

        let bias = match bias_name {
            Some(name) if weights.contains(name) => Some(weights.get_array1(name)?),
            _ => None,
        };

        Ok(Self {
            data,
            bias,
            layout: WeightLayout::OutIn,
        })
    }

    /// Load weight only (no bias)
    pub fn from_weight(
        weights: &ModelWeights,
        weight_name: &str,
        transpose: bool,
        dtype_override: Option<DType>,
    ) -> Result<Self> {
        Self::from_weight_and_bias(weights, weight_name, None, transpose, dtype_override)
    }
    pub fn from_weights(
        weights: &ModelWeights,
        name: &str,
        dtype_override: Option<DType>,
    ) -> Result<Self> {
        // Simple loader. No transpose logic.
        let raw_tensor = weights.get_raw(name)?;
        let file_dtype = raw_tensor.dtype;
        let target_dtype = dtype_override.unwrap_or(file_dtype);

        let data = match (file_dtype, target_dtype) {
            (DType::BF16, DType::BF16) => {
                let w = weights.get_linear_weight_bf16(name)?;
                LinearData::BF16(w.as_standard_layout().to_owned())
            }
            (DType::F32, DType::F32) | (DType::BF16, DType::F32) => {
                let w = weights.get_array2(name)?;
                LinearData::F32(w.as_standard_layout().to_owned())
            }
            _ => return Err(anyhow!("Unsupported conversion")),
        };

        let bias_key = if name.ends_with(".weight") {
            name.replace(".weight", ".bias")
        } else {
            format!("{}.bias", name.trim_end_matches(".weight"))
        };
        let bias = if weights.contains(&bias_key) {
            Some(weights.get_array1(&bias_key)?)
        } else {
            None
        };

        Ok(Self {
            data,
            bias,
            layout: WeightLayout::OutIn,
        })
    }

    

    pub fn shape(&self) -> [usize; 2] {
        match &self.data {
            LinearData::F32(w) => [w.shape()[0], w.shape()[1]],
            LinearData::BF16(w) => [w.shape()[0], w.shape()[1]],
        }
    }

    // --- Accessors ---
    pub fn out_features(&self) -> usize {
        match &self.data {
            LinearData::F32(w) => w.shape()[0],
            LinearData::BF16(w) => w.shape()[0],
        }
    }

    pub fn in_features(&self) -> usize {
        match &self.data {
            LinearData::F32(w) => w.shape()[1],
            LinearData::BF16(w) => w.shape()[1],
        }
    }

    pub fn dtype(&self) -> DType {
        match &self.data {
            LinearData::F32(_) => DType::F32,
            LinearData::BF16(_) => DType::BF16,
        }
    }

    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    // ==================== Conversions ====================

    pub fn to_f32(&self) -> Array2<f32> {
        match &self.data {
            LinearData::F32(w) => w.clone(),
            LinearData::BF16(w) => w.mapv(|x| f32::from_bits((x as u32) << 16)),
        }
    }

    pub fn bias_to_gpu(
        &self,
        ctx: &std::sync::Arc<crate::WgpuContext>,
    ) -> Result<Option<crate::gpu_ops::GpuTensor>> {
        if let Some(bias) = &self.bias {
            let gpu_bias = crate::gpu_ops::GpuTensor::from_ndarray(ctx, bias)?;
            Ok(Some(gpu_bias))
        } else {
            Ok(None)
        }
    }
    pub fn to_gpu(
        &self,
        ctx: &std::sync::Arc<crate::WgpuContext>,
    ) -> Result<crate::gpu_ops::GpuTensor> {
        crate::gpu_ops::GpuTensor::from_ndarray(ctx, &self.to_f32())
    }
}

impl From<Array2<f32>> for LinearLayer {
    fn from(arr: Array2<f32>) -> Self {
        LinearLayer::new_f32(arr, None)
    }
}
