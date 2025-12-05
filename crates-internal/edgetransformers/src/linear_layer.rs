use crate::utils::linear_algebra::{matmul_2d_mixed_bf16, matmul_2d_transposed};
use crate::weights::{DType, ModelWeights};
use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, ArrayView2};

pub struct LinearLayer {
    data: LinearData,
    bias: Option<Array1<f32>>,
}

pub enum LinearData {
    F32(Array2<f32>),
    BF16(Array2<u16>),
    // Q4(Q4Tensor), // Ready for future
}

impl LinearLayer {
    /// Computes X * W^T + b
    #[inline]
    pub fn matmul(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        // 1. Matrix Multiplication
        let mut result = match &self.data {
            LinearData::F32(w) => matmul_2d_transposed(input, &w.view()),
            LinearData::BF16(w) => matmul_2d_mixed_bf16(input, &w.view()),
        };

        // 2. Bias Addition (Broadcasting)
        if let Some(bias) = &self.bias {
            // ndarray handles [Batch, Dim] + [Dim] broadcasting automatically
            // if the last dimensions match.
            result += bias;
        }

        result
    }

    pub fn shape(&self) -> &[usize] {
        let result = match &self.data {
            LinearData::F32(w) => w.shape(),
            LinearData::BF16(w) => w.shape(),
        };

        result
    }

    /// Upload to GPU with transpose for standard A @ B matmul
    pub fn to_gpu(
        &self,
        ctx: &std::sync::Arc<crate::WgpuContext>,
    ) -> Result<crate::gpu_ops::GpuTensor> {
        crate::gpu_ops::GpuTensor::from_ndarray(ctx, &self.to_f32_transposed())
    }

    /// Converts internal weights to F32 Array2.
    /// Useful for GPU uploading or legacy debugging.
    pub fn to_f32(&self) -> Array2<f32> {
        match &self.data {
            LinearData::F32(w) => w.clone(),
            LinearData::BF16(w) => {
                // Convert u16 -> f32
                // We map over the array and do the bit shift logic
                w.mapv(|x| f32::from_bits((x as u32) << 16))
            }
        }
    }

    /// Returns a reference to F32 weights if they exist, else None.
    /// Used for the legacy DecoderLanguageModel::lm_head() trait.
    pub fn as_f32(&self) -> Option<&Array2<f32>> {
        match &self.data {
            LinearData::F32(w) => Some(w),
            _ => None,
        }
    }

    /// Returns a transposed copy of the weights as F32.
    /// Used for GPU upload (GPU expects [In, Out] usually, or [Out, In] depending on your kernel).
    /// Your GPU code seemed to expect .t(), so let's provide a helper.
    pub fn to_f32_transposed(&self) -> Array2<f32> {
        let f32_view = self.to_f32();
        f32_view.t().as_standard_layout().to_owned()
    }

    /// Smart Loading Logic
    pub fn from_weights(
        weights: &ModelWeights,
        name: &str,
        dtype_override: Option<DType>,
    ) -> Result<Self> {
        // 1. Peek at the file to see what it actually is
        let raw_tensor = weights.get_raw(name)?;
        let file_dtype = raw_tensor.dtype;

        // 2. Decide what we WANT it to be
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
            (src, tgt) => {
                return Err(anyhow!(
                    "Cannot load tensor '{}': Source is {:?}, Target is {:?} (Conversion not supported)",
                    name,
                    src,
                    tgt
                ));
            }
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

        Ok(Self { data, bias })
    }

    /// Returns the output dimension (number of rows in the weight matrix).
    pub fn out_features(&self) -> usize {
        match &self.data {
            LinearData::F32(w) => w.shape()[0],
            LinearData::BF16(w) => w.shape()[0],
        }
    }

    /// Returns the input dimension (number of columns in the weight matrix).
    pub fn in_features(&self) -> usize {
        match &self.data {
            LinearData::F32(w) => w.shape()[1],
            LinearData::BF16(w) => w.shape()[1],
        }
    }
}
// --- Backwards Compatibility Implementation ---
// This allows SwiGluFeedForward::new(Array2<f32>, ...) to still work!
impl From<Array2<f32>> for LinearLayer {
    fn from(arr: Array2<f32>) -> Self {
        // Assume input is [Out, In] or [In, Out]?
        // Usually manual construction in tests implies standard math [In, Out] or [Out, In].
        // Let's assume the user constructed it matching the weight shape [Out, In].
        LinearLayer {
            data: LinearData::F32(arr.as_standard_layout().to_owned()),
            bias: None,
        }
    }
}
