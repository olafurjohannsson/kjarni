use crate::utils::linear_algebra::{matmul_2d_mixed_bf16, matmul_2d_transposed};
use crate::weights::{DType, ModelWeights};
use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, ArrayView2};

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

    pub fn new_bf16(weight: Array2<u16>, bias: Option<Array1<f32>>) -> Self {
        Self {
            data: LinearData::BF16(weight.as_standard_layout().to_owned()),
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

        Ok(Self { data, bias, layout: WeightLayout::OutIn })
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

        Ok(Self { data, bias, layout: WeightLayout::OutIn })
    }
    /// Smart loader that converts whatever is in the file to the optimal format for this backend.
    ///
    /// For CPU (Faer), optimal is `InOut`.
    /// For GPU/SIMD, optimal is `OutIn`.
    pub fn from_weights_layout(
        weights: &ModelWeights,
        name: &str,
        source_layout_hint: Option<WeightLayout>, // None = Try to Auto-Detect
        dtype_override: Option<DType>,
    ) -> Result<Self> {
        let raw_tensor = weights.get_raw(name)?;
        let shape = raw_tensor.shape.clone();

        // --- 1. Load Bias (Crucial for auto-detection) ---
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

        // --- 2. Determine Source Layout ---
        let source_layout = match source_layout_hint {
            Some(l) => l,
            None => Self::infer_layout(&shape, bias.as_ref().map(|b| b.shape()[0]))?,
        };

        // --- 3. Determine Target Layout (Optimization Strategy) ---
        // For CPU F32, Faer is much faster with [In, Out].
        // For BF16 custom kernels, you might prefer [Out, In].
        let target_layout = match dtype_override.unwrap_or(raw_tensor.dtype) {
            DType::F32 => WeightLayout::InOut,
            _ => WeightLayout::OutIn,
        };

        let transpose_needed = source_layout != target_layout;

        // --- 4. Load & Transpose if needed ---
        let data = match (raw_tensor.dtype, dtype_override.unwrap_or(raw_tensor.dtype)) {
            (DType::F32, DType::F32) | (DType::BF16, DType::F32) => {
                let mut w = weights.get_array2(name)?;
                if transpose_needed {
                    w = w.t().as_standard_layout().to_owned();
                }
                LinearData::F32(w.as_standard_layout().to_owned())
            }
            (DType::BF16, DType::BF16) => {
                let mut w = weights.get_linear_weight_bf16(name)?;
                if transpose_needed {
                    w = w.t().as_standard_layout().to_owned();
                }
                LinearData::BF16(w.as_standard_layout().to_owned())
            }
            _ => return Err(anyhow!("Unsupported dtype conversion")),
        };

        Ok(Self { data, bias, layout: target_layout })
    }

    /// Heuristic to detect layout based on shapes and bias
    fn infer_layout(weight_shape: &[usize], bias_len: Option<usize>) -> Result<WeightLayout> {
        if weight_shape.len() != 2 {
            return Err(anyhow!("Weight must be 2D"));
        }
        let (dim0, dim1) = (weight_shape[0], weight_shape[1]);

        // Case A: Bias is present
        if let Some(bias_size) = bias_len {
            if dim0 == bias_size && dim1 != bias_size {
                // Bias matches Dim0 -> Dim0 is OutFeatures -> [Out, In]
                return Ok(WeightLayout::OutIn);
            }
            if dim1 == bias_size && dim0 != bias_size {
                // Bias matches Dim1 -> Dim1 is OutFeatures -> [In, Out]
                return Ok(WeightLayout::InOut);
            }
            if dim0 == bias_size && dim1 == bias_size {
                // Square matrix with matching bias. Ambiguous.
                // Default to PyTorch standard (OutIn) as it's most common today (Llama, Bart, Bert).
                // GPT-2 users MUST provide explicit hint.
                log::warn!("Ambiguous square matrix with bias. Defaulting to OutIn (PyTorch standard).");
                return Ok(WeightLayout::OutIn);
            }
        }

        // Case B: No Bias (Rectangular)
        // Hard to guess without model knowledge.
        // Usually [4096, 1024] -> OutIn (FFN Expansion)
        // Defaulting to OutIn is the safest bet for Safetensors/HuggingFace.
        Ok(WeightLayout::OutIn)
    }
    // ==================== Accessors ====================

    pub fn shape(&self) -> [usize; 2] {
        match &self.data {
            LinearData::F32(w) => [w.shape()[0], w.shape()[1]],
            LinearData::BF16(w) => [w.shape()[0], w.shape()[1]],
        }
    }

    pub fn out_features(&self) -> usize {
        self.shape()[0]
    }
    pub fn in_features(&self) -> usize {
        self.shape()[1]
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

    pub fn as_f32(&self) -> Option<&Array2<f32>> {
        match &self.data {
            LinearData::F32(w) => Some(w),
            _ => None,
        }
    }

    pub fn to_f32_transposed(&self) -> Array2<f32> {
        self.to_f32().t().as_standard_layout().to_owned()
    }

    pub fn to_gpu(
        &self,
        ctx: &std::sync::Arc<crate::WgpuContext>,
    ) -> Result<crate::gpu_ops::GpuTensor> {
        crate::gpu_ops::GpuTensor::from_ndarray(ctx, &self.to_f32())
    }
    pub fn to_gpu_transposed(
        &self,
        ctx: &std::sync::Arc<crate::WgpuContext>,
    ) -> Result<crate::gpu_ops::GpuTensor> {
        crate::gpu_ops::GpuTensor::from_ndarray(ctx, &self.to_f32_transposed())
    }
}

impl From<Array2<f32>> for LinearLayer {
    fn from(arr: Array2<f32>) -> Self {
        LinearLayer::new_f32(arr, None)
    }
}
