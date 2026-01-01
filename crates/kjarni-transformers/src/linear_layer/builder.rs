use crate::{
    linear_layer::{F32MatmulStrategy, LinearData, LinearLayer},
    tensor::{CpuTensor, DType, QuantizedMatrix},
    weights::ModelWeights,
};
use anyhow::{Result, anyhow};
use half::bf16;
use ndarray::{Ix1, Ix2};

/// A builder for constructing a `LinearLayer` from `ModelWeights`.
/// This encapsulates all the complex loading and conversion logic.
pub struct LinearLayerBuilder<'a> {
    weights: &'a ModelWeights,
    weight_name: String,
    bias_name: Option<String>,
    target_dtype: Option<DType>,
    f32_strategy: F32MatmulStrategy,
}

impl<'a> LinearLayerBuilder<'a> {
    pub fn new(weights: &'a ModelWeights, weight_name: &str) -> Self {
        Self {
            weights,
            weight_name: weight_name.to_string(),
            bias_name: None,
            target_dtype: None,
            f32_strategy: F32MatmulStrategy::CustomSimd,
        }
    }

    pub fn with_bias(mut self, bias_name: &str) -> Self {
        self.bias_name = Some(bias_name.to_string());
        self
    }

    pub fn with_optional_bias(mut self, bias_name: Option<&str>) -> Self {
        self.bias_name = bias_name.map(|s| s.to_string());
        self
    }

    pub fn with_target_dtype(mut self, dtype: Option<DType>) -> Self {
        self.target_dtype = dtype;
        self
    }

    pub fn with_f32_strategy(mut self, strategy: Option<F32MatmulStrategy>) -> Self {
        self.f32_strategy = strategy.unwrap_or(F32MatmulStrategy::CustomSimd);
        self
    }

    /// The main construction logic, moved from the old `from_weights`.
    pub fn build(self) -> Result<LinearLayer> {
        let weight_name = if self.weight_name.ends_with(".weight") {
            self.weight_name
        } else {
            format!("{}.weight", self.weight_name)
        };

        let typed_tensor = self.weights.get_typed_tensor(&weight_name)?;
        let target_dtype = self.target_dtype.unwrap_or_else(|| typed_tensor.dtype());

        let data = match (typed_tensor, target_dtype) {
            // --- Direct Paths (No Conversion) ---
            (CpuTensor::F32(arr), DType::F32) => LinearData::F32(arr.into_dimensionality::<Ix2>()?),
            (CpuTensor::BF16(arr), DType::BF16) => {
                LinearData::BF16(arr.into_dimensionality::<Ix2>()?)
            }
            (CpuTensor::Q8_0(m), DType::Q8_0) => LinearData::Q8_0(m),
            (CpuTensor::Q4_K(m), DType::Q4_K) => LinearData::Q4_K(m),
            (CpuTensor::Q6_K(m), DType::Q6_K) => LinearData::Q6_K(m),

            // --- Conversion Paths ---
            (CpuTensor::BF16(arr), DType::F32) => {
                LinearData::F32(arr.into_dimensionality::<Ix2>()?.mapv(|v| v.to_f32()))
            }
            (CpuTensor::F32(arr), DType::BF16) => {
                LinearData::BF16(arr.into_dimensionality::<Ix2>()?.mapv(bf16::from_f32))
            }

            // --- Dequantization Path ---
            (tensor, DType::F32) if tensor.is_quantized() => {
                log::info!("Dequantizing {:?} to F32 for LinearLayer", tensor.dtype());
                LinearData::F32(tensor.to_array2_f32()?)
            }

            // --- Quantization Paths (New) ---
            (CpuTensor::F32(arr), DType::Q8_0) => {
                log::info!("Quantizing F32 to Q8_0 for LinearLayer");
                let w = arr.into_dimensionality::<Ix2>()?;
                let blocks = crate::kernels::quantize::quantize_matrix_q8_0(&w)?;
                LinearData::Q8_0(QuantizedMatrix {
                    blocks,
                    shape: [w.shape()[0], w.shape()[1]],
                })
            }
            (CpuTensor::BF16(arr), DType::Q8_0) => {
                log::info!("Quantizing BF16 to Q8_0 for LinearLayer");
                let w = arr.into_dimensionality::<Ix2>()?;
                let w_f32 = w.mapv(|v| v.to_f32());
                let blocks = crate::kernels::quantize::quantize_matrix_q8_0(&w_f32)?;
                LinearData::Q8_0(QuantizedMatrix {
                    blocks,
                    shape: [w.shape()[0], w.shape()[1]],
                })
            }

            (t, d) => {
                return Err(anyhow!(
                    "Unsupported dtype conversion from {:?} to {:?}",
                    t.dtype(),
                    d
                ));
            }
        };

        let bias_name_to_load = self
            .bias_name
            .unwrap_or_else(|| weight_name.replace(".weight", ".bias"));

        let bias = if self.weights.contains(&bias_name_to_load) {
            match self.weights.get_typed_tensor(&bias_name_to_load)? {
                CpuTensor::F32(b) => Some(b.into_dimensionality::<Ix1>()?),
                CpuTensor::BF16(b) => {
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

        Ok(LinearLayer {
            data,
            bias,
            f32_strategy: self.f32_strategy,
        })
    }
}
