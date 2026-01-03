//! Builder pattern for constructing `LinearLayer` from model weights.
//!
//! This module provides [`LinearLayerBuilder`], which encapsulates the complex
//! loading and dtype conversion logic for creating linear layers from stored
//! model weights.
//!
//! # Overview
//!
//! The builder handles:
//! - Loading weight tensors from [`ModelWeights`] by name
//! - Automatic dtype conversion (e.g., BF16 to F32, or quantization)
//! - Optional bias loading with automatic name inference
//! - F32 matmul strategy selection
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::linear_layer::LinearLayer;
//! use kjarni_transformers::weights::ModelWeights;
//!
//! let weights = ModelWeights::load("model.safetensors")?;
//!
//! // Basic usage
//! let layer = LinearLayer::builder(&weights, "model.layer")
//!     .build()?;
//!
//! // With explicit bias and dtype conversion
//! let layer = LinearLayer::builder(&weights, "model.layer")
//!     .with_bias("model.layer.bias")
//!     .with_target_dtype(Some(DType::F32))
//!     .build()?;
//! ```
//!
//! # See Also
//!
//! - [`LinearLayer`] — The struct being constructed.
//! - [`ModelWeights`] — Source of weight tensors.

use crate::{
    linear_layer::{F32MatmulStrategy, LinearData, LinearLayer},
    tensor::{CpuTensor, DType, QuantizedMatrix},
    weights::ModelWeights,
};
use anyhow::{Result, anyhow};
use half::bf16;
use ndarray::{Array2, Ix1, Ix2};
use std::borrow::Cow;

/// Controls how bias loading is handled.
#[derive(Debug, Clone)]
enum BiasLoading {
    /// Infer bias name from weight name (default).
    Infer,
    /// Use an explicit bias name.
    Explicit(String),
    /// Skip bias loading entirely.
    Skip,
}

/// Extracts shape from a 2D array as `[rows, cols]`.
#[inline]
fn array2_shape<T>(arr: &Array2<T>) -> [usize; 2] {
    [arr.shape()[0], arr.shape()[1]]
}

/// Builder for constructing a `LinearLayer` from `ModelWeights`.
///
/// Encapsulates all the complex loading and dtype conversion logic, providing
/// a fluent interface for configuration.
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::linear_layer::LinearLayer;
///
/// let layer = LinearLayer::builder(&weights, "encoder.layer.0.attention.query")
///     .with_bias("encoder.layer.0.attention.query.bias")
///     .with_target_dtype(Some(DType::F32))
///     .with_f32_strategy(Some(F32MatmulStrategy::CustomSimd))
///     .build()?;
/// ```
///
/// # Dtype Conversion
///
/// The builder supports the following conversions:
///
/// | Source | Target | Notes |
/// |--------|--------|-------|
/// | F32    | F32    | Direct copy |
/// | BF16   | BF16   | Direct copy |
/// | BF16   | F32    | Upcasting |
/// | F32    | BF16   | Downcasting |
/// | Q8_0   | Q8_0   | Direct copy |
/// | Q4_K   | Q4_K   | Direct copy |
/// | Q6_K   | Q6_K   | Direct copy |
/// | Any quantized | F32 | Dequantization |
/// | F32/BF16 | Q8_0 | Quantization |
pub struct LinearLayerBuilder<'a> {
    weights: &'a ModelWeights,
    weight_name: Cow<'a, str>,
    bias_loading: BiasLoading,
    target_dtype: Option<DType>,
    f32_strategy: F32MatmulStrategy,
}

impl<'a> LinearLayerBuilder<'a> {
    /// Creates a new builder for the specified weight tensor.
    ///
    /// # Arguments
    ///
    /// * `weights` - Reference to the model weights container.
    /// * `weight_name` - Name of the weight tensor. The `.weight` suffix is added
    ///   automatically if not present.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // These are equivalent:
    /// let builder = LinearLayerBuilder::new(&weights, "model.layer");
    /// let builder = LinearLayerBuilder::new(&weights, "model.layer.weight");
    /// ```
    pub fn new(weights: &'a ModelWeights, weight_name: &'a str) -> Self {
        Self {
            weights,
            weight_name: Cow::Borrowed(weight_name),
            bias_loading: BiasLoading::Infer,
            target_dtype: None,
            f32_strategy: F32MatmulStrategy::CustomSimd,
        }
    }

    /// Sets an explicit bias tensor name.
    ///
    /// By default, the builder infers the bias name by replacing `.weight` with
    /// `.bias` in the weight name. Use this method to override that behavior.
    ///
    /// # Arguments
    ///
    /// * `bias_name` - The full name of the bias tensor.
    pub fn with_bias(mut self, bias_name: &str) -> Self {
        self.bias_loading = BiasLoading::Explicit(bias_name.to_string());
        self
    }

    /// Sets an optional bias tensor name.
    ///
    /// Convenience method for conditionally setting a bias name. If `None`,
    /// the builder falls back to the default bias name inference.
    ///
    /// # Arguments
    ///
    /// * `bias_name` - Optional bias tensor name.
    pub fn with_optional_bias(mut self, bias_name: Option<&str>) -> Self {
        if let Some(name) = bias_name {
            self.bias_loading = BiasLoading::Explicit(name.to_string());
        }
        self
    }

    /// Skips bias loading entirely.
    ///
    /// Use this when you know the layer has no bias or when you want to
    /// avoid the overhead of checking for a bias tensor.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let layer = LinearLayer::builder(&weights, "layer")
    ///     .without_bias()
    ///     .build()?;
    /// assert!(!layer.has_bias());
    /// ```
    pub fn without_bias(mut self) -> Self {
        self.bias_loading = BiasLoading::Skip;
        self
    }

    /// Sets the target dtype for the weight matrix.
    ///
    /// If set, the weights are converted to the specified dtype during loading.
    /// If `None`, the weights retain their original dtype.
    ///
    /// # Arguments
    ///
    /// * `dtype` - Target dtype, or `None` to use the original dtype.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Load BF16 weights and convert to F32
    /// let layer = LinearLayer::builder(&weights, "layer")
    ///     .with_target_dtype(Some(DType::F32))
    ///     .build()?;
    /// ```
    pub fn with_target_dtype(mut self, dtype: Option<DType>) -> Self {
        self.target_dtype = dtype;
        self
    }

    /// Sets the F32 matmul strategy.
    ///
    /// Only relevant when the final layer dtype is F32. If `None`, defaults
    /// to [`F32MatmulStrategy::CustomSimd`].
    ///
    /// # Arguments
    ///
    /// * `strategy` - The matmul strategy to use, or `None` for the default.
    pub fn with_f32_strategy(mut self, strategy: Option<F32MatmulStrategy>) -> Self {
        self.f32_strategy = strategy.unwrap_or(F32MatmulStrategy::CustomSimd);
        self
    }

    /// Builds the `LinearLayer` by loading and converting the weights.
    ///
    /// Loads the weight tensor from the model weights, applies any dtype
    /// conversions, and optionally loads the bias tensor.
    ///
    /// # Returns
    ///
    /// The constructed `LinearLayer`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * The weight tensor is not found.
    /// * The dtype conversion is not supported.
    /// * The tensor shapes are invalid (e.g., not 2D for weights, not 1D for bias).
    /// * The bias tensor has an unsupported dtype.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let layer = LinearLayer::builder(&weights, "model.layer.weight")
    ///     .with_target_dtype(Some(DType::F32))
    ///     .build()?;
    /// ```
    pub fn build(self) -> Result<LinearLayer> {
        // Normalize weight name: append ".weight" suffix if not already present.
        // This allows callers to use either "layer" or "layer.weight" interchangeably.
        let weight_name = if self.weight_name.ends_with(".weight") {
            self.weight_name.clone().into_owned()
        } else {
            format!("{}.weight", self.weight_name)
        };

        // Load the raw tensor from the weight store
        let typed_tensor = self.weights.get_typed_tensor(&weight_name)?;

        // Determine target dtype: use explicit override or keep original
        let target_dtype = self.target_dtype.unwrap_or_else(|| typed_tensor.dtype());

        // Convert tensor to LinearData, applying dtype conversion if needed.
        // The match arms are ordered: identity conversions first, then upcasts,
        // downcasts, dequantization, and finally quantization.
        let data = match (typed_tensor, target_dtype) {
            // === Identity conversions (no transformation needed) ===
            (CpuTensor::F32(arr), DType::F32) => LinearData::F32(arr.into_dimensionality::<Ix2>()?),
            (CpuTensor::BF16(arr), DType::BF16) => {
                LinearData::BF16(arr.into_dimensionality::<Ix2>()?)
            }
            (CpuTensor::Q8_0(m), DType::Q8_0) => LinearData::Q8_0(m),
            (CpuTensor::Q4_K(m), DType::Q4_K) => LinearData::Q4_K(m),
            (CpuTensor::Q6_K(m), DType::Q6_K) => LinearData::Q6_K(m),

            // === Upcast: BF16 -> F32 (lossless, 2x memory increase) ===
            (CpuTensor::BF16(arr), DType::F32) => {
                LinearData::F32(arr.into_dimensionality::<Ix2>()?.mapv(|v| v.to_f32()))
            }

            // === Downcast: F32 -> BF16 (lossy, 2x memory savings) ===
            (CpuTensor::F32(arr), DType::BF16) => {
                LinearData::BF16(arr.into_dimensionality::<Ix2>()?.mapv(bf16::from_f32))
            }

            // === Dequantization: Q* -> F32 (expensive, expands memory ~4-8x) ===
            (tensor, DType::F32) if tensor.is_quantized() => {
                LinearData::F32(tensor.to_array2_f32()?)
            }

            // === Quantization: F32 -> Q8_0 (expensive, ~4x memory savings) ===
            (CpuTensor::F32(arr), DType::Q8_0) => {
                let w = arr.into_dimensionality::<Ix2>()?;
                let blocks = crate::kernels::quantize::quantize_matrix_q8_0(&w)?;
                LinearData::Q8_0(QuantizedMatrix {
                    blocks,
                    shape: array2_shape(&w),
                })
            }

            // === Quantization: BF16 -> Q8_0 (two-step: upcast then quantize) ===
            (CpuTensor::BF16(arr), DType::Q8_0) => {
                let w = arr.into_dimensionality::<Ix2>()?;
                // Must convert to F32 first since quantization kernels expect F32 input
                let w_f32 = w.mapv(|v| v.to_f32());
                let blocks = crate::kernels::quantize::quantize_matrix_q8_0(&w_f32)?;
                LinearData::Q8_0(QuantizedMatrix {
                    blocks,
                    shape: array2_shape(&w),
                })
            }

            // === Unsupported conversion ===
            (t, d) => {
                return Err(anyhow!(
                    "Unsupported dtype conversion from {:?} to {:?}",
                    t.dtype(),
                    d
                ));
            }
        };

        // Load bias tensor based on the configured strategy
        let bias = match &self.bias_loading {
            BiasLoading::Skip => None,
            BiasLoading::Explicit(name) => self.load_bias(name)?,
            BiasLoading::Infer => {
                // Derive bias name by replacing ".weight" suffix with ".bias"
                let inferred_name = weight_name.replace(".weight", ".bias");
                self.load_bias(&inferred_name)?
            }
        };

        Ok(LinearLayer {
            data,
            bias,
            f32_strategy: self.f32_strategy,
        })
    }

    /// Loads the bias tensor if it exists, converting to F32 if necessary.
    fn load_bias(&self, name: &str) -> Result<Option<ndarray::Array1<f32>>> {
        if !self.weights.contains(name) {
            return Ok(None);
        }

        match self.weights.get_typed_tensor(name)? {
            CpuTensor::F32(b) => Ok(Some(b.into_dimensionality::<Ix1>()?)),
            CpuTensor::BF16(b) => Ok(Some(
                b.mapv(|val| val.to_f32()).into_dimensionality::<Ix1>()?,
            )),
            _ => Err(anyhow!("Bias tensor '{}' has an unsupported dtype", name)),
        }
    }
}
