//! CPU-based linear layer with multi-dtype support.
//!
//! This module provides [`LinearLayer`], the core building block for neural network
//! weight matrices. It supports F32, BF16, and quantized (Q4_K, Q6_K, Q8_0) storage
//! with automatic dispatch to optimized SIMD kernels.
//!
//! # Overview
//!
//! The [`LinearLayer`] struct wraps weight data in various formats and provides
//! a unified [`LinearLayer::matmul`] interface that dispatches to the optimal kernel
//! based on the underlying data type. A key design principle is the memory layout
//! convention: all weight tensors are stored in an `[OutFeatures, InFeatures]` layout,
//! which is optimal for custom SIMD/AVX kernels and matches the standard layout in
//! formats like safetensors.
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::linear_layer::LinearLayer;
//! use ndarray::Array2;
//!
//! // Create from raw arrays
//! let weights = Array2::<f32>::zeros((4096, 2048));
//! let layer = LinearLayer::new_f32(weights, None);
//!
//! // Forward pass
//! let input = Array2::<f32>::zeros((1, 2048));
//! let output = layer.matmul(&input.view());
//! ```
//!
//! # Performance
//!
//! - F32: Uses `faer` library or custom AVX2/FMA kernels.
//! - BF16: Mixed-precision with F32 accumulation.
//! - Q4_K/Q6_K/Q8_0: Quantized formats with on-the-fly dequantization.
//!
//! # See Also
//!
//! - [`ModelWeights`] — Loading weights from safetensors/GGUF files.
//! - [`DType`] — Supported data types.

use crate::{cpu::{
    kernels::q_common::{BlockQ4_K, BlockQ6_K, BlockQ8_0},
    ops,
}, tensor::raw_tensor::TensorView};

use crate::gpu_ops::GpuTensor;
use crate::linear_layer::LinearLayerBuilder;
use crate::tensor::{DType, QuantizedMatrix};
use crate::utils::tensor_ops;
use crate::weights::ModelWeights;
use crate::WgpuContext;
use anyhow::{anyhow, Result};
use half::bf16;
use ndarray::{Array1, Array2, ArrayView2};
use std::borrow::Cow;
use std::sync::Arc;

/// Strategy for F32 matrix multiplication dispatch.
///
/// Allows selection between different matrix multiplication backends for
/// benchmarking and performance tuning. Each strategy has different weight
/// layout requirements.
///
/// # Weight Layout
///
/// Different strategies expect weights in different memory layouts:
///
/// - [`CustomSimd`](F32MatmulStrategy::CustomSimd): `[OutFeatures, InFeatures]` — the standard
///   layout used by safetensors and most model formats. This is the default.
/// - [`Faer`](F32MatmulStrategy::Faer): `[InFeatures, OutFeatures]` — transposed layout.
///   Some model formats (e.g., certain GPT weight files) store weights in this layout.
///   Using Faer avoids the cost of transposing at load time or during matmul, which
///   can be significant for large weight matrices.
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::linear_layer::F32MatmulStrategy;
///
/// let strategy = F32MatmulStrategy::CustomSimd;
/// assert_eq!(strategy, F32MatmulStrategy::CustomSimd);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F32MatmulStrategy {
    /// Uses the highly optimized `faer` library.
    ///
    /// Expects weights in `[InFeatures, OutFeatures]` layout (transposed from the
    /// standard layout). This is useful when model weights are already stored in
    /// this format, avoiding costly transpose operations. The `faer` library's
    /// matmul explicitly accepts `[In, Out]` layout and performs the equivalent
    /// of `input @ weights` without transposition.
    Faer,

    /// Uses custom-written SIMD kernels with AVX2/FMA.
    ///
    /// Expects weights in `[OutFeatures, InFeatures]` layout (the standard layout).
    /// This is the default strategy and matches most model weight formats.
    CustomSimd,

    FaerOutIn,
}

/// A CPU-based linear transformation layer (y = xW^T + b).
///
/// `LinearLayer` wraps a weight matrix and optional bias vector, providing
/// efficient matrix multiplication with automatic dispatch to SIMD kernels.
/// Weights can be stored in F32, BF16, or quantized formats (Q4_K, Q6_K, Q8_0).
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::linear_layer::LinearLayer;
/// use ndarray::Array2;
///
/// // Create from raw arrays
/// let weights = Array2::<f32>::zeros((4096, 2048));
/// let layer = LinearLayer::new_f32(weights, None);
///
/// // Forward pass
/// let input = Array2::<f32>::zeros((1, 2048));
/// let output = layer.matmul(&input.view());
/// assert_eq!(output.shape(), &[1, 4096]);
/// ```
///
/// # Thread Safety
///
/// `LinearLayer` is `Send + Sync` and can be safely shared across threads.
/// The weight data is immutable after construction.
///
/// # See Also
///
/// - [`LinearData`] — The underlying weight storage enum.
/// - [`ModelWeights`] — Loading weights from files.
#[derive(Clone)]
pub struct LinearLayer {
    /// The weight matrix in one of several supported formats.
    pub data: LinearData,

    /// Optional bias vector, always stored as F32.
    pub bias: Option<Array1<f32>>,

    /// Strategy for F32 matmul dispatch (Faer vs CustomSimd).
    pub f32_strategy: F32MatmulStrategy,
}

/// Weight data storage for a linear layer.
///
/// Holds the actual weight matrix in one of several supported formats,
/// enabling type-safe dispatch to different compute kernels based on
/// the underlying data type.
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::linear_layer::LinearData;
/// use kjarni_transformers::tensor::DType;
/// use ndarray::Array2;
///
/// let weights = Array2::<f32>::zeros((4096, 2048));
/// let data = LinearData::F32(weights);
/// assert_eq!(data.dtype(), DType::F32);
/// ```
#[allow(non_camel_case_types)]
#[derive(Clone)]
pub enum LinearData {
    /// 32-bit IEEE 754 floating point weights.
    ///
    /// Highest precision, largest memory footprint. Use for accuracy-critical
    /// computations or when loading F32 model weights.
    F32(Arc<Array2<f32>>),

    /// 16-bit brain floating point weights.
    ///
    /// Same exponent range as F32 with reduced mantissa. Provides 2x memory
    /// savings with minimal quality loss for inference.
    BF16(Arc<Array2<bf16>>),

    /// 8-bit block-quantized weights (32 elements per block).
    ///
    /// Each block stores 32 int8 values with a shared F16 scale factor.
    /// Provides approximately 4x compression with minimal quality loss.
    Q8_0(Arc<QuantizedMatrix<BlockQ8_0>>),

    /// 4-bit block-quantized weights with K-quants (256 elements per block).
    ///
    /// Each block stores 256 4-bit values with multiple scale factors.
    /// Provides approximately 8x compression, suitable for large models
    /// on limited memory.
    Q4_K(Arc<QuantizedMatrix<BlockQ4_K>>),

    /// 6-bit block-quantized weights with K-quants.
    ///
    /// Higher precision than Q4_K with approximately 5x compression.
    Q6_K(Arc<QuantizedMatrix<BlockQ6_K>>),
}

impl LinearData {
    /// Returns the data type of the stored weights.
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
    /// Creates a new zero-initialized `LinearLayer` with the specified dimensions and dtype.
    ///
    /// # Arguments
    ///
    /// * `out_features` - Number of output features (rows in weight matrix).
    /// * `in_features` - Number of input features (columns in weight matrix).
    /// * `dtype` - Data type for the weight matrix. Only F32 and BF16 are supported.
    ///
    /// # Panics
    ///
    /// Panics if `dtype` is a quantized type (Q8_0, Q4_K, Q6_K).
    pub fn new(out_features: usize, in_features: usize, dtype: DType) -> Self {
        match dtype {
            DType::F32 => Self::new_f32(Array2::<f32>::zeros((out_features, in_features)), None),
            DType::BF16 => Self::new_bf16(Array2::<bf16>::zeros((out_features, in_features)), None),
            _ => {
                panic!("Unsupported dtype for LinearLayer::new: {:?}", dtype);
            }
        }
    }

    /// Returns a builder for constructing a `LinearLayer` from model weights.
    ///
    /// # Arguments
    ///
    /// * `weights` - The model weights container.
    /// * `name` - The name/key of the weight tensor in the model.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let layer = LinearLayer::builder(&weights, "model.layer.weight")
    ///     .with_bias("model.layer.bias")
    ///     .build()?;
    /// ```
    pub fn builder<'a>(weights: &'a ModelWeights, name: &'a str) -> LinearLayerBuilder<'a> {
        LinearLayerBuilder::new(weights, name)
    }
    /// Creates a new F32 `LinearLayer` from a weight matrix and an optional bias.
    ///
    /// Convenience constructor for testing or building models in code.
    /// Assumes the weight matrix is in `[OutFeatures, InFeatures]` layout
    /// and defaults to the high-performance [`F32MatmulStrategy::CustomSimd`] strategy.
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight matrix of shape `[out_features, in_features]`.
    /// * `bias` - Optional bias vector of length `out_features`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kjarni_transformers::linear_layer::LinearLayer;
    /// use ndarray::Array2;
    ///
    /// let weights = Array2::<f32>::zeros((4096, 2048));
    /// let layer = LinearLayer::new_f32(weights, None);
    /// ```
    pub fn new_f32(weights: Array2<f32>, bias: impl Into<Option<Array1<f32>>>) -> Self {
        Self {
            data: LinearData::F32(Arc::new(weights)),
            bias: bias.into(),
            f32_strategy: F32MatmulStrategy::CustomSimd,
        }
    }

    pub fn from_arc_f32(weights: Arc<Array2<f32>>, bias: impl Into<Option<Array1<f32>>>) -> Self {
        Self {
            data: LinearData::F32(weights), // No copy!
            bias: bias.into(),
            f32_strategy: F32MatmulStrategy::CustomSimd,
        }
    }

    /// Creates a new BF16 `LinearLayer` from a weight matrix and an optional bias.
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight matrix of shape `[out_features, in_features]` in BF16 format.
    /// * `bias` - Optional bias vector of length `out_features` (always F32).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kjarni_transformers::linear_layer::LinearLayer;
    /// use ndarray::Array2;
    /// use half::bf16;
    ///
    /// let weights = Array2::<bf16>::zeros((4096, 2048));
    /// let layer = LinearLayer::new_bf16(weights, None);
    /// ```
    pub fn new_bf16(weights: Array2<bf16>, bias: impl Into<Option<Array1<f32>>>) -> Self {
        Self {
            data: LinearData::BF16(Arc::new(weights)),
            bias: bias.into(),
            f32_strategy: F32MatmulStrategy::CustomSimd,
        }
    }

    pub fn from_arc_bf16(weights: Arc<Array2<bf16>>, bias: impl Into<Option<Array1<f32>>>) -> Self {
        Self {
            data: LinearData::BF16(weights), // No copy!
            bias: bias.into(),
            f32_strategy: F32MatmulStrategy::CustomSimd,
        }
    }

    // Q8_0 Sharing
    pub fn from_arc_q8_0(weights: Arc<QuantizedMatrix<BlockQ8_0>>, bias: impl Into<Option<Array1<f32>>>) -> Self {
        Self {
            data: LinearData::Q8_0(weights),
            bias: bias.into(),
            f32_strategy: F32MatmulStrategy::CustomSimd,
        }
    }

    pub fn new_f32_with_strategy(weights: Array2<f32>, bias: impl Into<Option<Array1<f32>>>, strategy: F32MatmulStrategy) -> Self {
        Self {
            data: LinearData::F32(Arc::new(weights)),
            bias: bias.into(),
            f32_strategy: strategy,
        }
    }

    /// Computes matrix multiplication with automatic kernel dispatch.
    ///
    /// Performs `C = A @ W^T + b` where `A` is the input activation and `W` is the
    /// weight matrix stored in this layer. The kernel is selected based on the
    /// weight's data type (F32, BF16, or quantized).
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape `[batch, in_features]`. Must be contiguous.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, out_features]`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let layer = LinearLayer::new_f32(weights, Some(bias));
    /// let input = Array2::<f32>::zeros((1, 2048));
    /// let output = layer.matmul(&input.view());
    /// assert_eq!(output.shape(), &[1, 4096]);
    /// ```
    #[inline]
    pub fn matmul(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        let mut result = match &self.data {
            LinearData::F32(w) => match self.f32_strategy {
                F32MatmulStrategy::Faer => tensor_ops::matmul_2d_faer(input, &w.view()),
                F32MatmulStrategy::CustomSimd => ops::matmul::matmul_2d_cpu_f32(input, &w.view()),
                F32MatmulStrategy::FaerOutIn => ops::matmul::matmul_2d_cpu_f32_faer(input, &w.view())
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
    /// Converts this layer to a quantized format.
    ///
    /// Creates a new `LinearLayer` with quantized weights for reduced memory usage.
    /// The bias is discarded during quantization (a warning is logged if present).
    ///
    /// # Arguments
    ///
    /// * `dtype` - Target quantized data type (currently only Q8_0 is supported).
    ///
    /// # Returns
    ///
    /// A new `LinearLayer` with quantized weights and no bias.
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion from the current dtype to the target dtype
    /// is not supported.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let f32_layer = LinearLayer::new_f32(weights, None);
    /// let quantized = f32_layer.to_quantized(DType::Q8_0)?;
    /// assert_eq!(quantized.dtype(), DType::Q8_0);
    /// ```
    pub fn to_quantized(&self, dtype: DType) -> Result<Self> {
        if self.bias.is_some() {
            log::warn!("Quantizing a layer that has a bias. Bias will be discarded.");
        }

        let quantized_data = match (&self.data, dtype) {
            (LinearData::BF16(w), DType::Q8_0) => {
                let w_f32 = w.mapv(|v| v.to_f32());
                let blocks = crate::cpu::kernels::quantize::quantize_matrix_q8_0(&w_f32)?;
                let shape = [w.shape()[0], w.shape()[1]];
                LinearData::Q8_0(Arc::new(QuantizedMatrix { blocks, shape }))
            }
            (LinearData::F32(w), DType::Q8_0) => {
                let blocks = crate::cpu::kernels::quantize::quantize_matrix_q8_0(w)?;
                let shape = [w.shape()[0], w.shape()[1]];
                LinearData::Q8_0(Arc::new(QuantizedMatrix { blocks, shape }))
            }
            (d, t) => {
                return Err(anyhow!(
                    "Unsupported quantization from {:?} to {:?}",
                    d.dtype(),
                    t
                ));
            }
        };

        Ok(Self {
            data: quantized_data,
            bias: None, // Bias is not used in quantized matmul for simplicity
            f32_strategy: self.f32_strategy,
        })
    }

    /// Returns the shape of the weight matrix as `[out_features, in_features]`.
    #[must_use]
    pub fn shape(&self) -> [usize; 2] {
        [self.out_features(), self.in_features()]
    }

    /// Returns the number of output features.
    #[must_use]
    pub fn out_features(&self) -> usize {
        match &self.data {
            LinearData::F32(w) => match self.f32_strategy {
                F32MatmulStrategy::Faer => w.shape()[1],
                F32MatmulStrategy::CustomSimd => w.shape()[0],
                F32MatmulStrategy::FaerOutIn => w.shape()[0]
            },
            LinearData::BF16(w) => w.shape()[0],
            LinearData::Q8_0(w) => w.shape[0],
            LinearData::Q4_K(w) => w.shape[0],
            LinearData::Q6_K(w) => w.shape[0],
        }
    }

    /// Returns the number of input features.
    #[must_use]
    pub fn in_features(&self) -> usize {
        match &self.data {
            LinearData::F32(w) => match self.f32_strategy {
                F32MatmulStrategy::Faer => w.shape()[0],
                F32MatmulStrategy::CustomSimd => w.shape()[1],
                F32MatmulStrategy::FaerOutIn => w.shape()[1]
            },
            LinearData::BF16(w) => w.shape()[1],
            LinearData::Q8_0(w) => w.shape[1],
            LinearData::Q4_K(w) => w.shape[1],
            LinearData::Q6_K(w) => w.shape[1],
        }
    }

    /// Returns the data type of the weight matrix.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    /// Returns `true` if this layer has a bias term.
    #[must_use]
    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    /// Uploads the bias tensor (if present) to the GPU.
    ///
    /// # Arguments
    ///
    /// * `ctx` - A reference-counted pointer to the `WgpuContext`.
    ///
    /// # Returns
    ///
    /// `Some(GpuTensor)` if a bias exists, `None` otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if GPU buffer creation fails.
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

    /// Uploads the weight matrix to the GPU.
    ///
    /// Convenience wrapper around [`Self::to_gpu_tensor`] with a default label.
    ///
    /// # Arguments
    ///
    /// * `ctx` - A reference-counted pointer to the `WgpuContext`.
    ///
    /// # Returns
    ///
    /// A `GpuTensor` containing the weight data.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not contiguous or GPU buffer creation fails.
    pub fn to_gpu(&self, ctx: &Arc<WgpuContext>) -> Result<GpuTensor> {
        self.to_gpu_tensor(ctx, "to_gpu")
    }

    /// Converts the linear layer's weight matrix to a `GpuTensor`.
    ///
    /// Efficiently handles all CPU data types (F32, BF16, quantized) and uploads
    /// them to the GPU's VRAM. Designed to be as zero-copy as possible on the
    /// CPU side before the GPU transfer.
    ///
    /// # Arguments
    ///
    /// * `ctx` - A reference-counted pointer to the `WgpuContext`.
    /// * `label` - A descriptive label for the GPU buffer (useful for debugging).
    ///
    /// # Returns
    ///
    /// A `GpuTensor` containing the weight data.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * The tensor is not contiguous in memory.
    /// * GPU buffer creation fails.
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
                let raw_tensor = TensorView {
                    name: label.to_string(),
                    bytes: Cow::Borrowed(bytemuck::cast_slice(&matrix.blocks)),
                    shape: matrix.shape.to_vec(),
                    dtype: DType::Q6_K,
                };
                GpuTensor::from_raw(ctx, &raw_tensor, label)
            }
        }
    }
}

/// Converts an F32 weight matrix into a `LinearLayer` with no bias.
///
/// Uses [`F32MatmulStrategy::CustomSimd`] as the default strategy.
impl From<Array2<f32>> for LinearLayer {
    fn from(weights: Array2<f32>) -> Self {
        LinearLayer::new_f32(weights, None)
    }
}

/// Converts an F32 weight matrix and bias into a `LinearLayer`.
///
/// Uses [`F32MatmulStrategy::CustomSimd`] as the default strategy.
impl From<(Array2<f32>, Array1<f32>)> for LinearLayer {
    fn from((weights, bias): (Array2<f32>, Array1<f32>)) -> Self {
        LinearLayer::new_f32(weights, Some(bias))
    }
}

/// Converts a BF16 weight matrix into a `LinearLayer` with no bias.
impl From<Array2<bf16>> for LinearLayer {
    fn from(weights: Array2<bf16>) -> Self {
        LinearLayer::new_bf16(weights, None)
    }
}

/// Converts a BF16 weight matrix and bias into a `LinearLayer`.
impl From<(Array2<bf16>, Array1<f32>)> for LinearLayer {
    fn from((weights, bias): (Array2<bf16>, Array1<f32>)) -> Self {
        LinearLayer::new_bf16(weights, Some(bias))
    }
}


#[cfg(test)]
mod matmul_speed_test {
    use super::*;
    use ndarray::{Array2, Array1};
    use std::time::{Instant, Duration};

    // ------------------------------------------------------------------
    // BENCHMARK SETTINGS (MiniLM Batch=120)
    // ------------------------------------------------------------------
    const BATCH: usize = 120;
    const SEQ_LEN: usize = 32;
    const M: usize = BATCH * SEQ_LEN; // 3840 rows (Tokens)
    const K: usize = 384;             // In Features (Hidden Dim)
    const N: usize = 1536;            // Out Features (Intermediate Dim)
    const ITERATIONS: u32 = 100;

    fn get_input() -> Array2<f32> {
        Array2::from_shape_fn((M, K), |(i, j)| ((i + j) % 100) as f32 / 100.0)
    }

    fn get_weights_standard() -> Array2<f32> {
        // [Out, In] -> This is how Safetensors/PyTorch saves them
        Array2::from_shape_fn((N, K), |(i, j)| ((i * j) % 100) as f32 / 100.0)
    }

    fn get_weights_transposed() -> Array2<f32> {
        // [In, Out] -> Simulating a load-time transpose
        Array2::from_shape_fn((K, N), |(i, j)| ((i * j) % 100) as f32 / 100.0)
    }

    fn run_benchmark(strategy: F32MatmulStrategy, weights: Array2<f32>, name: &str) {
        let input = get_input();
        let layer = LinearLayer::new_f32_with_strategy(weights, None, strategy);

        // Warmup
        let _ = layer.matmul(&input.view());

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let output = layer.matmul(&input.view());
            std::hint::black_box(output);
        }
        let duration = start.elapsed();
        
        // GFLOPS Calculation: 2 * M * N * K
        let total_ops = 2.0 * M as f64 * N as f64 * K as f64 * ITERATIONS as f64;
        let gflops = total_ops / (duration.as_secs_f64() * 1e9);

        println!(
            "Strategy: {:<20} | Latency: {:>8.2?} | Speed: {:>6.2} GFLOPS",
            name, duration / ITERATIONS, gflops
        );
    }

    // 1. Current Implementation (Standard Weights, Manual SIMD)
    #[test]
    fn bench_1_custom_simd() {
        let weights = get_weights_standard(); // [N, K]
        run_benchmark(F32MatmulStrategy::CustomSimd, weights, "Custom Simd");
    }

    // 2. Proposed Fix (Standard Weights, Faer with .transpose())
    #[test]
    fn bench_2_faer_out_in() {
        let weights = get_weights_standard(); // [N, K]
        run_benchmark(F32MatmulStrategy::FaerOutIn, weights, "Faer (Virtual Transpose)");
    }

    // 3. Alternative Fix (Pre-Transposed Weights, Normal Faer)
    #[test]
    fn bench_3_faer_normal() {
        let weights = get_weights_transposed(); // [K, N]
        // This simulates doing the transpose ONCE during model loading
        run_benchmark(F32MatmulStrategy::Faer, weights, "Faer (Load Transpose)");
    }
}