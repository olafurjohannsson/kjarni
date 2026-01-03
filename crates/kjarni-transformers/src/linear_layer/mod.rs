//! CPU-based linear transformation layers with multi-dtype support.
//!
//! This module provides [`LinearLayer`], the fundamental building block for
//! neural network weight matrices in transformer models. It supports multiple
//! data types (F32, BF16, Q4_K, Q6_K, Q8_0) with automatic dispatch to optimized
//! SIMD kernels, and includes a fluent builder API for loading from model files.
//!
//! # Overview
//!
//! Key components:
//! - [`LinearLayer`] — Core struct wrapping weight/bias data
//! - [`LinearLayerBuilder`] — Fluent API for loading from [`ModelWeights`](crate::weights::ModelWeights)
//! - [`LinearData`] — Enum storing weights in various data types
//! - [`F32MatmulStrategy`] — Backend selection for F32 matrix multiplication
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::linear_layer::LinearLayer;
//! use kjarni_transformers::weights::ModelWeights;
//! use ndarray::Array2;
//!
//! // Load from model weights
//! let weights = ModelWeights::load("model.safetensors")?;
//! let layer = LinearLayer::builder(&weights, "model.layers.0.mlp.gate_proj")
//!     .with_bias("model.layers.0.mlp.gate_proj.bias")
//!     .build()?;
//!
//! // Forward pass
//! let input = Array2::<f32>::zeros((batch_size, hidden_dim));
//! let output = layer.matmul(&input.view());
//! ```
//!
//! # Performance
//!
//! - **F32**: ~50 GFLOPS via `faer` or custom AVX2/FMA kernels
//! - **BF16**: ~40 GFLOPS mixed-precision with F32 accumulation
//! - **Q4_K**: ~30 GFLOPS 4-bit quantized with on-the-fly dequantization
//! - **Q8_0**: ~45 GFLOPS 8-bit quantized with optimized kernels
//!
//! # See Also
//!
//! - [`crate::weights::ModelWeights`] — Loading weights from disk
//! - [`crate::tensor::DType`] — Supported data types
//! - [`crate::kernels`] — Low-level SIMD kernels

mod linear_layer;
mod builder;

pub use linear_layer::{LinearLayer, LinearData, F32MatmulStrategy};
pub use builder::LinearLayerBuilder;