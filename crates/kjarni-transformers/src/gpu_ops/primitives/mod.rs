//! GPU compute primitives for transformer operations.
//!
//! This module provides low-level GPU kernels (compute shaders) for common
//! neural network operations. Each primitive is optimized for WebGPU and
//! supports multiple data types (F32, BF16) where applicable.
//!
//! # Overview
//!
//! Primitives are organized into categories:
//!
//! ## Core Math Operations
//! - [`matmul`] — Matrix multiplication with tiling and dtype dispatch
//! - [`bmm`] — Batched matrix multiplication
//! - [`add`] — Element-wise addition
//! - [`scale`] — Element-wise scaling
//!
//! ## Attention Operations
//! - [`softmax`] — Softmax with optional scaling and padding support
//! - [`apply_mask`] — Attention mask application
//! - [`repeat_kv`] — KV cache repeat for grouped-query attention
//!
//! ## Normalization
//! - [`layer_norm`] — Layer normalization with learnable affine parameters
//!
//! ## Layout Transformations
//! - [`layout`] — Reshape, permute, slice, concatenate operations
//!
//! ## Embeddings & Lookups
//! - [`lookup`] — Embedding table lookup
//! - [`lookup2`] — Alternative lookup implementation
//!
//! ## Specialized
//! - [`linear`] — Fused linear layer (matmul + bias)
//! - [`add_bias`] — Bias addition
//! - [`argmax`] — Find maximum values along dimension
//! - [`tanh`] — Hyperbolic tangent activation
//! - [`broadcast`] — Broadcasting operations
//!
//! # Architecture
//!
//! Each primitive follows a consistent pattern:
//! 1. Implements the [`Kernel`](crate::gpu_ops::Kernel) trait
//! 2. Compiles WGSL shaders at initialization
//! 3. Encodes compute passes to command encoders
//! 4. Supports profiling and validation
//!
//! # Performance
//!
//! - **Tiling**: Matrix operations use cooperative tiling for cache efficiency
//! - **Type specialization**: Separate pipelines for F32/BF16 optimize memory bandwidth
//! - **Uniform arena**: Reduces allocation overhead for per-frame uniforms
//! - **GEMV fast path**: Single-row matmul uses optimized vector-matrix kernel
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::gpu_ops::primitives::matmul::GpuMatMul;
//! use kjarni_transformers::gpu_ops::Kernel;
//!
//! let matmul = GpuMatMul::new(&context);
//! let mut encoder = device.create_command_encoder(&Default::default());
//!
//! // Encode: output = a @ b^T
//! matmul.encode(&mut encoder, &[&a, &b], &output);
//! ```
//!
//! # See Also
//!
//! - [`crate::gpu_ops::Kernel`] — Base trait for GPU operations
//! - [`crate::gpu_ops::GpuTensor`] — GPU tensor storage
//! - [`crate::WgpuContext`] — WebGPU device context

pub mod add;
pub mod add_bias;
pub mod apply_mask;
// pub mod reshape;
pub mod softmax;
pub mod matmul;
pub mod bmm;
pub mod layout;
pub mod scale;
pub mod lookup;
pub mod lookup2;
pub mod repeat_kv;
pub mod argmax;
pub mod linear;
pub mod broadcast;
pub mod tanh;