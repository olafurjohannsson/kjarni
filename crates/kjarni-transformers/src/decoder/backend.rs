//! Type-erased backend that dispatches to CPU or GPU implementations.
//!
//! This module provides `AnyDecoderBackend`, an enum that wraps both
//! `CpuDecoderBackend` and `GpuDecoderBackend` behind a unified interface.
//!
//! # Why Type Erasure?
//!
//! The `DecoderGenerator` needs to work with both CPU and GPU models without
//! knowing at compile time which backend will be used. Rust's type system
//! requires us to handle this somehow:
//!
//! | Approach | Pros | Cons |
//! |----------|------|------|
//! | Generics `<B: Backend>` | Zero-cost | Type must be known at compile time |
//! | `Box<dyn Backend>` | Runtime flexibility | Associated types don't work well |
//! | **Enum dispatch** | Runtime flexibility + type safety | Match arms for each variant |
//!
//! We use enum dispatch because the `Backend` trait has an associated type
//! (`Tensor`) that differs between CPU (`Array2<u32>`) and GPU (`GpuTensor`).
//!
//! # Tensor Type Erasure
//!
//! The `Tensor` associated type is erased to `Box<dyn Any + Send + Sync>`.
//! At runtime, we downcast back to the concrete type:
//!
//! ```text
//! User calls: backend.decode_one(&model, &tensor, ...)
//!                                        │
//!                                        ▼
//!                              Box<dyn Any + Send + Sync>
//!                                        │
//!                         ┌──────────────┴──────────────┐
//!                         ▼                              ▼
//!                 CPU: downcast to              GPU: downcast to
//!                     Array2<u32>                   GpuTensor
//! ```
//!
//! # Performance Impact
//!
//! The type erasure adds minimal overhead:
//! - One enum match per call (branch prediction handles this)
//! - One downcast check per call (single pointer comparison)
//! - No heap allocation in the hot path (tensor is pre-allocated)

use crate::cache::Cache;
use crate::decoder::prelude::*;
use crate::gpu_ops::GpuTensor;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ndarray::{Array1, Array2};
use std::any::Any;
use std::sync::Arc;

/// Unified backend that dispatches to CPU or GPU implementations.
///
/// This enum allows `DecoderGenerator` to work with either backend without
/// knowing which one at compile time. The correct variant is selected based
/// on the model's device configuration.
///
/// # Example
///
/// ```ignore
/// let backend = match model.device() {
///     Device::Cpu => AnyDecoderBackend::Cpu(CpuDecoderBackend),
///     Device::Wgpu => AnyDecoderBackend::Gpu(GpuDecoderBackend::new(context)?),
/// };
///
/// // Now use backend uniformly
/// let logits = backend.prefill(&model, &tokens, &mut cache).await?;
/// ```
///
/// # Type Safety
///
/// While the tensor type is erased, mismatches are caught at runtime with
/// clear error messages. A mismatch would indicate a bug in the generator
/// (mixing CPU tensors with GPU backend or vice versa).
#[derive(Clone)]
pub enum AnyDecoderBackend {
    /// CPU backend using ndarray
    Cpu(CpuDecoderBackend),
    /// GPU backend using WebGPU/Vulkan
    Gpu(Arc<GpuDecoderBackend>),
}

impl AnyDecoderBackend {
    /// Returns whether this is a CPU backend.
    pub fn is_cpu(&self) -> bool {
        matches!(self, AnyDecoderBackend::Cpu(_))
    }

    /// Returns whether this is a GPU backend.
    pub fn is_gpu(&self) -> bool {
        matches!(self, AnyDecoderBackend::Gpu(_))
    }
}

#[async_trait]
impl DecoderGenerationBackend for AnyDecoderBackend {
    /// Type-erased tensor that can hold either CPU or GPU tensors.
    ///
    /// - CPU: Contains `Array2<u32>`
    /// - GPU: Contains `GpuTensor`
    ///
    /// The concrete type is recovered via `downcast_ref`/`downcast_mut`.
    type Tensor = Box<dyn Any + Send + Sync>;

    /// Creates a token tensor from prompt token IDs.
    ///
    /// Delegates to the underlying backend and boxes the result.
    fn prime_tokens(&self, tokens: &[u32]) -> Result<Self::Tensor> {
        match self {
            AnyDecoderBackend::Cpu(backend) => {
                let tensor = backend.prime_tokens(tokens)?;
                Ok(Box::new(tensor))
            }
            AnyDecoderBackend::Gpu(backend) => {
                let tensor = backend.prime_tokens(tokens)?;
                Ok(Box::new(tensor))
            }
        }
    }

    /// Allocates a single-token tensor for the decode loop.
    fn new_token_tensor(&self) -> Result<Self::Tensor> {
        match self {
            AnyDecoderBackend::Cpu(backend) => {
                let tensor = backend.new_token_tensor()?;
                Ok(Box::new(tensor))
            }
            AnyDecoderBackend::Gpu(backend) => {
                let tensor = backend.new_token_tensor()?;
                Ok(Box::new(tensor))
            }
        }
    }

    /// Updates the token tensor with a newly sampled token.
    ///
    /// Downcasts the type-erased tensor to the concrete type expected
    /// by the underlying backend.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor type doesn't match the backend
    /// (indicates a bug in the calling code).
    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_token_id: u32) -> Result<()> {
        match self {
            AnyDecoderBackend::Cpu(backend) => {
                let concrete = tensor
                    .downcast_mut::<Array2<u32>>()
                    .ok_or_else(|| anyhow!(
                        "Type mismatch: CPU backend expected Array2<u32>, got different type. \
                         This is a bug - tensor was created by wrong backend."
                    ))?;
                backend.update_token_tensor(concrete, new_token_id)
            }
            AnyDecoderBackend::Gpu(backend) => {
                let concrete = tensor
                    .downcast_mut::<GpuTensor>()
                    .ok_or_else(|| anyhow!(
                        "Type mismatch: GPU backend expected GpuTensor, got different type. \
                         This is a bug - tensor was created by wrong backend."
                    ))?;
                backend.update_token_tensor(concrete, new_token_id)
            }
        }
    }

    /// Processes the prompt and returns logits.
    ///
    /// Delegates directly to the underlying backend (no tensor handling needed).
    async fn prefill(
        &self,
        model: &dyn DecoderLanguageModel,
        initial_tokens: &[u32],
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        match self {
            AnyDecoderBackend::Cpu(backend) => {
                backend.prefill(model, initial_tokens, cache).await
            }
            AnyDecoderBackend::Gpu(backend) => {
                backend.prefill(model, initial_tokens, cache).await
            }
        }
    }

    /// Processes a single token and returns logits.
    ///
    /// Downcasts the tensor and delegates to the underlying backend.
    async fn decode_one(
        &self,
        model: &dyn DecoderLanguageModel,
        token_tensor: &Self::Tensor,
        seq_len: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        match self {
            AnyDecoderBackend::Cpu(backend) => {
                let concrete = token_tensor
                    .downcast_ref::<Array2<u32>>()
                    .ok_or_else(|| anyhow!(
                        "Type mismatch: CPU backend expected Array2<u32>"
                    ))?;
                backend.decode_one(model, concrete, seq_len, cache).await
            }
            AnyDecoderBackend::Gpu(backend) => {
                let concrete = token_tensor
                    .downcast_ref::<GpuTensor>()
                    .ok_or_else(|| anyhow!(
                        "Type mismatch: GPU backend expected GpuTensor"
                    ))?;
                backend.decode_one(model, concrete, seq_len, cache).await
            }
        }
    }
}