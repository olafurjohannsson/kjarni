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
//! (`DecodeToken`) that differs between CPU (`Array2<u32>`) and GPU (`GpuTensor`).
//!
//! # Decode Token Type Erasure
//!
//! The `DecodeToken` associated type is erased to `Box<dyn Any + Send + Sync>`.
//! At runtime, we downcast back to the concrete type:
//!
//! ```text
//! User calls: backend.decode_one(&model, &token, ...)
//!                                         │
//!                                         ▼
//!                              Box<dyn Any + Send + Sync>
//!                                         │
//!                         ┌───────────────┴───────────────┐
//!                         ▼                               ▼
//!                 CPU: downcast to               GPU: downcast to
//!                     Array2<u32>                    GpuTensor
//! ```
//!
//! # Prefill Input
//!
//! Unlike the decode token, prefill always receives `&Array2<u32>` from CPU.
//! This is because tokens always originate from the tokenizer (CPU). The backend
//! handles any necessary upload to GPU internally.
//!
//! # Performance Impact
//!
//! The type erasure adds minimal overhead:
//! - One enum match per call (branch prediction handles this)
//! - One downcast check per call (single pointer comparison)
//! - No heap allocation in the hot path (decode token is pre-allocated)

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
///     Device::Cpu => AnyDecoderBackend::Cpu(CpuDecoderBackend::new()),
///     Device::Wgpu => AnyDecoderBackend::Gpu(Arc::new(GpuDecoderBackend::new(context)?)),
/// };
///
/// // Tokenize prompt to CPU array
/// let prompt_tokens = tokenizer.encode("Hello, world!")?;
/// let tokens = Array2::from_shape_vec((1, prompt_tokens.len()), prompt_tokens)?;
///
/// // Prefill - backend handles upload if needed
/// let logits = backend.prefill(&model, &tokens, &mut cache).await?;
///
/// // Decode loop with type-erased token
/// let mut decode_token = backend.new_decode_token()?;
/// backend.update_decode_token(&mut decode_token, next_id)?;
/// let logits = backend.decode_one(&model, &decode_token, seq_len, &mut cache).await?;
/// ```
///
/// # Type Safety
///
/// While the decode token type is erased, mismatches are caught at runtime with
/// clear error messages. A mismatch would indicate a bug in the generator
/// (mixing CPU tokens with GPU backend or vice versa).
#[derive(Clone)]
pub enum AnyDecoderBackend {
    /// CPU backend using ndarray
    Cpu(CpuDecoderBackend),
    /// GPU backend using WebGPU/Vulkan
    Gpu(Arc<GpuDecoderBackend>),
}

impl AnyDecoderBackend {
    /// Creates a CPU backend.
    pub fn cpu() -> Self {
        AnyDecoderBackend::Cpu(CpuDecoderBackend::new())
    }

    /// Creates a GPU backend.
    ///
    /// # Arguments
    ///
    /// * `backend` - Pre-configured GPU backend (requires WgpuContext)
    pub fn gpu(backend: Arc<GpuDecoderBackend>) -> Self {
        AnyDecoderBackend::Gpu(backend)
    }

    /// Returns whether this is a CPU backend.
    pub fn is_cpu(&self) -> bool {
        matches!(self, AnyDecoderBackend::Cpu(_))
    }

    /// Returns whether this is a GPU backend.
    pub fn is_gpu(&self) -> bool {
        matches!(self, AnyDecoderBackend::Gpu(_))
    }

    /// Returns a string description of the backend type.
    pub fn backend_type(&self) -> &'static str {
        match self {
            AnyDecoderBackend::Cpu(_) => "CPU",
            AnyDecoderBackend::Gpu(_) => "GPU",
        }
    }
}

#[async_trait]
impl DecoderGenerationBackend for AnyDecoderBackend {
    /// Type-erased decode token that can hold either CPU or GPU tensors.
    ///
    /// - CPU: Contains `Array2<u32>`
    /// - GPU: Contains `GpuTensor`
    ///
    /// The concrete type is recovered via `downcast_ref`/`downcast_mut`.
    type DecodeToken = Box<dyn Any + Send + Sync>;

    /// Allocates a single-token tensor for the decode loop.
    ///
    /// The returned token is type-erased but internally holds:
    /// - CPU: `Array2<u32>` of shape `[1, 1]`
    /// - GPU: `GpuTensor` of shape `[1, 1]`
    fn new_decode_token(&self) -> Result<Self::DecodeToken> {
        match self {
            AnyDecoderBackend::Cpu(backend) => {
                let token = backend.new_decode_token()?;
                Ok(Box::new(token))
            }
            AnyDecoderBackend::Gpu(backend) => {
                let token = backend.new_decode_token()?;
                Ok(Box::new(token))
            }
        }
    }

    /// Updates the decode token with a newly sampled token ID.
    ///
    /// Downcasts the type-erased token to the concrete type expected
    /// by the underlying backend.
    ///
    /// # Arguments
    ///
    /// * `token` - Type-erased decode token from `new_decode_token()`
    /// * `new_token_id` - The token ID to write
    ///
    /// # Errors
    ///
    /// Returns an error if the token type doesn't match the backend
    /// (indicates a bug in the calling code).
    fn update_decode_token(
        &self,
        token: &mut Self::DecodeToken,
        new_token_id: u32,
    ) -> Result<()> {
        match self {
            AnyDecoderBackend::Cpu(backend) => {
                let concrete = token.downcast_mut::<Array2<u32>>().ok_or_else(|| {
                    anyhow!(
                        "Type mismatch: CPU backend expected Array2<u32>, got different type. \
                         This is a bug - decode token was created by wrong backend."
                    )
                })?;
                backend.update_decode_token(concrete, new_token_id)
            }
            AnyDecoderBackend::Gpu(backend) => {
                let concrete = token.downcast_mut::<GpuTensor>().ok_or_else(|| {
                    anyhow!(
                        "Type mismatch: GPU backend expected GpuTensor, got different type. \
                         This is a bug - decode token was created by wrong backend."
                    )
                })?;
                backend.update_decode_token(concrete, new_token_id)
            }
        }
    }

    /// Processes the prompt and returns logits for the first generated token.
    ///
    /// The prompt tokens are always provided as a CPU `Array2<u32>`. The backend
    /// handles any necessary upload to GPU internally (for GPU backend).
    ///
    /// # Arguments
    ///
    /// * `model` - The decoder language model
    /// * `tokens` - Prompt token IDs as CPU array of shape `[batch, seq_len]`
    /// * `cache` - KV cache to populate
    ///
    /// # Returns
    ///
    /// Vocabulary logits as 1D array of shape `[vocab_size]`
    async fn prefill(
        &self,
        model: &dyn DecoderLanguageModel,
        tokens: &Array2<u32>,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        match self {
            AnyDecoderBackend::Cpu(backend) => backend.prefill(model, tokens, cache).await,
            AnyDecoderBackend::Gpu(backend) => backend.prefill(model, tokens, cache).await,
        }
    }

    /// Processes a single token and returns logits for the next token.
    ///
    /// Downcasts the type-erased decode token and delegates to the underlying backend.
    ///
    /// # Arguments
    ///
    /// * `model` - The decoder language model
    /// * `token` - Type-erased decode token containing the current token ID
    /// * `seq_len` - Total sequence length so far (prompt + generated)
    /// * `cache` - KV cache with previous keys/values
    ///
    /// # Returns
    ///
    /// Vocabulary logits as 1D array of shape `[vocab_size]`
    async fn decode_one(
        &self,
        model: &dyn DecoderLanguageModel,
        token: &Self::DecodeToken,
        seq_len: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        match self {
            AnyDecoderBackend::Cpu(backend) => {
                let concrete = token.downcast_ref::<Array2<u32>>().ok_or_else(|| {
                    anyhow!(
                        "Type mismatch: CPU backend expected Array2<u32>, got different type. \
                         This is a bug - decode token was created by wrong backend."
                    )
                })?;
                backend.decode_one(model, concrete, seq_len, cache).await
            }
            AnyDecoderBackend::Gpu(backend) => {
                let concrete = token.downcast_ref::<GpuTensor>().ok_or_else(|| {
                    anyhow!(
                        "Type mismatch: GPU backend expected GpuTensor, got different type. \
                         This is a bug - decode token was created by wrong backend."
                    )
                })?;
                backend.decode_one(model, concrete, seq_len, cache).await
            }
        }
    }
}

impl std::fmt::Debug for AnyDecoderBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnyDecoderBackend::Cpu(_) => write!(f, "AnyDecoderBackend::Cpu"),
            AnyDecoderBackend::Gpu(_) => write!(f, "AnyDecoderBackend::Gpu"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::decoder::CpuDecoderBackend;

    #[test]
    fn test_any_backend_cpu_constructor() {
        let backend = AnyDecoderBackend::cpu();
        assert!(backend.is_cpu());
        assert!(!backend.is_gpu());
        assert_eq!(backend.backend_type(), "CPU");
    }

    #[test]
    fn test_any_backend_is_cpu() {
        let cpu = CpuDecoderBackend::new();
        let any = AnyDecoderBackend::Cpu(cpu);

        assert!(any.is_cpu());
        assert!(!any.is_gpu());
    }

    #[test]
    fn test_any_backend_new_decode_token_cpu() {
        let any = AnyDecoderBackend::cpu();

        let token = any.new_decode_token();
        assert!(token.is_ok());

        let t = token.unwrap();
        // Verify internal type is Array2<u32>
        let concrete = t.downcast_ref::<Array2<u32>>();
        assert!(concrete.is_some());
        assert_eq!(concrete.unwrap().shape(), &[1, 1]);
    }

    #[test]
    fn test_any_backend_update_decode_token_cpu() {
        let any = AnyDecoderBackend::cpu();

        let mut token = any.new_decode_token().unwrap();
        let update = any.update_decode_token(&mut token, 123);
        assert!(update.is_ok());

        // Verify value was written
        let concrete = token.downcast_ref::<Array2<u32>>().unwrap();
        assert_eq!(concrete[[0, 0]], 123);
    }

    #[test]
    fn test_any_backend_update_decode_token_multiple_times() {
        let any = AnyDecoderBackend::cpu();
        let mut token = any.new_decode_token().unwrap();

        for i in 0..100 {
            any.update_decode_token(&mut token, i).unwrap();
            let concrete = token.downcast_ref::<Array2<u32>>().unwrap();
            assert_eq!(concrete[[0, 0]], i);
        }
    }

    #[test]
    fn test_decode_token_type_mismatch_error() {
        let any = AnyDecoderBackend::cpu();

        // Create a fake token with wrong type
        let mut fake_token: Box<dyn Any + Send + Sync> = Box::new(String::from("fake"));

        // CPU backend expects Array2<u32>, should fail gracefully
        let result = any.update_decode_token(&mut fake_token, 1);
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Type mismatch"));
        assert!(err_msg.contains("CPU backend expected Array2<u32>"));
    }

    #[test]
    fn test_any_backend_debug_impl() {
        let cpu = AnyDecoderBackend::cpu();
        assert_eq!(format!("{:?}", cpu), "AnyDecoderBackend::Cpu");
    }

    #[test]
    fn test_any_backend_clone() {
        let backend1 = AnyDecoderBackend::cpu();
        let backend2 = backend1.clone();

        assert!(backend1.is_cpu());
        assert!(backend2.is_cpu());

        // Both should work independently
        let token1 = backend1.new_decode_token().unwrap();
        let token2 = backend2.new_decode_token().unwrap();

        let concrete1 = token1.downcast_ref::<Array2<u32>>().unwrap();
        let concrete2 = token2.downcast_ref::<Array2<u32>>().unwrap();
        assert_eq!(concrete1, concrete2);
    }
}