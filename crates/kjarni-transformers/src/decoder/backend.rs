use anyhow::{Result, anyhow};
use crate::cache::Cache;
use crate::common::sampling::{GenerationConfig, apply_repetition_penalty, sample_token};
use crate::common::{StreamedToken, TokenType};
use crate::decoder::prelude::*;
use crate::gpu_ops::GpuTensor;
use crate::prelude::*;
use async_stream::try_stream;
use async_trait::async_trait;
use futures_core::stream::Stream;
use futures_util::TryStreamExt;
use log::{debug, error};
use ndarray::Array1;
use ndarray::Array2;
use std::any::Any;
use std::sync::Arc;

pub enum AnyDecoderBackend {
    Cpu(CpuDecoderBackend),
    Gpu(GpuDecoderBackend),
}

#[async_trait(?Send)]
impl DecoderGenerationBackend for AnyDecoderBackend {
    // The `Tensor` type is a type-erased box that can hold either a CPU or GPU tensor.
    type Tensor = Box<dyn Any + Send + Sync>;

    // --- Memory Management ---

    /// Creates the initial tensor populated with prompt tokens.
    fn prime_tokens(&self, tokens: &[u32]) -> Result<Self::Tensor> {
        match self {
            AnyDecoderBackend::Cpu(backend) => {
                let cpu_tensor = backend.prime_tokens(tokens)?;
                // Box the concrete tensor to fit the type-erased signature.
                Ok(Box::new(cpu_tensor))
            }
            AnyDecoderBackend::Gpu(backend) => {
                let gpu_tensor = backend.prime_tokens(tokens)?;
                Ok(Box::new(gpu_tensor))
            }
        }
    }

    /// Allocates a tensor to hold a single new token.
    fn new_token_tensor(&self) -> Result<Self::Tensor> {
        match self {
            AnyDecoderBackend::Cpu(backend) => {
                let cpu_tensor = backend.new_token_tensor()?;
                Ok(Box::new(cpu_tensor))
            }
            AnyDecoderBackend::Gpu(backend) => {
                let gpu_tensor = backend.new_token_tensor()?;
                Ok(Box::new(gpu_tensor))
            }
        }
    }

    /// Efficiently updates the single-token tensor with the next ID.
    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_token_id: u32) -> Result<()> {
        match self {
            AnyDecoderBackend::Cpu(backend) => {
                // Try to downcast the `Box<dyn Any>` back to the concrete type the backend expects.
                if let Some(concrete_tensor) = tensor.downcast_mut::<Array2<u32>>() {
                    backend.update_token_tensor(concrete_tensor, new_token_id)
                } else {
                    Err(anyhow!(
                        "Mismatched tensor type for CPU backend: expected Array2<u32>"
                    ))
                }
            }
            AnyDecoderBackend::Gpu(backend) => {
                // Similarly, downcast for the GPU backend.
                if let Some(concrete_tensor) = tensor.downcast_mut::<GpuTensor>() {
                    backend.update_token_tensor(concrete_tensor, new_token_id)
                } else {
                    Err(anyhow!(
                        "Mismatched tensor type for GPU backend: expected GpuTensor"
                    ))
                }
            }
        }
    }

    // --- Execution Phase (Unchanged) ---

    /// Processes the prompt tokens to populate the KV Cache.
    async fn prefill(
        &self,
        model: &dyn DecoderLanguageModel,
        initial_tokens: &[u32],
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        match self {
            AnyDecoderBackend::Cpu(backend) => backend.prefill(model, initial_tokens, cache).await,
            AnyDecoderBackend::Gpu(backend) => backend.prefill(model, initial_tokens, cache).await,
        }
    }

    /// Decodes a single step.
    async fn decode_one(
        &self,
        model: &dyn DecoderLanguageModel,
        token_tensor: &Self::Tensor, // This is &Box<dyn Any>
        seq_len: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        match self {
            AnyDecoderBackend::Cpu(backend) => {
                // Try to downcast the Box<dyn Any> to the type the CpuBackend expects.
                if let Some(concrete_tensor) = token_tensor.downcast_ref::<Array2<u32>>() {
                    // If successful, call the backend's method with the correctly-typed tensor.
                    backend
                        .decode_one(model, concrete_tensor, seq_len, cache)
                        .await
                } else {
                    // If the downcast fails, it's a logic error in the program.
                    Err(anyhow!(
                        "Mismatched tensor type for CPU backend: expected Array2<u32>"
                    ))
                }
            }
            AnyDecoderBackend::Gpu(backend) => {
                // Similarly, downcast for the GPU backend.
                if let Some(concrete_tensor) = token_tensor.downcast_ref::<GpuTensor>() {
                    backend
                        .decode_one(model, concrete_tensor, seq_len, cache)
                        .await
                } else {
                    Err(anyhow!(
                        "Mismatched tensor type for GPU backend: expected GpuTensor"
                    ))
                }
            }
        }
    }
}
