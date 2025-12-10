//! A generic, backend-agnostic transformer encoder implementation.
//!
//! This module provides `TransformerEncoder`, a reusable component that can represent
//! various encoder-only models like BERT, RoBERTa, etc. It is designed to be
//! backend-aware, containing either a CPU or a (future) GPU implementation.
//!
//! The encoder is constructed generically by relying on the `EncoderArchitecture`
//! trait, which provides the specific weight names and hyperparameters for a

//! given model, allowing for maximum code reuse.

mod cpu;
mod gpu;
pub mod encoder_self_attention;
pub mod traits;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ndarray::{Array2, Array3};
use std::sync::Arc;

use crate::encoder::traits::EncoderArchitecture;
use crate::gpu_context::WgpuContext;
use crate::traits::{Device, Encoder, EncoderOutput, TransformerModel};
use crate::weights::ModelWeights;
use cpu::CpuTransformerEncoder;
use gpu::GpuTransformerEncoder;

pub mod prelude {
    pub use crate::encoder::traits::{
        DType,
        EncoderLanguageModel,
        EncodingConfig,
        GpuClassificationHead,
        GpuEncoder,
        GpuEncoderInput,
        GpuEncoderOutput,
        PoolingStrategy,
    };
}


/// A generic, backend-agnostic transformer encoder stack.
///
/// This enum acts as a container for the backend-specific implementation
/// (e.g., `CpuTransformerEncoder`). It dispatches calls to the appropriate
/// backend, providing a single, consistent API for any encoder model.
pub enum TransformerEncoder {
    Cpu(CpuTransformerEncoder),
    Gpu(GpuTransformerEncoder),
}

impl TransformerEncoder {
    /// Creates a new generic `TransformerEncoder` for the specified device.
    ///
    /// This factory function is generic over any configuration `C` that implements
    /// the `EncoderArchitecture` trait. It uses the trait to dynamically load the
    /// correct weights and build the model stack for either the CPU or GPU backend.
    pub fn new(
        weights: &ModelWeights,
        config: Arc<dyn EncoderArchitecture + Send + Sync>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        match device {
            Device::Cpu => Ok(Self::Cpu(CpuTransformerEncoder::new(
                weights,
                config.clone(),
            )?)),
            Device::Wgpu => {
                let ctx = context.ok_or_else(|| {
                    anyhow!("A WGPU context is required to create a GPU-based encoder.")
                })?;
                Ok(Self::Gpu(GpuTransformerEncoder::new(
                    weights,
                    config.clone(),
                    ctx,
                )?))
            }
        }
    }
    /// Get the hidden size
    pub fn hidden_size(&self) -> usize {
        match self {
            Self::Cpu(model) => model.config().hidden_size(),
            Self::Gpu(model) => model.config().hidden_size(),
        }
    }

    /// Get the max sequence length
    pub fn max_length(&self) -> usize {
        match self {
            Self::Cpu(model) => model.config().max_position_embeddings(),
            Self::Gpu(model) => model.config().max_position_embeddings(),
        }
    }
}

/// Implements the base `TransformerModel` trait for the generic encoder, delegating to the backend.
impl TransformerModel for TransformerEncoder {
    fn device(&self) -> Device {
        match self {
            Self::Cpu(model) => model.device(),
            Self::Gpu(model) => model.device(),
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Implements the `Encoder` trait for the generic encoder, delegating to the backend.
#[async_trait]
impl Encoder for TransformerEncoder {
    type Input = Array2<u32>;
    type Output = EncoderOutput;

    async fn forward(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
        token_type_ids: Option<&Array2<u32>>,
    ) -> Result<Self::Output> {
        match self {
            Self::Cpu(model) => model.forward(input, attention_mask, token_type_ids).await,
            Self::Gpu(model) => model.forward(input, attention_mask, token_type_ids).await,
        }
    }
    async fn get_hidden_states(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
        token_type_ids: Option<&Array2<u32>>,
    ) -> Result<Array3<f32>> {
        match self {
            Self::Cpu(model) => {
                model
                    .get_hidden_states(input, attention_mask, token_type_ids)
                    .await
            }
            Self::Gpu(model) => {
                model
                    .get_hidden_states(input, attention_mask, token_type_ids)
                    .await
            }
        }
    }
}
