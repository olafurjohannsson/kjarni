//! A generic, backend-agnostic transformer decoder implementation.
//!
//! This module provides `TransformerDecoder`, a reusable component that can represent
//! various decoder-only models like GPT-2, GPT-J, etc. It is designed to be
//! backend-aware, containing either a CPU or GPU implementation.
//!
//! The decoder is constructed generically by relying on the `DecoderArchitecture`
//! trait, which provides the specific weight names and hyperparameters for a
//! given model, allowing for maximum code reuse.

mod cpu;
mod gpu;

use crate::traits::{
    Decoder, DecoderArchitecture, DecoderOutput, Device, TransformerModel,
};
use crate::weights::ModelWeights;
use crate::gpu_context::WgpuContext;
pub use crate::{Cache, CpuKVCache, GpuKVCache};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use cpu::CpuTransformerDecoder;
use gpu::GpuTransformerDecoder;
use ndarray::{Array2, Array3};
use std::sync::Arc;

/// A generic, backend-agnostic transformer decoder stack.
///
/// This enum acts as a container for the backend-specific implementation
/// (e.g., `CpuTransformerDecoder`, `GpuTransformerDecoder`). It dispatches calls to the appropriate
/// backend, providing a single, consistent API for any decoder model.
pub enum TransformerDecoder {
    Cpu(CpuTransformerDecoder),
    Gpu(GpuTransformerDecoder),
}

impl TransformerDecoder {
    /// Creates a new generic `TransformerDecoder` for the specified device.
    ///
    /// This factory function is generic over any configuration `C` that implements
    /// the `DecoderArchitecture` trait. It uses the trait to dynamically load the
    /// correct weights and build the model stack for either the CPU or GPU backend.
    pub fn new(
        weights: &ModelWeights,
        config: Arc<dyn DecoderArchitecture + Send + Sync>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self>
    {
        match device {
            Device::Cpu => Ok(Self::Cpu(CpuTransformerDecoder::new(
                weights,
                config.clone(),
            )?)),
            Device::Wgpu => {
                let ctx = context.ok_or_else(|| {
                    anyhow!("A WGPU context is required to create a GPU-based decoder.")
                })?;
                Ok(Self::Gpu(GpuTransformerDecoder::new(
                    weights,
                    config.clone(),
                    ctx,
                )?))
            }
        }
    }
}

/// Implements the base `Model` trait for the generic decoder, delegating to the backend.
impl TransformerModel for TransformerDecoder {
    fn device(&self) -> Device {
        match self {
            Self::Cpu(model) => model.device(),
            Self::Gpu(model) => model.device(),
        }
    }
}

/// Implements the `Decoder` trait for the generic decoder, delegating to the backend.
#[async_trait]
impl Decoder for TransformerDecoder {
    type Input = Array2<f32>;
    type Output = DecoderOutput;

    // pub async fn forward_cross_attention(
    //     &self,
    //     input: &Self::Input,
    //     decoder_attention_mask: &Array2<f32>,
    //     encoder_output: &EncoderOutput,
    //     encoder_attention_mask: &Array2<f32>,
    //     cache: Option<&mut dyn Cache>,
    // ) -> Result<DecoderOutput> {
    //     match self {
    //         Self::Cpu(model) => model.forward_cross_attention(input, decoder_attention_mask, encoder_output, encoder_attention_mask, cache).await,
    //         Self::Gpu(model) => model.forward_cross_attention(input, decoder_attention_mask, encoder_output, encoder_attention_mask, cache).await,
    //     }
    // }

    async fn forward(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Self::Output> {
        match self {
            Self::Cpu(model) => model.forward(input, attention_mask, cache).await,
            Self::Gpu(model) => model.forward(input, attention_mask, cache).await,
        }
    }
    async fn get_hidden_states(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
    ) -> Result<Array3<f32>> {
        match self {
            Self::Cpu(model) => model.get_hidden_states(input, attention_mask).await,
            Self::Gpu(model) => model.get_hidden_states(input, attention_mask).await,
        }
    }
}
