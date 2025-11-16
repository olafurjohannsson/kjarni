//! A generic, backend-agnostic transformer encoder-decoder implementation.
//!
//! This module provides `TransformerEncoderDecoder`, a reusable component that can represent
//! various seq2seq models like BART, T5, etc. It is designed to be
//! backend-aware, containing either a CPU or a GPU implementation.
//!
//! The model is constructed generically by relying on the `EncoderDecoderArchitecture`
//! trait, which provides the specific weight names and hyperparameters.

mod cpu;
mod gpu;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3};
use std::sync::Arc;

use crate::gpu_context::WgpuContext;
use crate::traits::{
    CrossAttentionDecoder, DecoderOutput, Device, EncoderDecoderArchitecture, EncoderOutput,
    TransformerModel,
};
use crate::weights::ModelWeights;
pub use crate::{Cache, CpuKVCache, GpuKVCache};
use cpu::CpuTransformerEncoderDecoder;
use gpu::GpuTransformerEncoderDecoder;
use crate::Encoder;

/// A generic, backend-agnostic transformer encoder-decoder stack.
pub enum TransformerEncoderDecoder {
    Cpu(CpuTransformerEncoderDecoder),
    Gpu(GpuTransformerEncoderDecoder),
}

impl TransformerEncoderDecoder {
    pub fn new(
        weights: &ModelWeights,
        config: Arc<dyn EncoderDecoderArchitecture + Send + Sync>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        match device {
            Device::Cpu => Ok(Self::Cpu(CpuTransformerEncoderDecoder::new(
                weights, config,
            )?)),
            Device::Wgpu => {
                let ctx = context.ok_or_else(|| {
                    anyhow!("A WGPU context is required for a GPU-based encoder-decoder.")
                })?;
                Ok(Self::Gpu(GpuTransformerEncoderDecoder::new(
                    weights, config, ctx,
                )?))
            }
        }
    }
    pub fn encoder(&self) -> &dyn Encoder<Input = Array2<u32>, Output = EncoderOutput> {
        match self {
            Self::Cpu(model) => model.encoder(),
            Self::Gpu(model) => model.encoder()
        }
    }

    pub fn decoder(
        &self,
    ) -> &dyn CrossAttentionDecoder<Input = Array2<u32>, Output = DecoderOutput> {
        match self {
            Self::Cpu(model) => model, // The CpuTransformerEncoderDecoder itself implements the trait
            Self::Gpu(model) => model,
        }
    }
}

// Implement the main forward pass via the CrossAttentionDecoder trait
#[async_trait(?Send)]
impl CrossAttentionDecoder for TransformerEncoderDecoder {
    type Input = Array2<u32>;
    type Output = DecoderOutput;

    async fn forward<'a>(
        &self,
        decoder_input_ids: &Self::Input,
        encoder_hidden_states: &'a Array3<f32>,
        encoder_attention_mask: Option<&'a Array2<f32>>,
        decoder_attention_mask: Option<&'a Array2<f32>>,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Self::Output> {
        match self {
            Self::Cpu(model) => {
                model
                    .forward(
                        decoder_input_ids,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        decoder_attention_mask,
                        cache,
                    )
                    .await
            }
            Self::Gpu(model) => {
                model
                    .forward(
                        decoder_input_ids,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        decoder_attention_mask,
                        cache,
                    )
                    .await
            }
        }
    }
}

impl TransformerModel for TransformerEncoderDecoder {
    fn device(&self) -> Device {
        match self {
            Self::Cpu(model) => model.device(),
            Self::Gpu(model) => model.device(),
        }
    }
}
