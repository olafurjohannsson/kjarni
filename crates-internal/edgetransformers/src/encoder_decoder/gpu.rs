use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3};
use std::sync::Arc;

use super::cpu::{DecoderConfigAdapter, EncoderConfigAdapter};
use crate::Cache;
use crate::decoder::TransformerDecoder;
use crate::encoder::TransformerEncoder;
use crate::gpu_context::WgpuContext;
use crate::traits::{
    CrossAttentionDecoder, DecoderOutput, Device, EncoderDecoderArchitecture, TransformerModel,
    EncoderOutput as EncoderOutputTrait
};
use crate::weights::ModelWeights; // Import adapters from the cpu module

/// The GPU backend implementation for a generic `TransformerEncoderDecoder`.
pub struct GpuTransformerEncoderDecoder {
    encoder: TransformerEncoder,
    // In the future, this will be a stack of GPU-specific layers
    // decoder: GpuCrossAttentionDecoder,
    config: Arc<dyn EncoderDecoderArchitecture + Send + Sync>,
    context: Arc<WgpuContext>,
}

impl GpuTransformerEncoderDecoder {
    pub fn new(weights: &ModelWeights, config: Arc<dyn EncoderDecoderArchitecture + Send + Sync>, context: Arc<WgpuContext>) -> Result<Self>
    {
        // Use the same adapter pattern as the CPU implementation
        let encoder_config_adapter = Arc::new(EncoderConfigAdapter(config.clone()));
        let decoder_config_adapter = Arc::new(DecoderConfigAdapter(config.clone()));

        let encoder = TransformerEncoder::new(
            weights,
            encoder_config_adapter,
            Device::Wgpu,
            Some(context.clone()),
        )?;
        let decoder = TransformerDecoder::new(
            weights,
            decoder_config_adapter,
            Device::Wgpu,
            Some(context.clone()),
            None,
        )?;

        Ok(Self {
            encoder,
            config,
            context,
        })
    }
}

#[async_trait]
impl CrossAttentionDecoder for GpuTransformerEncoderDecoder {
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
        todo!("Implement GPU pipeline for cross-attention decoding.");
    }
}

impl TransformerModel for GpuTransformerEncoderDecoder {
    fn device(&self) -> Device {
        Device::Wgpu
    }
}
