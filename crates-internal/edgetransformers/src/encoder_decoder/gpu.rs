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
        )?;

        // This is a placeholder. In reality, you'd build a GPU-specific decoder stack here
        // similar to the manual construction in `cpu.rs`.
        Ok(Self {
            encoder,
            // decoder, // Placeholder
            config,
            context,
        })
    }
}

#[async_trait]
impl CrossAttentionDecoder for GpuTransformerEncoderDecoder {
    type Input = Array2<f32>;
    type Output = DecoderOutput;

    async fn forward(
        &self,
        _encoder_input_ids: &Self::Input,
        _decoder_input_ids: &Self::Input,
        _encoder_attention_mask: &Array2<f32>,
        _decoder_attention_mask: &Array2<f32>,
        _cache: Option<&mut dyn Cache>,
        _encoder_output_opt: Option<&EncoderOutputTrait>,
    ) -> Result<Self::Output> {
        // The orchestration logic will be the same as the CPU.
        // 1. Run encoder pass on GPU.
        // 2. Run decoder pass with cross-attention using a new `GpuCrossAttentionPipeline`.
        todo!("Implement GPU pipeline for cross-attention decoding.");
    }
}

impl TransformerModel for GpuTransformerEncoderDecoder {
    fn device(&self) -> Device {
        Device::Wgpu
    }
}
