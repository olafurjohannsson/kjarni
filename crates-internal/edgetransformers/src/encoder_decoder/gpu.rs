use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3};
use std::sync::Arc;

use super::cpu::{DecoderConfigAdapter, EncoderConfigAdapter};
use crate::Cache;
use crate::decoder::TransformerDecoder;
use crate::encoder::TransformerEncoder;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::blocks::decoder_cross_attention::{
    GpuCrossAttentionDecoder, GpuCrossAttentionDecoderLayer,
};
use std::any::Any;
use crate::traits::{
    CrossAttentionDecoder, DecoderOutput, Device, EncoderDecoderArchitecture,
    EncoderOutput as EncoderOutputTrait, TransformerModel,
};
use crate::weights::ModelWeights; // Import adapters from the cpu module

/// The GPU backend implementation for a generic `TransformerEncoderDecoder`.
pub struct GpuTransformerEncoderDecoder {
    encoder: TransformerEncoder, // this supports CPU/GPU by dispatch
    pub decoder: GpuCrossAttentionDecoder,
    config: Arc<dyn EncoderDecoderArchitecture + Send + Sync>,
    context: Arc<WgpuContext>,
}

impl GpuTransformerEncoderDecoder {
     pub fn decoder(&self) -> &GpuCrossAttentionDecoder {
        &self.decoder
    }
    pub fn new(
        weights: &ModelWeights,
        config: Arc<dyn EncoderDecoderArchitecture + Send + Sync>,
        context: Arc<WgpuContext>,
    ) -> Result<Self> {
        // Use the same adapter pattern as the CPU implementation
        let encoder_config_adapter = Arc::new(EncoderConfigAdapter(config.clone()));
        let decoder_config_adapter = Arc::new(DecoderConfigAdapter(config.clone()));

        let encoder = TransformerEncoder::new(
            weights,
            encoder_config_adapter,
            Device::Wgpu,
            Some(context.clone()),
        )?;
        let decoder = GpuCrossAttentionDecoder::new(&context, weights, decoder_config_adapter)?;

        Ok(Self {
            encoder,
            decoder, // Store the new decoder
            config,
            context,
        })
    }
    pub fn encoder(&self) -> &TransformerEncoder {
        &self.encoder
    }
}

#[async_trait(?Send)]
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
        self.decoder.forward(
            decoder_input_ids,
            encoder_hidden_states,
            encoder_attention_mask,
            decoder_attention_mask,
            cache,
        ).await
    }
}

impl TransformerModel for GpuTransformerEncoderDecoder {
    fn device(&self) -> Device {
        Device::Wgpu
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        Some(self.context.clone())
    }
fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
