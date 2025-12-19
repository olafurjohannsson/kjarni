use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

use super::cpu::{DecoderConfigAdapter, EncoderConfigAdapter};
use crate::encoder::GpuTransformerEncoder;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::GpuTensor;
use crate::traits::{
    DecoderOutput, Device, EncoderDecoderArchitecture
    , TransformerModel,
};
use crate::encoder_decoder::traits::{EncoderDecoderLanguageModel, GpuCrossDecoder};
use crate::weights_old::ModelWeights;
use crate::Cache;
// Import adapters from the cpu module

// The GPU backend implementation for a generic `TransformerEncoderDecoder`.
// pub struct GpuTransformerEncoderDecoder {
//     encoder: GpuTransformerEncoder,
//     pub decoder: GpuCrossAttentionDecoder,
//     config: Arc<dyn EncoderDecoderArchitecture + Send + Sync>,
//     context: Arc<WgpuContext>,
// }

// impl GpuTransformerEncoderDecoder {
//     pub fn decoder(&self) -> &GpuCrossAttentionDecoder {
//         &self.decoder
//     }
//     pub fn new(
//         weights: &ModelWeights,
//         config: Arc<dyn EncoderDecoderArchitecture + Send + Sync>,
//         context: Arc<WgpuContext>,
//     ) -> Result<Self> {
//         // Use the same adapter pattern as the CPU implementation
//         let encoder_config_adapter = Arc::new(EncoderConfigAdapter(config.clone()));
//         let decoder_config_adapter = Arc::new(DecoderConfigAdapter(config.clone()));

//         let encoder = GpuTransformerEncoder::new(
//             weights,
//             encoder_config_adapter,
//             context.clone(),
//         )?;
//         let decoder = GpuCrossAttentionDecoder::new(&context, weights, decoder_config_adapter)?;

//         Ok(Self {
//             encoder,
//             decoder, // Store the new decoder
//             config,
//             context,
//         })
//     }
//     pub fn encoder(&self) -> &GpuTransformerEncoder {
//         &self.encoder
//     }
// }

// #[async_trait(?Send)]
// impl CrossAttentionDecoder for GpuTransformerEncoderDecoder {
//     type TokenInput = GpuTensor;
//     type EncoderStateInput = GpuTensor;
//     type MaskInput = GpuTensor;
//     type Output = DecoderOutput;
    

//     async fn forward<'a>(
//         &self,
//         decoder_input_ids: &Self::TokenInput,
//         encoder_hidden_states: &'a Self::EncoderStateInput,
//         encoder_attention_mask: Option<&'a Self::MaskInput>,
//         decoder_attention_mask: Option<&'a Self::MaskInput>,
//         cache: Option<&mut dyn Cache>,
//         // NEW: Optional pre-computed Cross KV
//         // Vector of tuples (K, V) matching the layers
//         cross_kv_caches: Option<&Vec<(ndarray::Array4<f32>, ndarray::Array4<f32>)>>, 
//     ) -> Result<Self::Output> {
//         self.decoder.forward(
//             decoder_input_ids,
//             encoder_hidden_states,
//             encoder_attention_mask,
//             decoder_attention_mask,
//             cache,
//             cross_kv_caches,
//         ).await
//     }
// }

// impl TransformerModel for GpuTransformerEncoderDecoder {
//     fn device(&self) -> Device {
//         Device::Wgpu
//     }
//     fn context(&self) -> Option<Arc<WgpuContext>> {
//         Some(self.context.clone())
//     }
//     fn as_any(&self) -> &dyn std::any::Any {
//         self
//     }
// }
