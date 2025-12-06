use anyhow::{anyhow, Result};
use async_trait::async_trait;
use bytemuck;
use edgetransformers::cache::{Cache, CpuBeamKVCache};
use edgetransformers::encoder_decoder::CpuTransformerEncoderDecoder;
use edgetransformers::models::base::EncoderDecoderLanguageModel;
use edgetransformers::prelude::*;
use edgetransformers::traits::EncoderOutput;
use ndarray::{Array2, Array3, Array4};

use edgetransformers::encoder_decoder::{GenerationBackend, HasShape, StepInput};

/// An enum to wrap different ndarray tensor types to satisfy the GenerationBackend trait,
/// which requires a single associated `Tensor` type.
#[derive(Debug)]
pub enum CpuTensor {
    U32(Array2<u32>),
    F32_2D(Array2<f32>),
    F32_3D(Array3<f32>),
    EncoderState {
        state: Array3<f32>,
        cross_cache: Vec<(Array4<f32>, Array4<f32>)>,
    },
}

impl HasShape for CpuTensor {
    fn shape(&self) -> &[usize] {
        match self {
            CpuTensor::U32(a) => a.shape(),
            CpuTensor::F32_2D(a) => a.shape(),
            CpuTensor::F32_3D(a) => a.shape(),
            CpuTensor::EncoderState { state, .. } => state.shape(),
        }
    }
}

pub struct CpuBackend;

#[async_trait(?Send)]
impl GenerationBackend for CpuBackend {
    type Cache = CpuBeamKVCache;
    type Tensor = CpuTensor;

    async fn forward<'a>(
        &'a self,
        model: &'a dyn EncoderDecoderLanguageModel,
        inputs: StepInput<'a, Self::Tensor>,
        cache: &'a mut dyn Cache,
    ) -> Result<Array3<f32>> {
        // 1. Downcast the generic inputs to concrete ndarray types
        let tokens = match inputs.tokens {
            CpuTensor::U32(t) => t,
            _ => return Err(anyhow!("Invalid tensor type for tokens, expected U32")),
        };
        // let encoder_state = match inputs.encoder_state.unwrap() {
        //     CpuTensor::F32_3D(s) => s,
        //     _ => return Err(anyhow!("Invalid tensor type for encoder_state, expected F32_3D")),
        // };
        let (encoder_state, cross_cache) = match inputs.encoder_state.unwrap() {
            CpuTensor::EncoderState { state, cross_cache } => (state, Some(cross_cache)),
            _ => return Err(anyhow!("Invalid encoder state tensor")),
        };
        let attention_mask = match inputs.attention_mask {
            CpuTensor::F32_2D(m) => m,
            _ => {
                return Err(anyhow!(
                    "Invalid tensor type for attention_mask, expected F32_2D"
                ));
            }
        };

        // // 2. Call the model's CPU decoder
        let decoder_output = model
            .decoder()
            .forward(
                tokens,
                encoder_state,
                None,
                Some(attention_mask),
                Some(cache),
                cross_cache,
            )
            .await?;

        Ok(decoder_output.last_hidden_state)
    }

    fn create_token_tensor(&self, tokens: &[u32], num_beams: usize) -> Result<Self::Tensor> {
        let seq_len = if num_beams > 0 {
            tokens.len() / num_beams
        } else {
            0
        };
        let tokens_ndarray = Array2::from_shape_vec((num_beams, seq_len), tokens.to_vec())?;
        Ok(CpuTensor::U32(tokens_ndarray))
    }

    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_tokens: &[u32]) -> Result<()> {
        let current_tensor = match tensor {
            CpuTensor::U32(t) => t,
            _ => {
                return Err(anyhow!(
                    "Invalid tensor type for update_token_tensor, expected U32"
                ));
            }
        };
        // The new tokens represent the next single token for each beam.
        let new_tokens_ndarray =
            Array2::from_shape_vec((new_tokens.len(), 1), new_tokens.to_vec())?;
        *current_tensor = new_tokens_ndarray;
        Ok(())
    }

    fn prepare_encoder_state(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        encoder_output: &EncoderOutput,
    ) -> Result<Self::Tensor> {
        // FIX: Access the decoder trait object first.
        // The 'model' is the wrapper (e.g. BartModel), which we cannot downcast to CpuTransformerEncoderDecoder.
        // The 'decoder' IS the CpuTransformerEncoderDecoder (on CPU).
        let decoder = model.decoder();

        // Now downcast the decoder trait object to the concrete CPU struct
        let cpu_decoder = decoder
            .as_any()
            .downcast_ref::<CpuTransformerEncoderDecoder>()
            .ok_or_else(|| {
                anyhow!(
                    "Decoder backend is not CpuTransformerEncoderDecoder. Are you running on GPU?"
                )
            })?;

        let state = encoder_output.last_hidden_state.clone();

        // PRE-COMPUTE CROSS-ATTENTION
        // We iterate over the decoder layers we just accessed
        let mut cross_cache = Vec::with_capacity(cpu_decoder.decoder_layers.len());
        for layer in &cpu_decoder.decoder_layers {
            // Pre-calculate Keys/Values for the entire encoder sequence
            let (k, v) = layer.cross_attn.precompute_encoder_kv(&state)?;
            cross_cache.push((k, v));
        }

        Ok(CpuTensor::EncoderState { state, cross_cache })
    }

    fn prepare_attention_mask(&self, seq_len: usize, num_beams: usize) -> Result<Self::Tensor> {
        let mask: Array2<f32> = Array2::ones((num_beams, seq_len));
        Ok(CpuTensor::F32_2D(mask))
    }

    fn reorder_cache(&self, cache: &mut dyn Cache, indices: &[usize]) -> Result<()> {
        let cpu_cache = cache
            .as_any_mut()
            .downcast_mut::<CpuBeamKVCache>() // <-- Change this to CpuBeamKVCache
            .ok_or_else(|| anyhow!("Failed to downcast to CpuBeamKVCache for reordering"))?;

        // This is now a simple, efficient call.
        cpu_cache.reorder(indices);

        Ok(())
    }
}
