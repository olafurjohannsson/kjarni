use anyhow::{anyhow, Result};
use async_trait::async_trait;
use bytemuck;
use edgetransformers::cache::{Cache, CpuBeamKVCache};
// use edgetransformers::encoder_decoder::CpuTransformerEncoderDecoder;
// use edgetransformers::models::base::EncoderDecoderLanguageModel;
use edgetransformers::prelude::*;
use edgetransformers::traits::EncoderOutput;
use ndarray::{Array2, Array3, Array4};

use edgetransformers::encoder_decoder::{
    EncoderDecoderGenerationBackend, EncoderDecoderLanguageModel, HasShape, StepInput,
};

/// An enum to wrap different ndarray tensor types to satisfy the GenerationBackend trait,
/// which requires a single associated `Tensor` type.
#[derive(Debug)]
pub enum CpuTensor {
    U32(Array2<u32>),
    EncoderState {
        state: Array3<f32>,
        cross_cache: Vec<(Array4<f32>, Array4<f32>)>,
    },
}

// impl HasShape for CpuTensor {
//     fn shape(&self) -> &[usize] {
//         match self {
//             CpuTensor::U32(a) => a.shape(),
//             CpuTensor::EncoderState { state, .. } => state.shape(),
//         }
//     }
// }

pub struct CpuBackend;

#[async_trait(?Send)]
impl EncoderDecoderGenerationBackend for CpuBackend {
    type Tensor = CpuTensor;

    async fn encode(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        tokens: &[u32],
        num_beams: usize,
    ) -> Result<Self::Tensor> {
        log::info!("[CpuBackend] Encoding {} tokens...", tokens.len());
        let t_start = std::time::Instant::now();

        let cpu_encoder = model.cpu_encoder()?;

        let input_ids = Array2::from_shape_vec((1, tokens.len()), tokens.to_vec())?;
        let attention_mask = Array2::ones(input_ids.dim());
        let encoder_output = cpu_encoder
            .forward(&input_ids, &attention_mask, None)?;
        // --- DEBUG LOG ---
        let hidden = &encoder_output.last_hidden_state;
        let slice = hidden.slice(ndarray::s![0, 0, 0..10]);
        log::error!("CHECKPOINT 2 - Encoder Hidden: {:?}", slice);
        // -----------------

        let original_state = &encoder_output.last_hidden_state;
        let state = if num_beams > 1 {
            original_state
                .broadcast((
                    num_beams,
                    original_state.shape()[1],
                    original_state.shape()[2],
                ))
                .ok_or_else(|| anyhow!("Failed to broadcast encoder state"))?
                .to_owned()
        } else {
            original_state.clone()
        };

        let cpu_decoder = model.cpu_decoder()?;
        let cross_cache = cpu_decoder.precompute_cross_attention_kv(&state)?;

        log::info!("[CpuBackend] Encoding finished in {:?}", t_start.elapsed());
        Ok(CpuTensor::EncoderState { state, cross_cache })
    }

    async fn decode_step(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        decoder_tokens: &Self::Tensor,
        encoder_state: &Self::Tensor,
        cache: &mut dyn Cache,
    ) -> Result<Array3<f32>> {
        let t_start = std::time::Instant::now();

        let cpu_decoder = model.cpu_decoder()?;
        let lm_head = model.lm_head_layer();
        let logits_bias = model.final_logits_bias();
        let tokens = match decoder_tokens {
            CpuTensor::U32(t) => t,
            _ => return Err(anyhow!("Invalid tensor type for decoder_tokens")),
        };
        let (enc_state, cross_kv) = match encoder_state {
            CpuTensor::EncoderState { state, cross_cache } => (state, Some(cross_cache)),
            _ => return Err(anyhow!("Invalid tensor type for encoder_state")),
        };


        let attention_mask = Array2::ones(tokens.dim());
        let decoder_output = cpu_decoder
            .forward(
                tokens,
                enc_state,
                None, //Some(&position_ids),
                Some(&attention_mask),
                Some(cache),
                cross_kv,
            )
            .await?;
        let hidden_states = decoder_output.last_hidden_state;
        let (batch, seq, hidden) = hidden_states.dim();
        let hidden_2d = hidden_states.view().into_shape_with_order((batch * seq, hidden))?;
        let mut logits_2d = lm_head.matmul(&hidden_2d);
        if let Some(bias) = logits_bias {
            logits_2d += bias;
        }
        let result = logits_2d.into_shape_with_order((batch, seq, model.vocab_size())).map_err(|e| anyhow!(e));
        log::info!("[CpuBackend] Decode step finished in {:?}", t_start.elapsed());
        result
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

    fn reorder_cache(&self, cache: &mut dyn Cache, indices: &[usize]) -> Result<()> {
        let cpu_cache = cache
            .as_any_mut()
            .downcast_mut::<CpuBeamKVCache>()
            .ok_or_else(|| anyhow!("CpuBackend requires a CpuBeamKVCache"))?;
        cpu_cache.reorder(indices);
        Ok(())
    }
}