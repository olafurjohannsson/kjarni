use crate::cache::{Cache, CpuBeamKVCache};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ndarray::{Array2, Array3};

use crate::encoder_decoder::{
    traits::{CpuCrossAttentionKVCache, CpuCrossDecoder, CpuEncoderDecoderOps}, EncoderDecoderGenerationBackend,
    EncoderDecoderLanguageModel,
};

#[derive(Debug)]
pub enum CpuSeq2SeqState {
    U32(Array2<u32>),
    EncoderState {
        /// The final hidden states from the encoder, broadcasted for beam search.
        hidden_states: Array3<f32>,
        /// The pre-computed cross-attention Key/Value cache for each decoder layer.
        cross_attention_kv_cache: CpuCrossAttentionKVCache,
        /// Identifies padding in the source sentence
        encoder_padding_mask: Array2<f32>,
    },
}

pub struct CpuBackend;

#[async_trait]
impl EncoderDecoderGenerationBackend for CpuBackend {
    type Tensor = CpuSeq2SeqState;

    async fn encode(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        tokens: &[u32],
        num_beams: usize,
    ) -> Result<Self::Tensor> {
        let seq2seq_ops = model
            .encoder_decoder_cpu_ops()
            .ok_or_else(|| anyhow!("Model does not support CPU execution"))?;
        let encoder_ops = model
            .encoder_cpu_ops()
            .ok_or_else(|| anyhow!("Model does not support CPU execution"))?;

        let input_ids = Array2::from_shape_vec((1, tokens.len()), tokens.to_vec())?;

        // Encoder padding mask
        let attention_mask = Array2::ones(input_ids.dim());

        let encoder_output = encoder_ops
            .encoder()
            .forward(&input_ids, &attention_mask, None)?
            .last_hidden_state;

        // let final_state = if num_beams > 1 {
        //     seq2seq_ops.broadcast_encoder_states(&encoder_output, num_beams)?
        // } else {
        //     encoder_output
        // };
        let (final_state, final_mask) = if num_beams > 1 {
            let s = seq2seq_ops.broadcast_encoder_states(&encoder_output, num_beams)?;
            let m = attention_mask.broadcast((num_beams, tokens.len()))
                .ok_or_else(|| anyhow!("Mask broadcast failed"))?.to_owned();
            (s, m)
        } else {
            (encoder_output, attention_mask)
        };

        let cross_cache = seq2seq_ops
            .decoder()
            .precompute_cross_attention_kv(&final_state)?;

        Ok(CpuSeq2SeqState::EncoderState {
            hidden_states: final_state,
            cross_attention_kv_cache: cross_cache,
            encoder_padding_mask: final_mask,
        })
    }

    async fn decode_step(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        decoder_tokens: &Self::Tensor,
        encoder_state: &Self::Tensor,
        cache: &mut dyn Cache,
    ) -> Result<Array3<f32>> {
        let ops = model
            .encoder_decoder_cpu_ops()
            .ok_or_else(|| anyhow!("Model does not support CPU execution"))?;

        let CpuSeq2SeqState::U32(tokens) = decoder_tokens else {
            return Err(anyhow!("Invalid tensor type for decoder_tokens"));
        };

        let CpuSeq2SeqState::EncoderState {
            hidden_states: enc_state,
            cross_attention_kv_cache: cross_kv,
            encoder_padding_mask,
        } = encoder_state
        else {
            return Err(anyhow!("Invalid tensor type for encoder_state"));
        };

        // create decoder padding mask, usually all 1s during auto regressive decode
        let attention_mask = Array2::ones(tokens.dim());

        // let decoder_output: CpuCrossDecoderOutput = ops.decoder().forward(
        //     tokens,
        //     enc_state,
        //     Some(&attention_mask),
        //     Some(cache),
        //     Some(cross_kv),
        // )?;
        let decoder_output = ops.decoder().forward2(
            tokens,
            enc_state,
            Some(&attention_mask),
            Some(encoder_padding_mask), // Pass the source mask here!
            Some(cache),
            Some(cross_kv),
        )?;

        let cpu_cache = cache
            .as_any_mut()
            .downcast_mut::<CpuBeamKVCache>()
            .ok_or_else(|| anyhow!("Expected CpuBeamKVCache"))?;
        for (i, (k, v)) in decoder_output.new_self_attn_kv.into_iter().enumerate() {
            cpu_cache.update(i, &k, &v)?;
        }
        let logits = ops.project_to_logits(&decoder_output.last_hidden_state)?;

        Ok(logits)
    }

    fn create_token_tensor(&self, tokens: &[u32], num_beams: usize) -> Result<Self::Tensor> {
        let seq_len = if num_beams > 0 {
            tokens.len() / num_beams
        } else {
            0
        };
        let tokens_ndarray = Array2::from_shape_vec((num_beams, seq_len), tokens.to_vec())?;
        Ok(CpuSeq2SeqState::U32(tokens_ndarray))
    }

    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_tokens: &[u32]) -> Result<()> {
        let current_tensor = match tensor {
            CpuSeq2SeqState::U32(t) => t,
            _ => {
                return Err(anyhow!(
                    "Invalid tensor type for update_token_tensor, expected U32"
                ));
            }
        };
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
