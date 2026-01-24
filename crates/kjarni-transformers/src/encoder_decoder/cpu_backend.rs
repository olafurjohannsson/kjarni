use core::panic;

use crate::cache::{Cache, CpuBeamKVCache};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3};

use crate::encoder_decoder::{
    EncoderDecoderGenerationBackend, EncoderDecoderLanguageModel,
    traits::{CpuCrossAttentionKVCache, CpuCrossDecoder, CpuEncoderDecoderOps},
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

#[derive(Debug)]
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

        let hidden_states = encoder_ops
            .embed_tokens(&input_ids, None, 0)?;

        let encoder_output = encoder_ops
            .encoder()
            .forward(&hidden_states, &attention_mask)?
            .last_hidden_state;

        let normed = encoder_ops.encoder().final_norm(&encoder_output)?;
        // println!("=== ENCODER DEBUG ===");
        // println!("Encoder shape: {:?}", normed.dim());
        // println!(
        //     "Encoder [0,0,:10]: {:?}",
        //     normed.slice(ndarray::s![0, 0, ..10])
        // );
        // println!("Encoder mean: {:?}", normed.mean());
        // let final_state = if num_beams > 1 {
        //     seq2seq_ops.broadcast_encoder_states(&encoder_output, num_beams)?
        // } else {
        //     encoder_output
        // };
        let (final_state, final_mask) = if num_beams > 1 {
            let s = seq2seq_ops.broadcast_encoder_states(&normed, num_beams)?;
            let m = attention_mask
                .broadcast((num_beams, tokens.len()))
                .ok_or_else(|| anyhow!("Mask broadcast failed"))?
                .to_owned();
            (s, m)
        } else {
            (normed, attention_mask)
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
        // println!("=== DECODE STEP ===");
        // println!("Decoder input tokens: {:?}", tokens);

        let CpuSeq2SeqState::EncoderState {
            hidden_states: enc_state,
            cross_attention_kv_cache: cross_kv,
            encoder_padding_mask,
        } = encoder_state
        else {
            return Err(anyhow!("Invalid tensor type for encoder_state"));
        };

        // let position_offset = cache.get_seq_length();
        // println!("Position offset (cache length): {}", position_offset);

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

        let hidden = &decoder_output.last_hidden_state;
    
        // println!("Decoder hidden state shape: {:?}", hidden.dim());
        // println!("Decoder hidden state [0,0,:10]: {:?}", hidden.slice(ndarray::s![0, 0, ..10]));

        // let dodo = decoder_output.last_hidden_state;

        //         println!("=== decoder_output DEBUG ===");
        // println!("decoder_output shape: {:?}", dodo.dim());
        // println!("decoder_output [0,0,:10]: {:?}", dodo.slice(ndarray::s![0, 0, ..10]));
        // println!("decoder_output mean: {:?}", dodo.mean());

        let cpu_cache = cache
            .as_any_mut()
            .downcast_mut::<CpuBeamKVCache>()
            .ok_or_else(|| anyhow!("Expected CpuBeamKVCache"))?;
        for (i, (k, v)) in decoder_output.new_self_attn_kv.into_iter().enumerate() {
            cpu_cache.update(i, &k, &v)?;
        }
        let logits = ops.project_to_logits(&decoder_output.last_hidden_state)?;

        // println!("Logits shape: {:?}", logits.dim());


        let last_logits = logits.slice(ndarray::s![0, -1, ..]);
        let mut indexed: Vec<(usize, f32)> = last_logits.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        // println!("Top 5 token IDs: {:?}", indexed[..5].iter().map(|(i, _)| i).collect::<Vec<_>>());
        // println!("Top 5 logits: {:?}", indexed[..5].iter().map(|(_, v)| v).collect::<Vec<_>>());



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
