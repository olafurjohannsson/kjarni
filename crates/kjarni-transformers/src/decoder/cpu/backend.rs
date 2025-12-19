use crate::cache::Cache;
use crate::decoder::prelude::*;
use crate::models::base::AutoregressiveLoop;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ndarray::{s, Array1, Array2};
use std::time::Instant;

/// A generation backend that runs the model entirely on the CPU using ndarray.
///
/// This backend acts as a controller. It does not know how to multiply matrices
/// or create masks; it delegates those tasks to the `CpuDecoderOps` provided
/// by the model.
pub struct CpuDecoderBackend;

#[async_trait(?Send)]
impl DecoderGenerationBackend for CpuDecoderBackend {
    // On CPU, we store tokens as a simple 2D Array [Batch, Seq]
    type Tensor = Array2<u32>;

    fn prime_tokens(&self, tokens: &[u32]) -> Result<Self::Tensor> {
        Array2::from_shape_vec((1, tokens.len()), tokens.to_vec()).map_err(|e| anyhow!(e))
    }

    fn new_token_tensor(&self) -> Result<Self::Tensor> {
        Ok(Array2::zeros((1, 1)))
    }

    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_token_id: u32) -> Result<()> {
        tensor[[0, 0]] = new_token_id;
        Ok(())
    }

    async fn prefill(
        &self,
        model: &dyn DecoderLanguageModel,
        initial_tokens: &[u32],
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        if initial_tokens.is_empty() {
            return Err(anyhow!("Cannot prefill with empty prompt."));
        }

        // 1. Get the CPU Operations Strategy
        let ops = model
            .decoder_cpu_ops()
            .ok_or_else(|| anyhow!("Model does not support CPU execution"))?;

        let prompt_len = initial_tokens.len();
        let t_start = Instant::now();

        match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => {
                // --- Modern Approach (Llama, Mistral, Phi) ---
                // We process the entire prompt in one go.

                // 2. Ask Model for Mask
                let attention_mask = ops.get_attention_mask(prompt_len, 0)?;

                // 3. Forward Pass
                let decoder_output = ops.decoder().forward(
                    DecoderInput::TokensCpu(initial_tokens),
                    &attention_mask,
                    0, // Position offset is 0 for prefill
                    Some(cache),
                )?;
                log::info!("[CPU] Prefill Forward: {:?}", t_start.elapsed());

                // 4. Project Hidden States -> Logits
                // The model knows if it needs Norm, Bias, or simple MatMul.
                let logits_3d = ops.project_to_logits(&decoder_output)?;

                // 5. Slice the last token's logits
                // [1, Seq, Vocab] -> [Vocab] (for the last position)
                Ok(logits_3d.slice(s![0, -1, ..]).to_owned())
            }
            AutoregressiveLoop::Legacy => {
                // --- Legacy Approach (GPT-2) ---
                // Process prompt to fill cache, then re-process last token to get logits.
                // TODO: Refactor GPT-2 attention to support Pipelined and remove this path.

                // Step A: Fill cache (Forward all tokens)
                // let mask_full = ops.get_attention_mask(prompt_len, 0)?;
                let mask_full = Array2::ones((1, prompt_len));
                ops.decoder().forward(
                    DecoderInput::TokensCpu(initial_tokens),
                    &mask_full,
                    0,
                    Some(cache),
                )?;

                // Step B: Reprocess LAST token to get prediction
                let last_token = initial_tokens[prompt_len - 1];
                let last_token_slice = [last_token];

                // Note: Legacy masking usually implies checking past_len
                // let mask_step = ops.get_attention_mask(1, prompt_len)?;
                let mask_step = ops.get_attention_mask(1, prompt_len)?;

                let decoder_output = ops.decoder().forward(
                    DecoderInput::TokensCpu(&last_token_slice),
                    &mask_step,
                    prompt_len, // Offset
                    Some(cache),
                )?;

                let logits_3d = ops.project_to_logits(&decoder_output)?;
                Ok(logits_3d.slice(s![0, 0, ..]).to_owned())
            }
        }
    }

    async fn decode_one(
        &self,
        model: &dyn DecoderLanguageModel,
        // token_id: u32,
        token_tensor: &Self::Tensor,
        seq_len: usize, // This is Total Length (Past + 1)
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        // 1. Get Ops
        let ops = model
            .decoder_cpu_ops()
            .ok_or_else(|| anyhow!("Model does not support CPU execution"))?;

        // 2. Prepare Input (Single Token)
        // let token_slice = [token_id];
        let token_id = token_tensor[[0, 0]];
        let token_slice = [token_id];
        let input = DecoderInput::TokensCpu(&token_slice);

        // 3. Prepare Mask
        // The model decides the mask shape based on sequence length.
        // For standard Causal, this is [1, Total_Len].
        // For Sliding Window, this handles the windowing.
        // let past_len = seq_len - 1;
        let mask_len = match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => seq_len,     // Llama: Matches Total Length
            AutoregressiveLoop::Legacy => seq_len + 1,    // GPT-2: Legacy +1
        };
        
        // Construct mask.
        // We pass `seq_len=1` and `past_len=mask_len-1` to `get_attention_mask`
        // so it creates a mask of total size `mask_len`.
        let attention_mask = ops.get_attention_mask(1, mask_len - 1)?;

        let past_len = seq_len - 1; // Offset is always based on actual sequence length

        let decoder_output = ops.decoder().forward(
            input,
            &attention_mask,
            past_len,
            Some(cache),
        )?;
        // log::debug!("[CPU] Decode Forward: {:?}", t_forward.elapsed());

        // 5. Project
        let logits_3d = ops.project_to_logits(&decoder_output)?;

        // 6. Return 1D Logits [Vocab]
        Ok(logits_3d.slice(s![0, 0, ..]).to_owned())
    }
}