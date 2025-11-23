use anyhow::{Result, anyhow};
use async_trait::async_trait;
use edgetransformers::cache::Cache;
use edgetransformers::models::DecoderLanguageModel;
use edgetransformers::models::base::AutoregressiveLoop;
use ndarray::{Array1, Array2, s};

use crate::generation::generator::DecoderGenerationBackend;

/// A generation backend that runs the model entirely on the CPU using ndarray.
pub struct CpuDecoderBackend;

#[async_trait(?Send)]
impl DecoderGenerationBackend for CpuDecoderBackend {
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

    fn prepare_attention_mask(&self, _seq_len: usize, _max_len: usize) -> Result<Self::Tensor> {
        Ok(Array2::zeros((0, 0)))
    }

    async fn prefill<'a>(
        &'a self,
        model: &'a dyn DecoderLanguageModel,
        initial_tokens: &[u32],
        cache: &'a mut dyn Cache,
    ) -> Result<Array1<f32>> {
        if initial_tokens.is_empty() {
            return Err(anyhow!("Cannot prefill with empty prompt."));
        }

        let decoder = model.decoder();
        let prompt_len = initial_tokens.len();
        let input_ids = self.prime_tokens(initial_tokens)?;

        match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => {
                // Llama: one pass
                let attention_mask = Array2::ones((1, prompt_len));
                let decoder_output = decoder
                    .forward(&input_ids, &attention_mask, Some(cache))
                    .await?;
                let logits_3d = model.project_to_logits(&decoder_output.last_hidden_state)?;
                Ok(logits_3d.slice(s![0, -1, ..]).to_owned())
            }
            AutoregressiveLoop::Legacy => {
                // GPT-2: process all, then reprocess last

                // Step 1: Fill cache with all tokens
                let attention_mask = Array2::ones((1, prompt_len));
                decoder
                    .forward(&input_ids, &attention_mask, Some(cache))
                    .await?;
                // Cache now has 11 tokens

                // Step 2: Reprocess LAST prompt token (mimics OLD code first iteration)
                let last_token = initial_tokens[prompt_len - 1];
                let mut single_token = self.new_token_tensor()?;
                self.update_token_tensor(&mut single_token, last_token)?;

                // Mask length = prompt_len + 1 (same as OLD code: current_len + 1)
                let attention_mask = Array2::ones((1, prompt_len + 1));

                let decoder_output = decoder
                    .forward(&single_token, &attention_mask, Some(cache))
                    .await?;
                // Cache now has 12 tokens (p10 processed twice)

                let logits_3d = model.project_to_logits(&decoder_output.last_hidden_state)?;
                Ok(logits_3d.slice(s![0, 0, ..]).to_owned())
            }
        }
    }

    async fn decode_one<'a>(
        &'a self,
        model: &'a dyn DecoderLanguageModel,
        token_id: u32,
        seq_len: usize, // tokens.len() AFTER push
        cache: &'a mut dyn Cache,
    ) -> Result<Array1<f32>> {
        let mut input_tensor = self.new_token_tensor()?;
        self.update_token_tensor(&mut input_tensor, token_id)?;

        // Both modes: mask = seq_len + 1
        // This matches OLD: current_len (before cache update) + 1
        let attention_mask = Array2::ones((1, seq_len + 1));

        let decoder_output = model
            .decoder()
            .forward(&input_tensor, &attention_mask, Some(cache))
            .await?;
        let logits_3d = model.project_to_logits(&decoder_output.last_hidden_state)?;

        Ok(logits_3d.slice(s![0, 0, ..]).to_owned())
    }
}
