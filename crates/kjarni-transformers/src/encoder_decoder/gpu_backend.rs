use crate::cache::{Cache, GpuBeamKVCache};
use crate::encoder::prelude::*;
use crate::encoder_decoder::traits::{
    EncoderDecoderGenerationBackend, EncoderDecoderLanguageModel,
};
use crate::encoder_decoder::traits::{GpuCrossAttentionKVCache, GpuCrossDecoderOutput};
use crate::gpu_ops::GpuFrameContext;
use crate::gpu_ops::GpuTensor;
use crate::gpu_ops::Kernel;
use crate::models::base::ModelInput;
use crate::prelude::*;
use crate::WgpuContext;
use anyhow::anyhow;
use anyhow::Result;
use async_trait::async_trait;
use bytemuck;
use ndarray::{Array1, Array2, Array3};
use std::sync::Arc;
pub struct GpuBackend {
    pub context: Arc<WgpuContext>,
}

impl GpuBackend {
    pub fn new(context: Arc<WgpuContext>) -> Result<Self> {
        Ok(Self { context })
    }
}

#[derive(Debug)]
pub enum GpuSeq2SeqState {
    TokenIds(GpuTensor),
    EncoderOutput {
        hidden_states: GpuTensor,
        cross_attention_kv_cache: GpuCrossAttentionKVCache,
    },
}

impl EncoderDecoderGenerationBackend for GpuBackend {
    type Tensor = GpuSeq2SeqState;

    async fn encode(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        tokens: &[u32],
        num_beams: usize,
    ) -> Result<Self::Tensor> {
        log::info!("[GpuBackend] Encoding {} tokens...", tokens.len());
        let t_start = std::time::Instant::now();

        // 1. Get the GPU Operations Strategy from the model
        let seq2seq_ops = model
            .encoder_decoder_gpu_ops()
            .ok_or_else(|| anyhow!("Model does not support GPU execution"))?;

        let encoder_ops = model
            .encoder_gpu_ops()
            .ok_or_else(|| anyhow!("Model does not support GPU execution"))?;

        // 2. Prepare inputs on GPU
        let pool = self.context.get_inference_pool();
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);
        // --- 3. Run the Encoder Pass ---
        let encoder_hidden_states = {
            let (encoder_cmd, pool_ref) = frame.resources();

            let input_ids_cpu = Array2::from_shape_vec((1, tokens.len()), tokens.to_vec())?;
            let input_ids_gpu = GpuTensor::from_ndarray(&self.context, &input_ids_cpu)?;
            let attention_mask_gpu =
                GpuTensor::from_ndarray(&self.context, &Array2::<f32>::ones(input_ids_cpu.dim()))?;

            encoder_ops
                .encoder()
                .forward(
                    encoder_cmd,
                    pool_ref,
                    ModelInput::TokensGpu(&input_ids_gpu),
                    &attention_mask_gpu,
                    None,
                )?
                .last_hidden_state
        };

        // The model's ops are responsible for broadcasting for beam search
        let final_hidden_states = if num_beams > 1 {
            seq2seq_ops.broadcast_encoder_states(&mut frame, &encoder_hidden_states, num_beams)?
        } else {
            encoder_hidden_states
        };

        let cross_attention_kv_cache = {
            let (encoder_cmd, pool_ref) = frame.resources(); // Re-borrow resources from the frame

            // Get the decoder from the ops and ask it to prepare the cache
            seq2seq_ops.decoder().precompute_cross_attention_kv(
                encoder_cmd,
                pool_ref,
                &final_hidden_states, // Use the final, broadcasted states
            )?
        };

        frame.finish();
        log::info!("[GpuBackend] Encoding finished in {:?}", t_start.elapsed());

        Ok(GpuSeq2SeqState::EncoderOutput {
            hidden_states: final_hidden_states,
            cross_attention_kv_cache,
        })
    }

    async fn decode_step(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        decoder_tokens: &Self::Tensor,
        encoder_state: &Self::Tensor,
        cache: &mut dyn Cache,
    ) -> Result<Array3<f32>> {
        let t_start = std::time::Instant::now();

        // 1. Get the GPU Operations Strategy from the model
        let ops = model
            .encoder_decoder_gpu_ops()
            .ok_or_else(|| anyhow!("Model does not support GPU execution"))?;

        // 2. Extract tensors
        let GpuSeq2SeqState::TokenIds(decoder_input_ids) = decoder_tokens else {
            return Err(anyhow!("Invalid tensor type for decoder_tokens"));
        };
        let GpuSeq2SeqState::EncoderOutput {
            hidden_states: encoder_hidden_states,
            cross_attention_kv_cache,
        } = encoder_state
        else {
            return Err(anyhow!("Invalid tensor type for encoder_state"));
        };

        // 3. Prepare for GPU execution
        let pool = self.context.get_inference_pool();
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);
        let (encoder_cmd, pool_ref) = frame.resources();

        // --- REFACTOR: All logic moves to the `ops` trait ---

        // a) Create the decoder attention mask (can also be moved to ops if it gets complex)
        let (batch_size, _) = decoder_input_ids.dims2();
        let seq_len = cache.get_seq_length() + 1;
        let mask_cpu = Array2::<f32>::ones((batch_size, seq_len));
        let attention_mask = GpuTensor::from_ndarray(&self.context, &mask_cpu)?;

        // b) Run the decoder stack
        let decoder_hidden_states: GpuCrossDecoderOutput = ops.decoder().forward(
            encoder_cmd,
            pool_ref,
            ModelInput::TokensGpu(decoder_input_ids),
            encoder_hidden_states,
            &attention_mask,
            Some(cache),
            Some(cross_attention_kv_cache), // Precomputed cross-KV is now an internal detail of the ops
        )?;

        let gpu_cache = cache.as_any_mut().downcast_mut::<GpuBeamKVCache>().unwrap();

        for (i, (k, v)) in decoder_hidden_states
            .new_self_attn_kv
            .into_iter()
            .enumerate()
        {
            // The update command is recorded into the SAME command encoder
            gpu_cache.update(encoder_cmd, i, &k, &v);
        }
        // gpu_cache.increment_len(1);
        // c) Project to logits
        let logits_gpu =
            ops.project_to_logits(&mut frame, &decoder_hidden_states.last_hidden_state)?;

        frame.finish();

        // d) Download the result
        let logits_cpu = logits_gpu.to_ndarray_3d::<f32>().await?;

        log::info!(
            "[GpuBackend] Decode step finished in {:?}",
            t_start.elapsed()
        );
        Ok(logits_cpu)
    }

    fn create_token_tensor(&self, tokens: &[u32], num_beams: usize) -> Result<Self::Tensor> {
        let seq_len = if num_beams > 0 {
            tokens.len() / num_beams
        } else {
            tokens.len()
        };
        let tokens_ndarray = Array2::from_shape_vec((num_beams, seq_len), tokens.to_vec())?;
        let tensor = GpuTensor::from_ndarray(&self.context, &tokens_ndarray)?;
        Ok(GpuSeq2SeqState::TokenIds(tensor))
    }

    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_tokens: &[u32]) -> Result<()> {
        let GpuSeq2SeqState::TokenIds(gpu_tensor) = tensor else {
            return Err(anyhow!("Invalid tensor type for update_token_tensor"));
        };
        let new_tokens_bytes: &[u8] = bytemuck::cast_slice(new_tokens);
        self.context
            .queue
            .write_buffer(gpu_tensor.buffer(), 0, new_tokens_bytes);
        Ok(())
    }

    fn reorder_cache(&self, cache: &mut dyn Cache, indices: &[usize]) -> Result<()> {
        let gpu_cache = cache
            .as_any_mut()
            .downcast_mut::<GpuBeamKVCache>()
            .ok_or_else(|| anyhow!("GpuBackend requires a GpuBeamKVCache"))?;
        let indices_ndarray = Array1::from_vec(indices.iter().map(|&i| i as u32).collect());
        let indices_gpu = GpuTensor::from_ndarray(&self.context, &indices_ndarray)?;
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&Default::default());
        gpu_cache.reorder(&mut encoder, &indices_gpu);
        self.context.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}
