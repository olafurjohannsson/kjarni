use anyhow::anyhow;
use anyhow::Result;
use async_trait::async_trait;
use bytemuck;
use edgetransformers::cache::{Cache, GpuBeamKVCache};
use edgetransformers::encoder::prelude::*;
use edgetransformers::encoder_decoder::traits::{
    EncoderDecoderGenerationBackend, EncoderDecoderLanguageModel, HasShape,
};
use edgetransformers::encoder_decoder::StepInput;
use edgetransformers::gpu_context::WgpuContext;
use edgetransformers::gpu_ops::primitives::add::GpuAdd;
use edgetransformers::gpu_ops::primitives::broadcast::GpuBroadcast;
use edgetransformers::gpu_ops::primitives::linear::GpuLinearLayer;
use edgetransformers::gpu_ops::GpuFrameContext;
use edgetransformers::gpu_ops::GpuTensor;
use edgetransformers::gpu_ops::Kernel;
use edgetransformers::prelude::*;
use edgetransformers::traits::EncoderOutput;
use ndarray::{Array1, Array2, Array3};
use std::sync::Arc;

#[derive(Debug)]
pub enum GpuSeq2SeqTensor {
    TokenIds(GpuTensor),
    EncoderOutput {
        hidden_states: GpuTensor,
        cross_attention_kv_cache: Vec<(GpuTensor, GpuTensor)>,
    },
}


pub struct GpuBackend {
    pub context: Arc<WgpuContext>,
    linear: GpuLinearLayer,
    broadcast: GpuBroadcast,
    add: GpuAdd,
}

impl GpuBackend {
    pub fn new(context: Arc<WgpuContext>) -> Result<Self> {
        Ok(Self {
            context: context.clone(),
            linear: GpuLinearLayer::new(&context),
            broadcast: GpuBroadcast::new(&context)?,
            add: GpuAdd::new(&context),
        })
    }
}

#[async_trait(?Send)]
impl EncoderDecoderGenerationBackend for GpuBackend {
    type Tensor = GpuSeq2SeqTensor;

    async fn encode(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        tokens: &[u32],
        num_beams: usize,
    ) -> Result<Self::Tensor> {
        log::info!("[GpuBackend] Encoding {} tokens...", tokens.len());
        let t_start = std::time::Instant::now();

        let gpu_encoder = model.gpu_encoder()?;
        let gpu_decoder = model.gpu_decoder()?;
        let input_ids_cpu = Array2::from_shape_vec((1, tokens.len()), tokens.to_vec())?;
        let input_ids = GpuTensor::from_ndarray(&self.context, &input_ids_cpu)?;
        let attention_mask =
            GpuTensor::from_ndarray(&self.context, &Array2::<f32>::ones(input_ids_cpu.dim()))?;
        let pool = self.context.get_inference_pool();
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);
        let (encoder_cmd, pool_ref) = frame.resources();

        let encode = gpu_encoder.forward(
            encoder_cmd,
            pool_ref,
            GpuEncoderInput::TokensGpu(&input_ids),
            &attention_mask,
            None,
        )?;
        let encoder_hidden_states = encode.last_hidden_state;

        let final_hidden_states = if num_beams > 1 {
            let mut expanded_shape = encoder_hidden_states.shape().to_vec();
            expanded_shape[0] = num_beams;
            let expanded_states = GpuTensor::uninitialized(
                &self.context,
                expanded_shape,
                encoder_hidden_states.dtype(),
                "expanded_encoder_states",
            );
            self.broadcast
                .encode(encoder_cmd, &encoder_hidden_states, &expanded_states, 0);
            expanded_states
        } else {
            encoder_hidden_states
        };

        let cross_attention_kv_cache = gpu_decoder.precompute_cross_attention_kv(
            encoder_cmd,
            pool_ref,
            &final_hidden_states,
        )?;

        frame.finish();
        log::info!("[GpuBackend] Encoding finished in {:?}", t_start.elapsed());

        Ok(GpuSeq2SeqTensor::EncoderOutput {
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
        let gpu_decoder = model.gpu_decoder()?;
        let lm_head_weights = model.gpu_lm_head_weights()?;
        let logits_bias = model.gpu_final_logits_bias()?;
        let GpuSeq2SeqTensor::TokenIds(decoder_input_ids) = decoder_tokens else {
            return Err(anyhow!("Invalid tensor type for decoder_tokens"));
        };
        let GpuSeq2SeqTensor::EncoderOutput {
            cross_attention_kv_cache,
            ..
        } = encoder_state
        else {
            return Err(anyhow!("Invalid tensor type for encoder_state"));
        };

        let pool = self.context.get_inference_pool();
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);
        let (encoder_cmd, pool_ref) = frame.resources();

        let (batch_size, _) = decoder_input_ids.dims2(); // (num_beams, 1)
        let seq_len = cache.get_seq_length() + 1;

        // During generation, there is no padding, so the mask is all ones.
        let mask_cpu = Array2::<f32>::ones((batch_size, seq_len));
        let attention_mask = GpuTensor::from_ndarray(&self.context, &mask_cpu)?;

        let decoder_hidden_states = gpu_decoder.forward(
            encoder_cmd,
            pool_ref,
            decoder_input_ids,
            None,
            Some(cross_attention_kv_cache),
            &attention_mask, // Pass the correct 2D mask
            cache,
        )?;

        let (batch, seq, hidden) = decoder_hidden_states.dims3();
        let vocab_size = lm_head_weights.shape()[0];
        let logits_shape = vec![batch * seq, vocab_size];
        let logits = pool_ref.get(logits_shape);
        let hidden_states_2d = decoder_hidden_states.view(vec![batch * seq, hidden]);
        self.linear
            .encode(encoder_cmd, &hidden_states_2d, lm_head_weights, &logits);

        let final_logits = if let Some(bias) = logits_bias {
            let logits_with_bias = pool_ref.get(logits.shape().to_vec());
            // UPDATED: Use the new broadcast method
            self.add
                .encode_broadcast_row(encoder_cmd, &logits, bias, &logits_with_bias);
            logits_with_bias
        } else {
            logits
        };

        frame.finish();

        let logits_cpu_2d = final_logits.to_ndarray_2d::<f32>().await?;

        let result = logits_cpu_2d
            .into_shape((batch, seq, model.vocab_size()))
            .map_err(|e| anyhow!(e));
        log::info!(
            "[GpuBackend] Decode step finished in {:?}",
            t_start.elapsed()
        );
        result
    }

    fn create_token_tensor(&self, tokens: &[u32], num_beams: usize) -> Result<Self::Tensor> {
        let seq_len = if num_beams > 0 {
            tokens.len() / num_beams
        } else {
            tokens.len()
        };
        let tokens_ndarray = Array2::from_shape_vec((num_beams, seq_len), tokens.to_vec())?;
        let tensor = GpuTensor::from_ndarray(&self.context, &tokens_ndarray)?;
        Ok(GpuSeq2SeqTensor::TokenIds(tensor))
    }

    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_tokens: &[u32]) -> Result<()> {
        let GpuSeq2SeqTensor::TokenIds(gpu_tensor) = tensor else {
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
