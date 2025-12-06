use anyhow::Result;
use async_trait::async_trait;
use bytemuck;
use edgetransformers::cache::{Cache, GpuBeamKVCache};
use edgetransformers::gpu_context::WgpuContext;
use edgetransformers::gpu_ops::GpuTensor;
use edgetransformers::models::base::EncoderDecoderLanguageModel;
use edgetransformers::prelude::*;
use edgetransformers::traits::EncoderOutput;
use ndarray::{Array1, Array2, Array3};
use std::sync::Arc;

use edgetransformers::encoder_decoder::{GenerationBackend, StepInput};

pub struct GpuBackend {
    pub context: Arc<WgpuContext>,
    // Removed pool - use shared pool from context!
}

impl GpuBackend {
    pub fn new(context: Arc<WgpuContext>) -> Result<Self> {
        Ok(Self { context })
    }
}

#[async_trait(?Send)]
impl GenerationBackend for GpuBackend {
    type Cache = GpuBeamKVCache;
    type Tensor = GpuTensor;

    async fn forward<'a>(
        &'a self,
        model: &'a dyn EncoderDecoderLanguageModel,
        inputs: StepInput<'a, Self::Tensor>,
        cache: &'a mut dyn Cache,
    ) -> Result<Array3<f32>> {
        // If you need the pool for GPU operations, get it from context:
        // let pool = self.context.get_inference_pool();
        // let pool_guard = pool.lock().await;
        // let mut frame = GpuFrameContext::new(&self.context, pool_guard);
        // let (encoder, pool) = frame.resources();

        let gpu_decoder = model.gpu_decoder();
        let decoder_output = gpu_decoder
            .forward(
                inputs.tokens,
                inputs.encoder_state.unwrap(),
                None,
                Some(inputs.attention_mask),
                Some(cache),
                None,
            )
            .await?;

        Ok(decoder_output.last_hidden_state)
    }

    fn create_token_tensor(&self, tokens: &[u32], num_beams: usize) -> Result<Self::Tensor> {
        let tokens_ndarray =
            Array2::from_shape_vec((num_beams, tokens.len() / num_beams), tokens.to_vec())?;
        GpuTensor::from_ndarray(&self.context, &tokens_ndarray)
    }

    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_tokens: &[u32]) -> Result<()> {
        let new_tokens_bytes: &[u8] = bytemuck::cast_slice(new_tokens);
        self.context
            .queue
            .write_buffer(tensor.buffer(), 0, new_tokens_bytes);
        Ok(())
    }

    fn prepare_encoder_state(&self, _model: &dyn EncoderDecoderLanguageModel, encoder_output: &EncoderOutput) -> Result<Self::Tensor> {
        GpuTensor::from_ndarray(&self.context, &encoder_output.last_hidden_state)
    }

    fn prepare_attention_mask(&self, seq_len: usize, num_beams: usize) -> Result<Self::Tensor> {
        let mask_cpu: Array2<f32> = Array2::ones((num_beams, seq_len));
        GpuTensor::from_ndarray(&self.context, &mask_cpu)
    }

    fn reorder_cache(&self, cache: &mut dyn Cache, indices: &[usize]) -> Result<()> {
        let gpu_cache = cache.as_any_mut().downcast_mut::<GpuBeamKVCache>().unwrap();
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