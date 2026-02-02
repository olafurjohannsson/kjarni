//! GPU-accelerated decoder generation backend.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array1, Array2, s};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, MapMode};

use crate::cache::{Cache, GpuKVCache};
use crate::decoder::prelude::*;
use crate::gpu_ops::primitives::layout::slice_last_token::GpuLastTokenSlice;
use crate::gpu_ops::timeout::{GpuTimeoutConfig, GpuTimeoutError};
use crate::gpu_ops::{GpuFrameContext, GpuTensor};
pub use crate::models::base::AutoregressiveLoop;
use crate::models::base::ModelInput;
use crate::WgpuContext;

pub struct GpuDecoderBackend {
    context: Arc<WgpuContext>,
    last_token_slicer: GpuLastTokenSlice,
    staging_buffer: std::sync::Mutex<Option<Buffer>>,
    staging_buffer_size: std::sync::Mutex<usize>,
    timeout_config: GpuTimeoutConfig,
}

impl GpuDecoderBackend {
    pub fn new(context: Arc<WgpuContext>) -> Result<Self> {
        Ok(Self {
            last_token_slicer: GpuLastTokenSlice::new(&context),
            context,
            staging_buffer: std::sync::Mutex::new(None),
            staging_buffer_size: std::sync::Mutex::new(0),
            timeout_config: GpuTimeoutConfig::default(),
        })
    }

    pub fn with_timeout(
        context: Arc<WgpuContext>,
        timeout_config: GpuTimeoutConfig,
    ) -> Result<Self> {
        Ok(Self {
            last_token_slicer: GpuLastTokenSlice::new(&context),
            context,
            staging_buffer: std::sync::Mutex::new(None),
            staging_buffer_size: std::sync::Mutex::new(0),
            timeout_config,
        })
    }

    pub fn context(&self) -> &Arc<WgpuContext> {
        &self.context
    }

    async fn forward_and_project(
        &self,
        model: &dyn DecoderLanguageModel,
        input: ModelInput<'_>,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        let input_len = match &input {
            ModelInput::TokensCpu(t) => t.shape()[1],
            ModelInput::TokensGpu(t) => t.shape()[1],
            ModelInput::HiddenGpu(t) => t.shape()[1],
            ModelInput::HiddenCpu(t) => t.shape()[1],
        };

        let ops = model
            .decoder_gpu_ops()
            .ok_or_else(|| anyhow!("model does not support gpu execution"))?;

        let gpu_cache = cache
            .as_any_mut()
            .downcast_mut::<GpuKVCache>()
            .ok_or_else(|| anyhow!("expected GpuKVCache for gpu backend"))?;

        let pool = self.context.get_inference_pool();
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);

        let cache_len = gpu_cache.get_seq_length();
        let logical_key_len = cache_len + input_len;

        let mask_size = match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => logical_key_len,
            AutoregressiveLoop::Legacy => gpu_cache.max_seq_len(),
        };

        let attention_mask = ops.get_attention_mask(&mut frame, logical_key_len, mask_size)?;

        let (encoder, pool) = frame.resources();

        let hidden_states = ops.decoder().forward(
            encoder,
            pool,
            input,
            &attention_mask,
            cache_len,
            Some(gpu_cache),
            None,
        )?;

        let (batch, _, hidden_dim) = hidden_states.dims3();
        let last_hidden = frame.pool_guard.get(vec![batch, 1, hidden_dim]);

        let (encoder, _pool) = frame.resources();
        self.last_token_slicer
            .encode(encoder, &hidden_states, &last_hidden);

        let logits = ops.project_to_logits(&mut frame, &last_hidden)?;

        let size = logits.buffer().size();
        let staging = self.get_or_create_staging_buffer(size as usize);

        let (encoder, _pool) = frame.resources();
        encoder.copy_buffer_to_buffer(logits.buffer(), 0, &staging, 0, size);

        frame.finish();

        gpu_cache.increment_len(input_len);

        self.read_logits_from_staging(&staging).await
    }

    async fn forward_only(
        &self,
        model: &dyn DecoderLanguageModel,
        input: ModelInput<'_>,
        cache: &mut dyn Cache,
    ) -> Result<()> {
        let input_len = match &input {
            ModelInput::TokensCpu(t) => t.shape()[1],
            ModelInput::TokensGpu(t) => t.shape()[1],
            ModelInput::HiddenGpu(t) => t.shape()[1],
            ModelInput::HiddenCpu(t) => t.shape()[1],
        };

        let ops = model
            .decoder_gpu_ops()
            .ok_or_else(|| anyhow!("model does not support gpu execution"))?;

        let gpu_cache = cache
            .as_any_mut()
            .downcast_mut::<GpuKVCache>()
            .ok_or_else(|| anyhow!("expected GpuKVCache for gpu backend"))?;

        let pool = self.context.get_inference_pool();
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);

        let cache_len = gpu_cache.get_seq_length();
        let logical_key_len = cache_len + input_len;

        let mask_size = match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => logical_key_len,
            AutoregressiveLoop::Legacy => gpu_cache.max_seq_len(),
        };

        let attention_mask = ops.get_attention_mask(&mut frame, logical_key_len, mask_size)?;

        let (encoder, pool) = frame.resources();
        let _ = ops.decoder().forward(
            encoder,
            pool,
            input,
            &attention_mask,
            cache_len,
            Some(gpu_cache),
            None,
        )?;

        frame.finish();
        gpu_cache.increment_len(input_len);

        Ok(())
    }

    fn get_or_create_staging_buffer(&self, required_bytes: usize) -> Buffer {
        let mut buffer_guard = self.staging_buffer.lock().unwrap();
        let mut size_guard = self.staging_buffer_size.lock().unwrap();

        if buffer_guard.is_some() && *size_guard == required_bytes {
            return buffer_guard.as_ref().unwrap().clone();
        }

        let _ = buffer_guard.take();

        let new_buffer = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("logits staging buffer"),
            size: required_bytes as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        *buffer_guard = Some(new_buffer.clone());
        *size_guard = required_bytes;
        new_buffer
    }

    async fn read_logits_from_staging(&self, buffer: &Buffer) -> Result<Array1<f32>> {
        let slice = buffer.slice(..);

        let (tx, mut rx) = tokio::sync::oneshot::channel();
        slice.map_async(MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        let start = std::time::Instant::now();
        let mut map_result = None;

        loop {
            match self.context.device.poll(wgpu::PollType::Poll) {
                Ok(_) => {}
                Err(e) => {
                    return Err(anyhow!("gpu poll failed: {:?}", e));
                }
            }

            match rx.try_recv() {
                Ok(result) => {
                    map_result = Some(result);
                    break;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {}
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    return Err(anyhow!("buffer mapping channel closed unexpectedly"));
                }
            }

            let elapsed = start.elapsed();
            if elapsed >= self.timeout_config.timeout {
                return Err(GpuTimeoutError {
                    operation: "buffer_map_read".to_string(),
                    elapsed,
                    timeout: self.timeout_config.timeout,
                }
                .into());
            }

            tokio::time::sleep(self.timeout_config.poll_interval).await;
        }

        map_result
            .ok_or_else(|| anyhow!("buffer mapping did not complete"))?
            .map_err(|e| anyhow!("buffer mapping failed: {:?}", e))?;

        let data = slice.get_mapped_range();
        let vec_data: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        buffer.unmap();

        Ok(Array1::from_vec(vec_data))
    }
}

#[async_trait]
impl DecoderGenerationBackend for GpuDecoderBackend {
    type DecodeToken = GpuTensor;

    fn new_decode_token(&self) -> Result<Self::DecodeToken> {
        let token_ndarray = Array2::<u32>::zeros((1, 1));
        GpuTensor::from_ndarray(&self.context, &token_ndarray)
    }

    fn update_decode_token(&self, tensor: &mut Self::DecodeToken, new_token_id: u32) -> Result<()> {
        self.context
            .queue
            .write_buffer(tensor.buffer(), 0, bytemuck::cast_slice(&[new_token_id]));
        Ok(())
    }

    async fn prefill(
        &self,
        model: &dyn DecoderLanguageModel,
        tokens: &Array2<u32>,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        if tokens.is_empty() {
            return Err(anyhow!("cannot prefill with empty tokens"));
        }

        let prompt_len = tokens.shape()[1];

        log::debug!(
            "gpu prefill: {} tokens, strategy={:?}",
            prompt_len,
            model.autoregressive_loop()
        );

        match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => {
                let input = ModelInput::TokensCpu(tokens.view());
                self.forward_and_project(model, input, cache).await
            }
            AutoregressiveLoop::Legacy => {
                let last_idx = prompt_len - 1;

                // stuff cache with context tokens (no logits needed)
                if last_idx > 0 {
                    let context_tokens = tokens.slice(s![.., 0..last_idx]);
                    let context_input = ModelInput::TokensCpu(context_tokens);
                    self.forward_only(model, context_input, cache).await?;
                }

                // get logits from last token only
                let last_token = tokens.slice(s![.., last_idx..]);
                let last_input = ModelInput::TokensCpu(last_token);
                self.forward_and_project(model, last_input, cache).await
            }
        }
    }

    async fn decode_one(
        &self,
        model: &dyn DecoderLanguageModel,
        token_tensor: &Self::DecodeToken,
        _seq_len: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        let input = ModelInput::TokensGpu(token_tensor);
        self.forward_and_project(model, input, cache).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_input_size_extraction() {
        let tokens = Array2::from_shape_vec((1, 5), vec![1u32, 2, 3, 4, 5]).unwrap();
        let input = ModelInput::TokensCpu(tokens.view());

        let size = match &input {
            ModelInput::TokensCpu(t) => t.shape()[1],
            _ => 0,
        };

        assert_eq!(size, 5);
    }
}