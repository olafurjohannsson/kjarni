//! GPU-accelerated decoder generation backend.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array1, Array2, s};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, MapMode};

use crate::WgpuContext;
use crate::cache::Cache;
use crate::decoder::prelude::*;
use crate::gpu::cache::GpuKVCache;
use crate::gpu::{GpuFrameContext, GpuTensor};
use crate::gpu_ops::primitives::layout::slice_last_token::GpuLastTokenSlice;
use crate::gpu_ops::timeout::{GpuTimeoutConfig, GpuTimeoutError};
pub use crate::models::base::AutoregressiveLoop;
use crate::models::base::ModelInput;

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
    use crate::gpu_ops::timeout::GpuTimeoutConfig;
    use std::time::Duration;
    async fn get_test_context() -> Arc<WgpuContext> {
        WgpuContext::new()
            .await
            .expect("Failed to create WgpuContext")
    }
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
    
    // ========================================================================
    //  Construction Tests
    // ========================================================================

    #[tokio::test]
    async fn test_gpu_decoder_backend_new() {
        let ctx = get_test_context().await;
        let backend = GpuDecoderBackend::new(ctx);
        
        assert!(backend.is_ok());
    }

    #[tokio::test]
    async fn test_gpu_decoder_backend_with_timeout() {
        let ctx = get_test_context().await;
        let timeout_config = GpuTimeoutConfig {
            timeout: Duration::from_secs(30),
            poll_interval: Duration::from_millis(5),
        };
        
        let backend = GpuDecoderBackend::with_timeout(ctx, timeout_config);
        
        assert!(backend.is_ok());
        let backend = backend.unwrap();
        assert_eq!(backend.timeout_config.timeout, Duration::from_secs(30));
        assert_eq!(backend.timeout_config.poll_interval, Duration::from_millis(5));
    }

    #[tokio::test]
    async fn test_gpu_decoder_backend_context_getter() {
        let ctx = get_test_context().await;
        let backend = GpuDecoderBackend::new(ctx.clone()).unwrap();
        
        // Should return the same context
        assert!(Arc::ptr_eq(&ctx, backend.context()));
    }

    // ========================================================================
    //  new_decode_token Tests
    // ========================================================================

    #[tokio::test]
    async fn test_new_decode_token() {
        let ctx = get_test_context().await;
        let backend = GpuDecoderBackend::new(ctx).unwrap();
        
        let token = backend.new_decode_token();
        
        assert!(token.is_ok());
        let token = token.unwrap();
        assert_eq!(token.shape(), &[1, 1]);
    }

    #[tokio::test]
    async fn test_new_decode_token_initial_value() {
        let ctx = get_test_context().await;
        let backend = GpuDecoderBackend::new(ctx).unwrap();
        
        let token = backend.new_decode_token().unwrap();
        
        // Download and verify it's initialized to zero
        let downloaded: Array2<u32> = token.to_ndarray_2d().await.unwrap();
        assert_eq!(downloaded[[0, 0]], 0);
    }

    // ========================================================================
    //  update_decode_token Tests
    // ========================================================================

    #[tokio::test]
    async fn test_update_decode_token() {
        let ctx = get_test_context().await;
        let backend = GpuDecoderBackend::new(ctx).unwrap();
        
        let mut token = backend.new_decode_token().unwrap();
        
        let result = backend.update_decode_token(&mut token, 42);
        assert!(result.is_ok());
        
        // Verify the update
        let downloaded: Array2<u32> = token.to_ndarray_2d().await.unwrap();
        assert_eq!(downloaded[[0, 0]], 42);
    }

    #[tokio::test]
    async fn test_update_decode_token_multiple_times() {
        let ctx = get_test_context().await;
        let backend = GpuDecoderBackend::new(ctx).unwrap();
        
        let mut token = backend.new_decode_token().unwrap();
        
        // Update multiple times
        backend.update_decode_token(&mut token, 100).unwrap();
        backend.update_decode_token(&mut token, 200).unwrap();
        backend.update_decode_token(&mut token, 300).unwrap();
        
        let downloaded: Array2<u32> = token.to_ndarray_2d().await.unwrap();
        assert_eq!(downloaded[[0, 0]], 300);
    }

    #[tokio::test]
    async fn test_update_decode_token_max_value() {
        let ctx = get_test_context().await;
        let backend = GpuDecoderBackend::new(ctx).unwrap();
        
        let mut token = backend.new_decode_token().unwrap();
        
        // Test with max u32 value
        backend.update_decode_token(&mut token, u32::MAX).unwrap();
        
        let downloaded: Array2<u32> = token.to_ndarray_2d().await.unwrap();
        assert_eq!(downloaded[[0, 0]], u32::MAX);
    }

    // ========================================================================
    //  get_or_create_staging_buffer Tests
    // ========================================================================

    #[tokio::test]
    async fn test_get_or_create_staging_buffer_creates_new() {
        let ctx = get_test_context().await;
        let backend = GpuDecoderBackend::new(ctx).unwrap();
        
        let buffer = backend.get_or_create_staging_buffer(1024);
        
        assert_eq!(buffer.size(), 1024);
    }

    #[tokio::test]
    async fn test_get_or_create_staging_buffer_reuses_same_size() {
        let ctx = get_test_context().await;
        let backend = GpuDecoderBackend::new(ctx).unwrap();
        
        let buffer1 = backend.get_or_create_staging_buffer(1024);
        let buffer2 = backend.get_or_create_staging_buffer(1024);
        
        // Should reuse the same buffer (same size)
        assert_eq!(buffer1.size(), buffer2.size());
    }

    #[tokio::test]
    async fn test_get_or_create_staging_buffer_recreates_different_size() {
        let ctx = get_test_context().await;
        let backend = GpuDecoderBackend::new(ctx).unwrap();
        
        let buffer1 = backend.get_or_create_staging_buffer(1024);
        assert_eq!(buffer1.size(), 1024);
        
        let buffer2 = backend.get_or_create_staging_buffer(2048);
        assert_eq!(buffer2.size(), 2048);
    }

    #[tokio::test]
    async fn test_get_or_create_staging_buffer_size_tracking() {
        let ctx = get_test_context().await;
        let backend = GpuDecoderBackend::new(ctx).unwrap();
        
        // Initially should be 0
        {
            let size_guard = backend.staging_buffer_size.lock().unwrap();
            assert_eq!(*size_guard, 0);
        }
        
        backend.get_or_create_staging_buffer(512);
        
        // Should be updated
        {
            let size_guard = backend.staging_buffer_size.lock().unwrap();
            assert_eq!(*size_guard, 512);
        }
    }

    // ========================================================================
    //  ModelInput Size Extraction Tests
    // ========================================================================

    #[test]
    fn test_model_input_size_extraction_tokens_cpu() {
        let tokens = Array2::from_shape_vec((1, 5), vec![1u32, 2, 3, 4, 5]).unwrap();
        let input = ModelInput::TokensCpu(tokens.view());

        let size = match &input {
            ModelInput::TokensCpu(t) => t.shape()[1],
            ModelInput::TokensGpu(t) => t.shape()[1],
            ModelInput::HiddenGpu(t) => t.shape()[1],
            ModelInput::HiddenCpu(t) => t.shape()[1],
        };

        assert_eq!(size, 5);
    }

    #[tokio::test]
    async fn test_model_input_size_extraction_tokens_gpu() {
        let ctx = get_test_context().await;
        let tokens = Array2::from_shape_vec((1, 7), vec![1u32, 2, 3, 4, 5, 6, 7]).unwrap();
        let gpu_tokens = GpuTensor::from_ndarray(&ctx, &tokens).unwrap();
        let input = ModelInput::TokensGpu(&gpu_tokens);

        let size = match &input {
            ModelInput::TokensCpu(t) => t.shape()[1],
            ModelInput::TokensGpu(t) => t.shape()[1],
            ModelInput::HiddenGpu(t) => t.shape()[1],
            ModelInput::HiddenCpu(t) => t.shape()[1],
        };

        assert_eq!(size, 7);
    }

    #[test]
    fn test_model_input_size_extraction_hidden_cpu() {
        let hidden = ndarray::Array3::from_shape_vec(
            (1, 10, 64),
            vec![0.0f32; 640],
        ).unwrap();
        let input = ModelInput::HiddenCpu(hidden.view());

        let size = match &input {
            ModelInput::TokensCpu(t) => t.shape()[1],
            ModelInput::TokensGpu(t) => t.shape()[1],
            ModelInput::HiddenGpu(t) => t.shape()[1],
            ModelInput::HiddenCpu(t) => t.shape()[1],
        };

        assert_eq!(size, 10);
    }

    #[tokio::test]
    async fn test_model_input_size_extraction_hidden_gpu() {
        let ctx = get_test_context().await;
        let hidden = ndarray::Array3::from_shape_vec(
            (1, 12, 64),
            vec![0.0f32; 768],
        ).unwrap();
        let gpu_hidden = GpuTensor::from_ndarray(&ctx, &hidden).unwrap();
        let input = ModelInput::HiddenGpu(&gpu_hidden);

        let size = match &input {
            ModelInput::TokensCpu(t) => t.shape()[1],
            ModelInput::TokensGpu(t) => t.shape()[1],
            ModelInput::HiddenGpu(t) => t.shape()[1],
            ModelInput::HiddenCpu(t) => t.shape()[1],
        };

        assert_eq!(size, 12);
    }

    // ========================================================================
    //  Batch Dimension Tests
    // ========================================================================

    #[test]
    fn test_model_input_batch_size() {
        let tokens = Array2::from_shape_vec((4, 5), vec![0u32; 20]).unwrap();
        let input = ModelInput::TokensCpu(tokens.view());

        let batch_size = match &input {
            ModelInput::TokensCpu(t) => t.shape()[0],
            ModelInput::TokensGpu(t) => t.shape()[0],
            ModelInput::HiddenGpu(t) => t.shape()[0],
            ModelInput::HiddenCpu(t) => t.shape()[0],
        };

        assert_eq!(batch_size, 4);
    }

    // ========================================================================
    //  Timeout Config Tests
    // ========================================================================

    #[tokio::test]
    async fn test_default_timeout_config() {
        let ctx = get_test_context().await;
        let backend = GpuDecoderBackend::new(ctx).unwrap();
        
        // Default config should be reasonable
        assert!(backend.timeout_config.timeout.as_secs() > 0);
        assert!(backend.timeout_config.poll_interval.as_millis() > 0);
    }

    #[tokio::test]
    async fn test_custom_timeout_config_preserved() {
        let ctx = get_test_context().await;
        let config = GpuTimeoutConfig {
            timeout: Duration::from_secs(60),
            poll_interval: Duration::from_millis(10),
        };
        
        let backend = GpuDecoderBackend::with_timeout(ctx, config).unwrap();
        
        assert_eq!(backend.timeout_config.timeout, Duration::from_secs(60));
        assert_eq!(backend.timeout_config.poll_interval, Duration::from_millis(10));
    }

    // ========================================================================
    //  Edge Cases
    // ========================================================================

    #[tokio::test]
    async fn test_staging_buffer_small_size() {
        let ctx = get_test_context().await;
        let backend = GpuDecoderBackend::new(ctx).unwrap();
        
        // Very small buffer
        let buffer = backend.get_or_create_staging_buffer(4);
        assert_eq!(buffer.size(), 4);
    }

    #[tokio::test]
    async fn test_staging_buffer_large_size() {
        let ctx = get_test_context().await;
        let backend = GpuDecoderBackend::new(ctx).unwrap();
        
        // Large buffer (1MB)
        let buffer = backend.get_or_create_staging_buffer(1024 * 1024);
        assert_eq!(buffer.size(), 1024 * 1024);
    }

    #[tokio::test]
    async fn test_multiple_backends_independent() {
        let ctx = get_test_context().await;
        
        let backend1 = GpuDecoderBackend::new(ctx.clone()).unwrap();
        let backend2 = GpuDecoderBackend::new(ctx.clone()).unwrap();
        
        // Each should have its own staging buffer
        backend1.get_or_create_staging_buffer(1024);
        backend2.get_or_create_staging_buffer(2048);
        
        {
            let size1 = backend1.staging_buffer_size.lock().unwrap();
            let size2 = backend2.staging_buffer_size.lock().unwrap();
            assert_eq!(*size1, 1024);
            assert_eq!(*size2, 2048);
        }
    }

    #[tokio::test]
    async fn test_decode_token_sequence() {
        let ctx = get_test_context().await;
        let backend = GpuDecoderBackend::new(ctx).unwrap();
        
        // Simulate a decode sequence
        let mut token = backend.new_decode_token().unwrap();
        
        let sequence = [101u32, 2003, 1037, 3231, 102]; // "this is a test"
        for &tok in &sequence {
            backend.update_decode_token(&mut token, tok).unwrap();
            
            // Verify each update
            let downloaded: Array2<u32> = token.to_ndarray_2d().await.unwrap();
            assert_eq!(downloaded[[0, 0]], tok);
        }
    }
}
