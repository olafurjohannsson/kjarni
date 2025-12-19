// use crate::generation::generator::DecoderGenerationBackend;
use anyhow::Result;
use async_trait::async_trait;
use crate::WgpuContext;
use crate::cache::{Cache, GpuKVCache};
use crate::gpu_ops::Kernel;
use crate::gpu_ops::primitives::layout::slice_last_token::GpuLastTokenSlice;
use crate::gpu_ops::{GpuFrameContext, GpuTensor};
use crate::decoder::prelude::*;
pub use crate::models::base::{AutoregressiveLoop};
use crate::common::{GenerationConfig, DecodingStrategy};
use ndarray::{Array1, Array2};
use std::sync::Arc;
use anyhow::{anyhow};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, MapMode};
pub struct GpuDecoderBackend {
    context: Arc<WgpuContext>,
    last_token_slicer: GpuLastTokenSlice,
    staging_buffer: std::sync::Mutex<Option<Buffer>>,
    staging_buffer_size: std::sync::Mutex<usize>,
}

impl GpuDecoderBackend {
    pub fn new(context: Arc<WgpuContext>) -> Result<Self> {
        Ok(Self {
            context: context.clone(),
            last_token_slicer: GpuLastTokenSlice::new(&context),
            staging_buffer: std::sync::Mutex::new(None),
            staging_buffer_size: std::sync::Mutex::new(0),
        })
    }

    async fn forward_and_project(
        &self,
        model: &dyn DecoderLanguageModel,
        input: DecoderInput<'_>,
        seq_len: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        // 1. Calculate length BEFORE moving 'input'
        let input_len = match input {
            DecoderInput::TokensCpu(t) => t.len(),
            DecoderInput::TokensGpu(t) => t.shape()[1],
            DecoderInput::HiddenGpu(t) => t.shape()[1],
            DecoderInput::HiddenCpu(t) => t.shape()[1],
        };

        // 2. Get Ops & Cache
        let ops = model
            .decoder_gpu_ops()
            .ok_or_else(|| anyhow!("Model does not support GPU execution"))?;
        
        let gpu_cache = cache.as_any_mut().downcast_mut::<GpuKVCache>().unwrap();

        // 3. Setup Frame Context
        let pool = self.context.get_inference_pool();
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);

        // 4. Ask Model for Mask
        let max_len = gpu_cache.max_seq_len();
        let attention_mask = ops.get_attention_mask(&mut frame, seq_len, max_len)?;

        // 5. Run Decoder Stack
        let (encoder, pool) = frame.resources();
        
        let hidden_states = ops.decoder().forward(
            encoder,
            pool,
            input, // <--- 'input' is moved here
            &attention_mask,
            gpu_cache.get_seq_length(), 
            Some(gpu_cache),
            None,
        ).await?;

        // 6. Slice Last Token
        let (batch, _, hidden_dim) = hidden_states.dims3();
        let last_hidden = frame.pool_guard.get(vec![batch, 1, hidden_dim]);
        
        let (encoder, _pool) = frame.resources();
        self.last_token_slicer.encode(encoder, &hidden_states, &last_hidden);

        // 7. Project to Logits
        let logits = ops.project_to_logits(&mut frame, &last_hidden)?;

        // 8. Submit & Readback
        let size = logits.buffer().size();
        let staging = self.get_staging_buffer(size as usize);
        
        let (encoder, _pool) = frame.resources();
        encoder.copy_buffer_to_buffer(logits.buffer(), 0, &staging, 0, size);
        
        frame.finish();

        // 9. Update Cache Length (using the pre-calculated length)
        gpu_cache.increment_len(input_len);

        self.sync_read_buffer(&staging).await
    }

    /// Internal Helper: Forward Only (No Projection).
    async fn forward_only(
        &self,
        model: &dyn DecoderLanguageModel,
        input: DecoderInput<'_>,
        seq_len: usize,
        cache: &mut dyn Cache,
    ) -> Result<()> {
        // 1. Calculate length BEFORE moving 'input'
        let input_len = match input {
            DecoderInput::TokensCpu(t) => t.len(),
            DecoderInput::TokensGpu(t) => t.shape()[1],
            DecoderInput::HiddenGpu(t) => t.shape()[1],
            DecoderInput::HiddenCpu(t) => t.shape()[1],
        };

        let ops = model.decoder_gpu_ops().ok_or_else(|| anyhow!("No GPU Ops"))?;
        let gpu_cache = cache.as_any_mut().downcast_mut::<GpuKVCache>().unwrap();

        let pool = self.context.get_inference_pool();
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);

        let max_len = gpu_cache.max_seq_len();
        let attention_mask = ops.get_attention_mask(&mut frame, seq_len, max_len)?;

        let (encoder, pool) = frame.resources();
        let _ = ops.decoder().forward(
            encoder,
            pool,
            input, // <--- 'input' is moved here
            &attention_mask,
            gpu_cache.get_seq_length(),
            Some(gpu_cache),
            None,
        ).await?;

        frame.finish();
        
        // 2. Update Cache Length
        gpu_cache.increment_len(input_len);

        Ok(())
    }

    fn get_staging_buffer(&self, required_bytes: usize) -> Buffer {
        let mut buffer_guard = self.staging_buffer.lock().unwrap();
        let mut size_guard = self.staging_buffer_size.lock().unwrap();

        if buffer_guard.is_some() && *size_guard == required_bytes {
            return buffer_guard.as_ref().unwrap().clone();
        }
        let _ = buffer_guard.take();
        
        let new_buffer = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("Readback Staging"),
            size: required_bytes as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        *buffer_guard = Some(new_buffer.clone());
        *size_guard = required_bytes;
        new_buffer
    }

    async fn sync_read_buffer(&self, buffer: &Buffer) -> Result<Array1<f32>> {
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |v| { let _ = tx.send(v); });
        
        self.context.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let vec_data: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        buffer.unmap();
        Ok(Array1::from_vec(vec_data))
    }
}

#[async_trait(?Send)]
impl DecoderGenerationBackend for GpuDecoderBackend {
    type Tensor = GpuTensor;

    fn prime_tokens(&self, tokens: &[u32]) -> Result<Self::Tensor> {
        let tokens_ndarray = Array2::from_shape_vec((1, tokens.len()), tokens.to_vec())?;
        GpuTensor::from_ndarray(&self.context, &tokens_ndarray)
    }

    fn new_token_tensor(&self) -> Result<Self::Tensor> {
        let token_ndarray = Array2::<u32>::zeros((1, 1));
        GpuTensor::from_ndarray(&self.context, &token_ndarray)
    }

    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_token_id: u32) -> Result<()> {
        self.context.queue.write_buffer(tensor.buffer(), 0, bytemuck::cast_slice(&[new_token_id]));
        Ok(())
    }

    async fn prefill(
        &self,
        model: &dyn DecoderLanguageModel,
        initial_tokens: &[u32],
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        if initial_tokens.is_empty() { return Err(anyhow!("Cannot prefill with empty tokens")); }
        let prompt_len = initial_tokens.len();

        match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => {
                self.forward_and_project(
                    model, 
                    DecoderInput::TokensCpu(initial_tokens), 
                    prompt_len, 
                    cache
                ).await
            }
            AutoregressiveLoop::Legacy => {
                let last_idx = prompt_len - 1;
                let context_tokens = &initial_tokens[..last_idx];
                let last_token = &initial_tokens[last_idx..];

                if !context_tokens.is_empty() {
                    log::info!("Filling cache with {} context tokens", context_tokens.len());
                    log::info!("Forward_only");
                    self.forward_only(
                        model,
                        DecoderInput::TokensCpu(context_tokens),
                        context_tokens.len(),
                        cache,
                    ).await?;
                }
                log::info!("Projecting last token");
                self.forward_and_project(
                    model,
                    DecoderInput::TokensCpu(last_token),
                    prompt_len,
                    cache,
                ).await
            }
        }
    }

    async fn decode_one(
        &self,
        model: &dyn DecoderLanguageModel,
        // token_id: u32,
        token_tensor: &Self::Tensor,
        seq_len: usize,
        cache: &mut dyn Cache,
    ) -> Result<Array1<f32>> {
        // let token_slice = [token_id];
        let input = DecoderInput::TokensGpu(token_tensor);
        self.forward_and_project(
            model,
            input,
            seq_len,
            cache,
        ).await
    }
}

// impl Drop for GpuDecoderBackend {
//     fn drop(&mut self) {
//         let _ = self.staging_buffer.lock();
//         self.context.device.poll(wgpu::PollType::Poll);
//     }
// }