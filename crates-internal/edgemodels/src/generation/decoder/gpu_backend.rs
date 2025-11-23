use crate::generation::generator::DecoderGenerationBackend;
use anyhow::Result;
use async_trait::async_trait;
use edgetransformers::cache::{Cache, GpuKVCache};
use edgetransformers::gpu_ops::primitives::bmm::GpuBatchedMatMul;
use edgetransformers::gpu_ops::primitives::layout::slice_last_token::GpuLastTokenSlice;
use edgetransformers::gpu_ops::Kernel;
use edgetransformers::gpu_ops::{GpuFrameContext, GpuTensor, GpuTensorPool};
pub use edgetransformers::models::base::{AutoregressiveLoop, DecodingStrategy, GenerationConfig};
use edgetransformers::models::DecoderLanguageModel;
use edgetransformers::WgpuContext;
use ndarray::{s, Array1, Array2};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use tokio::sync::Mutex as TokioMutex;
use wgpu::CommandEncoder;

/// Projects hidden states to vocabulary logits on the GPU using a batched matmul.
pub fn project_to_vocab_gpu(
    context: &Arc<WgpuContext>,
    encoder: &mut CommandEncoder,
    hidden_states: &GpuTensor,
    lm_head_weights_transposed: &GpuTensor,
    pool: &mut GpuTensorPool,
) -> Result<GpuTensor> {
    let batch_size = hidden_states.shape()[0];
    let seq_len = hidden_states.shape()[1];
    let vocab_size = lm_head_weights_transposed.shape()[1];

    let output_shape = vec![batch_size, seq_len, vocab_size];
    let logits_output = pool.get(output_shape);

    let bmm_kernel = GpuBatchedMatMul::new(context);
    bmm_kernel.encode(
        encoder,
        &[hidden_states, lm_head_weights_transposed],
        &logits_output,
    );

    Ok(logits_output)
}

pub struct GpuDecoderBackend {
    context: Arc<WgpuContext>,
    token_tensor: GpuTensor,
    last_token_slicer: GpuLastTokenSlice,

    // Ring buffer for async readback (triple buffering)
    staging_buffers: Mutex<Option<Vec<wgpu::Buffer>>>,
    buffer_size: Mutex<Option<usize>>,
    current_buffer_idx: AtomicUsize,
}

impl GpuDecoderBackend {
    pub fn new(context: Arc<WgpuContext>) -> Result<Self> {
        let token_ndarray = Array2::<u32>::zeros((1, 1));
        let token_tensor = GpuTensor::from_ndarray(&context, &token_ndarray)?;

        Ok(Self {
            context: context.clone(),
            token_tensor,
            last_token_slicer: GpuLastTokenSlice::new(&context),
            staging_buffers: Mutex::new(None),
            buffer_size: Mutex::new(None),
            current_buffer_idx: AtomicUsize::new(0),
        })
    }

    /// Ensure staging buffers exist and match the required size
    fn ensure_staging_buffers(&self, required_size_bytes: usize) -> Result<()> {
        let mut buffers_guard = self.staging_buffers.lock().unwrap();
        let mut size_guard = self.buffer_size.lock().unwrap();

        let needs_creation = match (*size_guard, buffers_guard.as_ref()) {
            (None, None) => true,
            (Some(size), Some(_)) if size != required_size_bytes => {
                log::debug!("Recreating staging buffers: {} -> {} bytes", size, required_size_bytes);
                true
            }
            _ => false,
        };

        if needs_creation {
            let new_buffers: Vec<_> = (0..3)
                .map(|i| {
                    self.context.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("Staging Buffer {} ({}KB)", i, required_size_bytes / 1024)),
                        size: required_size_bytes as u64,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    })
                })
                .collect();

            *buffers_guard = Some(new_buffers);
            *size_guard = Some(required_size_bytes);
            log::debug!("Created 3 staging buffers of {}KB each", required_size_bytes / 1024);
        }

        Ok(())
    }

    /// Internal helper to run a forward pass and project to logits.
    async fn forward_and_project<'a>(
        &'a self,
        model: &'a dyn DecoderLanguageModel,
        input_tensor: &'a GpuTensor,
        seq_len: usize,
        cache: &'a mut dyn Cache,
    ) -> Result<Array1<f32>> {
        let gpu_decoder = model.gpu_decoder()?;
        let gpu_lm_head_transposed = model.gpu_lm_head_transposed()?;
        let mut gpu_cache = cache.as_any_mut().downcast_mut::<GpuKVCache>().unwrap();

        // Use shared pool from context!
        let pool = self.context.get_inference_pool();
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);
        let (encoder, pool) = frame.resources();

        let position_offset = gpu_cache.get_seq_length();

        // Create attention mask
        let attention_mask = {
            let max_len = gpu_cache.max_seq_len();
            let mut mask_cpu = Array2::zeros((1, max_len));
            mask_cpu.slice_mut(s![.., 0..seq_len]).fill(1.0);
            GpuTensor::from_ndarray(&self.context, &mask_cpu)?
        };

        // 1. Forward pass
        let hidden_states = gpu_decoder
            .forward(
                encoder,
                pool,
                input_tensor,
                &attention_mask,
                position_offset,
                Some(gpu_cache),
                None,
            )
            .await?;

        // 2. Project to logits
        let logits_gpu = project_to_vocab_gpu(
            &self.context,
            encoder,
            &hidden_states,
            gpu_lm_head_transposed,
            pool,
        )?;

        // 3. Slice to last token on GPU
        let batch_size = logits_gpu.shape()[0];
        let vocab_size = logits_gpu.shape()[2];
        let last_token_logits = pool.get(vec![batch_size, vocab_size]);

        self.last_token_slicer.encode(encoder, &logits_gpu, &last_token_logits);

        // 4. Setup ring buffer for this vocab size
        let buffer_size_bytes = vocab_size * std::mem::size_of::<f32>();
        self.ensure_staging_buffers(buffer_size_bytes)?;

        // 5. Copy to NEXT staging buffer in ring
        let buffer_idx = self.current_buffer_idx.fetch_add(1, Ordering::Relaxed) % 3;

        // Clone the Arc (cheap!) so we can use buffers without holding the lock
        let buffers_arc = {
            let guard = self.staging_buffers.lock().unwrap();
            guard.as_ref().unwrap().clone()  // Clone the Arc, not the buffers!
        };

        // Copy to staging buffer (no lock needed, we have Arc)
        encoder.copy_buffer_to_buffer(
            last_token_logits.buffer(),
            0,
            &buffers_arc[buffer_idx],
            0,
            buffer_size_bytes as u64,
        );

        frame.finish();

        // 6. Read from PREVIOUS buffer (lag=1, GPU is working ahead!)
        let read_idx = if buffer_idx == 0 {
            if self.current_buffer_idx.load(Ordering::Relaxed) == 1 {
                0  // First iteration
            } else {
                2  // Wrapped around
            }
        } else {
            buffer_idx - 1
        };

        // Use buffers_arc directly - no lock needed!
        let read_buffer = &buffers_arc[read_idx];
        let slice = read_buffer.slice(..);

        let (tx, rx) = futures::channel::oneshot::channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Background polling makes this instant!
        rx.await??;

        // Zero-copy read with bytemuck
        let data = slice.get_mapped_range();
        let logits_cpu: Array1<f32> = bytemuck::cast_slice(&data).to_vec().into();

        drop(data);
        read_buffer.unmap();

        gpu_cache.increment_len(input_tensor.shape()[1]);

        Ok(logits_cpu)
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
        self.context.queue.write_buffer(
            tensor.buffer(),
            0,
            bytemuck::cast_slice(&[new_token_id]),
        );
        Ok(())
    }

    fn prepare_attention_mask(&self, seq_len: usize, max_len: usize) -> Result<Self::Tensor> {
        let mut mask_cpu = Array2::zeros((1, max_len));
        mask_cpu.slice_mut(s![.., 0..seq_len]).fill(1.0);
        GpuTensor::from_ndarray(&self.context, &mask_cpu)
    }

    async fn decode_one<'a>(
        &'a self,
        model: &'a dyn DecoderLanguageModel,
        token_id: u32,
        seq_len: usize,
        cache: &'a mut dyn Cache,
    ) -> Result<Array1<f32>> {
        // Update reusable token tensor
        self.context.queue.write_buffer(
            self.token_tensor.buffer(),
            0,
            bytemuck::cast_slice(&[token_id]),
        );

        // Adjust mask length based on autoregressive mode
        let mask_len = match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => seq_len,
            AutoregressiveLoop::Legacy => seq_len + 1,
        };

        self.forward_and_project(model, &self.token_tensor, mask_len, cache).await
    }

    async fn prefill<'a>(
        &'a self,
        model: &'a dyn DecoderLanguageModel,
        initial_tokens: &[u32],
        cache: &'a mut dyn Cache,
    ) -> Result<Array1<f32>> {
        if initial_tokens.is_empty() {
            return Err(anyhow::anyhow!("Cannot prefill with empty tokens"));
        }

        let prompt_len = initial_tokens.len();

        match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => {
                // Llama: Single efficient pass
                let prompt_tensor = {
                    let ndarray = Array2::from_shape_vec((1, prompt_len), initial_tokens.to_vec())?;
                    GpuTensor::from_ndarray(&self.context, &ndarray)?
                };
                self.forward_and_project(model, &prompt_tensor, prompt_len, cache).await
            }
            AutoregressiveLoop::Legacy => {
                // GPT-2: Batch process + reprocess last

                // Step 1: Process ALL tokens at once to fill cache
                let prompt_tensor = {
                    let ndarray = Array2::from_shape_vec((1, prompt_len), initial_tokens.to_vec())?;
                    GpuTensor::from_ndarray(&self.context, &ndarray)?
                };

                // Run forward but don't return these logits
                let _ = self.forward_and_project(model, &prompt_tensor, prompt_len, cache).await?;

                // Step 2: Reprocess the LAST prompt token
                let last_token = initial_tokens[prompt_len - 1];
                self.context.queue.write_buffer(
                    self.token_tensor.buffer(),
                    0,
                    bytemuck::cast_slice(&[last_token]),
                );

                // Forward pass with mask = prompt_len + 1
                self.forward_and_project(model, &self.token_tensor, prompt_len + 1, cache).await
            }
        }
    }
}