// use crate::generation::generator::DecoderGenerationBackend;
use anyhow::Result;
use async_trait::async_trait;
use edgetransformers::WgpuContext;
use edgetransformers::cache::{Cache, GpuKVCache};
use edgetransformers::decoder::DecoderGenerationBackend;
use edgetransformers::gpu_ops::DType;
use edgetransformers::gpu_ops::Kernel;
use edgetransformers::gpu_ops::primitives::argmax::GpuArgMax;
use edgetransformers::gpu_ops::primitives::bmm::GpuBatchedMatMul;
use edgetransformers::gpu_ops::primitives::layout::slice_last_token::GpuLastTokenSlice;
use edgetransformers::gpu_ops::primitives::linear::GpuLinearLayer;
use edgetransformers::gpu_ops::primitives::matmul::GpuMatMul;
use edgetransformers::gpu_ops::{GpuFrameContext, GpuTensor, GpuTensorPool};
use edgetransformers::models::DecoderLanguageModel;
use edgetransformers::models::base::DecoderInput;
pub use edgetransformers::models::base::{AutoregressiveLoop, DecodingStrategy, GenerationConfig};
use ndarray::{Array1, Array2, Array3, s};
use std::sync::Arc;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, MapMode};

pub struct GpuDecoderBackend {
    context: Arc<WgpuContext>,
    token_tensor: GpuTensor,

    // Persistent kernels
    last_token_slicer: GpuLastTokenSlice,

    linear_layer: Option<GpuLinearLayer>,
    // Optional: Only exists if we are projecting on GPU
    vocab_projection_kernel: Option<GpuBatchedMatMul>,
    gemv_bf16_kernel: Option<GpuMatMul>,
    argmax_kernel: GpuArgMax,

    staging_buffer: std::sync::Mutex<Option<Buffer>>,
    staging_buffer_size: std::sync::Mutex<usize>,

    attention_mask: std::sync::Mutex<Option<GpuTensor>>,
    attention_mask_max_len: std::sync::atomic::AtomicUsize,
}
impl GpuDecoderBackend {
    pub fn new(context: Arc<WgpuContext>, model: &dyn DecoderLanguageModel) -> Result<Self> {
        let token_ndarray = Array2::<u32>::zeros((1, 1));
        let token_tensor = GpuTensor::from_ndarray(&context, &token_ndarray)?;

        // Check if the model loaded the head to VRAM
        let has_gpu_head = model.gpu_lm_head_transposed().is_ok();

        let (proj_kernel, gemv_kernel) = if has_gpu_head {
            (
                Some(GpuBatchedMatMul::new(&context)),
                Some(GpuMatMul::new(&context)),
            )
        } else {
            // If head is on CPU, we don't need this kernel
            (None, None)
        };

        let linear_layer = if has_gpu_head {
            Some(GpuLinearLayer::new(&context))
        } else {
            None
        };

        Ok(Self {
            context: context.clone(),
            token_tensor,
            last_token_slicer: GpuLastTokenSlice::new(&context),
            linear_layer,
            vocab_projection_kernel: proj_kernel,
            argmax_kernel: GpuArgMax::new(&context),
            staging_buffer: std::sync::Mutex::new(None),
            staging_buffer_size: std::sync::Mutex::new(0),
            gemv_bf16_kernel: gemv_kernel,
            attention_mask: std::sync::Mutex::new(None),
            attention_mask_max_len: std::sync::atomic::AtomicUsize::new(0),
        })
    }
    /// Get or create attention mask, updating values in-place
    fn get_attention_mask(&self, seq_len: usize, max_len: usize) -> Result<GpuTensor> {
        use std::sync::atomic::Ordering;

        let mut guard = self.attention_mask.lock().unwrap();
        let current_max = self.attention_mask_max_len.load(Ordering::Relaxed);

        // Reallocate only if max_len changed (new generation)
        if guard.is_none() || current_max != max_len {
            log::debug!("Allocating new attention mask for max_len={}", max_len);
            let mask =
                GpuTensor::zeros(&self.context, vec![1, max_len], DType::F32, "AttentionMask")?;
            self.attention_mask_max_len
                .store(max_len, Ordering::Relaxed);
            *guard = Some(mask);
        }

        // Update values via queue.write_buffer (fast!)
        let mask = guard.as_ref().unwrap();
        let mask_data: Vec<f32> = (0..max_len)
            .map(|i| if i < seq_len { 1.0 } else { 0.0 })
            .collect();

        self.context
            .queue
            .write_buffer(mask.buffer(), 0, bytemuck::cast_slice(&mask_data));

        Ok(mask.clone())
    }
    fn get_staging_buffer(&self, required_bytes: usize) -> Buffer {
        let mut buffer_guard = self.staging_buffer.lock().unwrap();
        let mut size_guard = self.staging_buffer_size.lock().unwrap();

        if buffer_guard.is_some() && *size_guard == required_bytes {
            return buffer_guard.as_ref().unwrap().clone();
        }

        // Safer cleanup: Don't call destroy() manually here to avoid race conditions.
        // Just dropping the old Arc<Buffer> handles cleanup when the GPU is done with it.
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

    async fn forward_only<'a>(
        &'a self,
        model: &'a dyn DecoderLanguageModel,
        input: DecoderInput<'_>,
        seq_len: usize,
        cache: &'a mut dyn Cache,
    ) -> Result<()> {
        let gpu_decoder = model.gpu_decoder()?;
        let gpu_cache = cache.as_any_mut().downcast_mut::<GpuKVCache>().unwrap();

        let pool = self.context.get_inference_pool();
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);
        let (encoder, pool) = frame.resources();

        let position_offset = gpu_cache.get_seq_length();

        let input_len = match input {
            DecoderInput::Cpu(ids) => ids.len(),
            DecoderInput::Gpu(t) => t.shape()[1],
        };

        let attention_mask = {
            let max_len = gpu_cache.max_seq_len();
            let mut mask_cpu = Array2::zeros((1, max_len));
            mask_cpu.slice_mut(s![.., 0..seq_len]).fill(1.0);
            GpuTensor::from_ndarray(&self.context, &mask_cpu)?
        };

        let _ = gpu_decoder
            .forward(
                encoder,
                pool,
                input,
                &attention_mask,
                position_offset,
                Some(gpu_cache),
                None,
            )
            .await?;

        frame.finish();

        gpu_cache.increment_len(input_len);
        Ok(())
    }

    async fn forward_and_project<'a>(
        &'a self,
        model: &'a dyn DecoderLanguageModel,
        input: DecoderInput<'_>,
        seq_len: usize,
        cache: &'a mut dyn Cache,
    ) -> Result<Array1<f32>> {
        let total_start = std::time::Instant::now();
        let gpu_decoder = model.gpu_decoder()?;
        let gpu_cache = cache.as_any_mut().downcast_mut::<GpuKVCache>().unwrap();

        self.context.uniform_arena.reset();

        log::info!(
            "forward_and_project: seq_len={}, cache_pos={}/{}",
            seq_len,
            gpu_cache.get_seq_length(),
            gpu_cache.max_seq_len()
        );
        let pool = self.context.get_inference_pool();
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);
        let (encoder, pool) = frame.resources();

        let position_offset = gpu_cache.get_seq_length();

        let input_len = match input {
            DecoderInput::Cpu(ids) => ids.len(),
            DecoderInput::Gpu(t) => t.shape()[1],
        };
        let t_mask = std::time::Instant::now();
        let attention_mask = {
            let max_len = gpu_cache.max_seq_len();
            let mut mask_cpu = Array2::zeros((1, max_len));
            mask_cpu.slice_mut(s![.., 0..seq_len]).fill(1.0);
            GpuTensor::from_ndarray(&self.context, &mask_cpu)?
        };
        log::info!("  Mask creation: {:?}", t_mask.elapsed());

        // 1. Forward Pass -> [Batch, Seq, Hidden]
        let t_forward = std::time::Instant::now();
        let hidden_states = gpu_decoder
            .forward(
                encoder,
                pool,
                input,
                &attention_mask,
                position_offset,
                Some(gpu_cache),
                None,
            )
            .await?;
        log::info!("  Forward encode: {:?}", t_forward.elapsed());
        let batch_size = hidden_states.shape()[0];

        let has_gpu_head =
            self.vocab_projection_kernel.is_some() && model.gpu_lm_head_transposed().is_ok();
        log::info!(
            "  Using {} LM head path",
            if has_gpu_head { "GPU" } else { "CPU" }
        );

        let t_project = std::time::Instant::now();
        let logits_cpu = if let (Some(linear), Some(proj_kernel), Ok(gpu_lm_head)) = (
            &self.linear_layer,
            &self.vocab_projection_kernel,
            model.gpu_lm_head_transposed(),
        ) {
            // GPU HEAD PATH
            // let seq_len_out = hidden_states.shape()[1];
            // // let vocab_size = gpu_lm_head.shape()[1];
            // let vocab_size = if gpu_lm_head.dtype() == DType::BF16 {
            //     gpu_lm_head.shape()[0] // [Vocab, Hidden]
            // } else {
            //     gpu_lm_head.shape()[1] // [Hidden, Vocab] (Transposed)
            // };
            // No more Transpose logic checks. gpu_lm_head is always [Vocab, Hidden]
            let batch_size = hidden_states.shape()[0];
            let seq_len_out = hidden_states.shape()[1];
            let vocab_size = gpu_lm_head.shape()[0]; // It's [Vocab, Hidden] now
            log::info!(
                "  GPU project: hidden [{},{},{}] @ lm_head [{},{}]",
                batch_size,
                seq_len_out,
                hidden_states.shape()[2],
                gpu_lm_head.shape()[0],
                gpu_lm_head.shape()[1] // FIX: was shape()[0] twice
            );

            let logits_output = pool.get(vec![batch_size, seq_len_out, vocab_size]);
            linear.encode(encoder, &hidden_states, gpu_lm_head, &logits_output);

            let last_token_logits = pool.get(vec![batch_size, vocab_size]);
            self.last_token_slicer
                .encode(encoder, &logits_output, &last_token_logits);

            let buffer_size = vocab_size * 4;
            log::info!(
                "  Readback size: {} bytes ({:.1} KB)",
                buffer_size,
                buffer_size as f64 / 1024.0
            );

            let staging = self.get_staging_buffer(buffer_size);
            encoder.copy_buffer_to_buffer(
                last_token_logits.buffer(),
                0,
                &staging,
                0,
                buffer_size as u64,
            );

            let t_submit = std::time::Instant::now();

            self.context.profiler.process_results(encoder);

            frame.finish();

            self.context.profiler.print_stats(&self.context).await;

            log::info!("  Submit: {:?}", t_submit.elapsed());

            let t_readback = std::time::Instant::now();
            let result = self.sync_read_buffer(&staging).await?;
            log::info!(
                "  Readback: {:?} ← THIS IS YOUR BOTTLENECK IF SLOW",
                t_readback.elapsed()
            );

            result
        } else {
            // CPU HEAD PATH
            let hidden_size = hidden_states.shape()[2];
            log::info!(
                "  CPU project: downloading {} floats ({:.1} KB)",
                hidden_size,
                hidden_size as f64 * 4.0 / 1024.0
            );

            let last_hidden_state = pool.get(vec![batch_size, hidden_size]);
            self.last_token_slicer
                .encode(encoder, &hidden_states, &last_hidden_state);

            let buffer_size = hidden_size * 4;
            let staging = self.get_staging_buffer(buffer_size);
            encoder.copy_buffer_to_buffer(
                last_hidden_state.buffer(),
                0,
                &staging,
                0,
                buffer_size as u64,
            );

            self.context.profiler.process_results(encoder);

            frame.finish();

            self.context.profiler.print_stats(&self.context).await;

            let t_readback = std::time::Instant::now();
            let hidden_vec = self.sync_read_buffer(&staging).await?;
            log::info!("  Hidden readback: {:?}", t_readback.elapsed());

            let t_cpu_proj = std::time::Instant::now();
            let hidden_3d = Array3::from_shape_vec((1, 1, hidden_size), hidden_vec.to_vec())?;

            // 2. Use the Model's projection logic (Handles BF16/LinearLayer automatically)
            // Note: project_to_logits returns [Batch, Seq, Vocab] -> [1, 1, Vocab]
            let logits_3d = model.project_to_logits(&hidden_3d)?;

            // 3. Flatten back to 1D for the generator
            // Get the first (and only) row
            let logits_1d = logits_3d
                .index_axis(ndarray::Axis(0), 0) // Remove Batch
                .index_axis(ndarray::Axis(0), 0) // Remove Seq
                .to_owned();
            // --- FIX ENDS HERE ---

            log::info!("  CPU matmul: {:?}", t_cpu_proj.elapsed());

            logits_1d
        };
        log::info!("  Project total: {:?}", t_project.elapsed());

        gpu_cache.increment_len(input_len);
        log::info!("forward_and_project total: {:?}", total_start.elapsed());

        Ok(logits_cpu)
    }

    /// Helper to handle the poll-wait-read cycle cleanly
    async fn sync_read_buffer(&self, buffer: &Buffer) -> Result<Array1<f32>> {
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();

        slice.map_async(MapMode::Read, move |v| {
            let _ = tx.send(v);
        });

        self.context
            .device
            .poll(wgpu::PollType::wait_indefinitely());

        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let vec_data: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        buffer.unmap();

        Ok(Array1::from_vec(vec_data))
    }
}

// ... Trait implementations below ...
#[async_trait(?Send)]
impl DecoderGenerationBackend for GpuDecoderBackend {
    type Tensor = GpuTensor;

    // ... prime_tokens, new_token_tensor, update_token_tensor, prepare_attention_mask same as before ...
    fn prime_tokens(&self, tokens: &[u32]) -> Result<Self::Tensor> {
        let tokens_ndarray = Array2::from_shape_vec((1, tokens.len()), tokens.to_vec())?;
        GpuTensor::from_ndarray(&self.context, &tokens_ndarray)
    }

    fn new_token_tensor(&self) -> Result<Self::Tensor> {
        let token_ndarray = Array2::<u32>::zeros((1, 1));
        GpuTensor::from_ndarray(&self.context, &token_ndarray)
    }

    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_token_id: u32) -> Result<()> {
        self.context
            .queue
            .write_buffer(tensor.buffer(), 0, bytemuck::cast_slice(&[new_token_id]));
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
        // FAST PATH: Create CPU slice.
        // If the model is CPU-Embed, it stays on CPU.
        // If the model is GPU-Embed, the backend uploads this slice.
        let input_ids = [token_id];
        let input = DecoderInput::Cpu(&input_ids);

        let mask_len = match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => seq_len,
            AutoregressiveLoop::Legacy => seq_len + 1,
        };

        self.forward_and_project(model, input, mask_len, cache)
            .await
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
                let input = DecoderInput::Cpu(initial_tokens);
                self.forward_and_project(model, input, prompt_len, cache)
                    .await
            }
            AutoregressiveLoop::Legacy => {
                let input = DecoderInput::Cpu(initial_tokens);
                self.forward_only(model, input, prompt_len, cache).await?;

                let last_token_id = initial_tokens[prompt_len - 1];
                let last_token_slice = [last_token_id];
                let input_last = DecoderInput::Cpu(&last_token_slice);

                self.forward_and_project(model, input_last, prompt_len + 1, cache)
                    .await
            }
        }
    }
}

impl Drop for GpuDecoderBackend {
    fn drop(&mut self) {
        // 1. Just release the lock and option.
        // DO NOT call buffer.destroy(). Let the Arc drop naturally.
        let _unused = self.staging_buffer.lock();

        // 2. Poll once to allow WGPU to process any pending unmaps/drops in queue.
        self.context.device.poll(wgpu::PollType::Poll);
    }
}
