use crate::generation::generator::DecoderGenerationBackend;
use anyhow::Result;
use async_trait::async_trait;
use edgetransformers::WgpuContext;
use edgetransformers::cache::{Cache, CpuKVCache, GpuKVCache};
use edgetransformers::gpu_ops::Kernel;
use edgetransformers::gpu_ops::primitives::bmm::GpuBatchedMatMul;
use edgetransformers::gpu_ops::primitives::layout::slice::GpuSlice;
use edgetransformers::gpu_ops::primitives::matmul::GpuMatMul;
use edgetransformers::gpu_ops::{GpuFrameContext, GpuTensor, GpuTensorPool};
use edgetransformers::gpu_ops::primitives::layout::slice_last_token::GpuLastTokenSlice;
use edgetransformers::models::DecoderLanguageModel;
pub use edgetransformers::models::base::{AutoregressiveLoop, DecodingStrategy, GenerationConfig};
use log::debug;
use ndarray::{Array1, Array2, s};
use std::sync::Arc;
use tokio::sync::Mutex;
use wgpu::CommandEncoder;

/// Projects hidden states to vocabulary logits on the GPU using a batched matmul.
pub fn project_to_vocab_gpu(
    context: &Arc<WgpuContext>,
    encoder: &mut CommandEncoder,
    hidden_states: &GpuTensor,
    lm_head_weights_transposed: &GpuTensor, // <-- Expects the transposed weights
    pool: &mut GpuTensorPool,
) -> Result<GpuTensor> {
    // hidden_states:      [Batch, SeqLen, HiddenDim]
    // lm_head_weights_T:  [HiddenDim, VocabSize]

    let batch_size = hidden_states.shape()[0];
    let seq_len = hidden_states.shape()[1];
    let vocab_size = lm_head_weights_transposed.shape()[1];

    // Allocate the output tensor for the logits
    let output_shape = vec![batch_size, seq_len, vocab_size];
    let logits_output = pool.get(output_shape);

    // Instantiate your batched matmul kernel
    let bmm_kernel = GpuBatchedMatMul::new(context);

    // Encode the operation. Your kernel correctly handles broadcasting the 2D
    // lm_head_weights_transposed tensor across the batch dimension of hidden_states.
    bmm_kernel.encode(
        encoder,
        &[hidden_states, lm_head_weights_transposed],
        &logits_output,
    );

    Ok(logits_output)
}

pub struct GpuDecoderBackend {
    context: Arc<WgpuContext>,
    pool: Arc<Mutex<GpuTensorPool>>,
    // A reusable 1x1 tensor for single-token decoding, avoiding reallocation.
    token_tensor: GpuTensor,
    slice_kernel: GpuSlice,
    last_token_slicer: GpuLastTokenSlice,
}

impl GpuDecoderBackend {
    pub fn new(context: Arc<WgpuContext>, pool: Arc<Mutex<GpuTensorPool>>) -> Result<Self> {
        let token_ndarray = Array2::<u32>::zeros((1, 1));
        let token_tensor = GpuTensor::from_ndarray(&context.clone(), &token_ndarray)?;
        let c = context.clone();
        Ok(Self {
            context,
            pool,
            token_tensor,
            slice_kernel: GpuSlice::new(&c),
            last_token_slicer: GpuLastTokenSlice::new(&c)
        })
    }

    /// Internal helper to run a forward pass and project to logits.
    /// This is the core logic shared by prefill and decode_one.
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

        let pool_guard = self.pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);
        let (encoder, pool) = frame.resources();

        let position_offset = gpu_cache.get_seq_length();

        // --- FIX 1: CREATE AND USE THE ATTENTION MASK ---
        let attention_mask = {
            // Assumes your GpuKVCache can expose its max capacity.
            let max_len = gpu_cache.max_seq_len();
            let mut mask_cpu = Array2::zeros((1, max_len));
            mask_cpu.slice_mut(s![.., 0..seq_len]).fill(1.0);
            GpuTensor::from_ndarray(&self.context, &mask_cpu)?
        };
        let decoder_fwd_start = std::time::Instant::now();
        // 1. Run the decoder forward pass
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
        log::info!(
            "    └─ Decoder forward pass: {:?}",
            decoder_fwd_start.elapsed()
        );
        // 2. Project to logits on the GPU
        let projection_start = std::time::Instant::now();
        let logits_gpu = project_to_vocab_gpu(
            &self.context,
            encoder,
            &hidden_states,
            gpu_lm_head_transposed,
            pool,
        )?;
        log::info!(
            "    └─ Projection to vocab: {:?}",
            projection_start.elapsed()
        );
        // --- FIX 2: CORRECTLY SLICE THE LOGITS TENSOR ---
        let output_seq_len = logits_gpu.shape()[1];
        let batch_size = logits_gpu.shape()[0];
        let vocab_size = logits_gpu.shape()[2];
        let last_token_logits = pool.get(vec![batch_size, vocab_size]);
        // Use your slice kernel to get the logits for the last token only
        // let logits_slice = logits_gpu.slice(
        //     encoder,
        //     &self.slice_kernel,
        //     &[0, output_seq_len - 1, 0], // Offset: [batch, last_token, start_of_vocab]
        //     &[1, 1, vocab_size],         // Shape: [1, 1, vocab_size]
        // )?;

        self.last_token_slicer.encode(encoder, &logits_gpu, &last_token_logits);

        frame.finish();

        // 3. Copy the small slice back to the CPU
        // let logits_cpu = logits_slice.to_ndarray_1d().await?;

        let copy_start = std::time::Instant::now();
        // let logits_3d_cpu: ndarray::Array3<f32> = logits_gpu.to_ndarray_3d().await?;
        
        let logits_cpu_2d = last_token_logits.to_ndarray_2d().await?;
        let logits_cpu = logits_cpu_2d.slice(s![0, ..]).to_owned();
        log::info!("    └─ GPU->CPU copy: {:?}", copy_start.elapsed());
        // let logits_cpu = logits_3d_cpu.slice(s![0, -1, ..]).to_owned();

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
        // Pre-allocate a 1x1 buffer on the GPU that we will reuse
        let token_ndarray = Array2::<u32>::zeros((1, 1));
        GpuTensor::from_ndarray(&self.context, &token_ndarray)
    }

    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_token_id: u32) -> Result<()> {
        // The optimization: write the new token ID into the existing GPU buffer.
        // This is extremely fast compared to creating a new ndarray and copying it.
        let nti = &[new_token_id];
        let bytes: &[u8] = bytemuck::cast_slice(nti);
        self.context.queue.write_buffer(tensor.buffer(), 0, bytes);
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
        // Update the reusable token tensor
        self.context.queue.write_buffer(
            self.token_tensor.buffer(),
            0,
            bytemuck::cast_slice(&[token_id]),
        );

        let project_start = std::time::Instant::now();

        // FIX: Adjust mask length based on autoregressive mode
        let mask_len = match model.autoregressive_loop() {
            AutoregressiveLoop::Pipelined => seq_len,
            AutoregressiveLoop::Legacy => seq_len + 1, // <-- Add this!
        };

        let o = self
            .forward_and_project(model, &self.token_tensor, mask_len, cache)
            .await;

        log::info!(
            "  └─ forward_and_project total: {:?}",
            project_start.elapsed()
        );

        o
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
                self.forward_and_project(model, &prompt_tensor, prompt_len, cache)
                    .await
            }
            AutoregressiveLoop::Legacy => {
                // GPT-2: Batch process + reprocess last (matches CPU backend)

                // Step 1: Process ALL tokens at once to fill cache
                let prompt_tensor = {
                    let ndarray = Array2::from_shape_vec((1, prompt_len), initial_tokens.to_vec())?;
                    GpuTensor::from_ndarray(&self.context, &ndarray)?
                };

                // Run forward but don't return these logits
                let _ = self
                    .forward_and_project(model, &prompt_tensor, prompt_len, cache)
                    .await?;
                // Cache now has all prompt tokens

                // Step 2: Reprocess the LAST prompt token to get its logits
                // This matches OLD generator behavior and CPU backend
                let last_token = initial_tokens[prompt_len - 1];

                // Update reusable token tensor

                self.context.queue.write_buffer(
                    self.token_tensor.buffer(),
                    0,
                    bytemuck::cast_slice(&[last_token]),
                );

                // Forward pass with mask = prompt_len + 1
                self.forward_and_project(model, &self.token_tensor, prompt_len + 1, cache)
                    .await
            }
        }
    }
}
