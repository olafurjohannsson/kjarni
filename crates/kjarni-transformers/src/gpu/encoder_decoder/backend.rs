use crate::WgpuContext;
use crate::cache::Cache;
use crate::encoder_decoder::traits::{
    EncoderDecoderGenerationBackend, EncoderDecoderLanguageModel,
};
use crate::encoder_decoder::traits::{GpuCrossAttentionKVCache, GpuCrossDecoderOutput};
use crate::gpu::cache::GpuBeamKVCache;
use crate::gpu::{GpuFrameContext, GpuTensor};
use crate::models::base::ModelInput;
use anyhow::Result;
use anyhow::anyhow;
use async_trait::async_trait;
use bytemuck;
use ndarray::{Array1, Array2, Array3};
use std::sync::Arc;

#[derive(Debug)]
pub struct GpuEncoderDecoderBackend {
    pub context: Arc<WgpuContext>,
}

impl std::fmt::Debug for WgpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuContext").finish()
    }
}
impl GpuEncoderDecoderBackend {
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

#[async_trait]
impl EncoderDecoderGenerationBackend for GpuEncoderDecoderBackend {
    type Tensor = GpuSeq2SeqState;

    async fn encode(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        tokens: &[u32],
        num_beams: usize,
    ) -> Result<Self::Tensor> {
        log::info!("[GpuBackend] Encoding {} tokens...", tokens.len());
        let t_start = std::time::Instant::now();

        let seq2seq_ops = model
            .encoder_decoder_gpu_ops()
            .ok_or_else(|| anyhow!("Model does not support GPU execution"))?;

        let encoder_ops = model
            .encoder_gpu_ops()
            .ok_or_else(|| anyhow!("Model does not support GPU execution"))?;

        let pool = self.context.get_inference_pool();
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);

        let encoder_hidden_states = {
            let (encoder_cmd, pool_ref) = frame.resources();

            let input_ids_cpu = Array2::from_shape_vec((1, tokens.len()), tokens.to_vec())?;
            let input_ids_gpu = GpuTensor::from_ndarray(&self.context, &input_ids_cpu)?;
            let attention_mask_gpu =
                GpuTensor::from_ndarray(&self.context, &Array2::<f32>::ones(input_ids_cpu.dim()))?;
            encoder_ops
                .forward_tokens(
                    encoder_cmd,
                    pool_ref,
                    &self.context,
                    ModelInput::TokensGpu(&input_ids_gpu),
                    Some(&attention_mask_gpu),
                    None,
                    0,
                )?
                .last_hidden_state
        };

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
                &final_hidden_states, 
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

        let ops = model
            .encoder_decoder_gpu_ops()
            .ok_or_else(|| anyhow!("Model does not support GPU execution"))?;

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

        let (batch_size, _) = decoder_input_ids.dims2();
        let seq_len = cache.get_seq_length() + 1;
        let mask_cpu = Array2::<f32>::ones((batch_size, seq_len));
        let attention_mask = GpuTensor::from_ndarray(&self.context, &mask_cpu)?;
        let position_offset = cache.get_seq_length();

        let embed = ops.embed_decoder_tokens(
            encoder_cmd,
            pool_ref,
            ModelInput::TokensGpu(&decoder_input_ids),
            position_offset,
        )?;

        let embed_normed = ops.decoder().embed_norm(encoder_cmd, pool_ref, &embed)?;

        // Run the decoder stack
        let decoder_hidden_states = ops.decoder().forward_layers(
            encoder_cmd,
            pool_ref,
            &embed_normed,
            encoder_hidden_states,
            &attention_mask,
            position_offset,
            Some(cache),
            Some(&cross_attention_kv_cache),
            0,
            ops.decoder().num_layers(),
        )?;
        let final_hidden = ops.decoder().final_norm(
            encoder_cmd,
            pool_ref,
            &decoder_hidden_states.last_hidden_state,
        )?;

        let gpu_cache = cache.as_any_mut().downcast_mut::<GpuBeamKVCache>().unwrap();

        for (i, (k, v)) in decoder_hidden_states
            .new_self_attn_kv
            .into_iter()
            .enumerate()
        {
            match gpu_cache.update(encoder_cmd, i, &k, &v) {
                Ok(()) => {}
                Err(e) => log::error!("Failed to update GPU cache: {:?}", e),
            }
        }
        let logits_gpu = ops.project_to_logits(&mut frame, &final_hidden)?;

        frame.finish();

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, Array4};

    async fn get_test_context() -> Arc<WgpuContext> {
        WgpuContext::new()
            .await
            .expect("Failed to create WgpuContext")
    }

    #[tokio::test]
    async fn test_gpu_seq2seq_state_token_ids() {
        let ctx = get_test_context().await;
        let tokens = Array2::from_shape_vec((1, 3), vec![1u32, 2, 3]).unwrap();
        let tensor = GpuTensor::from_ndarray(&ctx, &tokens).unwrap();

        let state = GpuSeq2SeqState::TokenIds(tensor);

        match &state {
            GpuSeq2SeqState::TokenIds(t) => {
                assert_eq!(t.shape(), &[1, 3]);
            }
            _ => panic!("Expected TokenIds state"),
        }
    }

    #[tokio::test]
    async fn test_gpu_seq2seq_state_encoder_output() {
        let ctx = get_test_context().await;
        let hidden: Array3<f32> = Array3::zeros((1, 10, 64));
        let hidden_gpu = GpuTensor::from_ndarray(&ctx, &hidden).unwrap();
        let cross_kv = GpuCrossAttentionKVCache::default();

        let state = GpuSeq2SeqState::EncoderOutput {
            hidden_states: hidden_gpu,
            cross_attention_kv_cache: cross_kv,
        };

        match &state {
            GpuSeq2SeqState::EncoderOutput {
                hidden_states,
                cross_attention_kv_cache,
            } => {
                assert_eq!(hidden_states.shape(), &[1, 10, 64]);
                assert!(cross_attention_kv_cache.0.is_empty());
            }
            _ => panic!("Expected EncoderOutput state"),
        }
    }

    #[tokio::test]
    async fn test_gpu_seq2seq_state_debug() {
        let ctx = get_test_context().await;
        let tokens = Array2::from_shape_vec((1, 2), vec![1u32, 2]).unwrap();
        let tensor = GpuTensor::from_ndarray(&ctx, &tokens).unwrap();
        let state = GpuSeq2SeqState::TokenIds(tensor);

        // Should not panic - Debug is derived
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("TokenIds"));
    }

    #[tokio::test]
    async fn test_backend_new() {
        let ctx = get_test_context().await;
        let backend = GpuEncoderDecoderBackend::new(ctx.clone());

        assert!(backend.is_ok());
    }

    #[tokio::test]
    async fn test_backend_debug() {
        let ctx = get_test_context().await;
        let backend = GpuEncoderDecoderBackend::new(ctx).unwrap();

        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("GpuEncoderDecoderBackend"));
    }

    #[tokio::test]
    async fn test_create_token_tensor_single_beam() {
        let ctx = get_test_context().await;
        let backend = GpuEncoderDecoderBackend::new(ctx.clone()).unwrap();

        let tokens = vec![1u32, 2, 3, 4, 5];
        let state = backend.create_token_tensor(&tokens, 1).unwrap();

        match &state {
            GpuSeq2SeqState::TokenIds(t) => {
                assert_eq!(t.shape(), &[1, 5]);

                // Verify contents by downloading
                let downloaded: Array2<u32> = t.to_ndarray_2d().await.unwrap();
                assert_eq!(downloaded[[0, 0]], 1);
                assert_eq!(downloaded[[0, 4]], 5);
            }
            _ => panic!("Expected TokenIds state"),
        }
    }

    #[tokio::test]
    async fn test_create_token_tensor_multiple_beams() {
        let ctx = get_test_context().await;
        let backend = GpuEncoderDecoderBackend::new(ctx.clone()).unwrap();

        // 4 beams, 3 tokens each = 12 total tokens
        let tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let state = backend.create_token_tensor(&tokens, 4).unwrap();

        match &state {
            GpuSeq2SeqState::TokenIds(t) => {
                assert_eq!(t.shape(), &[4, 3]);

                let downloaded: Array2<u32> = t.to_ndarray_2d().await.unwrap();
                // First beam: [1, 2, 3]
                assert_eq!(downloaded[[0, 0]], 1);
                assert_eq!(downloaded[[0, 2]], 3);
                // Last beam: [10, 11, 12]
                assert_eq!(downloaded[[3, 0]], 10);
                assert_eq!(downloaded[[3, 2]], 12);
            }
            _ => panic!("Expected TokenIds state"),
        }
    }

    #[tokio::test]
    async fn test_create_token_tensor_zero_beams() {
        let ctx = get_test_context().await;
        let backend = GpuEncoderDecoderBackend::new(ctx.clone()).unwrap();

        let tokens = vec![1u32, 2, 3];
        // Zero beams - tokens.len() / 0 would panic, but the code handles it
        let state = backend.create_token_tensor(&tokens, 0);

        match state {
            Ok(GpuSeq2SeqState::TokenIds(t)) => {
                assert_eq!(t.shape()[0], 0); // 0 beams
            }
            Err(_) => {
            }
            _ => panic!("Unexpected state type"),
        }
    }

    #[tokio::test]
    async fn test_update_token_tensor_basic() {
        let ctx = get_test_context().await;
        let backend = GpuEncoderDecoderBackend::new(ctx.clone()).unwrap();

        // Create initial state
        let tokens = vec![1u32, 2, 3, 4];
        let mut state = backend.create_token_tensor(&tokens, 2).unwrap(); // 2 beams, 2 tokens each

        // Update with new tokens
        let new_tokens = vec![10u32, 20];
        backend
            .update_token_tensor(&mut state, &new_tokens)
            .unwrap();

        match &state {
            GpuSeq2SeqState::TokenIds(t) => {
                let downloaded: Array2<u32> = t.to_ndarray_2d().await.unwrap();
                // The write_buffer writes from offset 0, so first elements change
                assert_eq!(downloaded[[0, 0]], 10);
                assert_eq!(downloaded[[0, 1]], 20);
            }
            _ => panic!("Expected TokenIds state"),
        }
    }

    #[tokio::test]
    async fn test_update_token_tensor_wrong_type() {
        let ctx = get_test_context().await;
        let backend = GpuEncoderDecoderBackend::new(ctx.clone()).unwrap();

        // Create EncoderOutput state instead of TokenIds
        let hidden = Array3::<f32>::zeros((1, 10, 64));
        let hidden_gpu = GpuTensor::from_ndarray(&ctx, &hidden).unwrap();
        let mut state = GpuSeq2SeqState::EncoderOutput {
            hidden_states: hidden_gpu,
            cross_attention_kv_cache: GpuCrossAttentionKVCache::default(),
        };

        let new_tokens = vec![1u32];
        let result = backend.update_token_tensor(&mut state, &new_tokens);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid tensor type")
        );
    }

    // ========================================================================
    //  reorder_cache Tests
    // ========================================================================
    // ========================================================================
    //  Cache helper with correct parameters
    // ========================================================================

    /// Helper to create a properly configured GpuBeamKVCache
    fn create_test_cache(
        ctx: &Arc<WgpuContext>,
        num_layers: usize,
        num_beams: usize,
        num_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> GpuBeamKVCache {
        GpuBeamKVCache::new(ctx, num_layers, num_beams, num_heads, head_dim, capacity)
            .expect("Failed to create GpuBeamKVCache")
    }

    /// Helper to populate GPU cache with dummy data
    /// Cache expects shape: [num_beams, num_heads, seq_len, head_dim]
    async fn populate_gpu_cache(
        ctx: &Arc<WgpuContext>,
        cache: &mut GpuBeamKVCache,
        num_layers: usize,
        num_beams: usize,
        num_heads: usize,
        head_dim: usize,
    ) {
        let hidden_size = num_heads * head_dim;

        // update() expects 3D: [num_beams, seq_len=1, hidden_size]
        let k_data = Array3::from_shape_fn((num_beams, 1, hidden_size), |(b, _, h)| {
            (b * hidden_size + h) as f32
        });
        let v_data = Array3::from_shape_fn((num_beams, 1, hidden_size), |(b, _, h)| {
            (b * hidden_size + h) as f32 * 0.5
        });

        let k_tensor = GpuTensor::from_ndarray(ctx, &k_data).unwrap();
        let v_tensor = GpuTensor::from_ndarray(ctx, &v_data).unwrap();

        let mut encoder = ctx.device.create_command_encoder(&Default::default());

        for layer in 0..num_layers {
            cache
                .update(&mut encoder, layer, &k_tensor, &v_tensor)
                .unwrap();
        }

        ctx.queue.submit(std::iter::once(encoder.finish()));
        cache.increment_len(1);
    }

    // ========================================================================
    //  reorder_cache Tests
    // ========================================================================

    #[tokio::test]
    async fn test_reorder_cache_basic() {
        let ctx = get_test_context().await;
        let backend = GpuEncoderDecoderBackend::new(ctx.clone()).unwrap();

        let num_layers = 2;
        let num_beams = 4;
        let num_heads = 8;
        let head_dim = 64;
        let capacity = 128;

        let mut cache =
            create_test_cache(&ctx, num_layers, num_beams, num_heads, head_dim, capacity);
        populate_gpu_cache(&ctx, &mut cache, num_layers, num_beams, num_heads, head_dim).await;

        let indices = vec![1, 0, 2, 3];
        let result = backend.reorder_cache(&mut cache, &indices);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_reorder_cache_identity() {
        let ctx = get_test_context().await;
        let backend = GpuEncoderDecoderBackend::new(ctx.clone()).unwrap();

        let num_layers = 1;
        let num_beams = 3;
        let num_heads = 4;
        let head_dim = 32;
        let capacity = 128;

        let mut cache =
            create_test_cache(&ctx, num_layers, num_beams, num_heads, head_dim, capacity);
        populate_gpu_cache(&ctx, &mut cache, num_layers, num_beams, num_heads, head_dim).await;

        let indices = vec![0, 1, 2];
        let result = backend.reorder_cache(&mut cache, &indices);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_reorder_cache_duplicate_indices() {
        let ctx = get_test_context().await;
        let backend = GpuEncoderDecoderBackend::new(ctx.clone()).unwrap();

        let num_layers = 1;
        let num_beams = 4;
        let num_heads = 8;
        let head_dim = 64;
        let capacity = 128;

        let mut cache =
            create_test_cache(&ctx, num_layers, num_beams, num_heads, head_dim, capacity);
        populate_gpu_cache(&ctx, &mut cache, num_layers, num_beams, num_heads, head_dim).await;

        let indices = vec![0, 0, 0, 0];
        let result = backend.reorder_cache(&mut cache, &indices);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_reorder_cache_after_multiple_tokens() {
        let ctx = get_test_context().await;
        let backend = GpuEncoderDecoderBackend::new(ctx.clone()).unwrap();

        let num_layers = 2;
        let num_beams = 4;
        let num_heads = 8;
        let head_dim = 64;
        let hidden_size = num_heads * head_dim;
        let capacity = 128;

        let mut cache =
            create_test_cache(&ctx, num_layers, num_beams, num_heads, head_dim, capacity);

        // Simulate 3 decode steps
        for _step in 0..3 {
            // Shape: [num_beams, 1, hidden_size] (3D, not 4D)
            let k_data =
                Array3::from_shape_fn((num_beams, 1, hidden_size), |(b, _, h)| (b + h) as f32);
            let v_data =
                Array3::from_shape_fn((num_beams, 1, hidden_size), |(b, _, h)| (b + h) as f32);

            let k_tensor = GpuTensor::from_ndarray(&ctx, &k_data).unwrap();
            let v_tensor = GpuTensor::from_ndarray(&ctx, &v_data).unwrap();

            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            for layer in 0..num_layers {
                cache
                    .update(&mut encoder, layer, &k_tensor, &v_tensor)
                    .unwrap();
            }
            ctx.queue.submit(std::iter::once(encoder.finish()));
            cache.increment_len(1);
        }

        assert_eq!(cache.get_seq_length(), 3);

        let indices = vec![3, 2, 1, 0];
        let result = backend.reorder_cache(&mut cache, &indices);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_beam_search_flow_states() {
        let ctx = get_test_context().await;
        let backend = GpuEncoderDecoderBackend::new(ctx.clone()).unwrap();

        let num_beams = 4;
        let num_layers = 6;
        let num_heads = 8;
        let head_dim = 64;
        let capacity = 128;

        // 1. Create initial tokens
        let initial_tokens: Vec<u32> = vec![2; num_beams];
        let decoder_state = backend
            .create_token_tensor(&initial_tokens, num_beams)
            .unwrap();

        match &decoder_state {
            GpuSeq2SeqState::TokenIds(t) => {
                assert_eq!(t.shape(), &[4, 1]);
            }
            _ => panic!("Expected TokenIds"),
        }

        // 2. Update with selected tokens
        let mut decoder_state = decoder_state;
        let selected_tokens = vec![10u32, 20, 30, 40];
        backend
            .update_token_tensor(&mut decoder_state, &selected_tokens)
            .unwrap();

        // 3. Create and populate cache
        let mut cache =
            create_test_cache(&ctx, num_layers, num_beams, num_heads, head_dim, capacity);
        populate_gpu_cache(&ctx, &mut cache, num_layers, num_beams, num_heads, head_dim).await;

        // 4. Reorder cache
        let reorder_indices = vec![2, 2, 0, 1];
        backend.reorder_cache(&mut cache, &reorder_indices).unwrap();
    }

    // ========================================================================
    //  Integration Flow Tests (state management only)
    // ========================================================================

    #[tokio::test]
    async fn test_typical_generation_flow_states() {
        let ctx = get_test_context().await;
        let backend = GpuEncoderDecoderBackend::new(ctx.clone()).unwrap();

        // 1. Create initial decoder tokens (decoder_start_token)
        let initial_tokens = vec![2u32]; // e.g., <s> token
        let decoder_state = backend.create_token_tensor(&initial_tokens, 1).unwrap();

        match &decoder_state {
            GpuSeq2SeqState::TokenIds(t) => {
                assert_eq!(t.shape(), &[1, 1]);
            }
            _ => panic!("Expected TokenIds"),
        }

        // 2. Update with new token
        let mut decoder_state = decoder_state;
        let new_tokens = vec![100u32];
        backend
            .update_token_tensor(&mut decoder_state, &new_tokens)
            .unwrap();

        match &decoder_state {
            GpuSeq2SeqState::TokenIds(t) => {
                let downloaded: Array2<u32> = t.to_ndarray_2d().await.unwrap();
                assert_eq!(downloaded[[0, 0]], 100);
            }
            _ => panic!("Expected TokenIds"),
        }
    }

    // ========================================================================
    //  WgpuContext Debug Test
    // ========================================================================

    #[tokio::test]
    async fn test_wgpu_context_debug() {
        let ctx = get_test_context().await;
        let debug_str = format!("{:?}", ctx);
        assert!(debug_str.contains("WgpuContext"));
    }
}
