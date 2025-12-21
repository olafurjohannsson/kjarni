//! A GPU-accelerated Embedding block.
//!
//! This module defines a `GpuEmbeddings` struct that encapsulates the logic for
//! performing token lookups and combining them with positional and token-type
//! embeddings, all on the GPU. It is designed for performance by minimizing
//! CPU-GPU data transfers.
//!
//! # Architecture
//!
//! 1.  **`GpuEmbeddingWeights` struct:** A container for the embedding tables (word,
//!     position, token type) that have been preloaded onto the GPU. Its constructor
//!     is the gatekeeper for ensuring weights are present.
//! 2.  **`GpuEmbeddings` struct:** The main public-facing struct. It owns the compiled
//!     GPU kernels required for the embedding process (lookup, add, scale).
//! 3.  **`encode` method:** The primary entry point. It orchestrates a sequence of
//!     GPU kernels to produce the final embedding tensor.
//! 4.  **Specialized Kernels:**
//!     - A `lookup` kernel to translate `u32` token IDs into `f32` vectors.
//!     - An `add` kernel (potentially with offset support) to combine embeddings.
//!     - A `scale` kernel to apply conditional scaling.
//!
//! # INVARIANT
//!
//! The constructor for `GpuEmbeddingWeights` handles loading the weights to the GPU.
//! The `GpuEmbeddings` struct is stateless and simply orchestrates the kernels.

use crate::gpu_context::WgpuContext;
use crate::gpu_ops::primitives::add::GpuAdd;
use crate::gpu_ops::primitives::lookup::GpuLookup;
use crate::gpu_ops::primitives::scale::GpuScale;
use crate::gpu_ops::{GpuFrameContext, GpuTensor, GpuTensorPool};
use crate::traits::LanguageModelConfig;
use crate::weights::ModelWeights;
use anyhow::Result;
use std::sync::Arc;

/// Holds all embedding tables as GPU tensors.
pub struct GpuEmbeddingWeights {
    pub word_embeddings: GpuTensor,
    pub position_embeddings: Option<GpuTensor>,
    pub token_type_embeddings: Option<GpuTensor>,
}

impl GpuEmbeddingWeights {
    /// Creates and uploads embedding weights to the GPU from CPU-side ndarrays.
    pub fn new(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        config: &dyn LanguageModelConfig,
    ) -> Result<Self> {
        let (word_w, pos_w, type_w) = config.get_embedding_weight_names();

        let word_embeddings_cpu = weights.get_array2(word_w)?;
        let word_embeddings = GpuTensor::from_ndarray(context, &word_embeddings_cpu)?;

        let position_embeddings = if !pos_w.is_empty() {
            let pos_cpu = weights.get_array2(pos_w)?;
            Some(GpuTensor::from_ndarray(context, &pos_cpu)?)
        } else {
            None
        };

        let token_type_embeddings = if let Some(name) = type_w {
            let type_cpu = weights.get_array2(name)?;
            Some(GpuTensor::from_ndarray(context, &type_cpu)?)
        } else {
            None
        };

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
        })
    }
}

/// A GPU-accelerated Embeddings block.
pub struct GpuEmbeddings {
    pub lookup: GpuLookup,
    pub add: GpuAdd,
    pub scale: GpuScale,
    context: Arc<WgpuContext>,
}

impl GpuEmbeddings {
    /// Creates a new `GpuEmbeddings` block.
    ///
    /// This struct is stateless and holds the compiled kernels needed to perform
    /// the embedding operations on the GPU.
    pub fn new(context: &Arc<WgpuContext>) -> Result<Self> {
        Ok(Self {
            lookup: GpuLookup::new(context),
            add: GpuAdd::new(context),
            scale: GpuScale::new(context),
            context: context.clone(),
        })
    }

    /// Encodes the complete embedding generation pass into the command encoder.
    ///
    /// This method avoids CPU-GPU transfers by performing all lookups and additions
    /// directly on the GPU.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        weights: &GpuEmbeddingWeights,
        input_ids: &GpuTensor,              // u32 tensor
        token_type_ids: Option<&GpuTensor>, // u32 tensor
        position_offset: usize,
        config: &dyn LanguageModelConfig,
        pool: &mut GpuTensorPool,
    ) -> Result<GpuTensor> {
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];
        let hidden_size = config.hidden_size();

        // 1. Word Embedding Lookup
        let mut hidden_states = pool.get(vec![batch_size, seq_len, hidden_size]);
        self.lookup
            .encode(encoder, &weights.word_embeddings, input_ids, &hidden_states);

        // 2. Add Positional Embeddings (with offset)
        // if let Some(pos_embeddings) = &weights.position_embeddings {
        //     // 2a. Calculate the final offset.
        //     let final_offset = position_offset + config.extra_pos_embeddings();

        //     // 2b. Create the positional IDs on the CPU.
        //     // This is efficient because seq_len is usually small (1 during generation).
        //     let pos_ids_cpu: Vec<u32> = (0..seq_len as u32)
        //         .map(|i| final_offset as u32 + i)
        //         .collect();
        //     let pos_ids_ndarray = ndarray::Array2::from_shape_vec((1, seq_len), pos_ids_cpu)?;

        //     // 2c. Upload the positional IDs to the GPU.
        //     let pos_ids_gpu = GpuTensor::from_ndarray(&self.context, &pos_ids_ndarray)?;

        //     // 2d. Perform a standard lookup to get the positional embedding vectors.
        //     let pos_embedding_vectors = pool.get(hidden_states.shape().to_vec());
        //     self.lookup.encode(
        //         encoder,
        //         pos_embeddings,
        //         &pos_ids_gpu,
        //         &pos_embedding_vectors,
        //     );

        //     // 2e. Perform a standard element-wise add.
        //     let pos_add_out = pool.get(hidden_states.shape().to_vec());
        //     self.add.encode_elementwise(
        //         encoder,
        //         &hidden_states,
        //         &pos_embedding_vectors,
        //         &pos_add_out,
        //     );
        //     hidden_states = pos_add_out;
        // }
        if let Some(pos_embeddings) = &weights.position_embeddings {
            let pos_add_out = pool.get(hidden_states.shape().to_vec());
            // GPT-style models often have a fixed offset (e.g., 2 for BART) in their
            // position embedding table that needs to be added to the dynamic offset.
            let final_offset = position_offset + config.extra_pos_embeddings();
            // Call the new, specific method
            self.add.encode_broadcast_offset(
                encoder,
                &hidden_states,
                pos_embeddings,
                final_offset,
                &pos_add_out,
            );
            hidden_states = pos_add_out;
        }

        // 3. Add Token Type Embeddings (using composition)
        if let Some(token_type_embeddings) = &weights.token_type_embeddings {
            // Step 3a: Create the tensor of token type vectors to add.
            let token_type_vectors = pool.get(hidden_states.shape().to_vec());

            if let Some(type_ids) = token_type_ids {
                self.lookup.encode(
                    encoder,
                    token_type_embeddings,
                    type_ids,
                    &token_type_vectors,
                );
            } else {
                // No IDs provided. Create a temporary [batch, seq] tensor of zeros and use that for the lookup to get the type 0 embedding everywhere.
                let zeros_cpu = ndarray::Array2::<u32>::zeros((batch_size, seq_len));
                let zeros_gpu = GpuTensor::from_ndarray(&self.context, &zeros_cpu)?;
                self.lookup.encode(
                    encoder,
                    token_type_embeddings,
                    &zeros_gpu,
                    &token_type_vectors,
                );
            }

            // Step 3b: Add the resulting vectors to the hidden states.
            let type_add_out = pool.get(hidden_states.shape().to_vec());
            self.add.encode_elementwise(
                encoder,
                &hidden_states,
                &token_type_vectors,
                &type_add_out,
            );
            hidden_states = type_add_out;
        }

        // 4. Apply Scaling
        if config.scale_embeddings() {
            let scale_factor = (hidden_size as f32).sqrt();
            let scale_out = pool.get(hidden_states.shape().to_vec());

            // Call the new, correct method
            self.scale.encode_out_of_place(
                encoder,
                &hidden_states, // input
                &scale_out,     // output
                scale_factor,
            );

            hidden_states = scale_out;
        }

        Ok(hidden_states)
    }
}

#[cfg(test)]
mod embedding_parity_tests {
    use super::*;
    use crate::WgpuContext;
    use crate::activations::Activation;
    use crate::embeddings::Embeddings;
    use crate::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
    use crate::gpu_ops::{GpuFrameContext, GpuTensor, GpuTensorPool};
    use crate::traits::LanguageModelConfig;
    use crate::traits::TransformerConfig;
    use anyhow::Result;
    use ndarray::{Array2, Array3};
    use std::any::Any;
    use std::sync::Arc;
    /// Mock config for testing embeddings
    struct MockEmbedConfig {
        hidden_size: usize,
        vocab_size: usize,
        max_position: usize,
        scale: bool,
    }

    impl TransformerConfig for MockEmbedConfig {
        fn hidden_size(&self) -> usize {
            self.hidden_size
        }
        fn num_attention_heads(&self) -> usize {
            4
        }
        fn num_hidden_layers(&self) -> usize {
            1
        }
        fn layer_norm_eps(&self) -> f32 {
            1e-5
        }
        fn is_causal(&self) -> bool {
            false
        }
        fn is_prenorm(&self) -> bool {
            false
        }
    }

    impl LanguageModelConfig for MockEmbedConfig {
        fn vocab_size(&self) -> usize {
            self.vocab_size
        }
        fn max_position_embeddings(&self) -> usize {
            self.max_position
        }
        fn intermediate_size(&self) -> usize {
            self.hidden_size * 4
        }

        fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>) {
            ("word", "pos", None)
        }

        fn scale_embeddings(&self) -> bool {
            self.scale
        }

        fn activation_function(&self) -> Activation {
            Activation::Gelu
        }
        fn decoder_start_token_id(&self) -> u32 {
            0
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn assert_close(cpu: &Array3<f32>, gpu: &Array3<f32>, atol: f32, name: &str) {
        assert_eq!(cpu.shape(), gpu.shape(), "{} shape mismatch", name);
        let max_diff = cpu
            .iter()
            .zip(gpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("[{}] Max diff: {:.6}", name, max_diff);
        println!(
            "  CPU first 5: {:?}",
            cpu.iter().take(5).collect::<Vec<_>>()
        );
        println!(
            "  GPU first 5: {:?}",
            gpu.iter().take(5).collect::<Vec<_>>()
        );

        if max_diff > atol {
            panic!("[FAIL] {} - max_diff {} > atol {}", name, max_diff, atol);
        }
        println!("[PASS] {}\n", name);
    }

    fn make_embeddings(
        vocab_size: usize,
        max_pos: usize,
        hidden_size: usize,
    ) -> (Array2<f32>, Array2<f32>) {
        let word_emb = Array2::from_shape_fn((vocab_size, hidden_size), |(i, j)| {
            ((i * hidden_size + j) % 1000) as f32 * 0.001 - 0.5
        });
        let pos_emb = Array2::from_shape_fn((max_pos, hidden_size), |(i, j)| {
            ((i * hidden_size + j + 5000) % 1000) as f32 * 0.001 - 0.5
        });
        (word_emb, pos_emb)
    }

    #[tokio::test]
    async fn test_word_embedding_lookup_only() -> Result<()> {
        let ctx = Arc::new(WgpuContext::new().await?);

        let vocab_size = 1000;
        let hidden_size = 64;
        let batch_size = 2;
        let seq_len = 6;

        let (word_emb, _) = make_embeddings(vocab_size, 128, hidden_size);

        // CPU

        let cpu_embed = Embeddings::new(
            crate::embeddings::EmbeddingData::F32(word_emb.clone()),
            None,
            None,
        );

        // GPU
        let gpu_weights = GpuEmbeddingWeights {
            word_embeddings: GpuTensor::from_ndarray(&ctx, &word_emb)?,
            position_embeddings: None,
            token_type_embeddings: None,
        };
        let gpu_embed = GpuEmbeddings::new(&ctx)?;

        let config = MockEmbedConfig {
            hidden_size,
            vocab_size,
            max_position: 128,
            scale: false,
        };

        let input_ids: Vec<u32> = vec![1, 50, 100, 200, 500, 999, 0, 10, 20, 30, 40, 50];
        let input_ids_cpu = Array2::from_shape_vec((batch_size, seq_len), input_ids.clone())?;
        let input_ids_gpu = GpuTensor::from_ndarray(&ctx, &input_ids_cpu)?;

        let cpu_output = cpu_embed.forward(&input_ids_cpu, None, 0, false);

        let pool = ctx.get_inference_pool();
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (encoder, pool_ref) = frame.resources();

        let gpu_output = gpu_embed.encode(
            encoder,
            &gpu_weights,
            &input_ids_gpu,
            None,
            0,
            &config,
            pool_ref,
        )?;

        frame.finish();
        let gpu_output_cpu = gpu_output.to_ndarray_3d::<f32>().await?;

        assert_close(&cpu_output, &gpu_output_cpu, 1e-5, "Word Embedding Lookup");
        Ok(())
    }

    #[tokio::test]
    async fn test_word_plus_position_embeddings() -> Result<()> {
        let ctx = Arc::new(WgpuContext::new().await?);

        let vocab_size = 1000;
        let hidden_size = 64;
        let max_pos = 128;
        let batch_size = 2;
        let seq_len = 6;

        let (word_emb, pos_emb) = make_embeddings(vocab_size, max_pos, hidden_size);

        let cpu_embed = Embeddings::new(
            crate::embeddings::EmbeddingData::F32(word_emb.clone()),
            Some(pos_emb.clone()),
            None,
        );

        let gpu_weights = GpuEmbeddingWeights {
            word_embeddings: GpuTensor::from_ndarray(&ctx, &word_emb)?,
            position_embeddings: Some(GpuTensor::from_ndarray(&ctx, &pos_emb)?),
            token_type_embeddings: None,
        };
        let gpu_embed = GpuEmbeddings::new(&ctx)?;

        let config = MockEmbedConfig {
            hidden_size,
            vocab_size,
            max_position: max_pos,
            scale: false,
        };

        let input_ids: Vec<u32> = vec![1, 50, 100, 200, 500, 999, 0, 10, 20, 30, 40, 50];
        let input_ids_cpu = Array2::from_shape_vec((batch_size, seq_len), input_ids.clone())?;
        let input_ids_gpu = GpuTensor::from_ndarray(&ctx, &input_ids_cpu)?;

        let cpu_output = cpu_embed.forward(&input_ids_cpu, None, 0, false);

        let pool = ctx.get_inference_pool();
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (encoder, pool_ref) = frame.resources();

        let gpu_output = gpu_embed.encode(
            encoder,
            &gpu_weights,
            &input_ids_gpu,
            None,
            0,
            &config,
            pool_ref,
        )?;

        frame.finish();
        let gpu_output_cpu = gpu_output.to_ndarray_3d::<f32>().await?;

        assert_close(
            &cpu_output,
            &gpu_output_cpu,
            1e-5,
            "Word + Position Embeddings",
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_embeddings_with_position_offset() -> Result<()> {
        let ctx = Arc::new(WgpuContext::new().await?);

        let vocab_size = 1000;
        let hidden_size = 64;
        let max_pos = 128;
        let batch_size = 1;
        let seq_len = 4;
        let position_offset = 2; // BART style

        let (word_emb, pos_emb) = make_embeddings(vocab_size, max_pos, hidden_size);

        let cpu_embed = Embeddings::new(
            crate::embeddings::EmbeddingData::F32(word_emb.clone()),
            Some(pos_emb.clone()),
            None,
        );

        let gpu_weights = GpuEmbeddingWeights {
            word_embeddings: GpuTensor::from_ndarray(&ctx, &word_emb)?,
            position_embeddings: Some(GpuTensor::from_ndarray(&ctx, &pos_emb)?),
            token_type_embeddings: None,
        };
        let gpu_embed = GpuEmbeddings::new(&ctx)?;

        let config = MockEmbedConfig {
            hidden_size,
            vocab_size,
            max_position: max_pos,
            scale: false,
        };

        let input_ids: Vec<u32> = vec![1, 50, 100, 200];
        let input_ids_cpu = Array2::from_shape_vec((batch_size, seq_len), input_ids.clone())?;
        let input_ids_gpu = GpuTensor::from_ndarray(&ctx, &input_ids_cpu)?;

        // CPU with offset
        let cpu_output = cpu_embed.forward(&input_ids_cpu, None, position_offset, false);

        let pool = ctx.get_inference_pool();
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (encoder, pool_ref) = frame.resources();

        // GPU with offset
        let gpu_output = gpu_embed.encode(
            encoder,
            &gpu_weights,
            &input_ids_gpu,
            None,
            position_offset,
            &config,
            pool_ref,
        )?;

        frame.finish();
        let gpu_output_cpu = gpu_output.to_ndarray_3d::<f32>().await?;

        assert_close(
            &cpu_output,
            &gpu_output_cpu,
            1e-5,
            "Embeddings with offset=2",
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_embeddings_with_scaling() -> Result<()> {
        let ctx = Arc::new(WgpuContext::new().await?);

        let vocab_size = 1000;
        let hidden_size = 64;
        let max_pos = 128;
        let batch_size = 1;
        let seq_len = 4;

        let (word_emb, pos_emb) = make_embeddings(vocab_size, max_pos, hidden_size);

        let cpu_embed = Embeddings::new(
            crate::embeddings::EmbeddingData::F32(word_emb.clone()),
            Some(pos_emb.clone()),
            None,
        );

        let gpu_weights = GpuEmbeddingWeights {
            word_embeddings: GpuTensor::from_ndarray(&ctx, &word_emb)?,
            position_embeddings: Some(GpuTensor::from_ndarray(&ctx, &pos_emb)?),
            token_type_embeddings: None,
        };
        let gpu_embed = GpuEmbeddings::new(&ctx)?;

        let config = MockEmbedConfig {
            hidden_size,
            vocab_size,
            max_position: max_pos,
            scale: true, // Enable scaling
        };

        let input_ids: Vec<u32> = vec![1, 50, 100, 200];
        let input_ids_cpu = Array2::from_shape_vec((batch_size, seq_len), input_ids.clone())?;
        let input_ids_gpu = GpuTensor::from_ndarray(&ctx, &input_ids_cpu)?;

        // CPU with scaling
        let cpu_output = cpu_embed.forward(&input_ids_cpu, None, 0, true);

        let pool = ctx.get_inference_pool();
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (encoder, pool_ref) = frame.resources();

        // GPU with scaling
        let gpu_output = gpu_embed.encode(
            encoder,
            &gpu_weights,
            &input_ids_gpu,
            None,
            0,
            &config,
            pool_ref,
        )?;

        frame.finish();
        let gpu_output_cpu = gpu_output.to_ndarray_3d::<f32>().await?;

        assert_close(
            &cpu_output,
            &gpu_output_cpu,
            1e-4,
            "Embeddings with scaling",
        );
        Ok(())
    }
}
