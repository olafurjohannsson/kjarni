//! Embedding layers for transformers

use crate::gpu_ops::primitives::scale;
use ndarray::parallel::prelude::*;
use ndarray::{Array2, Array3, Axis, s};
use tokio::sync::Mutex;
/// Configuration for embedding layers
pub struct EmbeddingConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
}

/// Combined embeddings (word + position + token type)
pub struct Embeddings {
    pub word_embeddings: Array2<f32>,
    pub position_embeddings: Option<Array2<f32>>,
    pub token_type_embeddings: Option<Array2<f32>>,
}

impl Embeddings {
    pub fn new(
        word_embeddings: Array2<f32>,
        position_embeddings: Option<Array2<f32>>,
        token_type_embeddings: Option<Array2<f32>>,
    ) -> Self {
        Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
        }
    }

    /// Embed input tokens without adding positional or token_type embeddings.
    pub fn forward_word_only(&self, input_ids: &Array2<u32>) -> Array3<f32> {
        let (batch_size, seq_len) = input_ids.dim();
        let hidden_size = self.word_embeddings.shape()[1];
        let vocab_size = self.word_embeddings.shape()[0];

        let mut hidden = Array3::<f32>::zeros((batch_size, seq_len, hidden_size));

        hidden
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(input_ids.axis_iter(Axis(0)))
            .for_each(|(mut hidden_slice, ids)| {
                for (j, &token_id) in ids.iter().enumerate() {
                    let token_id = token_id as usize;
                    // Bounds check
                    if token_id >= vocab_size {
                        panic!(
                            "Token ID {} is out of vocabulary range [0, {})",
                            token_id, vocab_size
                        );
                    }
                    let word_emb = self.word_embeddings.row(token_id);
                    hidden_slice.slice_mut(s![j, ..]).assign(&word_emb);
                }
            });
        hidden
    }

    /// Embed input tokens
    pub fn forward(
        &self,
        input_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
        position_offset: usize,
        scale_embeddings: bool,
    ) -> Array3<f32> {
        let (batch_size, seq_len) = input_ids.dim();
        let hidden_size = self.word_embeddings.shape()[1];
        let vocab_size = self.word_embeddings.shape()[0];

        // Check position embeddings constraint (only if we have them)
        if let Some(ref pos_emb) = self.position_embeddings {
            let max_position = pos_emb.shape()[0];
            if seq_len > max_position {
                panic!(
                    "Sequence length {} exceeds max position embeddings {}. Please truncate your input.",
                    seq_len, max_position
                );
            }
        }

        let mut hidden = Array3::<f32>::zeros((batch_size, seq_len, hidden_size));

        hidden
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(input_ids.axis_iter(Axis(0)))
            .for_each(|(mut hidden_slice, ids)| {
                for (j, &token_id) in ids.iter().enumerate() {
                    let token_id = token_id as usize;
                    // Bounds check
                    if token_id >= vocab_size {
                        panic!(
                            "Token ID {} is out of vocabulary range [0, {})",
                            token_id, vocab_size
                        );
                    }
                    let word_emb = self.word_embeddings.row(token_id);
                    hidden_slice.slice_mut(s![j, ..]).assign(&word_emb);
                }
            });
// --- START CPU DEBUG LOG ---
        if position_offset >= 2 { // A simple check to only log during the decoder pass
             println!("[CPU] [Decoder] Word Embeddings ONLY: shape={:?}, data={:?}", hidden.shape(), hidden.slice(s![0, 0, 0..8]));
        }
        // --- END CPU DEBUG LOG ---

        if let Some(ref pos_emb) = self.position_embeddings {
            let start_idx = position_offset;
            let end_idx = position_offset + seq_len;

            // Check if the required slice is out of bounds.
            let max_position = pos_emb.shape()[0];
            if end_idx > max_position {
                panic!(
                    "Sequence length {} with offset {} exceeds max position embeddings {}.",
                    seq_len, position_offset, max_position
                );
            }

            let pos_embeddings_to_add = pos_emb.slice(s![start_idx..end_idx, ..]);
            hidden += &pos_embeddings_to_add;
            // --- START NEW CPU DEBUG LOG ---
            if position_offset >= 2 {
                println!("[CPU] [Decoder] After Positional Add: shape={:?}, data={:?}", hidden.shape(), hidden.slice(s![0, 0, 0..8]));
            }
            // --- END NEW CPU DEBUG LOG ---
        }

        // Token type embeddings (only if present)
        if let Some(ref token_type_emb) = self.token_type_embeddings {
            let type_vocab_size = token_type_emb.shape()[0];

            if type_vocab_size == 0 {
                return hidden;
            }

            if let Some(type_ids) = token_type_ids {
                hidden
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .zip(type_ids.axis_iter(Axis(0)))
                    .for_each(|(mut hidden_slice, type_ids_row)| {
                        for (j, &type_id) in type_ids_row.iter().enumerate() {
                            let type_id = type_id as usize;
                            // Bounds check
                            if type_id >= type_vocab_size {
                                panic!(
                                    "Token type ID {} is out of range [0, {})",
                                    type_id, type_vocab_size
                                );
                            }
                            let type_emb = token_type_emb.row(type_id);
                            let mut slice = hidden_slice.slice_mut(s![j, ..]);
                            slice += &type_emb;
                        }
                    });
            } else {
                // Default to type 0 for all positions
                let type_embeddings = token_type_emb.row(0);
                hidden += &type_embeddings;
            }
        }
        if scale_embeddings {
            let hidden_size: f32 = hidden_size as f32;
            let scale_factor: f32 = hidden_size.sqrt();
            hidden *= scale_factor;
        }
        hidden
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::*;
    use crate::gpu_context::WgpuContext;
    use crate::gpu_ops::{GpuTensor, GpuTensorPool, GpuFrameContext};
    use crate::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
    use crate::traits::{LanguageModelConfig, TransformerConfig}; // Make sure traits are in scope
    use anyhow::Result;
    use ndarray::{Array2, arr2};
    use std::path::Path;
    use std::sync::Arc;

    // --- Mock Config for testing ---
    struct TestConfig {
        extra_pos_embeddings: usize,
        scale_embed: bool,
    }
    impl TransformerConfig for TestConfig {
        fn hidden_size(&self) -> usize {
            384
        }
        fn num_attention_heads(&self) -> usize {
            12
        }
        fn num_hidden_layers(&self) -> usize {
            6
        }
        fn layer_norm_eps(&self) -> f32 {
            1e-12
        }
        fn is_causal(&self) -> bool {
            false
        }
        fn is_prenorm(&self) -> bool {
            false
        }
        fn extra_pos_embeddings(&self) -> usize { 0 }
    }
    impl LanguageModelConfig for TestConfig {
        fn vocab_size(&self) -> usize {
            30522
        }
        fn decoder_start_token_id(&self) -> u32 {
            0
        }
        fn max_position_embeddings(&self) -> usize {
            512
        }
        fn intermediate_size(&self) -> usize {
            1536
        }
        fn scale_embeddings(&self) -> bool {
            self.scale_embed
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn activation_function(&self) -> crate::activations::Activation {
            crate::activations::Activation::Gelu
        }
        // Dummy names, not used in this test
        fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>) {
            (
                "embeddings.word_embeddings.weight",
                "embeddings.position_embeddings.weight",
                Some("embeddings.token_type_embeddings.weight"),
            )
        }
    }

    // Helper for float comparison
    fn assert_tensors_are_close(a: &Array3<f32>, b: &Array3<f32>, epsilon: f32) {
        assert_eq!(a.shape(), b.shape(), "Array shapes do not match");
        for (val_a, val_b) in a.iter().zip(b.iter()) {
            assert!(
                (val_a - val_b).abs() < epsilon,
                "Values differ: {} vs {}",
                val_a,
                val_b
            );
        }
    }

    #[tokio::test]
async fn test_gpu_vs_cpu_embeddings_parity() -> Result<()> {
    // --- 1. Setup Common Data and Config ---
    let context = Arc::new(WgpuContext::new().await?);
    let config = TestConfig {
        extra_pos_embeddings: 2,
        scale_embed: false,
    };

    // Create mock inputs on CPU
    let input_ids_cpu: Array2<u32> = arr2(&[[10, 20, 30], [40, 50, 60]]);
    let token_type_ids_cpu: Array2<u32> = arr2(&[[0, 0, 1], [1, 1, 0]]);

    // Keep the hardcoded path for now as requested
    let p = "/home/olafurj/.cache/edgegpt/sentence-transformers_all-MiniLM-L6-v2/";
    let weights = crate::weights::ModelWeights::new(Path::new(p))?;

    // --- 2. Run CPU Path (Expected Result) ---
    let (word_w, pos_w, type_w) = config.get_embedding_weight_names();
    let token_type_embeddings = if let Some(name) = type_w {
        Some(weights.get_array2(name)?)
    } else {
        None
    };

    let cpu_embeddings = Embeddings::new(
        weights.get_array2(word_w)?,
        Some(weights.get_array2(pos_w)?),
        token_type_embeddings,
    );
    let expected_output = cpu_embeddings.forward(
        &input_ids_cpu,
        Some(&token_type_ids_cpu),
        config.extra_pos_embeddings(),
        config.scale_embeddings(),
    );

    // --- 3. Setup GPU Modules and Inputs ---
    let gpu_embedding_weights = GpuEmbeddingWeights::new(&context, &weights, &config)?;
    let gpu_embeddings = GpuEmbeddings::new(&context)?;

    // Upload inputs
    let input_ids_gpu = GpuTensor::from_ndarray(&context, &input_ids_cpu)?;
    let token_type_ids_gpu = GpuTensor::from_ndarray(&context, &token_type_ids_cpu)?;
    
    // --- START CORRECTION ---

    // 1. Create the encoder and pool directly for the test.
    let mut encoder = context.device.create_command_encoder(&Default::default());
    let mut pool = GpuTensorPool::new(context.clone());

    // 2. Call the encode function with the raw &mut encoder and &mut pool.
    let output_gpu = gpu_embeddings.encode(
        &mut encoder,
        &gpu_embedding_weights,
        &input_ids_gpu,
        Some(&token_type_ids_gpu),
        0, // position_offset
        &config,
        &mut pool,
    )?;

    // 3. Submit the work and advance the pool's frame.
    context.queue.submit(Some(encoder.finish()));
    pool.next_frame();

    // --- END CORRECTION ---

    // --- 4. Verify Results ---
    let actual_output = output_gpu.to_ndarray_3d().await?;

    // Using a slightly more relaxed tolerance for embeddings is often wise,
    // as it involves multiple additions which can accumulate small errors.
    assert_tensors_are_close(&expected_output, &actual_output, 1e-5);
    
    Ok(())
}
    #[tokio::test]
    async fn test_gpu_vs_cpu_embeddings_parity_no_token_type_ids() -> Result<()> {
        // --- 1. Setup Common Data and Config ---
        let context = Arc::new(WgpuContext::new().await?);
        let config = TestConfig {
            extra_pos_embeddings: 2,
            scale_embed: false,
        }; // Test BART-like settings

        // Create mock inputs on CPU
        let input_ids_cpu: Array2<u32> = arr2(&[[10, 20, 30], [40, 50, 60]]);

        let p = "/home/olafurj/.cache/edgegpt/sentence-transformers_all-MiniLM-L6-v2/";

        // --- 2. Run CPU Path (Expected Result) ---
        let mut weights = crate::weights::ModelWeights::new(Path::new(p))?;
        let (word_w, pos_w, type_w) = config.get_embedding_weight_names();
        let token_type_embeddings = match type_w {
            Some(name) => Some(weights.get_array2(name)?), // Load if present
            None => None,
        };

        let cpu_embeddings = Embeddings::new(
            weights.get_array2(word_w)?,
            Some(weights.get_array2(pos_w)?),
            token_type_embeddings,
        );
        let expected_output = cpu_embeddings.forward(
            &input_ids_cpu,
            None,
            config.extra_pos_embeddings(),
            config.scale_embeddings(),
        );

        let gpu_embedding_weights = GpuEmbeddingWeights::new(&context, &weights, &config)?;
        let gpu_embeddings = GpuEmbeddings::new(&context)?;

        // Upload inputs
        let input_ids_gpu = GpuTensor::from_ndarray(&context, &input_ids_cpu)?;
        let mut pool = GpuTensorPool::new(context.clone());

        let mut encoder = context.device.create_command_encoder(&Default::default());
        let output_gpu = gpu_embeddings.encode(
            &mut encoder,
            &gpu_embedding_weights,
            &input_ids_gpu,
            None,
            0,
            &config,
            &mut pool,
        )?;
        context.queue.submit(Some(encoder.finish()));
        pool.next_frame();

        let actual_output = output_gpu.to_ndarray_3d().await?;

        // --- 4. Verification ---
        assert_tensors_are_close(&expected_output, &actual_output, 1e-4);

        Ok(())
    }
    #[test]
    fn test_embeddings_with_position() {
        // GPT-2 / BERT style
        let word_emb = Array2::ones((100, 64));
        let pos_emb = Array2::ones((512, 64));

        let embeddings = Embeddings::new(word_emb, Some(pos_emb), None);

        let input_ids = Array2::zeros((2, 10));
        let output = embeddings.forward(&input_ids, None, 0, false);

        assert_eq!(output.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_embeddings_without_position() {
        // LLaMA / RoPE style
        let word_emb = Array2::ones((100, 64));

        let embeddings = Embeddings::new(word_emb, None, None);

        let input_ids = Array2::zeros((2, 10));
        let output = embeddings.forward(&input_ids, None, 0, false);

        assert_eq!(output.shape(), &[2, 10, 64]);
        // Should work without position embeddings
    }

    #[test]
    fn test_embeddings_with_token_types() {
        let word_emb = Array2::ones((100, 64));
        let pos_emb = Array2::ones((512, 64));
        let token_type_emb = Array2::ones((2, 64));

        let embeddings = Embeddings::new(word_emb, Some(pos_emb), Some(token_type_emb));

        let input_ids = Array2::zeros((2, 10));
        let token_type_ids = Array2::zeros((2, 10));
        let output = embeddings.forward(&input_ids, Some(&token_type_ids), 0, false);

        assert_eq!(output.shape(), &[2, 10, 64]);
    }

    #[test]
    #[should_panic(expected = "exceeds max position embeddings")]
    fn test_sequence_too_long() {
        let word_emb = Array2::ones((100, 64));
        let pos_emb = Array2::ones((10, 64)); // Only 10 positions

        let embeddings = Embeddings::new(word_emb, Some(pos_emb), None);

        let input_ids = Array2::zeros((2, 20)); // 20 tokens - too long!
        let _ = embeddings.forward(&input_ids, None, 0, false);
    }

    #[test]
    fn test_llama_long_sequence() {
        // LLaMA without position embeddings can handle any length
        let word_emb = Array2::ones((100, 64));

        let embeddings = Embeddings::new(word_emb, None, None);

        let input_ids = Array2::zeros((2, 1000)); // Very long sequence
        let output = embeddings.forward(&input_ids, None, 0, false);

        assert_eq!(output.shape(), &[2, 1000, 64]);
        // Should work fine - RoPE handles position in attention layer
    }

    #[test]
    #[should_panic(expected = "out of vocabulary range")]
    fn test_invalid_token_id() {
        let word_emb = Array2::ones((100, 64));
        let pos_emb = Array2::ones((512, 64));

        let embeddings = Embeddings::new(word_emb, Some(pos_emb), None);

        let mut input_ids = Array2::zeros((2, 10));
        input_ids[[0, 0]] = 150 as u32; // Out of vocab range [0, 100)

        let _ = embeddings.forward(&input_ids, None, 0, false);
    }
}
