use crate::cache::{Cache, CpuBeamKVCache};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3};

use crate::encoder_decoder::{
    EncoderDecoderGenerationBackend, EncoderDecoderLanguageModel, traits::CpuCrossAttentionKVCache,
};

#[derive(Debug)]
pub enum CpuSeq2SeqState {
    U32(Array2<u32>),
    EncoderState {
        /// The final hidden states from the encoder, broadcasted for beam search.
        hidden_states: Array3<f32>,
        /// The pre-computed cross-attention Key/Value cache for each decoder layer.
        cross_attention_kv_cache: CpuCrossAttentionKVCache,
        /// Identifies padding in the source sentence
        encoder_padding_mask: Array2<f32>,
    },
}

#[derive(Debug)]
pub struct CpuBackend;

#[async_trait]
impl EncoderDecoderGenerationBackend for CpuBackend {
    type Tensor = CpuSeq2SeqState;

    async fn encode(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        tokens: &[u32],
        num_beams: usize,
    ) -> Result<Self::Tensor> {
        let seq2seq_ops = model
            .encoder_decoder_cpu_ops()
            .ok_or_else(|| anyhow!("Model does not support CPU execution"))?;
        let encoder_ops = model
            .encoder_cpu_ops()
            .ok_or_else(|| anyhow!("Model does not support CPU execution"))?;

        let input_ids = Array2::from_shape_vec((1, tokens.len()), tokens.to_vec())?;
        let attention_mask = Array2::ones(input_ids.dim());

        // Use the new trait path: embed_tokens → embed_norm → forward (layers + final_norm)
        let encoder_output = encoder_ops
            .forward_tokens(&input_ids, Some(&attention_mask), None, 0)?
            .last_hidden_state;

        let (final_state, final_mask) = if num_beams > 1 {
            let s = seq2seq_ops.broadcast_encoder_states(&encoder_output, num_beams)?;
            let m = attention_mask
                .broadcast((num_beams, tokens.len()))
                .ok_or_else(|| anyhow!("Mask broadcast failed"))?
                .to_owned();
            (s, m)
        } else {
            (encoder_output, attention_mask)
        };

        let cross_cache = seq2seq_ops
            .decoder()
            .precompute_cross_attention_kv(&final_state)?;

        Ok(CpuSeq2SeqState::EncoderState {
            hidden_states: final_state,
            cross_attention_kv_cache: cross_cache,
            encoder_padding_mask: final_mask,
        })
    }

    async fn decode_step(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        decoder_tokens: &Self::Tensor,
        encoder_state: &Self::Tensor,
        cache: &mut dyn Cache,
    ) -> Result<Array3<f32>> {
        let ops = model
            .encoder_decoder_cpu_ops()
            .ok_or_else(|| anyhow!("Model does not support CPU execution"))?;

        let CpuSeq2SeqState::U32(tokens) = decoder_tokens else {
            return Err(anyhow!("Invalid tensor type for decoder_tokens"));
        };
        // println!("=== DECODE STEP ===");
        // println!("Decoder input tokens: {:?}", tokens);

        let CpuSeq2SeqState::EncoderState {
            hidden_states: enc_state,
            cross_attention_kv_cache: cross_kv,
            encoder_padding_mask,
        } = encoder_state
        else {
            return Err(anyhow!("Invalid tensor type for encoder_state"));
        };

        // create decoder padding mask, usually all 1s during auto regressive decode
        let attention_mask = Array2::ones(tokens.dim());

        let decoder_output = ops.decoder().forward(
            tokens,
            enc_state,
            Some(&attention_mask),
            Some(encoder_padding_mask),
            Some(cache),
            Some(cross_kv),
        )?;

        let cpu_cache = cache
            .as_any_mut()
            .downcast_mut::<CpuBeamKVCache>()
            .ok_or_else(|| anyhow!("Expected CpuBeamKVCache"))?;
        for (i, (k, v)) in decoder_output.new_self_attn_kv.into_iter().enumerate() {
            cpu_cache.update(i, &k, &v)?;
        }
        // TODO: missing increment_len ?
        let logits = ops.project_to_logits(&decoder_output.last_hidden_state)?;

        let last_logits = logits.slice(ndarray::s![0, -1, ..]);
        let mut indexed: Vec<(usize, f32)> = last_logits.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(logits)
    }

    fn create_token_tensor(&self, tokens: &[u32], num_beams: usize) -> Result<Self::Tensor> {
        let seq_len = if num_beams > 0 {
            tokens.len() / num_beams
        } else {
            0
        };
        let tokens_ndarray = Array2::from_shape_vec((num_beams, seq_len), tokens.to_vec())?;
        Ok(CpuSeq2SeqState::U32(tokens_ndarray))
    }

    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_tokens: &[u32]) -> Result<()> {
        let current_tensor = match tensor {
            CpuSeq2SeqState::U32(t) => t,
            _ => {
                return Err(anyhow!(
                    "Invalid tensor type for update_token_tensor, expected U32"
                ));
            }
        };
        let new_tokens_ndarray =
            Array2::from_shape_vec((new_tokens.len(), 1), new_tokens.to_vec())?;
        *current_tensor = new_tokens_ndarray;
        Ok(())
    }

    fn reorder_cache(&self, cache: &mut dyn Cache, indices: &[usize]) -> Result<()> {
        let cpu_cache = cache
            .as_any_mut()
            .downcast_mut::<CpuBeamKVCache>()
            .ok_or_else(|| anyhow!("CpuBackend requires a CpuBeamKVCache"))?;
        cpu_cache.reorder(indices);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // ========================================================================
    //  CpuSeq2SeqState Tests
    // ========================================================================

    #[test]
    fn test_cpu_seq2seq_state_u32_creation() {
        let tokens = Array2::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let state = CpuSeq2SeqState::U32(tokens.clone());

        match state {
            CpuSeq2SeqState::U32(t) => {
                assert_eq!(t.shape(), &[2, 3]);
                assert_eq!(t[[0, 0]], 1);
                assert_eq!(t[[1, 2]], 6);
            }
            _ => panic!("Expected U32 state"),
        }
    }

    #[test]
    fn test_cpu_seq2seq_state_encoder_state_creation() {
        let hidden = Array3::zeros((2, 10, 64));
        let mask = Array2::ones((2, 10));
        let cross_kv = CpuCrossAttentionKVCache::default();

        let state = CpuSeq2SeqState::EncoderState {
            hidden_states: hidden.clone(),
            cross_attention_kv_cache: cross_kv,
            encoder_padding_mask: mask.clone(),
        };

        match state {
            CpuSeq2SeqState::EncoderState {
                hidden_states,
                cross_attention_kv_cache,
                encoder_padding_mask,
            } => {
                assert_eq!(hidden_states.shape(), &[2, 10, 64]);
                assert_eq!(encoder_padding_mask.shape(), &[2, 10]);
                assert!(cross_attention_kv_cache.0.is_empty());
            }
            _ => panic!("Expected EncoderState"),
        }
    }

    #[test]
    fn test_cpu_seq2seq_state_debug() {
        let tokens = Array2::from_shape_vec((1, 2), vec![1, 2]).unwrap();
        let state = CpuSeq2SeqState::U32(tokens);

        // Should not panic - Debug is derived
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("U32"));
    }

    // ========================================================================
    //  CpuBackend::create_token_tensor Tests
    // ========================================================================

    #[test]
    fn test_create_token_tensor_single_beam() {
        let backend = CpuBackend;
        let tokens = vec![1u32, 2, 3, 4, 5];

        let state = backend.create_token_tensor(&tokens, 1).unwrap();

        match state {
            CpuSeq2SeqState::U32(t) => {
                assert_eq!(t.shape(), &[1, 5]);
                assert_eq!(t[[0, 0]], 1);
                assert_eq!(t[[0, 4]], 5);
            }
            _ => panic!("Expected U32 state"),
        }
    }

    #[test]
    fn test_create_token_tensor_multiple_beams() {
        let backend = CpuBackend;
        // 4 beams, 3 tokens each = 12 total tokens
        let tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        let state = backend.create_token_tensor(&tokens, 4).unwrap();

        match state {
            CpuSeq2SeqState::U32(t) => {
                assert_eq!(t.shape(), &[4, 3]);
                // First beam: [1, 2, 3]
                assert_eq!(t[[0, 0]], 1);
                assert_eq!(t[[0, 2]], 3);
                // Last beam: [10, 11, 12]
                assert_eq!(t[[3, 0]], 10);
                assert_eq!(t[[3, 2]], 12);
            }
            _ => panic!("Expected U32 state"),
        }
    }

    #[test]
    fn test_create_token_tensor_empty() {
        let backend = CpuBackend;
        let tokens: Vec<u32> = vec![];

        let state = backend.create_token_tensor(&tokens, 1).unwrap();

        match state {
            CpuSeq2SeqState::U32(t) => {
                assert_eq!(t.shape(), &[1, 0]);
            }
            _ => panic!("Expected U32 state"),
        }
    }

    #[test]
    fn test_create_token_tensor_zero_beams() {
        let backend = CpuBackend {};
        let tokens = &[1u32, 2, 3];
        let num_beams = 0;

        // Should return error, not panic
        let result = backend.create_token_tensor(tokens, num_beams);
        assert!(result.is_err(), "Expected error for zero beams");
    }

    // ========================================================================
    //  CpuBackend::update_token_tensor Tests
    // ========================================================================

    #[test]
    fn test_update_token_tensor_basic() {
        let backend = CpuBackend;
        let mut state = CpuSeq2SeqState::U32(Array2::zeros((2, 3)));

        let new_tokens = vec![10u32, 20];
        backend
            .update_token_tensor(&mut state, &new_tokens)
            .unwrap();

        match state {
            CpuSeq2SeqState::U32(t) => {
                // Shape becomes (num_new_tokens, 1)
                assert_eq!(t.shape(), &[2, 1]);
                assert_eq!(t[[0, 0]], 10);
                assert_eq!(t[[1, 0]], 20);
            }
            _ => panic!("Expected U32 state"),
        }
    }

    #[test]
    fn test_update_token_tensor_single_token() {
        let backend = CpuBackend;
        let mut state = CpuSeq2SeqState::U32(Array2::zeros((1, 5)));

        let new_tokens = vec![42u32];
        backend
            .update_token_tensor(&mut state, &new_tokens)
            .unwrap();

        match state {
            CpuSeq2SeqState::U32(t) => {
                assert_eq!(t.shape(), &[1, 1]);
                assert_eq!(t[[0, 0]], 42);
            }
            _ => panic!("Expected U32 state"),
        }
    }

    #[test]
    fn test_update_token_tensor_wrong_type() {
        let backend = CpuBackend;
        let mut state = CpuSeq2SeqState::EncoderState {
            hidden_states: Array3::zeros((1, 1, 1)),
            cross_attention_kv_cache: CpuCrossAttentionKVCache::default(),
            encoder_padding_mask: Array2::zeros((1, 1)),
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
    //  CpuBackend::reorder_cache Tests
    // ===========================================
    #[test]
    fn test_reorder_cache_basic() {
        let backend = CpuBackend;
        let num_layers = 2;
        let num_beams = 4;
        let max_len = 128;
        let hidden_size = 64;

        let mut cache = CpuBeamKVCache::new(num_layers, num_beams, max_len, hidden_size);

        // Populate cache with one token for all layers
        // Shape: (num_beams, seq_len=1, hidden_size)
        let k = Array3::from_shape_fn((num_beams, 1, hidden_size), |(b, _, h)| {
            (b * hidden_size + h) as f32
        });
        let v = Array3::from_shape_fn((num_beams, 1, hidden_size), |(b, _, h)| {
            (b * hidden_size + h) as f32 * 0.5
        });

        // Update all layers
        for layer in 0..num_layers {
            cache.update(layer, &k, &v).unwrap();
        }
        // CRITICAL: increment seq_length after updating all layers
        cache.increment_len(1);

        // Now reorder should work
        let indices = vec![1, 0, 2, 3];
        let result = backend.reorder_cache(&mut cache, &indices);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reorder_cache_identity() {
        let backend = CpuBackend;
        let num_layers = 1;
        let num_beams = 3;
        let hidden_size = 64;

        let mut cache = CpuBeamKVCache::new(num_layers, num_beams, 128, hidden_size);

        // Populate and increment
        let k = Array3::zeros((num_beams, 1, hidden_size));
        let v = Array3::zeros((num_beams, 1, hidden_size));
        cache.update(0, &k, &v).unwrap();
        cache.increment_len(1);

        // Identity reorder
        let indices = vec![0, 1, 2];
        let result = backend.reorder_cache(&mut cache, &indices);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reorder_cache_duplicate_indices() {
        let backend = CpuBackend;
        let num_layers = 1;
        let num_beams = 4;
        let hidden_size = 64;

        let mut cache = CpuBeamKVCache::new(num_layers, num_beams, 128, hidden_size);

        // Populate with distinct values per beam
        let k = Array3::from_shape_fn((num_beams, 1, hidden_size), |(b, _, h)| {
            (b * 100 + h) as f32
        });
        let v = Array3::from_shape_fn((num_beams, 1, hidden_size), |(b, _, h)| {
            (b * 100 + h) as f32
        });
        cache.update(0, &k, &v).unwrap();
        cache.increment_len(1);

        // Duplicate beam 0 to all positions
        let indices = vec![0, 0, 0, 0];
        let result = backend.reorder_cache(&mut cache, &indices);
        assert!(result.is_ok());
    }

    #[test]
    fn test_beam_search_flow_states() {
        let backend = CpuBackend;
        let num_beams = 4;
        let num_layers = 6;
        let hidden_size = 64;

        // 1. Create initial tokens for beam search
        let initial_tokens: Vec<u32> = vec![2; num_beams];
        let decoder_state = backend
            .create_token_tensor(&initial_tokens, num_beams)
            .unwrap();

        match &decoder_state {
            CpuSeq2SeqState::U32(t) => {
                assert_eq!(t.shape(), &[4, 1]);
            }
            _ => panic!("Expected U32"),
        }

        // 2. After first decode step, update with selected tokens
        let mut decoder_state = decoder_state;
        let selected_tokens = vec![10u32, 20, 30, 40];
        backend
            .update_token_tensor(&mut decoder_state, &selected_tokens)
            .unwrap();

        match &decoder_state {
            CpuSeq2SeqState::U32(t) => {
                assert_eq!(t.shape(), &[4, 1]);
                assert_eq!(t[[0, 0]], 10);
                assert_eq!(t[[3, 0]], 40);
            }
            _ => panic!("Expected U32"),
        }

        // 3. Create and populate cache before reordering
        let mut cache = CpuBeamKVCache::new(num_layers, num_beams, 128, hidden_size);

        // Simulate one decode step - populate all layers then increment
        let k = Array3::zeros((num_beams, 1, hidden_size));
        let v = Array3::zeros((num_beams, 1, hidden_size));
        for layer in 0..num_layers {
            cache.update(layer, &k, &v).unwrap();
        }
        cache.increment_len(1); // <-- Critical!

        // 4. Now reorder cache based on beam scores
        let reorder_indices = vec![2, 2, 0, 1];
        backend.reorder_cache(&mut cache, &reorder_indices).unwrap();
    }

    #[test]
    fn test_reorder_cache_after_multiple_tokens() {
        let backend = CpuBackend;
        let num_layers = 2;
        let num_beams = 4;
        let hidden_size = 64;

        let mut cache = CpuBeamKVCache::new(num_layers, num_beams, 128, hidden_size);

        // Simulate 3 decode steps
        for _step in 0..3 {
            let k = Array3::from_shape_fn((num_beams, 1, hidden_size), |(b, _, h)| (b + h) as f32);
            let v = Array3::from_shape_fn((num_beams, 1, hidden_size), |(b, _, h)| (b + h) as f32);

            for layer in 0..num_layers {
                cache.update(layer, &k, &v).unwrap();
            }
            cache.increment_len(1); // Increment after each step
        }

        assert_eq!(cache.get_seq_length(), 3);

        // Reorder after accumulating multiple tokens
        let indices = vec![3, 2, 1, 0]; // Reverse order
        let result = backend.reorder_cache(&mut cache, &indices);
        assert!(result.is_ok());
    }
    // ========================================================================
    //  CpuBackend Debug
    // ========================================================================

    #[test]
    fn test_cpu_backend_debug() {
        let backend = CpuBackend;
        let debug_str = format!("{:?}", backend);
        assert_eq!(debug_str, "CpuBackend");
    }

    //

    #[test]
    fn test_typical_generation_flow_states() {
        let backend = CpuBackend;

        // 1. Create initial decoder tokens (decoder_start_token)
        let initial_tokens = vec![2u32]; // e.g., <s> token
        let decoder_state = backend.create_token_tensor(&initial_tokens, 1).unwrap();

        // Verify initial state
        match &decoder_state {
            CpuSeq2SeqState::U32(t) => {
                assert_eq!(t.shape(), &[1, 1]);
                assert_eq!(t[[0, 0]], 2);
            }
            _ => panic!("Expected U32"),
        }

        // 2. Simulate getting new tokens and updating
        let mut decoder_state = decoder_state;
        let new_tokens = vec![100u32]; // Generated token
        backend
            .update_token_tensor(&mut decoder_state, &new_tokens)
            .unwrap();

        match &decoder_state {
            CpuSeq2SeqState::U32(t) => {
                assert_eq!(t.shape(), &[1, 1]);
                assert_eq!(t[[0, 0]], 100);
            }
            _ => panic!("Expected U32"),
        }
    }


}
