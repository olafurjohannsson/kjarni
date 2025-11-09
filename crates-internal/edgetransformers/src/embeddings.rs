//! Embedding layers for transformers

use ndarray::{Array2, Array3, Axis, s};

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
        position_embeddings: Array2<f32>,
        token_type_embeddings: Option<Array2<f32>>,
    ) -> Self {
        Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            token_type_embeddings,
        }
    }
    /// Create embeddings without position embeddings (for RoPE-based models)
    pub fn new_without_position(
        word_embeddings: Array2<f32>,
        token_type_embeddings: Option<Array2<f32>>,
    ) -> Self {
        Self {
            word_embeddings,
            position_embeddings: None,
            token_type_embeddings,
        }
    }
    /// Embed input tokens without adding positional or token_type embeddings.
    pub fn forward_word_only(&self, input_ids: &Array2<f32>) -> Array3<f32> {
        let (batch_size, seq_len) = input_ids.dim();
        let hidden_size = self.word_embeddings.shape()[1];
        let vocab_size = self.word_embeddings.shape()[0];

        let mut hidden = Array3::<f32>::zeros((batch_size, seq_len, hidden_size));

        // This can be parallelized in the future if needed
        for i in 0..batch_size {
            for j in 0..seq_len {
                let token_id = input_ids[[i, j]] as usize;
                if token_id >= vocab_size {
                    panic!(
                        "Token ID {} is out of vocabulary range [0, {})",
                        token_id, vocab_size
                    );
                }
                let word_emb = self.word_embeddings.row(token_id);
                hidden.slice_mut(s![i, j, ..]).assign(&word_emb);
            }
        }
        hidden
    }

    /// Embed input tokens
pub fn forward(
        &self,
        input_ids: &Array2<f32>,
        token_type_ids: Option<&Array2<f32>>,
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

        // Word embeddings (parallelized on non-WASM)
        #[cfg(not(target_arch = "wasm32"))]
        {
            use ndarray::parallel::prelude::*;
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
        }

        #[cfg(target_arch = "wasm32")]
        {
            for i in 0..batch_size {
                for j in 0..seq_len {
                    let token_id = input_ids[[i, j]] as usize;
                    // Bounds check
                    if token_id >= vocab_size {
                        panic!(
                            "Token ID {} is out of vocabulary range [0, {})",
                            token_id, vocab_size
                        );
                    }
                    let word_emb = self.word_embeddings.row(token_id);
                    hidden.slice_mut(s![i, j, ..]).assign(&word_emb);
                }
            }
        }

        // Position embeddings (only if present - not used by LLaMA/RoPE models)
        if let Some(ref pos_emb) = self.position_embeddings {
            let pos_embeddings = pos_emb.slice(s![0..seq_len, ..]);
            hidden += &pos_embeddings;
        }

        // Token type embeddings (only if present)
        if let Some(ref token_type_emb) = self.token_type_embeddings {
            let type_vocab_size = token_type_emb.shape()[0];

            if type_vocab_size == 0 {
                return hidden;
            }

            if let Some(type_ids) = token_type_ids {
                // Specific token type IDs provided
                #[cfg(not(target_arch = "wasm32"))]
                {
                    use ndarray::parallel::prelude::*;
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
                }

                #[cfg(target_arch = "wasm32")]
                {
                    for i in 0..batch_size {
                        for j in 0..seq_len {
                            let type_id = type_ids[[i, j]] as usize;
                            // Bounds check
                            if type_id >= type_vocab_size {
                                panic!(
                                    "Token type ID {} is out of range [0, {})",
                                    type_id, type_vocab_size
                                );
                            }
                            let type_emb = token_type_emb.row(type_id);
                            let mut slice = hidden.slice_mut(s![i, j, ..]);
                            slice += &type_emb;
                        }
                    }
                }
            } else {
                // Default to type 0 for all positions
                let type_embeddings = token_type_emb.row(0);
                hidden += &type_embeddings;
            }
        }

        hidden
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_embeddings_with_position() {
        // GPT-2 / BERT style
        let word_emb = Array2::ones((100, 64));
        let pos_emb = Array2::ones((512, 64));
        
        let embeddings = Embeddings::new(word_emb, pos_emb, None);
        
        let input_ids = Array2::zeros((2, 10));
        let output = embeddings.forward(&input_ids, None);
        
        assert_eq!(output.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_embeddings_without_position() {
        // LLaMA / RoPE style
        let word_emb = Array2::ones((100, 64));
        
        let embeddings = Embeddings::new_without_position(word_emb, None);
        
        let input_ids = Array2::zeros((2, 10));
        let output = embeddings.forward(&input_ids, None);
        
        assert_eq!(output.shape(), &[2, 10, 64]);
        // Should work without position embeddings
    }

    #[test]
    fn test_embeddings_with_token_types() {
        let word_emb = Array2::ones((100, 64));
        let pos_emb = Array2::ones((512, 64));
        let token_type_emb = Array2::ones((2, 64));
        
        let embeddings = Embeddings::new(word_emb, pos_emb, Some(token_type_emb));
        
        let input_ids = Array2::zeros((2, 10));
        let token_type_ids = Array2::zeros((2, 10));
        let output = embeddings.forward(&input_ids, Some(&token_type_ids));
        
        assert_eq!(output.shape(), &[2, 10, 64]);
    }

    #[test]
    #[should_panic(expected = "exceeds max position embeddings")]
    fn test_sequence_too_long() {
        let word_emb = Array2::ones((100, 64));
        let pos_emb = Array2::ones((10, 64));  // Only 10 positions
        
        let embeddings = Embeddings::new(word_emb, pos_emb, None);
        
        let input_ids = Array2::zeros((2, 20));  // 20 tokens - too long!
        let _ = embeddings.forward(&input_ids, None);
    }

    #[test]
    fn test_llama_long_sequence() {
        // LLaMA without position embeddings can handle any length
        let word_emb = Array2::ones((100, 64));
        
        let embeddings = Embeddings::new_without_position(word_emb, None);
        
        let input_ids = Array2::zeros((2, 1000));  // Very long sequence
        let output = embeddings.forward(&input_ids, None);
        
        assert_eq!(output.shape(), &[2, 1000, 64]);
        // Should work fine - RoPE handles position in attention layer
    }

    #[test]
    #[should_panic(expected = "out of vocabulary range")]
    fn test_invalid_token_id() {
        let word_emb = Array2::ones((100, 64));
        let pos_emb = Array2::ones((512, 64));
        
        let embeddings = Embeddings::new(word_emb, pos_emb, None);
        
        let mut input_ids = Array2::zeros((2, 10));
        input_ids[[0, 0]] = 150.0;  // Out of vocab range [0, 100)
        
        let _ = embeddings.forward(&input_ids, None);
    }
}