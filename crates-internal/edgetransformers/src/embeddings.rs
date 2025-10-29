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
    pub position_embeddings: Array2<f32>,
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
            position_embeddings,
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
                    panic!("Token ID {} is out of vocabulary range [0, {})", token_id, vocab_size);
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
        let max_position = self.position_embeddings.shape()[0];
        let vocab_size = self.word_embeddings.shape()[0];

        if seq_len > max_position {
            panic!(
                "Sequence length {} exceeds max position embeddings {}. Please truncate your input.",
                seq_len, max_position
            );
        }

        let mut hidden = Array3::<f32>::zeros((batch_size, seq_len, hidden_size));

        // Word embeddings
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

        // Position embeddings
        let pos_embeddings = self.position_embeddings.slice(s![0..seq_len, ..]);
        hidden += &pos_embeddings;

        // Token type embeddings
        if let Some(ref token_type_emb) = self.token_type_embeddings {
            let type_vocab_size = token_type_emb.shape()[0];
            
            if type_vocab_size == 0 {
                return hidden;
            }

            if let Some(type_ids) = token_type_ids {
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
            } else {
                // Default to type 0
                let type_embeddings = token_type_emb.row(0);
                hidden += &type_embeddings;
            }
        }

        hidden
    }
}
