use crate::tensor::TypedCpuTensor;
use crate::weights::ModelWeights;
use anyhow::Result;
use half::bf16;
use ndarray::{Array2, Array3, Axis, s};
use rayon::prelude::*;

/// An enum to hold the word embeddings table in its native, memory-efficient format.
pub enum EmbeddingData {
    F32(Array2<f32>),
    BF16(Array2<bf16>),
}

/// A CPU-based embedding layer that handles word, position, and token type embeddings.
/// This struct is now type-aware and avoids unnecessary upcasting of the word embedding table.
pub struct Embeddings {
    pub word_embeddings: EmbeddingData,
    pub position_embeddings: Option<Array2<f32>>,
    pub token_type_embeddings: Option<Array2<f32>>,
}

impl Embeddings {
    /// Creates a new `Embeddings` layer from pre-loaded components.
    pub fn new(
        word_embeddings: EmbeddingData,
        position_embeddings: Option<Array2<f32>>,
        token_type_embeddings: Option<Array2<f32>>,
    ) -> Self {
        Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
        }
    }

    /// Loads the embedding layer from model weights, preserving the native dtype
    /// of the word embedding table to save memory.
    pub fn from_weights(
        weights: &ModelWeights,
        word_embedding_name: &str,
        pos_embedding_name: Option<&str>,
        type_embedding_name: Option<&str>,
    ) -> Result<Self> {
        let word_tensor = weights.get_typed_tensor(word_embedding_name)?;
        let word_embeddings = match word_tensor {
            TypedCpuTensor::F32(arr) => EmbeddingData::F32(arr.into_dimensionality()?),
            TypedCpuTensor::BF16(arr) => EmbeddingData::BF16(arr.into_dimensionality()?),
            TypedCpuTensor::Q8_0(_) | TypedCpuTensor::Q4_K(_) | TypedCpuTensor::Q6_K(_) => {
                log::info!(
                    "Dequantizing GGUF embeddings for '{}' to F32",
                    word_embedding_name
                );
                EmbeddingData::F32(word_tensor.to_array2_f32()?)
            }

            _ => anyhow::bail!(
                "Unsupported dtype for word embeddings: {:?}",
                word_tensor.dtype()
            ),
        };

        let position_embeddings = if let Some(name) = pos_embedding_name {
            if weights.contains(name) {
                Some(weights.get_array2(name)?)
            } else {
                None
            }
        } else {
            None
        };

        let token_type_embeddings = if let Some(name) = type_embedding_name {
            if weights.contains(name) {
                Some(weights.get_array2(name)?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self::new(
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
        ))
    }

    fn vocab_size(&self) -> usize {
        match &self.word_embeddings {
            EmbeddingData::F32(w) => w.shape()[0],
            EmbeddingData::BF16(w) => w.shape()[0],
        }
    }

    fn hidden_size(&self) -> usize {
        match &self.word_embeddings {
            EmbeddingData::F32(w) => w.shape()[1],
            EmbeddingData::BF16(w) => w.shape()[1],
        }
    }

    /// Performs the complete embedding forward pass.
    pub fn forward(
        &self,
        input_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
        position_offset: usize,
        scale_embeddings: bool,
    ) -> Array3<f32> {
        let (batch_size, seq_len) = input_ids.dim();
        let hidden_size = self.hidden_size();

        let mut hidden = Array3::<f32>::zeros((batch_size, seq_len, hidden_size));
        self.perform_word_lookup(&mut hidden, input_ids);

        if let Some(ref pos_emb) = self.position_embeddings {
            let start_idx = position_offset;
            let end_idx = position_offset + seq_len;
            let max_position = pos_emb.shape()[0];
            if end_idx > max_position {
                panic!(
                    "Sequence length {} with offset {} exceeds max position embeddings {}.",
                    seq_len, position_offset, max_position
                );
            }
            let pos_embeddings_to_add = pos_emb.slice(s![start_idx..end_idx, ..]);
            hidden += &pos_embeddings_to_add;
        }

        if let Some(ref token_type_emb) = self.token_type_embeddings {
            if let Some(type_ids) = token_type_ids {
                self.add_token_type_embeddings(&mut hidden, type_ids, token_type_emb);
            } else {
                let type_embeddings = token_type_emb.row(0);
                hidden += &type_embeddings;
            }
        }

        if scale_embeddings {
            let scale_factor = (hidden_size as f32).sqrt();
            hidden *= scale_factor;
        }

        hidden
    }

    /// Internal helper to perform the parallelized word embedding lookup.
    /// This function is type-aware and handles the conversion from BF16 to F32.
    fn perform_word_lookup(&self, hidden: &mut Array3<f32>, input_ids: &Array2<u32>) {
        let vocab_size = self.vocab_size();

        match &self.word_embeddings {
            EmbeddingData::F32(word_embeddings) => {
                hidden
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .zip(input_ids.axis_iter(Axis(0)))
                    .for_each(|(mut hidden_slice, ids)| {
                        for (j, &token_id) in ids.iter().enumerate() {
                            let token_id = token_id as usize;
                            if token_id >= vocab_size {
                                panic!(
                                    "Token ID {} is out of vocabulary range [0, {})",
                                    token_id, vocab_size
                                );
                            }
                            let word_emb = word_embeddings.row(token_id);
                            hidden_slice.slice_mut(s![j, ..]).assign(&word_emb);
                        }
                    });
            }
            EmbeddingData::BF16(word_embeddings) => {
                hidden
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .zip(input_ids.axis_iter(Axis(0)))
                    .for_each(|(mut hidden_slice, ids)| {
                        for (j, &token_id) in ids.iter().enumerate() {
                            let token_id = token_id as usize;
                            if token_id >= vocab_size {
                                panic!(
                                    "Token ID {} is out of vocabulary range [0, {})",
                                    token_id, vocab_size
                                );
                            }
                            // The key optimization: convert to f32 *after* the lookup.
                            // This creates a temporary owned Array1<f32>.
                            let word_emb = word_embeddings.row(token_id).mapv(|v| v.to_f32());
                            hidden_slice.slice_mut(s![j, ..]).assign(&word_emb);
                        }
                    });
            }
        }
    }

    /// Internal helper to add token type embeddings.
    fn add_token_type_embeddings(
        &self,
        hidden: &mut Array3<f32>,
        type_ids: &Array2<u32>,
        token_type_emb: &Array2<f32>,
    ) {
        let type_vocab_size = token_type_emb.shape()[0];
        if type_vocab_size == 0 {
            return;
        }

        hidden
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(type_ids.axis_iter(Axis(0)))
            .for_each(|(mut hidden_slice, type_ids_row)| {
                for (j, &type_id) in type_ids_row.iter().enumerate() {
                    let type_id = type_id as usize;
                    if type_id >= type_vocab_size {
                        panic!(
                            "Token type ID {} is out of range [0, {})",
                            type_id, type_vocab_size
                        );
                    }
                    let type_emb = token_type_emb.row(type_id);
                    // CORRECTED: Bind to a mutable variable before using `+=`.
                    let mut slice = hidden_slice.slice_mut(s![j, ..]);
                    slice += &type_emb;
                }
            });
    }
}
