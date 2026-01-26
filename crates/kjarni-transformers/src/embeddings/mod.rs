mod loader;

use std::ops::AddAssign;
use std::sync::Arc;

use crate::cpu::kernels::dequantize::dequantize_q8_0_block;
use crate::cpu::kernels::q_common::BlockQ8_0;
use crate::cpu::kernels::quantize::quantize_matrix_q8_0;
use crate::linear_layer::LinearLayer;
use crate::tensor::{CpuTensor, DType, QuantizedMatrix};
use crate::weights::ModelWeights;
use anyhow::Result;
use half::bf16;
use ndarray::{Array2, Array3, Axis, s};
use rayon::prelude::*;

pub use loader::{EmbeddingConfig, EmbeddingConfigBuilder, EmbeddingInput, LoadedEmbeddings};

#[cfg(test)]
mod tests;

/// An enum to hold the word embeddings table in its native, memory-efficient format.
#[derive(Clone)]
pub enum EmbeddingData {
    F32(Arc<Array2<f32>>),
    BF16(Arc<Array2<bf16>>),
    Q8_0(Arc<QuantizedMatrix<BlockQ8_0>>),
}

impl EmbeddingData {
    
    pub fn to_linear_layer(&self) -> LinearLayer {
        match self {
            EmbeddingData::F32(arc) => LinearLayer::from_arc_f32(arc.clone(), None),
            EmbeddingData::BF16(arc) => LinearLayer::from_arc_bf16(arc.clone(), None),
            EmbeddingData::Q8_0(arc) => LinearLayer::from_arc_q8_0(arc.clone(), None),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            EmbeddingData::F32(_) => DType::F32,
            EmbeddingData::BF16(_) => DType::BF16,
            EmbeddingData::Q8_0(_) => DType::Q8_0,
        }
    }
    
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

    pub fn with_shared_words(
        shared_words: EmbeddingData,
        weights: &ModelWeights,
        position_embedding_key: Option<&str>,
        token_type_embedding_key: Option<&str>,
    ) -> Result<Self> {
        let pos_emb = position_embedding_key
            .map(|k| weights.get_array2(k))
            .transpose()?;
        let type_emb = token_type_embedding_key
            .map(|k| weights.get_array2(k))
            .transpose()?;

        Ok(Self {
            word_embeddings: shared_words,
            position_embeddings: pos_emb,
            token_type_embeddings: type_emb,
        })
    }

    pub fn from_weights(
        weights: &ModelWeights,
        word_embedding_name: &str,
        pos_embedding_name: Option<&str>,
        type_embedding_name: Option<&str>,
        target_dtype: Option<DType>,
    ) -> Result<Self> {
        let word_tensor = weights.get_typed_tensor(word_embedding_name)?;
        let effective_dtype = target_dtype.unwrap_or(word_tensor.dtype());

        let word_embeddings = match (word_tensor, effective_dtype) {
            (CpuTensor::Q8_0(m), DType::Q8_0) => {
                log::info!("Loading embeddings as Native Q8_0 (Zero Copy)");
                EmbeddingData::Q8_0(Arc::new(m))
            }

            // === CASE 2: Quantize F32 -> Q8_0 (Best for Safetensors RAM Saving) ===
            (CpuTensor::F32(arr), DType::Q8_0) => {
                log::info!("Quantizing F32 embeddings to Q8_0");
                let w = arr.into_dimensionality::<ndarray::Ix2>()?;
                let blocks = quantize_matrix_q8_0(&w)?;
                let shape = [w.shape()[0], w.shape()[1]];
                EmbeddingData::Q8_0(Arc::new(QuantizedMatrix { blocks, shape }))
            }

            // === CASE 3: Quantize BF16 -> Q8_0 ===
            (CpuTensor::BF16(arr), DType::Q8_0) => {
                log::info!("Quantizing BF16 embeddings to Q8_0");
                let w = arr.into_dimensionality::<ndarray::Ix2>()?;
                let w_f32 = w.mapv(|v| v.to_f32()); // Need F32 for quantization kernel
                let blocks = quantize_matrix_q8_0(&w_f32)?;
                let shape = [w.shape()[0], w.shape()[1]];
                EmbeddingData::Q8_0(Arc::new(QuantizedMatrix { blocks, shape }))
            }

            // === CASE 4: Standard F32 ===
            (CpuTensor::F32(arr), DType::F32) => {
                EmbeddingData::F32(Arc::new(arr.into_dimensionality()?))
            }

            // === CASE 5: Standard BF16 ===
            (CpuTensor::BF16(arr), DType::BF16) => {
                EmbeddingData::BF16(Arc::new(arr.into_dimensionality()?))
            }

            // === CASE 6: Conversions (BF16 <-> F32) ===
            (CpuTensor::BF16(arr), DType::F32) => {
                let w = arr
                    .into_dimensionality::<ndarray::Ix2>()?
                    .mapv(|v| v.to_f32());
                EmbeddingData::F32(Arc::new(w))
            }
            (CpuTensor::F32(arr), DType::BF16) => {
                let w = arr
                    .into_dimensionality::<ndarray::Ix2>()?
                    .mapv(bf16::from_f32);
                EmbeddingData::BF16(Arc::new(w))
            }

            // === CASE 7: Fallback (Q4_K/Q6_K -> F32) ===
            // Until we implement specific gather kernels for Q4/Q6, we expand to F32.
            (t, _) => {
                log::warn!("Implicitly dequantizing {:?} embeddings to F32", t.dtype());
                EmbeddingData::F32(Arc::new(t.to_array2_f32()?))
            }
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
            EmbeddingData::Q8_0(w) => w.shape[0],
        }
    }

    fn hidden_size(&self) -> usize {
        match &self.word_embeddings {
            EmbeddingData::F32(w) => w.shape()[1],
            EmbeddingData::BF16(w) => w.shape()[1],
            EmbeddingData::Q8_0(w) => w.shape[1],
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

        if scale_embeddings {
            let scale_factor = (hidden_size as f32).sqrt();
            hidden.mapv_inplace(|x| x * scale_factor);
        }

        if let Some(ref pos_emb) = self.position_embeddings {
            let start_idx = position_offset;
            let end_idx = position_offset + seq_len;
            let max_position = pos_emb.shape()[0];

            // Safe slice logic
            let effective_end = end_idx.min(max_position);
            let len = effective_end.saturating_sub(start_idx);
            if len > 0 {
                let pos_slice = pos_emb.slice(s![start_idx..start_idx + len, ..]);
                // Shape [len, hidden] -> add batch axis to broadcast
                let pos_broadcast = pos_slice.insert_axis(Axis(0)); // shape [1, len, hidden_size]

                // Slice hidden and broadcast addition
                hidden
                    .slice_mut(s![.., 0..len, ..])
                    .add_assign(&pos_broadcast);
            }
            // if len > 0 {
            //     let pos_slice = pos_emb.slice(s![start_idx..start_idx + len, ..]);
            //     // Add to the corresponding part of hidden
            //     // Note: If batch > 1, this needs broadcasting or iteration
            //     // Assuming standard broadcasting works for shape [seq, hidden] vs [batch, seq, hidden]
            //     for mut batch_slice in hidden.axis_iter_mut(Axis(0)) {
            //         let mut seq_slice = batch_slice.slice_mut(s![0..len, ..]);
            //         seq_slice += &pos_slice;
            //     }
            // }
        }

        if let Some(ref token_type_emb) = self.token_type_embeddings {
            if let Some(type_ids) = token_type_ids {
                self.add_token_type_embeddings(&mut hidden, type_ids, token_type_emb);
            } else {
                let type_embeddings = token_type_emb.row(0);
                hidden += &type_embeddings;
            }
        }

        // if scale_embeddings {
        //     let scale_factor = (hidden_size as f32).sqrt();
        //     hidden *= scale_factor;
        // }

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
                            if token_id < vocab_size {
                                hidden_slice
                                    .slice_mut(s![j, ..])
                                    .assign(&word_embeddings.row(token_id));
                            }
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
                            if token_id < vocab_size {
                                let emb = word_embeddings.row(token_id).mapv(|v| v.to_f32());
                                hidden_slice.slice_mut(s![j, ..]).assign(&emb);
                            }
                        }
                    });
            }
            EmbeddingData::Q8_0(matrix) => {
                // Q8_0: 32 elements per block.
                let blocks_per_row = matrix.shape[1] / 32;

                hidden
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .zip(input_ids.axis_iter(Axis(0)))
                    .for_each(|(mut hidden_slice, ids)| {
                        for (j, &token_id) in ids.iter().enumerate() {
                            let token_id = token_id as usize;
                            if token_id >= vocab_size {
                                continue;
                            }

                            // Locate compressed data
                            let start = token_id * blocks_per_row;
                            let row_blocks = &matrix.blocks[start..start + blocks_per_row];

                            // Get output slice
                            let out_row = hidden_slice.slice_mut(s![j, ..]);
                            let out_slice =
                                out_row.into_slice().expect("Non-contiguous output slice");

                            // Dequantize block-by-block
                            for (b_idx, block) in row_blocks.iter().enumerate() {
                                let dest = &mut out_slice[b_idx * 32..(b_idx + 1) * 32];
                                dequantize_q8_0_block(block, dest);
                            }
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
                    let mut slice = hidden_slice.slice_mut(s![j, ..]);
                    slice += &type_emb;
                }
            });
    }
}
