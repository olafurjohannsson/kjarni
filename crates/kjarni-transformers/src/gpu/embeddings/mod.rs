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

use crate::WgpuContext;
use crate::gpu_ops::primitives::add::GpuAdd;
use crate::gpu_ops::primitives::lookup2::GpuLookup2;
use crate::gpu_ops::primitives::scale::GpuScale;
use crate::gpu::{GpuTensor, GpuTensorPool};
use crate::tensor::DType;
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
        word_name: &str,
        pos_name: Option<&str>,
        type_name: Option<&str>,
        target_dtype: Option<DType>,
    ) -> Result<Self> {
        // Use from_model_weights - it handles quantized dequantization automatically
        let word_embeddings = GpuTensor::from_model_weights(
            context,
            weights,
            word_name,
            target_dtype,
            "word_embeddings",
        )?;

        let position_embeddings = if let Some(name) = pos_name {
            if weights.contains(name) {
                Some(GpuTensor::from_model_weights(
                    context,
                    weights,
                    name,
                    Some(DType::F32), // Position embeddings use F32
                    "pos_embeddings",
                )?)
            } else {
                None
            }
        } else {
            None
        };

        let token_type_embeddings = if let Some(name) = type_name {
            if weights.contains(name) {
                Some(GpuTensor::from_model_weights(
                    context,
                    weights,
                    name,
                    Some(DType::F32),
                    "type_embeddings",
                )?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
        })
    }
    /// Creates embedding weights using pre-loaded shared word embeddings.
    /// Only loads position and token type embeddings from weights.
    pub fn with_shared_words(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        shared_word_embeddings: GpuTensor,
        pos_name: Option<&str>,
        type_name: Option<&str>,
        target_dtype: Option<DType>,
    ) -> Result<Self> {
        let position_embeddings = if let Some(name) = pos_name {
            if weights.contains(name) {
                Some(GpuTensor::from_model_weights(
                    context,
                    weights,
                    name,
                    Some(DType::F32),
                    "pos_embeddings",
                )?)
            } else {
                None
            }
        } else {
            None
        };

        let token_type_embeddings = if let Some(name) = type_name {
            if weights.contains(name) {
                Some(GpuTensor::from_model_weights(
                    context,
                    weights,
                    name,
                    Some(DType::F32),
                    "type_embeddings",
                )?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            word_embeddings: shared_word_embeddings,
            position_embeddings,
            token_type_embeddings,
        })
    }
}

/// A GPU-accelerated Embeddings block.
pub struct GpuEmbeddings {
    pub lookup: GpuLookup2,
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
            lookup: GpuLookup2::new(context),
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
        input_ids: &GpuTensor,              // Shape: [batch, seq]
        token_type_ids: Option<&GpuTensor>, // Shape: [batch, seq]
        position_offset: usize,
        hidden_size: usize,
        extra_pos_embeddings: usize,
        scale_embeddings: bool,
        pool: &mut GpuTensorPool,
    ) -> Result<GpuTensor> {
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];

        // 1. Word Embedding Lookup
        let mut hidden_states = pool.get(vec![batch_size, seq_len, hidden_size]);
        self.lookup
            .encode(encoder, &weights.word_embeddings, input_ids, &hidden_states);

        // 4. Apply Scaling (BART/T5 style)
        if scale_embeddings {
            let scale_factor = (hidden_size as f32).sqrt();
            let scale_out = pool.get(hidden_states.shape().to_vec());

            self.scale
                .encode_out_of_place(encoder, &hidden_states, &scale_out, scale_factor);

            hidden_states = scale_out;
        }

        // 2. Add Positional Embeddings (with offset)
        if let Some(pos_embeddings) = &weights.position_embeddings {
            let pos_add_out = pool.get(hidden_states.shape().to_vec());

            // Use the passed extra_pos_embeddings (e.g., 2 for BART, 0 for others)
            let final_offset = position_offset + extra_pos_embeddings;

            self.add.encode_broadcast_offset(
                encoder,
                &hidden_states,
                pos_embeddings,
                final_offset,
                &pos_add_out,
            );
            hidden_states = pos_add_out;
        }

        // 3. Add Token Type Embeddings (BERT style)
        if let Some(token_type_embeddings) = &weights.token_type_embeddings {
            let token_type_vectors = pool.get(hidden_states.shape().to_vec());

            if let Some(type_ids) = token_type_ids {
                self.lookup.encode(
                    encoder,
                    token_type_embeddings,
                    type_ids,
                    &token_type_vectors,
                );
            } else {
                // Default to type 0 if no IDs provided
                let zeros_cpu = ndarray::Array2::<u32>::zeros((batch_size, seq_len));
                let zeros_gpu = GpuTensor::from_ndarray(&self.context, &zeros_cpu)?;
                self.lookup.encode(
                    encoder,
                    token_type_embeddings,
                    &zeros_gpu,
                    &token_type_vectors,
                );
            }

            let type_add_out = pool.get(hidden_states.shape().to_vec());
            self.add.encode_elementwise(
                encoder,
                &hidden_states,
                &token_type_vectors,
                &type_add_out,
            );
            hidden_states = type_add_out;
        }

        // // 4. Apply Scaling (BART/T5 style)
        // if scale_embeddings {
        //     let scale_factor = (hidden_size as f32).sqrt();
        //     let scale_out = pool.get(hidden_states.shape().to_vec());

        //     self.scale
        //         .encode_out_of_place(encoder, &hidden_states, &scale_out, scale_factor);

        //     hidden_states = scale_out;
        // }

        Ok(hidden_states)
    }
}

#[cfg(test)]
mod tests;
