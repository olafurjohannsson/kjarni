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
use crate::gpu_ops::GpuTensor;
use crate::gpu_ops::blocks::attention::TempStorage;
use crate::gpu_ops::primitives::add::GpuAdd;
use crate::gpu_ops::primitives::lookup::GpuLookup;
use crate::gpu_ops::primitives::scale::GpuScale;
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
    lookup: GpuLookup,
    add: GpuAdd,
    scale: GpuScale,
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
        temp: &mut TempStorage,
    ) -> Result<GpuTensor> {
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];
        let hidden_size = config.hidden_size();

        // 1. Word Embedding Lookup
        let mut hidden_states = temp.get(vec![batch_size, seq_len, hidden_size]);
        self.lookup
            .encode(encoder, &weights.word_embeddings, input_ids, &hidden_states);

        // 2. Add Positional Embeddings (with offset)
        // if let Some(pos_embeddings) = &weights.position_embeddings {
        //     // 2a. Calculate the final offset.
        //     let final_offset = position_offset + config.extra_pos_embeddings();
        //
        //     // 2b. Create the positional IDs on the CPU.
        //     // This is efficient because seq_len is usually small (1 during generation).
        //     let pos_ids_cpu: Vec<u32> = (0..seq_len as u32)
        //         .map(|i| final_offset as u32 + i)
        //         .collect();
        //     let pos_ids_ndarray = ndarray::Array2::from_shape_vec((1, seq_len), pos_ids_cpu)?;
        //
        //     // 2c. Upload the positional IDs to the GPU.
        //     let pos_ids_gpu = GpuTensor::from_ndarray(&self.context, &pos_ids_ndarray)?;
        //
        //     // 2d. Perform a standard lookup to get the positional embedding vectors.
        //     let pos_embedding_vectors = temp.get(hidden_states.shape().to_vec());
        //     self.lookup.encode(
        //         encoder,
        //         pos_embeddings,
        //         &pos_ids_gpu,
        //         &pos_embedding_vectors,
        //     );
        //
        //     // 2e. Perform a standard element-wise add.
        //     let pos_add_out = temp.get(hidden_states.shape().to_vec());
        //     self.add.encode_elementwise(
        //         encoder,
        //         &hidden_states,
        //         &pos_embedding_vectors,
        //         &pos_add_out,
        //     );
        //     hidden_states = pos_add_out;
        // }
        if let Some(pos_embeddings) = &weights.position_embeddings {
            let pos_add_out = temp.get(hidden_states.shape().to_vec());
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
            let token_type_vectors = temp.get(hidden_states.shape().to_vec());

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
            let type_add_out = temp.get(hidden_states.shape().to_vec());
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
            let scale_out = temp.get(hidden_states.shape().to_vec());

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
