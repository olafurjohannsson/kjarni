//! A GPU-accelerated Embedding block

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
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
        })
    }
    
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
    pub fn new(context: &Arc<WgpuContext>) -> Result<Self> {
        Ok(Self {
            lookup: GpuLookup2::new(context),
            add: GpuAdd::new(context),
            scale: GpuScale::new(context),
            context: context.clone(),
        })
    }

    /// Encodes the complete embedding generation pass into the command encoder
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

        let mut hidden_states = pool.get(vec![batch_size, seq_len, hidden_size]);
        self.lookup
            .encode(encoder, &weights.word_embeddings, input_ids, &hidden_states);

        if scale_embeddings {
            let scale_factor = (hidden_size as f32).sqrt();
            let scale_out = pool.get(hidden_states.shape().to_vec());

            self.scale
                .encode_out_of_place(encoder, &hidden_states, &scale_out, scale_factor);

            hidden_states = scale_out;
        }

        if let Some(pos_embeddings) = &weights.position_embeddings {
            let pos_add_out = pool.get(hidden_states.shape().to_vec());

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
        Ok(hidden_states)
    }
}

#[cfg(test)]
mod tests;
