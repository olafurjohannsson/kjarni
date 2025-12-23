use crate::{
    embeddings::Embeddings,
    gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings},
    models::base::ModelLoadConfig,
    traits::ModelLayout,
    weights::ModelWeights,
    WgpuContext,
};
use anyhow::{Context, Result};
use std::sync::Arc;

pub struct LoadedEmbeddings {
    pub cpu: Option<Embeddings>,
    pub gpu_weights: Option<GpuEmbeddingWeights>,
    pub gpu_layer: Option<GpuEmbeddings>,
}

impl LoadedEmbeddings {
    pub fn from_layout(
        ctx: &Arc<WgpuContext>,
        weights: &ModelWeights,
        layout: &ModelLayout,
        load_config: ModelLoadConfig,
    ) -> Result<Self> {
        let word_name = &layout.token_embedding;
        let encoder_layout = layout.encoder.as_ref().context("ModelLayout is missing the required 'encoder' layout")?;

        let pos_name = encoder_layout.position_embedding.as_deref();
        // Llama doesn't have type embeddings, but BERT does
        let type_name = None;

        if load_config.offload_embeddings {
            log::info!("Offloading embeddings to CPU RAM.");
            // Falls back to file DType (BF16/F32/Q6_K)
            let cpu_embs = Embeddings::from_weights(weights, word_name, pos_name, type_name)?;
            Ok(Self {
                cpu: Some(cpu_embs),
                gpu_weights: None,
                gpu_layer: None,
            })
        } else {
            log::info!("Loading embeddings to GPU VRAM.");
            let gpu_weights = GpuEmbeddingWeights::from_layout(
                ctx,
                weights,
                word_name,
                pos_name,
                type_name,
                load_config.target_dtype,
            )?;
            let gpu_layer = GpuEmbeddings::new(ctx)?;
            Ok(Self {
                cpu: None,
                gpu_weights: Some(gpu_weights),
                gpu_layer: Some(gpu_layer),
            })
        }
    }
}
