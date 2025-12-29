//! Unified embedding loading for all model architectures.
//!
//! This module provides `LoadedEmbeddings`, which handles the complexity of
//! loading embeddings to CPU or GPU based on user configuration.
//!
//! # Supported Architectures
//!
//! - **Decoder-only** (Llama, Mistral, Phi): Word embeddings only, RoPE for positions
//! - **Encoder-only** (BERT, RoBERTa): Word + position + type embeddings
//! - **Encoder-decoder** (BART, T5): Separate encoder/decoder embeddings
//!
//! # Example
//!
//! ```ignore
//! let config = EmbeddingConfig::builder("model.embed_tokens.weight", 2048).build();
//! let embs = LoadedEmbeddings::new(&ctx, &weights, config, load_config)?;
//! ```

use crate::{
    WgpuContext, embeddings::Embeddings, gpu_ops::{GpuTensor, GpuTensorPool, blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings}}, linear_layer::LinearLayer, models::base::ModelLoadConfig, tensor::DType, weights::ModelWeights
};
use anyhow::Result;
use ndarray::Array2;
use std::sync::Arc;

/// Configuration for embedding loading.
///
/// Describes what embeddings a model needs, independent of architecture type.
/// Use the builder pattern for clean construction.
///
/// # Example
///
/// ```ignore
/// // Minimal (Llama)
/// let config = EmbeddingConfig::builder("model.embed_tokens.weight", 2048).build();
///
/// // With position embeddings (GPT-2)
/// let config = EmbeddingConfig::builder("wte.weight", 768)
///     .position_embedding("wpe.weight")
///     .build();
///
/// // Full (BERT)
/// let config = EmbeddingConfig::builder("embeddings.word_embeddings.weight", 768)
///     .position_embedding("embeddings.position_embeddings.weight")
///     .type_embedding("embeddings.token_type_embeddings.weight")
///     .build();
///
/// // With scaling and offset (BART)
/// let config = EmbeddingConfig::builder("shared.weight", 1024)
///     .position_embedding("encoder.embed_positions.weight")
///     .position_offset(2)
///     .scale_embeddings(true)
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct EmbeddingConfig {
    /// Weight name for word/token embeddings. Always required.
    pub word_embedding: String,

    /// Weight name for position embeddings. None for RoPE models.
    pub position_embedding: Option<String>,

    /// Weight name for token type embeddings. None for most models.
    pub type_embedding: Option<String>,

    /// Hidden size of the model.
    pub hidden_size: usize,

    /// Position embedding offset (e.g., 2 for BART special tokens).
    pub position_offset: usize,

    /// Whether to scale embeddings by sqrt(hidden_size).
    pub scale_embeddings: bool,
}
impl EmbeddingConfig {
    /// Creates a simple config for RoPE models (word embeddings only).
    ///
    /// Use this for Llama, Mistral, Phi, Gemma, etc.
    pub fn new(word_embedding: impl Into<String>, hidden_size: usize) -> Self {
        Self {
            word_embedding: word_embedding.into(),
            hidden_size,
            ..Default::default()
        }
    }

    /// Creates a builder for more complex configurations.
    pub fn builder(word_embedding: impl Into<String>, hidden_size: usize) -> EmbeddingConfigBuilder {
        EmbeddingConfigBuilder {
            word_embedding: word_embedding.into(),
            hidden_size,
            position_embedding: None,
            type_embedding: None,
            position_offset: 0,
            scale_embeddings: false,
        }
    }
}

/// Builder for `EmbeddingConfig`.
pub struct EmbeddingConfigBuilder {
    word_embedding: String,
    hidden_size: usize,
    position_embedding: Option<String>,
    type_embedding: Option<String>,
    position_offset: usize,
    scale_embeddings: bool,
}

impl EmbeddingConfigBuilder {
    /// Sets the position embedding weight name.
    pub fn position_embedding(mut self, name: impl Into<String>) -> Self {
        self.position_embedding = Some(name.into());
        self
    }

    /// Sets the token type embedding weight name.
    pub fn type_embedding(mut self, name: impl Into<String>) -> Self {
        self.type_embedding = Some(name.into());
        self
    }

    /// Sets the position offset (e.g., 2 for BART).
    pub fn position_offset(mut self, offset: usize) -> Self {
        self.position_offset = offset;
        self
    }

    /// Enables embedding scaling by sqrt(hidden_size).
    pub fn scale_embeddings(mut self, scale: bool) -> Self {
        self.scale_embeddings = scale;
        self
    }

    /// Builds the `EmbeddingConfig`.
    pub fn build(self) -> EmbeddingConfig {
        EmbeddingConfig {
            word_embedding: self.word_embedding,
            position_embedding: self.position_embedding,
            type_embedding: self.type_embedding,
            hidden_size: self.hidden_size,
            position_offset: self.position_offset,
            scale_embeddings: self.scale_embeddings,
        }
    }
}

/// Loaded embeddings that can be on CPU, GPU, or hybrid.
///
/// Encapsulates device placement complexity. Callers just call `forward`.
pub struct LoadedEmbeddings {
    /// CPU embeddings (when offloaded).
    pub cpu: Option<Embeddings>,

    /// GPU embedding weights.
    pub gpu_weights: Option<GpuEmbeddingWeights>,

    /// GPU embedding kernel.
    pub gpu_layer: Option<GpuEmbeddings>,

    /// Configuration.
    pub config: EmbeddingConfig,

    /// GPU context for transfers.
    context: Option<Arc<WgpuContext>>,
}

impl LoadedEmbeddings {
    /// Creates a new `LoadedEmbeddings` from configuration.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The WGPU context.
    /// * `weights` - Model weights to load from.
    /// * `config` - Embedding configuration.
    /// * `load_config` - User's load preferences (offload, dtype).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = EmbeddingConfig::builder("model.embed_tokens.weight", 2048).build();
    /// let embs = LoadedEmbeddings::new(&ctx, &weights, config, load_config)?;
    /// ```
    pub fn new(
        ctx: Option<&Arc<WgpuContext>>,
        weights: &ModelWeights,
        config: EmbeddingConfig,
        load_cpu: bool,
        load_gpu: bool,
        target_dtype: Option<DType>,
    ) -> Result<Self> {
        // CPU embeddings
        let cpu = if load_cpu {
            log::info!("Loading embeddings to CPU RAM");
            Some(Embeddings::from_weights(
                weights,
                &config.word_embedding,
                config.position_embedding.as_deref(),
                config.type_embedding.as_deref(),
            )?)
        } else {
            None
        };

        // GPU embeddings
        let (gpu_weights, gpu_layer) = if load_gpu {
            let ctx = ctx.ok_or_else(|| anyhow::anyhow!("GPU embeddings require WgpuContext"))?;
            log::info!("Loading embeddings to GPU VRAM");
            
            let gpu_weights = GpuEmbeddingWeights::new(
                ctx,
                weights,
                &config.word_embedding,
                config.position_embedding.as_deref(),
                config.type_embedding.as_deref(),
                target_dtype,
            )?;
            let gpu_layer = GpuEmbeddings::new(ctx)?;
            (Some(gpu_weights), Some(gpu_layer))
        } else {
            (None, None)
        };

        // Need at least one
        if cpu.is_none() && gpu_weights.is_none() {
            return Err(anyhow::anyhow!("Must load embeddings to at least one device"));
        }

        Ok(Self {
            cpu,
            gpu_weights,
            gpu_layer,
            config,
            context: ctx.cloned(),
        })
    }

    pub fn word_embeddings_gpu(&self) -> Option<GpuTensor> {
        self.gpu_weights.as_ref().map(|w| w.word_embeddings.clone())
    }

    /// Returns the raw word embedding weights for CPU weight sharing.
    pub fn word_embeddings_cpu(&self) -> Option<LinearLayer> {
        // This assumes your Embeddings struct has a public/internal word_embeddings field
        self.cpu.as_ref().and_then(|e| match &e.word_embeddings {
            crate::embeddings::EmbeddingData::F32(w) => Some(LinearLayer::new_f32(w.clone(), None)),
            crate::embeddings::EmbeddingData::BF16(w) => Some(LinearLayer::new_bf16(w.clone(), None)),
        })
    }

    /// Returns true if embeddings are on CPU.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        self.cpu.is_some()
    }

    /// Returns true if embeddings are on GPU.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        self.gpu_weights.is_some()
    }

    /// Returns the embedding configuration.
    #[inline]
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }

    /// Returns the hidden size.
    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Performs embedding lookup from CPU token IDs.
    ///
    /// Handles CPU/GPU placement automatically.
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        token_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        if let Some(cpu_embs) = &self.cpu {
            // CPU path: compute on CPU, upload to GPU
            let embeddings = cpu_embs.forward(
                token_ids,
                token_type_ids,
                position_offset + self.config.position_offset,
                self.config.scale_embeddings,
            );
            GpuTensor::from_ndarray(&self.context.as_ref().unwrap(), &embeddings)
        } else {
            // GPU path
            let gpu_layer = self.gpu_layer.as_ref().unwrap();
            let gpu_weights = self.gpu_weights.as_ref().unwrap();

            let ids_tensor = GpuTensor::from_ndarray(&self.context.as_ref().unwrap(), token_ids)?;
            let type_tensor = token_type_ids
                .map(|t| GpuTensor::from_ndarray(&self.context.as_ref().unwrap(), t))
                .transpose()?;

            gpu_layer.encode(
                encoder,
                gpu_weights,
                &ids_tensor,
                type_tensor.as_ref(),
                position_offset,
                self.config.hidden_size,
                self.config.position_offset, // extra_pos_embeddings
                self.config.scale_embeddings,
                pool,
            )
        }
    }

    /// Performs embedding lookup from GPU token IDs.
    ///
    /// If embeddings are offloaded to CPU, downloads tokens first.
    pub async fn forward_gpu(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        token_ids: &GpuTensor,
        token_type_ids: Option<&GpuTensor>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        if let Some(cpu_embs) = &self.cpu {
            // Offloaded: download tokens, compute on CPU, upload result
            let token_data: Vec<u32> =
                bytemuck::cast_slice(&token_ids.read_raw_data().await?).to_vec();
            let shape = token_ids.shape();
            let token_array = Array2::from_shape_vec((shape[0], shape[1]), token_data)?;

            let type_array = if let Some(type_ids) = token_type_ids {
                let type_data: Vec<u32> =
                    bytemuck::cast_slice(&type_ids.read_raw_data().await?).to_vec();
                Some(Array2::from_shape_vec((shape[0], shape[1]), type_data)?)
            } else {
                None
            };

            let embeddings = cpu_embs.forward(
                &token_array,
                type_array.as_ref(),
                position_offset + self.config.position_offset,
                self.config.scale_embeddings,
            );
            GpuTensor::from_ndarray(&self.context.as_ref().unwrap(), &embeddings)
        } else {
            // GPU path
            let gpu_layer = self.gpu_layer.as_ref().unwrap();
            let gpu_weights = self.gpu_weights.as_ref().unwrap();

            gpu_layer.encode(
                encoder,
                gpu_weights,
                token_ids,
                token_type_ids,
                position_offset,
                self.config.hidden_size,
                self.config.position_offset,
                self.config.scale_embeddings,
                pool,
            )
        }
    }
}