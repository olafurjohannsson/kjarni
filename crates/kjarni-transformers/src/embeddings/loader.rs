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
    WgpuContext,
    embeddings::{EmbeddingData, Embeddings},
    gpu_ops::{
        GpuTensor, GpuTensorPool,
        blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings},
    },
    linear_layer::LinearLayer,
    models::base::ModelInput,
    tensor::DType,
    weights::ModelWeights,
};
use anyhow::{Result, anyhow};
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

    pub fn with_position_embedding(mut self, position_embedding: Option<String>) -> Self {
        self.position_embedding = position_embedding;
        self
    }

    pub fn with_type_embedding(mut self, type_embedding: Option<String>) -> Self {
        self.type_embedding = type_embedding;
        self
    }

    /// Creates a builder for more complex configurations.
    pub fn builder(
        word_embedding: impl Into<String>,
        hidden_size: usize,
    ) -> EmbeddingConfigBuilder {
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

    pub fn with_token_type_embedding(mut self, name: Option<String>) -> Self {
        self.type_embedding = name;
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
pub enum EmbeddingInput<'a> {
    Cpu(&'a Array2<u32>),
    Gpu(&'a GpuTensor),
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
                target_dtype,
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
            return Err(anyhow::anyhow!(
                "Must load embeddings to at least one device"
            ));
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
        self.cpu.as_ref().and_then(|e| match &e.word_embeddings {
            EmbeddingData::F32(arc_w) => {
                // Use the new LinearLayer::from_arc_f32
                Some(LinearLayer::from_arc_f32(arc_w.clone(), None))
            },
            EmbeddingData::BF16(arc_w) => {
                // You need to add LinearLayer::from_arc_bf16 too!
                Some(LinearLayer::from_arc_bf16(arc_w.clone(), None))
            },
            EmbeddingData::Q8_0(arc_q) => {
                Some(LinearLayer::from_arc_q8_0(arc_q.clone(), None))
            }
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
    pub fn is_cpu_loaded(&self) -> bool {
        self.cpu.is_some()
    }
    pub fn is_gpu_loaded(&self) -> bool {
        self.gpu_weights.is_some()
    }


    /// CPU-Native embedding lookup.
    /// Returns Array3<f32> (Hidden States) for pure CPU execution.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - The token IDs [batch, seq_len]
    /// * `token_type_ids` - Optional token type IDs [batch, seq_len] (e.g., for BERT)
    /// * `position_offset` - The starting position for positional embeddings
    ///
    pub fn embed_cpu(
        &self, 
        token_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
        position_offset: usize
    ) -> Result<ndarray::Array3<f32>> {
        let cpu_layer = self.cpu.as_ref().ok_or_else(|| 
            anyhow::anyhow!("Cannot run embed_cpu: Embeddings are not loaded on CPU")
        )?;

        // Forward pass on CPU
        // This delegates directly to the underlying Embeddings logic
        Ok(cpu_layer.forward(
            token_ids,
            token_type_ids,
            self.config.position_offset,
            self.config.scale_embeddings,
        ))
    }

    /// The "Universal" embedding function.
    ///
    /// Automatically handles data movement and compute placement.
    /// - Returns `GpuTensor` (hidden states ready for the first layer).
    /// - Fully Synchronous (safe for CommandEncoder).
    pub fn embed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        token_type_ids: Option<ModelInput<'_>>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        let ctx = self
            .context
            .as_ref()
            .ok_or_else(|| anyhow!("No GPU context for embeddings"))?;

        // 1. Handle Pre-computed Hidden States (Pass-through)
        match input {
            ModelInput::HiddenGpu(t) => return Ok(t.clone()),
            ModelInput::HiddenCpu(view) => {
                // Sync Upload
                return GpuTensor::from_ndarray(ctx, &view.as_standard_layout().to_owned());
            }
            _ => {} // Continue to token processing
        }

        // 2. Handle Token Inputs
        match input {
            ModelInput::TokensCpu(ids) => {
                // === CPU INPUT PATH ===

                // OPTION A: Hybrid (Weights on CPU) -> Compute CPU, Upload Result
                if let Some(cpu_layer) = &self.cpu {
                    // 1. Resolve Aux Types to an OWNED variable (Option<Array2>)
                    // We need to own the data here so it lives long enough for the forward call.
                    let cpu_types_storage: Option<Array2<u32>> = match token_type_ids {
                        Some(ModelInput::TokensCpu(t)) => Some(t.to_owned()),
                        None => None,
                        _ => {
                            return Err(anyhow::anyhow!(
                                "Hybrid Error: Input is CPU but TokenTypes are GPU"
                            ));
                        }
                    };

                    // 2. Prepare Input IDs (Owned)
                    let ids_owned = ids.to_owned();

                    // 3. Compute
                    // .as_ref() converts Option<Array2> -> Option<&Array2>
                    let hidden = cpu_layer.forward(
                        &ids_owned,
                        cpu_types_storage.as_ref(),
                        position_offset + self.config.position_offset,
                        self.config.scale_embeddings,
                    );

                    // 4. Sync Upload
                    return GpuTensor::from_ndarray(ctx, &hidden);
                }

                // OPTION B: Upload (Weights on GPU) -> Upload Input, Compute GPU
                if let Some(gpu_layer) = &self.gpu_layer {
                    // Upload Tokens
                    let ids_gpu = GpuTensor::from_ndarray(ctx, &ids.to_owned())?;

                    // Upload Types
                    let types_gpu = match token_type_ids {
                        Some(ModelInput::TokensCpu(t)) => {
                            Some(GpuTensor::from_ndarray(ctx, &t.to_owned())?)
                        }
                        None => None,
                        _ => {
                            return Err(anyhow!(
                                "Upload Error: Input is CPU but TokenTypes are GPU"
                            ));
                        }
                    };

                    return gpu_layer.encode(
                        encoder,
                        self.gpu_weights.as_ref().unwrap(),
                        &ids_gpu,
                        types_gpu.as_ref(),
                        position_offset,
                        self.config.hidden_size,
                        self.config.position_offset,
                        self.config.scale_embeddings,
                        pool,
                    );
                }
            }

            ModelInput::TokensGpu(ids) => {
                // === GPU INPUT PATH ===
                if let Some(gpu_layer) = &self.gpu_layer {
                    // Resolve Aux (Must be GPU)
                    let types_gpu = match token_type_ids {
                        Some(ModelInput::TokensGpu(t)) => Some(t),
                        None => None,
                        _ => {
                            return Err(anyhow!(
                                "Pure GPU Error: Input is GPU but TokenTypes are CPU"
                            ));
                        }
                    };

                    return gpu_layer.encode(
                        encoder,
                        self.gpu_weights.as_ref().unwrap(),
                        ids,
                        types_gpu,
                        position_offset,
                        self.config.hidden_size,
                        self.config.position_offset,
                        self.config.scale_embeddings,
                        pool,
                    );
                } else {
                    return Err(anyhow!(
                        "Invalid Config: Tokens on GPU but Embeddings are CPU-only. Cannot compute synchronously."
                    ));
                }
            }
            _ => unreachable!(),
        }

        Err(anyhow!("No embeddings loaded!"))
    }
    /// The "One Stop Shop" for embedding.
    /// Handles CPU/GPU location mismatch automatically.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: EmbeddingInput,
        token_type_ids: Option<EmbeddingInput>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        let ctx = self
            .context
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No GPU context"))?;

        // CASE 1: CPU Input Tokens
        if let EmbeddingInput::Cpu(cpu_ids) = input {
            // A. Hybrid Path: Weights on CPU -> Compute CPU -> Upload Result
            if let Some(cpu_layer) = &self.cpu {
                // Resolve CPU token types if present
                let cpu_types = match token_type_ids {
                    Some(EmbeddingInput::Cpu(t)) => Some(t),
                    None => None,
                    _ => {
                        return Err(anyhow::anyhow!(
                            "Cannot mix CPU tokens with GPU token_types in Hybrid mode"
                        ));
                    }
                };

                let hidden = cpu_layer.forward(
                    cpu_ids,
                    cpu_types,
                    position_offset + self.config.position_offset,
                    self.config.scale_embeddings,
                );

                // Synchronous Upload of result
                return GpuTensor::from_ndarray(ctx, &hidden);
            }

            // B. Upload Path: Weights on GPU -> Upload Tokens -> Compute GPU
            // (We fall through to GPU logic below)
        }

        // Prepare GPU Input Tensors
        // If input was CPU, we upload it now. If GPU, we use it as is.
        let ids_gpu = match input {
            EmbeddingInput::Cpu(t) => GpuTensor::from_ndarray(ctx, t)?,
            EmbeddingInput::Gpu(t) => t.clone(), // Clone is cheap (Arc)
        };

        let types_gpu = match token_type_ids {
            Some(EmbeddingInput::Cpu(t)) => Some(GpuTensor::from_ndarray(ctx, t)?),
            Some(EmbeddingInput::Gpu(t)) => Some(t.clone()),
            None => None,
        };

        // CASE 2: GPU Computation
        if let Some(gpu_layer) = &self.gpu_layer {
            let gpu_weights = self.gpu_weights.as_ref().unwrap();

            return gpu_layer.encode(
                encoder,
                gpu_weights,
                &ids_gpu,
                types_gpu.as_ref(),
                position_offset,
                self.config.hidden_size,
                self.config.position_offset,
                self.config.scale_embeddings,
                pool,
            );
        }

        Err(anyhow::anyhow!(
            "Invalid state: Tokens on GPU but Embeddings only on CPU. Cannot compute synchronously."
        ))
    }
    pub fn encode_cpu(
        &self,
        context: &Arc<WgpuContext>,
        token_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        let cpu_embs = self
            .cpu
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("CPU embeddings not loaded"))?;

        let hidden = cpu_embs.forward(
            token_ids,
            token_type_ids,
            position_offset + self.config.position_offset,
            self.config.scale_embeddings,
        );

        // Synchronous Upload
        GpuTensor::from_ndarray(context, &hidden)
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
