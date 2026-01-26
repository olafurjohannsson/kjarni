use crate::models::bart::config::BartConfig;
use anyhow::{anyhow, Result};
use kjarni_transformers::cpu::encoder::prelude::*;
use kjarni_transformers::{EmbeddingConfig, Embeddings, LoadedEmbeddings};
use kjarni_transformers::gpu_ops::blocks::attention::GpuAttentionWeights;
use kjarni_transformers::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use kjarni_transformers::gpu_ops::blocks::encoder::GpuEncoderLayer;
use kjarni_transformers::gpu_ops::blocks::GpuFeedForwardWeights;
use kjarni_transformers::gpu_ops::blocks::{
    layer_norm::{GpuLayerNorm, GpuLayerNormWeights}, GpuFeedForwardWeightsStd, GpuNormalization,
    GpuNormalizationWeights,
};
use kjarni_transformers::gpu_ops::{GpuTensor, GpuTensorPool};
use kjarni_transformers::models::base::{ModelInput, ModelLoadConfig};
use kjarni_transformers::traits::{ModelConfig, ModelLayout, ModelMetadata};
use kjarni_transformers::weights::ModelWeights;
use kjarni_transformers::WgpuContext;
use std::sync::Arc;
use wgpu::CommandEncoder;

pub struct BartGpuEncoder {
    context: Arc<WgpuContext>,
    config: Arc<BartConfig>,
    load_config: ModelLoadConfig,

    // --- GPU Kernels ---
    gpu_embeddings: GpuEmbeddings,
    embed_layer_norm: GpuNormalization,

    // --- GPU Weights (None if using CPU embeddings) ---
    gpu_embedding_weights: Option<GpuEmbeddingWeights>,
    embed_ln_weights: GpuNormalizationWeights,

    // --- CPU Embeddings (None if using GPU embeddings) ---
    cpu_embeddings: Option<Embeddings>,

    // --- Encoder Layers ---
    layers: Vec<GpuEncoderLayer>,

    pub embeddings: LoadedEmbeddings,

    pub meta: ModelMetadata,
    pub layout: ModelLayout,
}

impl BartGpuEncoder {
    // ========================================================================
    // CONSTRUCTORS
    // ========================================================================

    /// Create a new BART GPU encoder with default configuration (full GPU).
    ///
    /// All model components are loaded to GPU for maximum performance.
    pub fn new(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        config: Arc<BartConfig>,
        load_config: ModelLoadConfig,
    ) -> Result<Self> {
        Self::with_config(context, weights, config, ModelLoadConfig::default())
    }
    /// Create encoder with CPU embeddings (convenience method).
    ///
    /// Equivalent to `with_config(..., ModelLoadConfig::offload_embeddings())`.
    pub fn with_cpu_embeddings(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        config: Arc<BartConfig>,
    ) -> Result<Self> {
        Self::with_config(
            context,
            weights,
            config,
            ModelLoadConfig::set_offload_embeddings(),
        )
    }

    pub fn with_config(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        config: Arc<BartConfig>, // Assuming BartConfig, adjust if it's a different config type
        load_config: ModelLoadConfig,
    ) -> Result<Self> {
        let meta = config.metadata();
        let layout = config.layout();
        let encoder_layout = layout
            .encoder
            .as_ref()
            .expect("Model layout must have an encoder section");
        let target_dt = load_config.target_dtype;
        // ====================================================================
        // 1. EMBEDDINGS - CPU or GPU based on config (Original Logic)
        // ====================================================================
        let (cpu_embeddings, gpu_embedding_weights) = if load_config.offload_embeddings {
            log::info!("Encoder: Loading embeddings to CPU (VRAM saving mode).");

            let word_embeddings = weights.get_array2(&layout.token_embedding)?;
            let pos_emb =
                weights.get_array2(encoder_layout.position_embedding.as_ref().unwrap())?;

            let embed = kjarni_transformers::EmbeddingData::F32(Arc::new(word_embeddings));
            let cpu_emb = Embeddings::new(embed, Some(pos_emb), None);
            (Some(cpu_emb), None)
        } else {
            log::info!("Encoder: Loading embeddings to GPU.");

            let gpu_weights = GpuEmbeddingWeights::new(
                context,
                weights,
                &layout.token_embedding,
                encoder_layout.position_embedding.as_deref(),
                encoder_layout.token_type_embedding.as_deref(),
                load_config.target_dtype,
            )?;
            (None, Some(gpu_weights))
        };

        let gpu_embeddings = GpuEmbeddings::new(context)?;

        let position_embedding = encoder_layout.position_embedding.as_ref().unwrap();

        let embedding_builder = EmbeddingConfig::builder(&layout.token_embedding, meta.hidden_size)
            .position_embedding(position_embedding) // BART Encoder specific
            .with_token_type_embedding(encoder_layout.token_type_embedding.clone())
            .position_offset(2)
            .scale_embeddings(meta.scale_embeddings);

        // if let(Some(type_embedding)) = &encoder_layout.token_type_embedding {
        //    embedding_builder.type_embedding(type_embedding);
        // };
        let embedding_config = embedding_builder.build();
        let embeddings = LoadedEmbeddings::new(
            Some(context),
            weights,
            embedding_config,
            load_config.offload_embeddings, // CPU offload pref
            true,                           // Always load GPU weights if not offloading
            target_dt,
        )?;

        // ====================================================================
        // 2. EMBEDDING LAYER NORM (Original Logic)
        // ====================================================================
        let embed_ln_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
            GpuTensor::from_model_weights(
                context,
                weights,
                encoder_layout.embedding_norm_weight.as_ref().unwrap(),
                load_config.target_dtype,
                "embed_ln_w",
            )?,
            GpuTensor::from_model_weights(
                context,
                weights,
                encoder_layout.embedding_norm_bias.as_ref().unwrap(),
                load_config.target_dtype,
                "embed_ln_b",
            )?,
        )?);

        let embed_layer_norm =
            GpuNormalization::LayerNorm(GpuLayerNorm::new(context, meta.norm_eps));

        // ====================================================================
        // 3. ENCODER LAYERS
        // ====================================================================
        let layers = Self::build_layers(context, weights, &meta, &layout, &load_config)?;

        Ok(Self {
            context: context.clone(),
            config,
            load_config,
            gpu_embeddings,
            embed_layer_norm,
            gpu_embedding_weights,
            embed_ln_weights,
            cpu_embeddings,
            layers,
            meta,
            layout,
            embeddings,
        })
    }

    // ========================================================================
    // PRIVATE HELPERS
    // ========================================================================

    fn build_layers(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        load_config: &ModelLoadConfig,
    ) -> Result<Vec<GpuEncoderLayer>> {
        let mut layers = Vec::with_capacity(meta.num_layers);
        let encoder_layout = layout
            .encoder
            .as_ref()
            .expect("Model layout must have an encoder section");
        let target_dt = load_config.target_dtype;

        for i in 0..meta.num_layers {
            // ================================================================
            // SELF-ATTENTION WEIGHTS (Using the reusable constructor)
            // ================================================================
            let self_attn_weights = GpuAttentionWeights::from_encoder_self_attn_layout(
                context, weights, layout, // Pass the whole layout
                i, target_dt,
            )?;

            // ================================================================
            // SELF-ATTENTION LAYER NORM (Original Logic, corrected names)
            // ================================================================

            let self_attn_ln_weights =
                GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_model_weights(
                        context,
                        weights,
                        &encoder_layout
                            .layer
                            .self_attn
                            .norm_weight
                            .replace("{}", &i.to_string()),
                        target_dt,
                        "sa_ln_w",
                    )?,
                    GpuTensor::from_model_weights(
                        context,
                        weights,
                        &encoder_layout
                            .layer
                            .self_attn
                            .norm_bias
                            .as_ref()
                            .unwrap()
                            .replace("{}", &i.to_string()),
                        target_dt,
                        "sa_ln_b",
                    )?,
                )?);

            // ================================================================
            // FEED-FORWARD WEIGHTS (Original Logic, corrected names)
            // ================================================================
            let raw_inter_w = weights.get_array2(
                &encoder_layout
                    .layer
                    .ffn
                    .up_weight
                    .replace("{}", &i.to_string()),
            )?;
            let raw_out_w = weights.get_array2(
                &encoder_layout
                    .layer
                    .ffn
                    .down_weight
                    .replace("{}", &i.to_string()),
            )?;

            let fc1_w = if meta.transpose_ffn_weights {
                raw_inter_w.t().as_standard_layout().to_owned()
            } else {
                raw_inter_w
            };
            let fc2_w = if meta.transpose_ffn_weights {
                raw_out_w.t().as_standard_layout().to_owned()
            } else {
                raw_out_w
            };

            let fc1_b = weights.get_array1(
                &encoder_layout
                    .layer
                    .ffn
                    .up_bias
                    .as_ref()
                    .unwrap()
                    .replace("{}", &i.to_string()),
            )?;
            let fc2_b = weights.get_array1(
                &encoder_layout
                    .layer
                    .ffn
                    .down_bias
                    .as_ref()
                    .unwrap()
                    .replace("{}", &i.to_string()),
            )?;

            let ff_weights = GpuFeedForwardWeights::Standard(GpuFeedForwardWeightsStd::new(
                GpuTensor::from_ndarray(context, &fc1_w)?,
                GpuTensor::from_ndarray(context, &fc1_b)?,
                GpuTensor::from_ndarray(context, &fc2_w)?,
                GpuTensor::from_ndarray(context, &fc2_b)?,
            )?);

            // ================================================================
            // FFN LAYER NORM (Original Logic, corrected names)
            // ================================================================

            let ffn_ln_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                GpuTensor::from_model_weights(
                    context,
                    weights,
                    &encoder_layout
                        .layer
                        .ffn
                        .norm_weight
                        .replace("{}", &i.to_string()),
                    target_dt,
                    "ffn_ln_w",
                )?,
                GpuTensor::from_model_weights(
                    context,
                    weights,
                    &encoder_layout
                        .layer
                        .ffn
                        .norm_bias
                        .as_ref()
                        .unwrap()
                        .replace("{}", &i.to_string()),
                    target_dt,
                    "ffn_ln_b",
                )?,
            )?);

            // ================================================================
            // BUILD LAYER (Original Logic)
            // ================================================================
            layers.push(GpuEncoderLayer::new(
                context,
                self_attn_weights,
                self_attn_ln_weights,
                ff_weights,
                ffn_ln_weights,
                meta.activation,
                meta,
            )?);
        }

        Ok(layers)
    }

    /// Validate weight tensor shape.
    fn validate_weight_shape(
        tensor: &GpuTensor,
        expected: &[usize],
        layer_idx: usize,
        name: &str,
    ) -> Result<()> {
        if tensor.shape() != expected {
            return Err(anyhow!(
                "Layer {} {} weight shape mismatch: expected {:?}, got {:?}",
                layer_idx,
                name,
                expected,
                tensor.shape()
            ));
        }
        Ok(())
    }

    // ========================================================================
    // PUBLIC ACCESSORS
    // ========================================================================

    /// Get the model configuration.
    pub fn config(&self) -> &BartConfig {
        &self.config
    }

    /// Get the load configuration.
    pub fn load_config(&self) -> &ModelLoadConfig {
        &self.load_config
    }

    /// Check if embeddings are on CPU.
    pub fn has_cpu_embeddings(&self) -> bool {
        self.cpu_embeddings.is_some()
    }

    /// Get the GPU context.
    pub fn context(&self) -> &Arc<WgpuContext> {
        &self.context
    }
}

// ============================================================================
// GPU ENCODER TRAIT IMPLEMENTATION
// ============================================================================

impl GpuEncoder for BartGpuEncoder {
    fn embed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        token_type_ids: Option<ModelInput<'_>>,
    ) -> Result<GpuTensor> {
        self.embeddings.embed(
            encoder,
            pool,
            input,
            token_type_ids,
            0, // position_offset (always 0 for Encoders)
        )
    }

    fn embed_and_normalize(
        &self,
        cmd_encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        token_type_ids: Option<ModelInput<'_>>,
    ) -> Result<GpuTensor> {
        // First, get hidden states on GPU (handles all input variants)
        let hidden_states = self.embed(cmd_encoder, pool, input, token_type_ids)?;

        // Apply embedding layer normalization
        let ln_output = pool.get(hidden_states.shape().to_vec());
        if self.config.normalize_embedding {
            self.embed_layer_norm.encode(
                cmd_encoder,
                &self.embed_ln_weights,
                &hidden_states,
                &ln_output,
            );
        }

        Ok(ln_output)
    }

    fn forward_layers(
        &self,
        cmd_encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<GpuTensor> {
        // Validate layer range
        let num_layers = self.layers.len();
        if start_layer > end_layer {
            return Err(anyhow!(
                "Invalid layer range: start ({}) > end ({})",
                start_layer,
                end_layer
            ));
        }
        if end_layer > num_layers {
            return Err(anyhow!(
                "Layer range end ({}) exceeds number of layers ({})",
                end_layer,
                num_layers
            ));
        }

        // Handle empty range
        if start_layer == end_layer {
            return Ok(hidden_states.clone());
        }

        // Process layers
        let mut hidden = hidden_states.clone();
        for (i, layer) in self.layers[start_layer..end_layer].iter().enumerate() {
            let layer_idx = start_layer + i;
            log::trace!("Processing encoder layer {}", layer_idx);

            // BART uses post-norm architecture
            hidden = layer.forward_postnorm(cmd_encoder, &hidden, attention_mask, pool)?;
        }

        Ok(hidden)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_size(&self) -> usize {
        self.meta.hidden_size
    }

    // forward() uses the default implementation from the trait:
    // fn forward(...) -> Result<GpuEncoderOutput> {
    //     let hidden = self.embed_and_normalize(...)?;
    //     let output = self.forward_layers(..., 0, self.num_layers())?;
    //     Ok(GpuEncoderOutput { last_hidden_state: output })
    // }
}
