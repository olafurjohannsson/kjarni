use crate::models::bart::config::BartConfig;
use anyhow::{anyhow, Result};
use kjarni_transformers::activations::Activation;
use kjarni_transformers::embeddings::Embeddings;
use kjarni_transformers::encoder::config::EncoderLoadConfig;
use kjarni_transformers::encoder::prelude::*;
use kjarni_transformers::gpu_ops::blocks::attention::GpuAttentionWeights;
use kjarni_transformers::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use kjarni_transformers::gpu_ops::blocks::encoder::GpuEncoderLayer;
use kjarni_transformers::gpu_ops::blocks::{
    layer_norm::{GpuLayerNorm, GpuLayerNormWeights}, GpuFeedForwardWeightsStd, GpuNormalization,
    GpuNormalizationWeights,
};
use kjarni_transformers::gpu_ops::{GpuTensor, GpuTensorPool};
use kjarni_transformers::traits::EncoderDecoderArchitecture;
use kjarni_transformers::traits::{LanguageModelConfig, TransformerConfig};
use kjarni_transformers::weights::ModelWeights;
use kjarni_transformers::WgpuContext;
use std::sync::Arc;
use wgpu::CommandEncoder;

/// GPU-accelerated BART encoder.
///
/// Supports flexible CPU/GPU memory placement via `EncoderLoadConfig`:
/// - Full GPU execution (default)
/// - CPU embeddings with GPU layers (saves VRAM)
/// - Partial layer execution for hybrid workflows

pub struct BartGpuEncoder {
    context: Arc<WgpuContext>,
    config: Arc<BartConfig>,
    load_config: EncoderLoadConfig,

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
    ) -> Result<Self> {
        Self::with_config(context, weights, config, EncoderLoadConfig::default())
    }

    /// Create a new BART GPU encoder with custom load configuration.
    ///
    /// # Arguments
    /// * `context` - WGPU context
    /// * `weights` - Model weights
    /// * `config` - BART model configuration
    /// * `load_config` - Memory placement and dtype configuration
    pub fn with_config(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        config: Arc<BartConfig>,
        load_config: EncoderLoadConfig,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size();
        let intermediate_size = config.encoder_ffn_dim;

        // ====================================================================
        // 1. EMBEDDINGS - CPU or GPU based on config
        // ====================================================================
        let (cpu_embeddings, gpu_embedding_weights) = if load_config.cpu_embeddings {
            log::info!(
                "BART Encoder: Loading embeddings to CPU (VRAM saving mode). \
                Vocab: {}, Hidden: {}, Saving ~{:.1}MB",
                config.vocab_size,
                hidden_size,
                (config.vocab_size * hidden_size * 4) as f64 / 1_000_000.0
            );

            // let word_emb = weights.get_array2(config.get_shared_embedding_weight_name())?;
            let pos_emb = weights.get_array2("model.encoder.embed_positions.weight")?;

            let word_embeddings = weights.get_array2(config.get_shared_embedding_weight_name())?;

            log::debug!(
                "  Word embeddings: {:?}, Position embeddings: {:?}",
                word_embeddings.shape(),
                pos_emb.shape()
            );
            let embed = kjarni_transformers::embeddings::EmbeddingData::F32(word_embeddings);
            let cpu_emb = Embeddings::new(embed, Some(pos_emb), None);
            (Some(cpu_emb), None)
        } else {
            log::info!(
                "BART Encoder: Loading embeddings to GPU. Vocab: {}, Hidden: {}",
                config.vocab_size,
                hidden_size
            );

            let gpu_weights = GpuEmbeddingWeights::new(context, weights, config.as_ref())?;
            (None, Some(gpu_weights))
        };

        // GPU embedding kernel (needed for GPU path)
        let gpu_embeddings = GpuEmbeddings::new(context)?;

        // ====================================================================
        // 2. EMBEDDING LAYER NORM (always on GPU)
        // ====================================================================
        let embed_ln_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
            GpuTensor::from_raw(
                context,
                &weights.get_raw("model.encoder.layernorm_embedding.weight")?,
                "embed_ln_w",
            )?,
            GpuTensor::from_raw(
                context,
                &weights.get_raw("model.encoder.layernorm_embedding.bias")?,
                "embed_ln_b",
            )?,
        )?);

        let embed_layer_norm =
            GpuNormalization::LayerNorm(GpuLayerNorm::new(context, config.layer_norm_eps));

        // ====================================================================
        // 3. ENCODER LAYERS
        // ====================================================================
        let layers = Self::build_layers(context, weights, &config, &load_config)?;

        log::info!(
            "BART Encoder initialized: {} layers, hidden_size={}, intermediate_size={}, \
            embeddings_on_cpu={}",
            layers.len(),
            hidden_size,
            intermediate_size,
            load_config.cpu_embeddings
        );

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
        })
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
            EncoderLoadConfig::offload_embeddings(),
        )
    }

    // ========================================================================
    // PRIVATE HELPERS
    // ========================================================================

    /// Build all encoder layers.
    fn build_layers(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        config: &BartConfig,
        _load_config: &EncoderLoadConfig,
    ) -> Result<Vec<GpuEncoderLayer>> {
        let hidden_size = config.hidden_size();
        let intermediate_size = config.encoder_ffn_dim;
        let num_layers = config.encoder_layers;

        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let prefix = format!("model.encoder.layers.{}", i);

            // ================================================================
            // SELF-ATTENTION WEIGHTS
            // ================================================================
            let q_w = GpuTensor::from_raw(
                context,
                &weights.get_raw(&format!("{}.self_attn.q_proj.weight", prefix))?,
                &format!("layer{}_q_w", i),
            )?;
            let k_w = GpuTensor::from_raw(
                context,
                &weights.get_raw(&format!("{}.self_attn.k_proj.weight", prefix))?,
                &format!("layer{}_k_w", i),
            )?;
            let v_w = GpuTensor::from_raw(
                context,
                &weights.get_raw(&format!("{}.self_attn.v_proj.weight", prefix))?,
                &format!("layer{}_v_w", i),
            )?;
            let o_w = GpuTensor::from_raw(
                context,
                &weights.get_raw(&format!("{}.self_attn.out_proj.weight", prefix))?,
                &format!("layer{}_o_w", i),
            )?;

            // Validate [Out, In] layout for attention weights
            Self::validate_weight_shape(&q_w, &[hidden_size, hidden_size], i, "Q")?;
            Self::validate_weight_shape(&k_w, &[hidden_size, hidden_size], i, "K")?;
            Self::validate_weight_shape(&v_w, &[hidden_size, hidden_size], i, "V")?;
            Self::validate_weight_shape(&o_w, &[hidden_size, hidden_size], i, "O")?;

            let q_b = GpuTensor::from_raw(
                context,
                &weights.get_raw(&format!("{}.self_attn.q_proj.bias", prefix))?,
                &format!("layer{}_q_b", i),
            )?;
            let k_b = GpuTensor::from_raw(
                context,
                &weights.get_raw(&format!("{}.self_attn.k_proj.bias", prefix))?,
                &format!("layer{}_k_b", i),
            )?;
            let v_b = GpuTensor::from_raw(
                context,
                &weights.get_raw(&format!("{}.self_attn.v_proj.bias", prefix))?,
                &format!("layer{}_v_b", i),
            )?;
            let o_b = GpuTensor::from_raw(
                context,
                &weights.get_raw(&format!("{}.self_attn.out_proj.bias", prefix))?,
                &format!("layer{}_o_b", i),
            )?;

            let self_attn_weights =
                GpuAttentionWeights::new(q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b)?;

            // ================================================================
            // SELF-ATTENTION LAYER NORM
            // ================================================================
            let self_attn_ln_weights = GpuLayerNormWeights::new(
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw(&format!("{}.self_attn_layer_norm.weight", prefix))?,
                    &format!("layer{}_sa_ln_w", i),
                )?,
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw(&format!("{}.self_attn_layer_norm.bias", prefix))?,
                    &format!("layer{}_sa_ln_b", i),
                )?,
            )?;

            // ================================================================
            // FEED-FORWARD WEIGHTS
            // ================================================================
            let fc1_w = GpuTensor::from_raw(
                context,
                &weights.get_raw(&format!("{}.fc1.weight", prefix))?,
                &format!("layer{}_fc1_w", i),
            )?;
            let fc2_w = GpuTensor::from_raw(
                context,
                &weights.get_raw(&format!("{}.fc2.weight", prefix))?,
                &format!("layer{}_fc2_w", i),
            )?;

            // Validate [Out, In] layout for FFN weights
            Self::validate_weight_shape(&fc1_w, &[intermediate_size, hidden_size], i, "FC1")?;
            Self::validate_weight_shape(&fc2_w, &[hidden_size, intermediate_size], i, "FC2")?;

            let fc1_b = GpuTensor::from_raw(
                context,
                &weights.get_raw(&format!("{}.fc1.bias", prefix))?,
                &format!("layer{}_fc1_b", i),
            )?;
            let fc2_b = GpuTensor::from_raw(
                context,
                &weights.get_raw(&format!("{}.fc2.bias", prefix))?,
                &format!("layer{}_fc2_b", i),
            )?;

            let ff_weights = GpuFeedForwardWeightsStd::new(fc1_w, fc1_b, fc2_w, fc2_b)?;

            // ================================================================
            // FFN LAYER NORM
            // ================================================================
            let ffn_ln_weights = GpuLayerNormWeights::new(
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw(&format!("{}.final_layer_norm.weight", prefix))?,
                    &format!("layer{}_ffn_ln_w", i),
                )?,
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw(&format!("{}.final_layer_norm.bias", prefix))?,
                    &format!("layer{}_ffn_ln_b", i),
                )?,
            )?;

            // ================================================================
            // BUILD LAYER
            // ================================================================
            let layer = GpuEncoderLayer::new(
                context,
                self_attn_weights,
                self_attn_ln_weights,
                ff_weights,
                ffn_ln_weights,
                Activation::Gelu,
                config,
            )?;

            log::debug!(
                "  Layer {}: attn [{}, {}], ffn [{} → {} → {}]",
                i,
                hidden_size,
                hidden_size,
                hidden_size,
                intermediate_size,
                hidden_size
            );

            layers.push(layer);
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
    pub fn load_config(&self) -> &EncoderLoadConfig {
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
        cmd_encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        input: GpuEncoderInput,
        token_type_ids: Option<&GpuTensor>,
    ) -> Result<GpuTensor> {
        match input {
            GpuEncoderInput::TokensGpu(input_ids) => {
                // Full GPU path - requires GPU embedding weights
                let weights = self.gpu_embedding_weights.as_ref().ok_or_else(|| {
                    anyhow!(
                        "GPU embedding weights not loaded. \
                        Model was created with cpu_embeddings=true. \
                        Use GpuEncoderInput::TokensCpu instead."
                    )
                })?;
                unimplemented!()
                // self.gpu_embeddings.encode(
                //     cmd_encoder,
                //     weights,
                //     input_ids,
                //     token_type_ids,
                //     0, // Position offset handled by config.extra_pos_embeddings()
                //     self.config.as_ref(),
                //     pool,
                // )
            }

            GpuEncoderInput::TokensCpu(input_ids) => {
                // CPU embedding path - requires CPU embeddings
                let cpu_emb = self.cpu_embeddings.as_ref().ok_or_else(|| {
                    anyhow!(
                        "CPU embeddings not loaded. \
                        Model was created with cpu_embeddings=false. \
                        Use GpuEncoderInput::TokensGpu instead."
                    )
                })?;

                // BART uses position offset of 2 (from config.extra_pos_embeddings())
                let position_offset = self.config.extra_pos_embeddings();
                let scale = self.config.scale_embeddings();

                // Perform CPU embedding lookup
                let hidden = cpu_emb.forward(input_ids, None, position_offset, scale);

                // Upload to GPU
                GpuTensor::from_ndarray(&self.context, &hidden)
            }

            GpuEncoderInput::HiddenGpu(hidden) => {
                // Already on GPU - just clone the reference
                Ok(hidden.clone())
            }

            GpuEncoderInput::HiddenCpu(hidden) => {
                // Upload pre-computed hidden states to GPU
                GpuTensor::from_ndarray(&self.context, hidden)
            }
        }
    }

    fn embed_and_normalize(
        &self,
        cmd_encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        input: GpuEncoderInput,
        token_type_ids: Option<&GpuTensor>,
    ) -> Result<GpuTensor> {
        // First, get hidden states on GPU (handles all input variants)
        let hidden_states = self.embed(cmd_encoder, pool, input, token_type_ids)?;

        // Apply embedding layer normalization
        let ln_output = pool.get(hidden_states.shape().to_vec());
        self.embed_layer_norm.encode(
            cmd_encoder,
            &self.embed_ln_weights,
            &hidden_states,
            &ln_output,
        );

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
        self.config.hidden_size()
    }

    // forward() uses the default implementation from the trait:
    // fn forward(...) -> Result<GpuEncoderOutput> {
    //     let hidden = self.embed_and_normalize(...)?;
    //     let output = self.forward_layers(..., 0, self.num_layers())?;
    //     Ok(GpuEncoderOutput { last_hidden_state: output })
    // }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that encoder can be created with default config
    #[tokio::test]
    async fn test_encoder_creation_default() -> Result<()> {
        // This test requires actual model weights, so skip in CI
        // Just verify the code compiles
        Ok(())
    }

    /// Test that encoder can be created with CPU embeddings
    #[tokio::test]
    async fn test_encoder_creation_cpu_embeddings() -> Result<()> {
        // This test requires actual model weights, so skip in CI
        // Just verify the code compiles
        Ok(())
    }

    /// Test input validation
    #[test]
    fn test_model_load_config() {
        let config = EncoderLoadConfig::default();
        assert!(!config.cpu_embeddings);
        // assert!(!config.cpu_output_head);
        assert!(config.gpu_layer_range.is_none());
        assert!(config.dtype.is_none());

        let config = EncoderLoadConfig::offload_embeddings();
        assert!(config.cpu_embeddings);

        // let config = EncoderLoadConfig::max_offload();
        // assert!(config.cpu_embeddings);
        // assert!(config.cpu_output_head);

        let config = EncoderLoadConfig::partial_gpu(0, 6);
        assert_eq!(config.gpu_layer_range, Some((0, 6)));

        // Test builder pattern
        let config = EncoderLoadConfig::default()
            .with_cpu_embeddings(true)
            .with_gpu_layer_range(0, 8);
        assert!(config.cpu_embeddings);
        assert_eq!(config.gpu_layer_range, Some((0, 8)));
    }
}
