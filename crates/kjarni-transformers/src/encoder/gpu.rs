use anyhow::{Context, Result};
use std::sync::Arc;

use crate::encoder::traits::{GpuEncoder, GpuEncoderInput};
use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
use crate::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use crate::gpu_ops::blocks::encoder::GpuEncoderLayer;
use crate::gpu_ops::blocks::ffn::GpuFeedForwardWeights;
use crate::gpu_ops::blocks::layer_norm::{GpuLayerNorm, GpuLayerNormWeights};
use crate::gpu_ops::{GpuTensor, GpuTensorPool};
use crate::models::base::ModelLoadConfig;
use crate::traits::{Device, InferenceModel, ModelLayout, ModelMetadata};
use crate::weights::ModelWeights;
use crate::Embeddings;
use crate::WgpuContext;

pub struct GpuTransformerEncoder {
    embedding_weights: GpuEmbeddingWeights,
    embeddings: GpuEmbeddings,
    embedding_layer_norm: GpuLayerNorm,
    embedding_ln_weights: GpuLayerNormWeights,
    layers: Vec<GpuEncoderLayer>,
    context: Arc<WgpuContext>,
    cpu_embeddings: Embeddings,
    pub metadata: ModelMetadata,
}

impl GpuTransformerEncoder {
    pub fn new(
        weights: &ModelWeights,
        context: Arc<WgpuContext>,
        meta: ModelMetadata,
        layout: ModelLayout,
        load_cfg: ModelLoadConfig,
    ) -> Result<Self> {
        let target_dt = load_cfg.target_dtype;

        // 1. Load Embeddings for CPU (as a fallback/prefill tool)
        let cpu_embeddings = Embeddings::from_weights(
            weights,
            &layout.token_embedding,
            layout.position_embedding.as_deref(),
            layout.token_type_embedding.as_deref(),
        )?;

        // 2. Load Embeddings for GPU
        // Using the generic from_layout helper we discussed
        let embedding_weights = GpuEmbeddingWeights::from_layout(
            &context,
            weights,
            &layout.token_embedding,
            layout.position_embedding.as_deref(),
            layout.token_type_embedding.as_deref(),
            target_dt,
        )?;
        let embeddings = GpuEmbeddings::new(&context)?;

        // 3. Load Embedding LayerNorm (Gamma and Beta)
        let emb_norm_name = layout
            .embedding_norm
            .as_ref()
            .context("Encoder requires embedding_norm")?;
        let emb_norm_bias = layout
            .embedding_norm_bias
            .as_ref()
            .context("Encoder requires embedding_norm_bias")?;

        let embedding_ln_weights = GpuLayerNormWeights::new(
            GpuTensor::from_raw(
                &context,
                &weights.get_raw_resolved(emb_norm_name, target_dt)?,
                "emb_ln_gamma",
            )?,
            GpuTensor::from_raw(
                &context,
                &weights.get_raw_resolved(emb_norm_bias, target_dt)?,
                "emb_ln_beta",
            )?,
        )?;
        let embedding_layer_norm = GpuLayerNorm::new(&context, meta.norm_eps);

        // 4. Build Layers Loop
        let mut layers = Vec::with_capacity(meta.num_layers);
        for i in 0..meta.num_layers {
            let idx = i.to_string();
            let name = |t: &String| t.replace("{}", &idx);
            let opt_name = |t: &Option<String>| t.as_ref().map(|s| s.replace("{}", &idx));

            // --- Attention Weights ---
            let self_attn_weights = GpuAttentionWeights::new(
                GpuTensor::from_raw(
                    &context,
                    &weights.get_raw_resolved(&name(&layout.attn_q), target_dt)?,
                    "q",
                )?,
                GpuTensor::from_raw(
                    &context,
                    &weights
                        .get_raw_resolved(&opt_name(&layout.attn_q_bias).unwrap(), target_dt)?,
                    "q_b",
                )?,
                GpuTensor::from_raw(
                    &context,
                    &weights.get_raw_resolved(&name(&layout.attn_k), target_dt)?,
                    "k",
                )?,
                GpuTensor::from_raw(
                    &context,
                    &weights
                        .get_raw_resolved(&opt_name(&layout.attn_k_bias).unwrap(), target_dt)?,
                    "k_b",
                )?,
                GpuTensor::from_raw(
                    &context,
                    &weights.get_raw_resolved(&name(&layout.attn_v), target_dt)?,
                    "v",
                )?,
                GpuTensor::from_raw(
                    &context,
                    &weights
                        .get_raw_resolved(&opt_name(&layout.attn_v_bias).unwrap(), target_dt)?,
                    "v_b",
                )?,
                GpuTensor::from_raw(
                    &context,
                    &weights.get_raw_resolved(&name(&layout.attn_o), target_dt)?,
                    "o",
                )?,
                GpuTensor::from_raw(
                    &context,
                    &weights
                        .get_raw_resolved(&opt_name(&layout.attn_o_bias).unwrap(), target_dt)?,
                    "o_b",
                )?,
            )?;

            let self_attn_ln_weights = GpuLayerNormWeights::new(
                GpuTensor::from_raw(
                    &context,
                    &weights.get_raw_resolved(&name(&layout.attn_norm), target_dt)?,
                    "attn_ln_g",
                )?,
                GpuTensor::from_raw(
                    &context,
                    &weights
                        .get_raw_resolved(&opt_name(&layout.attn_norm_bias).unwrap(), target_dt)?,
                    "attn_ln_b",
                )?,
            )?;

            // --- FFN Weights (Legacy Path for Encoders) ---
            let up_w_raw = weights.get_array2(&name(&layout.ffn_up))?;
            let down_w_raw = weights.get_array2(&name(&layout.ffn_down))?;

            // Handle Transposition
            let up_w = if meta.transpose_ffn_weights {
                up_w_raw.t().as_standard_layout().to_owned()
            } else {
                up_w_raw
            };
            let down_w = if meta.transpose_ffn_weights {
                down_w_raw.t().as_standard_layout().to_owned()
            } else {
                down_w_raw
            };

            let ff_weights = GpuFeedForwardWeights::from_ndarrays(
                &context,
                &up_w,
                &weights.get_array1(&opt_name(&layout.ffn_up_bias).unwrap())?,
                &down_w,
                &weights.get_array1(&opt_name(&layout.ffn_down_bias).unwrap())?,
            )?;

            let ffn_ln_weights = GpuLayerNormWeights::new(
                GpuTensor::from_raw(
                    &context,
                    &weights.get_raw_resolved(&name(&layout.ffn_norm), target_dt)?,
                    "ffn_ln_g",
                )?,
                GpuTensor::from_raw(
                    &context,
                    &weights
                        .get_raw_resolved(&opt_name(&layout.ffn_norm_bias).unwrap(), target_dt)?,
                    "ffn_ln_b",
                )?,
            )?;

            // 5. Build the Encoder Layer
            layers.push(GpuEncoderLayer::new(
                &context,
                self_attn_weights,
                self_attn_ln_weights,
                ff_weights,
                ffn_ln_weights,
                meta.activation,
                &meta, // Pass metadata instead of config
            )?);
        }

        Ok(Self {
            embedding_weights,
            embeddings,
            embedding_layer_norm,
            embedding_ln_weights,
            layers,
            metadata: meta,
            context,
            cpu_embeddings,
        })
    }
}

impl InferenceModel for GpuTransformerEncoder {
    fn device(&self) -> Device {
        Device::Wgpu
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        Some(self.context.clone())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl GpuEncoder for GpuTransformerEncoder {
    fn embed(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: GpuEncoderInput,
        token_type_ids: Option<&GpuTensor>,
    ) -> Result<GpuTensor> {
        match input {
            GpuEncoderInput::TokensGpu(input_ids) => {
                // Standard pure-GPU path
                self.embeddings.encode(
                    cmd_encoder,
                    &self.embedding_weights,
                    input_ids,
                    token_type_ids,
                    0, // Encoders don't use a rolling position offset
                    self.metadata.hidden_size,
                    0,
                    self.metadata.scale_embeddings, // from ModelMetadata
                    pool,
                )
            }
            GpuEncoderInput::TokensCpu(input_ids_cpu) => {
                // Hybrid path: embeddings on CPU, layers on GPU.
                let hidden_cpu = self.cpu_embeddings.forward(
                    input_ids_cpu,
                    None, // Assuming CPU path doesn't get token_type_ids for now
                    self.metadata.extra_pos_embeddings,
                    self.metadata.scale_embeddings,
                );
                GpuTensor::from_ndarray(&self.context, &hidden_cpu)
            }
            GpuEncoderInput::HiddenGpu(hidden_states) => Ok(hidden_states.clone()),
            GpuEncoderInput::HiddenCpu(hidden_states_cpu) => {
                GpuTensor::from_ndarray(&self.context, hidden_states_cpu)
            }
        }
    }

    fn embed_and_normalize(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: GpuEncoderInput,
        token_type_ids: Option<&GpuTensor>,
    ) -> Result<GpuTensor> {
        // This logic is taken directly from your old `forward` method.
        let hidden_states = self.embed(cmd_encoder, pool, input, token_type_ids)?;

        // This logic correctly handles post-norm models like BERT/BART.
        // For a pre-norm model, this would just return `hidden_states`.
        if !self.metadata.is_prenorm {
            let ln_output = pool.get(hidden_states.shape().to_vec());
            self.embedding_layer_norm.encode(
                cmd_encoder,
                &self.embedding_ln_weights,
                &hidden_states,
                &ln_output,
            );
            Ok(ln_output)
        } else {
            Ok(hidden_states)
        }
    }

    fn forward_layers(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<GpuTensor> {
        // This logic is also taken directly from your old `forward` method.
        let mut current_states = hidden_states.clone();
        for layer in &self.layers[start_layer..end_layer] {
            current_states = layer.forward(
                cmd_encoder,
                &current_states,
                attention_mask,
                &self.metadata,
                pool,
            )?;
        }
        Ok(current_states)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_size(&self) -> usize {
        self.metadata.hidden_size
    }
}
