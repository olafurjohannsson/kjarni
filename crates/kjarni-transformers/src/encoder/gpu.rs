use anyhow::{Context, Result};
use async_trait::async_trait;
use ndarray::s;
use std::sync::Arc;

use crate::Embeddings;
use crate::WgpuContext;
use crate::embeddings::EmbeddingConfig;
use crate::embeddings::LoadedEmbeddings;
use crate::encoder::traits::{GpuEncoder};
use crate::gpu_ops::blocks::GpuFeedForward;
use crate::gpu_ops::blocks::GpuNormalization;
use crate::gpu_ops::blocks::GpuNormalizationWeights;
use crate::gpu_ops::blocks::GpuRMSNorm;
use crate::gpu_ops::blocks::GpuRMSNormWeights;
use crate::gpu_ops::blocks::GpuSwiGLUFFNWeights;
use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
use crate::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use crate::gpu_ops::blocks::encoder::GpuEncoderLayer;
use crate::gpu_ops::blocks::ffn::GpuFeedForwardWeights;
use crate::gpu_ops::blocks::layer_norm::{GpuLayerNorm, GpuLayerNormWeights};
use crate::gpu_ops::{GpuTensor, GpuTensorPool};
use crate::models::base::ModelInput;
use crate::models::base::ModelLoadConfig;
use crate::traits::NormalizationStrategy;
use crate::traits::{Device, InferenceModel, ModelLayout, ModelMetadata};
use crate::weights::ModelWeights;

pub struct GpuTransformerEncoder {
    embeddings: LoadedEmbeddings,
    embedding_layer_norm: GpuNormalization,
    embedding_ln_weights: GpuNormalizationWeights,
    layers: Vec<GpuEncoderLayer>,
    context: Arc<WgpuContext>,
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

        // 1. Layout
        let encoder_layout = layout
            .encoder
            .as_ref()
            .context("ModelLayout is missing the required 'encoder' layout")?;

        // 2. Embeddings (Unified)
        let embeddings = LoadedEmbeddings::new(
            Some(&context),
            weights,
            EmbeddingConfig::new(&layout.token_embedding, meta.hidden_size)
                .with_position_embedding(encoder_layout.position_embedding.clone())
                .with_type_embedding(encoder_layout.token_type_embedding.clone()),
            load_cfg.offload_embeddings,
            true, // Load to GPU
            target_dt,
        )?;

        // 3. Embedding Normalization (LayerNorm or RMSNorm)
        // Nomic/BERT use LayerNorm. Future models might use RMSNorm.
        let emb_norm_w_name = encoder_layout
            .embedding_norm_weight
            .as_ref()
            .context("missing emb_norm")?;
        let emb_norm_b_name = encoder_layout.embedding_norm_bias.as_ref(); // Optional for RMSNorm

        let (embedding_layer_norm, embedding_ln_weights) = match meta.normalization_strategy {
            NormalizationStrategy::LayerNorm => {
                let ln = GpuLayerNorm::new(&context, meta.norm_eps);
                let w = GpuLayerNormWeights::new(
                    GpuTensor::from_model_weights(
                        &context,
                        weights,
                        emb_norm_w_name,
                        target_dt,
                        "emb_ln_g",
                    )?,
                    GpuTensor::from_model_weights(
                        &context,
                        weights,
                        emb_norm_b_name.unwrap(),
                        target_dt,
                        "emb_ln_b",
                    )?,
                )?;
                (
                    GpuNormalization::LayerNorm(ln),
                    GpuNormalizationWeights::LayerNorm(w),
                )
            }
            NormalizationStrategy::RMSNorm => {
                let rms = GpuRMSNorm::new(&context, meta.norm_eps);
                let w = GpuRMSNormWeights::new(GpuTensor::from_model_weights(
                    &context,
                    weights,
                    emb_norm_w_name,
                    target_dt,
                    "emb_rms_g",
                )?)?;
                (
                    GpuNormalization::RMSNorm(rms),
                    GpuNormalizationWeights::RMSNorm(w),
                )
            }
        };

        // 4. Build Layers
        let mut layers = Vec::with_capacity(meta.num_layers);
        for i in 0..meta.num_layers {
            let idx = i.to_string();
            let name = |t: &String| t.replace("{}", &idx);
            let resolve_bias = |opt: &Option<String>| opt.as_ref().map(|s| name(s));

            // --- ATTENTION LOADING (Handles Nomic Fused QKV) ---
            let q_name = name(&encoder_layout.layer.self_attn.q_weight);
            let k_name = name(&encoder_layout.layer.self_attn.k_weight);
            let v_name = name(&encoder_layout.layer.self_attn.v_weight);

            let (q_t, k_t, v_t, q_b, k_b, v_b) = if q_name == k_name && k_name == v_name {
                // Fused Path
                let fused_w = weights.get_array2(&q_name)?;
                let hidden = meta.hidden_size;
                let q_w = fused_w.slice(s![0..hidden, ..]).to_owned();
                let k_w = fused_w.slice(s![hidden..2 * hidden, ..]).to_owned();
                let v_w = fused_w.slice(s![2 * hidden..3 * hidden, ..]).to_owned();

                let (qb, kb, vb) =
                    if let Some(b_name) = resolve_bias(&encoder_layout.layer.self_attn.q_bias) {
                        let fused_b = weights.get_array1(&b_name)?;
                        (
                            Some(fused_b.slice(s![0..hidden]).to_owned()),
                            Some(fused_b.slice(s![hidden..2 * hidden]).to_owned()),
                            Some(fused_b.slice(s![2 * hidden..3 * hidden]).to_owned()),
                        )
                    } else {
                        (None, None, None)
                    };

                (
                    GpuTensor::from_ndarray(&context, &q_w)?,
                    GpuTensor::from_ndarray(&context, &k_w)?,
                    GpuTensor::from_ndarray(&context, &v_w)?,
                    qb.map(|b| GpuTensor::from_ndarray(&context, &b))
                        .transpose()?,
                    kb.map(|b| GpuTensor::from_ndarray(&context, &b))
                        .transpose()?,
                    vb.map(|b| GpuTensor::from_ndarray(&context, &b))
                        .transpose()?,
                )
            } else {
                // Standard Path
                let load =
                    |n: &str| GpuTensor::from_model_weights(&context, weights, n, target_dt, n);
                let load_bias = |n: Option<String>| {
                    n.map(|s| GpuTensor::from_model_weights(&context, weights, &s, target_dt, &s))
                        .transpose()
                };
                (
                    load(&q_name)?,
                    load(&k_name)?,
                    load(&v_name)?,
                    load_bias(resolve_bias(&encoder_layout.layer.self_attn.q_bias))?,
                    load_bias(resolve_bias(&encoder_layout.layer.self_attn.k_bias))?,
                    load_bias(resolve_bias(&encoder_layout.layer.self_attn.v_bias))?,
                )
            };

            let o_t = GpuTensor::from_model_weights(
                &context,
                weights,
                &name(&encoder_layout.layer.self_attn.o_weight),
                target_dt,
                "o",
            )?;

            let o_b = resolve_bias(&encoder_layout.layer.self_attn.o_bias)
                .map(|s| GpuTensor::from_model_weights(&context, weights, &s, target_dt, "o_b"))
                .transpose()?;

            let self_attn_weights =
                GpuAttentionWeights::new(q_t, q_b, k_t, k_b, v_t, v_b, o_t, o_b)?;

            // --- FFN LOADING (Standard vs SwiGLU) ---
            let up_name = name(&encoder_layout.layer.ffn.up_weight);
            let down_name = name(&encoder_layout.layer.ffn.down_weight);
            let gate_name = encoder_layout
                .layer
                .ffn
                .gate_weight
                .as_ref()
                .map(|s| name(s));

            // Helper for transposition
            let load_transposed = |n: &str| -> Result<ndarray::Array2<f32>> {
                let arr = weights.get_array2(n)?;
                if meta.transpose_ffn_weights {
                    Ok(arr.t().as_standard_layout().to_owned())
                } else {
                    Ok(arr)
                }
            };

            let is_fused_swiglu = gate_name.as_ref() == Some(&up_name);

            // Construct the specific weights struct first
            let ff_weights_enum = if is_fused_swiglu {
                // Fused SwiGLU (Nomic GGUF/SafeTensors fused)
                let fused_w = load_transposed(&up_name)?;
                let half_dim = fused_w.shape()[0] / 2;
                let gate_w = fused_w.slice(s![0..half_dim, ..]).to_owned();
                let up_w = fused_w.slice(s![half_dim.., ..]).to_owned();
                let down_w = load_transposed(&down_name)?;

                // Biases (Nomic has none, but handled if present)
                let (gate_b, up_b) =
                    if let Some(b_name) = resolve_bias(&encoder_layout.layer.ffn.up_bias) {
                        let fused_b = weights.get_array1(&b_name)?;
                        (
                            Some(fused_b.slice(s![0..half_dim]).to_owned()),
                            Some(fused_b.slice(s![half_dim..]).to_owned()),
                        )
                    } else {
                        (None, None)
                    };

                let down_b = resolve_bias(&encoder_layout.layer.ffn.down_bias)
                    .map(|s| weights.get_array1(&s))
                    .transpose()?;

                let swiglu_weights = GpuSwiGLUFFNWeights::new(
                    GpuTensor::from_ndarray(&context, &gate_w)?,
                    // gate_b.map(|b| GpuTensor::from_ndarray(&context, &b)).transpose()?,
                    GpuTensor::from_ndarray(&context, &up_w)?,
                    // up_b.map(|b| GpuTensor::from_ndarray(&context, &b)).transpose()?,
                    GpuTensor::from_ndarray(&context, &down_w)?,
                    // down_b.map(|b| GpuTensor::from_ndarray(&context, &b)).transpose()?,
                )?;
                crate::gpu_ops::blocks::GpuFeedForwardWeights::SwiGLU(swiglu_weights)
            } else if let Some(g_name) = gate_name {
                // Separate SwiGLU (Llama style)
                let up_w = load_transposed(&up_name)?;
                let gate_w = load_transposed(&g_name)?;
                let down_w = load_transposed(&down_name)?;

                // Assuming Nomic/Llama naming conventions
                let up_b = resolve_bias(&encoder_layout.layer.ffn.up_bias)
                    .map(|s| weights.get_array1(&s))
                    .transpose()?;
                let down_b = resolve_bias(&encoder_layout.layer.ffn.down_bias)
                    .map(|s| weights.get_array1(&s))
                    .transpose()?;
                // let gate_b = None; // Usually implicit or shared?

                let swiglu_weights = crate::gpu_ops::blocks::ffn_swiglu::GpuSwiGLUFFNWeights::new(
                    GpuTensor::from_ndarray(&context, &gate_w)?,
                    // gate_b.map(|b| GpuTensor::from_ndarray(&context, &b)).transpose()?,
                    GpuTensor::from_ndarray(&context, &up_w)?,
                    // up_b.map(|b| GpuTensor::from_ndarray(&context, &b)).transpose()?,
                    GpuTensor::from_ndarray(&context, &down_w)?,
                    // down_b.map(|b| GpuTensor::from_ndarray(&context, &b)).transpose()?,
                )?;
                crate::gpu_ops::blocks::GpuFeedForwardWeights::SwiGLU(swiglu_weights)
            } else {
                // Standard FeedForward (MiniLM/BERT)
                let up_w = load_transposed(&up_name)?;
                let down_w = load_transposed(&down_name)?;
                let up_b = resolve_bias(&encoder_layout.layer.ffn.up_bias)
                    .map(|s| weights.get_array1(&s))
                    .transpose()?;
                let down_b = resolve_bias(&encoder_layout.layer.ffn.down_bias)
                    .map(|s| weights.get_array1(&s))
                    .transpose()?;

                let std_weights = crate::gpu_ops::blocks::GpuFeedForwardWeightsStd::new(
                    GpuTensor::from_ndarray(&context, &up_w)?,
                    GpuTensor::from_ndarray(&context, &up_b.expect("Standard FFN requires bias"))?,
                    GpuTensor::from_ndarray(&context, &down_w)?,
                    GpuTensor::from_ndarray(
                        &context,
                        &down_b.expect("Standard FFN requires bias"),
                    )?,
                )?;

                crate::gpu_ops::blocks::GpuFeedForwardWeights::Standard(std_weights)
            };

            // --- LAYER NORMS (Enum Based) ---
            let build_norm = |w: String, b: Option<String>| -> Result<GpuNormalizationWeights> {
                match meta.normalization_strategy {
                    NormalizationStrategy::LayerNorm => Ok(GpuNormalizationWeights::LayerNorm(
                        GpuLayerNormWeights::new(
                            GpuTensor::from_model_weights(
                                &context, weights, &w, target_dt, "ln_g",
                            )?,
                            GpuTensor::from_model_weights(
                                &context,
                                weights,
                                &b.unwrap(),
                                target_dt,
                                "ln_b",
                            )?,
                        )?,
                    )),
                    NormalizationStrategy::RMSNorm => Ok(GpuNormalizationWeights::RMSNorm(
                        GpuRMSNormWeights::new(GpuTensor::from_model_weights(
                            &context, weights, &w, target_dt, "rms_g",
                        )?)?,
                    )),
                }
            };

            let self_attn_ln_weights = build_norm(
                name(&encoder_layout.layer.self_attn.norm_weight),
                resolve_bias(&encoder_layout.layer.self_attn.norm_bias),
            )?;

            let ffn_ln_weights = build_norm(
                name(&encoder_layout.layer.ffn.norm_weight),
                resolve_bias(&encoder_layout.layer.ffn.norm_bias),
            )?;

            // 6. Build Layer
            layers.push(GpuEncoderLayer::new(
                &context,
                self_attn_weights,
                self_attn_ln_weights,
                ff_weights_enum, // Pass the Enum
                ffn_ln_weights,
                meta.activation,
                &meta,
            )?);
        }

        Ok(Self {
            embeddings,
            embedding_layer_norm,
            embedding_ln_weights,
            layers,
            metadata: meta,
            context,
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
            0
        )
    }

    fn embed_and_normalize(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        token_type_ids: Option<ModelInput<'_>>,
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
