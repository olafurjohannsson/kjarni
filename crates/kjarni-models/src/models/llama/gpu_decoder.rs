// --- Standard Library ---
use std::sync::Arc;

// --- Crate-Specific ---
use crate::models::llama::config::LlamaConfig;
// --- External Crates ---
use anyhow::{Context, Result};
use async_trait::async_trait;
// --- Workspace Crates ---
use kjarni_transformers::{
    cache::GpuKVCache,
    decoder::prelude::*,
    embeddings::{Embeddings, LoadedEmbeddings},
    gpu_ops::{
        blocks::{
            attention::{GpuAttention, GpuAttentionWeights}, embeddings::{GpuEmbeddingWeights, GpuEmbeddings}, rms_norm::{GpuRMSNorm, GpuRMSNormWeights}, rope::GpuRoPE,
            GpuFeedForward, GpuFeedForwardWeights,
            GpuNormalization,
            GpuNormalizationWeights,
            GpuSwiGLUFFN,
            GpuSwiGLUFFNWeights,
        }, primitives::add::GpuAdd, GpuTensor,
        GpuTensorPool,
        Kernel,
    },
    models::base::ModelLoadConfig,
    tensor::DType,
    traits::{ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
    WgpuContext,
};
use ndarray::Array2;

/// The GPU-native implementation of the Llama decoder architecture.
pub struct LlamaGpuDecoder {
    gpu_embeddings: Option<GpuEmbeddings>,
    gpu_embedding_weights: Option<GpuEmbeddingWeights>,
    cpu_embeddings: Option<Embeddings>,
    layers: Vec<LlamaGpuDecoderLayer>,
    final_layer_norm: GpuNormalization,
    final_ln_weights: GpuNormalizationWeights,
    gpu_rope: GpuRoPE,
    context: Arc<WgpuContext>,
    config: Arc<LlamaConfig>,
    load_config: ModelLoadConfig,
}

impl LlamaGpuDecoder {
    pub fn context(&self) -> &Arc<WgpuContext> {
        &self.context
    }

    pub fn new(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        config: Arc<LlamaConfig>,
        gpu_rope: GpuRoPE,
        load_config: ModelLoadConfig,
    ) -> Result<Self> {
        let meta = config.metadata();
        let layout = config.layout();

        // 1. Unified Embedding Loading
        let embs = LoadedEmbeddings::from_layout(context, weights, &layout, load_config)?;

        // 2. Final Layer Norm
        let norm_raw = weights.get_raw_resolved(&layout.final_norm, load_config.target_dtype)?;
        let final_layer_norm = GpuNormalization::RMSNorm(GpuRMSNorm::new(context, meta.norm_eps));
        let final_ln_weights = GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(
            GpuTensor::from_raw(context, &norm_raw, "final_norm")?,
        )?);

        // 3. Decoder Layers
        let mut layers = Vec::with_capacity(meta.num_layers);
        for i in 0..meta.num_layers {
            // No more casting to dyn DecoderArchitecture!
            let decoder_layer = Self::build_layer(
                context.clone(),
                weights,
                &meta,
                &layout,
                i,
                load_config,
            )?;
            layers.push(decoder_layer);
        }

        Ok(Self {
            gpu_embedding_weights: embs.gpu_weights,
            gpu_embeddings: embs.gpu_layer,
            cpu_embeddings: embs.cpu,
            layers,
            final_layer_norm,
            final_ln_weights,
            gpu_rope,
            context: context.clone(),
            config,
            load_config,
        })
    }

    fn build_layer(
        context: Arc<WgpuContext>,
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        i: usize,
        load_config: ModelLoadConfig,
    ) -> Result<LlamaGpuDecoderLayer> {
        let idx = i.to_string();
        let name = |template: &String| template.replace("{}", &idx);
        let target_dt = load_config.target_dtype;

        // --- 1. Attention Weights ---
        let q_t = GpuTensor::from_raw(
            &context,
            &weights.get_raw_resolved(&name(&layout.attn_q), target_dt)?,
            "q",
        )?;
        let k_t = GpuTensor::from_raw(
            &context,
            &weights.get_raw_resolved(&name(&layout.attn_k), target_dt)?,
            "k",
        )?;
        let v_t = GpuTensor::from_raw(
            &context,
            &weights.get_raw_resolved(&name(&layout.attn_v), target_dt)?,
            "v",
        )?;
        let o_t = GpuTensor::from_raw(
            &context,
            &weights.get_raw_resolved(&name(&layout.attn_o), target_dt)?,
            "o",
        )?;

        let q_bias = GpuTensor::zeros(&context, vec![meta.hidden_size], DType::F32, "q_b")?;
        let head_dim = meta.hidden_size / meta.num_attention_heads;
        let k_bias = GpuTensor::zeros(
            &context,
            vec![meta.num_kv_heads * head_dim],
            DType::F32,
            "k_b",
        )?;

        let self_attn_weights = GpuAttentionWeights::new(
            q_t,
            q_bias.clone(),
            k_t,
            k_bias.clone(),
            v_t,
            k_bias,
            o_t,
            q_bias,
        )?;
        let self_attn_norm_w =
            GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(GpuTensor::from_raw(
                &context,
                &weights.get_raw_resolved(&name(&layout.attn_norm), target_dt)?,
                "attn_norm",
            )?)?);

        // --- 2. FFN Weights ---
        let gate_name = layout.ffn_gate.as_ref().context("SwiGLU requires gate")?;
        let gate_t = GpuTensor::from_raw(
            &context,
            &weights.get_raw_resolved(&name(gate_name), target_dt)?,
            "gate",
        )?;
        let up_t = GpuTensor::from_raw(
            &context,
            &weights.get_raw_resolved(&name(&layout.ffn_up), target_dt)?,
            "up",
        )?;
        let down_t = GpuTensor::from_raw(
            &context,
            &weights.get_raw_resolved(&name(&layout.ffn_down), target_dt)?,
            "down",
        )?;

        let ff_weights =
            GpuFeedForwardWeights::SwiGLU(GpuSwiGLUFFNWeights::new(gate_t, up_t, down_t)?);
        let ffn_norm_w =
            GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(GpuTensor::from_raw(
                &context,
                &weights.get_raw_resolved(&name(&layout.ffn_norm), target_dt)?,
                "ffn_norm",
            )?)?);

        // --- 3. Construct Dedicated Layer ---
        LlamaGpuDecoderLayer::new(
            &context,
            meta.hidden_size,
            meta.num_attention_heads,
            meta.num_kv_heads,
            self_attn_weights,
            self_attn_norm_w,
            ff_weights,
            ffn_norm_w,
            meta.norm_eps,
        )
    }
}

#[async_trait(?Send)]
impl GpuDecoder for LlamaGpuDecoder {
    async fn embed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: DecoderInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        let metadata = self.config.metadata();
        match input {
            DecoderInput::TokensCpu(ids) => {
                // This path is for prefill. It should use CPU embeddings if available.
                if let Some(cpu_embeds) = &self.cpu_embeddings {
                    let input_array = Array2::from_shape_vec((1, ids.len()), ids.to_vec())?;
                    let embeddings_cpu = cpu_embeds.forward(
                        &input_array,
                        None,
                        position_offset,
                        metadata.scale_embeddings,
                    );
                    GpuTensor::from_ndarray(&self.context, &embeddings_cpu)
                } else {
                    // This case handles prefill when embeddings are on GPU.
                    let gpu_embeds = self.gpu_embeddings.as_ref().unwrap();
                    let gpu_weights = self.gpu_embedding_weights.as_ref().unwrap();
                    let ids_tensor = GpuTensor::from_ndarray(
                        &self.context,
                        &Array2::from_shape_vec((1, ids.len()), ids.to_vec())?,
                    )?;
                    gpu_embeds.encode(
                        encoder,
                        gpu_weights,
                        &ids_tensor,
                        None,
                        position_offset,
                        metadata.hidden_size,
                        0,                         // no extra pos embeddings for Llama
                        metadata.scale_embeddings, // from ModelMetadata
                        pool,
                    )
                }
            }
            DecoderInput::TokensGpu(ids_tensor) => {
                if let Some(cpu_embeds) = &self.cpu_embeddings {
                    let token_id_vec: Vec<u32> =
                        bytemuck::cast_slice(&ids_tensor.read_raw_data().await?).to_vec();
                    let token_id = token_id_vec[0];

                    let input_array = Array2::from_elem((1, 1), token_id);
                    let embeddings_cpu = cpu_embeds.forward(
                        &input_array,
                        None,
                        position_offset,
                        metadata.scale_embeddings,
                    );

                    GpuTensor::from_ndarray(&self.context, &embeddings_cpu)
                } else {
                    let gpu_embeds = self.gpu_embeddings.as_ref().unwrap(); // Safe to unwrap here
                    let gpu_weights = self.gpu_embedding_weights.as_ref().unwrap();
                    gpu_embeds.encode(
                        encoder,
                        gpu_weights,
                        &ids_tensor,
                        None, // no token type ids for Llama
                        position_offset,
                        metadata.hidden_size,
                        0,                         // no extra pos embeddings for Llama
                        metadata.scale_embeddings, // from ModelMetadata
                        pool,
                    )
                }
            }
            DecoderInput::HiddenGpu(t) => Ok(t.clone()),
            DecoderInput::HiddenCpu(t) => GpuTensor::from_ndarray(&self.context, t),
        }
    }

    async fn embed_and_normalize(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: DecoderInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        // Llama is Pre-Norm (Norm is inside the layer), so we just embed.
        self.embed(encoder, pool, input, position_offset).await
    }

    fn forward_layers(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        position_offset: usize,
        mut cache: Option<&mut GpuKVCache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<GpuTensor> {
        let mut current_state = hidden_states.clone();

        for i in start_layer..end_layer {
            let layer = &self.layers[i];
            let layer_cache = cache.as_deref_mut();

            current_state = layer.forward(
                encoder,
                &current_state,
                attention_mask,
                i,
                position_offset,
                layer_cache,
                pool,
                &self.gpu_rope, // Always passed for Llama
            )?;
        }

        Ok(current_state)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    // Override default forward because Llama needs Final Layer Norm
    async fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: DecoderInput<'_>,
        attention_mask: &GpuTensor,
        position_offset: usize,
        cache: Option<&mut GpuKVCache>,
        _encoder_hidden_states: Option<&GpuTensor>,
    ) -> Result<GpuTensor> {
        // 1. Embed
        let hidden = self
            .embed_and_normalize(encoder, pool, input, position_offset)
            .await?;

        // 2. Layers
        let mut hidden = self.forward_layers(
            encoder,
            pool,
            &hidden,
            attention_mask,
            position_offset,
            cache,
            0,
            self.num_layers(),
        )?;

        // 3. Final Layer Norm (Llama Specific)
        let final_ln_output = pool.get(hidden.shape().to_vec());
        self.final_layer_norm
            .encode(encoder, &self.final_ln_weights, &hidden, &final_ln_output);
        hidden = final_ln_output;

        Ok(hidden)
    }
}

pub struct LlamaGpuDecoderLayer {
    pub self_attn: GpuAttention,
    pub self_attn_weights: GpuAttentionWeights,
    pub self_attn_norm: GpuNormalization,
    pub self_attn_norm_weights: GpuNormalizationWeights,
    pub feedforward: GpuFeedForward,
    pub ff_weights: GpuFeedForwardWeights,
    pub ffn_norm: GpuNormalization,
    pub ffn_norm_weights: GpuNormalizationWeights,
    add: GpuAdd,
}

impl LlamaGpuDecoderLayer {
    pub fn new(
        context: &Arc<WgpuContext>,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        self_attn_weights: GpuAttentionWeights,
        self_attn_norm_weights: GpuNormalizationWeights,
        ff_weights: GpuFeedForwardWeights,
        ffn_norm_weights: GpuNormalizationWeights,
        norm_eps: f32,
    ) -> Result<Self> {
        let self_attn = GpuAttention::new(
            context,
            hidden_size as u32,
            num_heads as u32,
            num_kv_heads as u32,
        );
        let add = GpuAdd::new(context);

        // Llama specific blocks
        let self_attn_norm = GpuNormalization::RMSNorm(GpuRMSNorm::new(context, norm_eps));
        let ffn_norm = GpuNormalization::RMSNorm(GpuRMSNorm::new(context, norm_eps));
        let feedforward = GpuFeedForward::SwiGLU(GpuSwiGLUFFN::new(context)?);

        Ok(Self {
            self_attn,
            self_attn_weights,
            self_attn_norm,
            self_attn_norm_weights,
            feedforward,
            ff_weights,
            ffn_norm,
            ffn_norm_weights,
            add,
        })
    }

    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        layer_idx: usize,
        position_offset: usize,
        gpu_cache: Option<&mut GpuKVCache>,
        pool: &mut GpuTensorPool,
        rope: &GpuRoPE, // Llama always uses RoPE
    ) -> Result<GpuTensor> {
        // --- 1. Self-Attention Block (Pre-Norm) ---
        let residual = hidden_states;
        let ln1_out = pool.get(hidden_states.shape().to_vec());
        self.self_attn_norm.encode(
            encoder,
            &self.self_attn_norm_weights,
            hidden_states,
            &ln1_out,
        );

        // Project Q, K, V
        let q_proj = self.self_attn.project(
            encoder,
            &ln1_out,
            &self.self_attn_weights.q_weight,
            &self.self_attn_weights.q_bias,
            pool,
        );
        let k_proj = self.self_attn.project(
            encoder,
            &ln1_out,
            &self.self_attn_weights.k_weight,
            &self.self_attn_weights.k_bias,
            pool,
        );
        let v_proj = self.self_attn.project(
            encoder,
            &ln1_out,
            &self.self_attn_weights.v_weight,
            &self.self_attn_weights.v_bias,
            pool,
        );

        // Split heads
        let q_split = self.self_attn.split_heads(encoder, &q_proj, pool);
        let k_split = self.self_attn.split_heads(encoder, &k_proj, pool);
        let v_split = self.self_attn.split_heads(encoder, &v_proj, pool);

        // Apply RoPE
        let q_rotated = pool.get(q_split.shape().to_vec());
        let k_rotated = pool.get(k_split.shape().to_vec());
        rope.encode(encoder, &q_split, &q_rotated, position_offset);
        rope.encode(encoder, &k_split, &k_rotated, position_offset);

        // Attention with Cache
        let attn_out = if let Some(cache) = gpu_cache {
            let k_rotated_3d = self.self_attn.merge_heads(encoder, &k_rotated, pool);
            cache.update(encoder, layer_idx, &k_rotated_3d, &v_proj, position_offset)?;

            let (cached_k, cached_v) = cache.get(layer_idx).unwrap();
            self.self_attn.llama_attention(
                encoder,
                &q_rotated,
                &cached_k,
                &cached_v,
                attention_mask,
                position_offset,
                pool,
                &self.self_attn_weights,
            )
        } else {
            self.self_attn.llama_attention(
                encoder,
                &q_rotated,
                &k_rotated,
                &v_split,
                attention_mask,
                position_offset,
                pool,
                &self.self_attn_weights,
            )
        };

        let attn_block_output = pool.get(hidden_states.shape().to_vec());
        self.add
            .encode(encoder, &[residual, &attn_out], &attn_block_output);

        // --- 2. Feed-Forward Block (Pre-Norm) ---
        let residual_2 = &attn_block_output;
        let ln2_out = pool.get(residual_2.shape().to_vec());
        self.ffn_norm
            .encode(encoder, &self.ffn_norm_weights, residual_2, &ln2_out);

        // FFN Reshape (Llama needs 2D FFN)
        let (b, s, h) = ln2_out.dims3();
        let ln2_out_2d = ln2_out.view(vec![b * s, h]);
        let ffn_out_2d = pool.get(vec![b * s, h]);

        self.feedforward
            .encode(encoder, &self.ff_weights, &ln2_out_2d, &ffn_out_2d, pool);
        let ffn_out = ffn_out_2d.view(vec![b, s, h]);

        let final_output = pool.get(residual_2.shape().to_vec());
        self.add
            .encode(encoder, &[residual_2, &ffn_out], &final_output);

        Ok(final_output)
    }
}
