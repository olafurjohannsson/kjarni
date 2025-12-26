use crate::models::llama::config::LlamaConfig;
use anyhow::{Context, Result};
use async_trait::async_trait;
use kjarni_transformers::{
    WgpuContext,
    cache::GpuKVCache,
    decoder::prelude::*,
    embeddings::{EmbeddingConfig, Embeddings, LoadedEmbeddings},
    gpu_ops::{
        GpuTensor, GpuTensorPool, Kernel,
        blocks::{
            GpuFeedForward, GpuFeedForwardWeights, GpuNormalization, GpuNormalizationWeights,
            GpuSwiGLUFFN, GpuSwiGLUFFNWeights,
            attention::{
                GpuAttention, GpuAttentionWeights, GpuDecoderSelfAttention, GpuRoPEAttention,
            },
            embeddings::{GpuEmbeddingWeights, GpuEmbeddings},
            rms_norm::{GpuRMSNorm, GpuRMSNormWeights},
            rope::GpuRoPE,
        },
        primitives::add::GpuAdd,
    },
    models::base::{ModelInput, ModelLoadConfig},
    tensor::DType,
    traits::{ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};
use std::sync::Arc;

/// The GPU-native implementation of the Llama decoder architecture.
pub struct LlamaGpuDecoder {
    // gpu_embeddings: Option<GpuEmbeddings>,
    // gpu_embedding_weights: Option<GpuEmbeddingWeights>,
    // cpu_embeddings: Option<Embeddings>,
    pub layers: Vec<LlamaGpuDecoderLayer>,
    pub final_layer_norm: GpuNormalization,
    pub final_ln_weights: GpuNormalizationWeights,
    pub gpu_rope: GpuRoPE,
    context: Arc<WgpuContext>,
    config: Arc<LlamaConfig>,
    load_config: ModelLoadConfig,
    embeddings: LoadedEmbeddings,
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
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("Llama layout must have a decoder section");

        // 1. Unified Embedding Loading
let target_dtype = load_config.target_dtype;

        // GPU decoder always loads to GPU, CPU offload is handled at model level
        let embeddings = LoadedEmbeddings::new(
            Some(context),  // Option<&Arc<WgpuContext>>
            weights,
            EmbeddingConfig::new(&layout.token_embedding, meta.hidden_size),
            false,  // load_cpu - GPU decoder doesn't need CPU embeddings
            true,   // load_gpu - always load to GPU for GPU decoder
            target_dtype,
        )?;

        // 2. Final Layer Norm
        let final_norm_name = decoder_layout.final_norm_weight.as_ref().unwrap();
        let final_layer_norm = GpuNormalization::RMSNorm(GpuRMSNorm::new(context, meta.norm_eps));
        let final_ln_weights = GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(
            GpuTensor::from_model_weights(
                context,
                weights,
                final_norm_name,
                load_config.target_dtype,
                "final_norm",
            )?,
        )?);

        // 3. Decoder Layers
        let mut layers = Vec::with_capacity(meta.num_layers);
        for i in 0..meta.num_layers {
            let decoder_layer =
                Self::build_layer(context.clone(), weights, &meta, &layout, i, load_config)?;
            layers.push(decoder_layer);
        }

        Ok(Self {
            // gpu_embedding_weights: gpu_weights,
            // gpu_embeddings: gpu_layer,
            // cpu_embeddings: cpu,
            layers,
            final_layer_norm,
            final_ln_weights,
            gpu_rope,
            context: context.clone(),
            config,
            load_config,
            embeddings,
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
        // Get the specific nested layouts for the decoder.
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("Llama layout must have a decoder section");
        let layer_layout = &decoder_layout.layer;
        let self_attn_layout = &layer_layout.self_attn;
        let ffn_layout = &layer_layout.ffn;

        let idx = i.to_string();
        let name = |template: &String| template.replace("{}", &idx);
        let target_dt = load_config.target_dtype;

        // for now force layers to use F32
        let layer_dt = Some(DType::BF16);

        // --- 1. Attention Weights (Using original GpuTensor::from_raw calls) ---
        let q_t = GpuTensor::from_model_weights(
            &context,
            weights,
            &name(&self_attn_layout.q_weight),
            layer_dt,
            "q",
        )?;
        let k_t = GpuTensor::from_model_weights(
            &context,
            weights,
            &name(&self_attn_layout.k_weight),
            layer_dt,
            "k",
        )?;
        let v_t = GpuTensor::from_model_weights(
            &context,
            weights,
            &name(&self_attn_layout.v_weight),
            layer_dt,
            "v",
        )?;
        let o_t = GpuTensor::from_model_weights(
            &context,
            weights,
            &name(&self_attn_layout.o_weight),
            layer_dt,
            "o",
        )?;

        // Dummy biases for Llama (logic preserved)
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

        let self_attn_norm_w = GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(
            GpuTensor::from_model_weights(
                &context,
                weights,
                &name(&self_attn_layout.norm_weight),
                layer_dt,
                "attn_norm",
            )?,
        )?);

        let gate_name = ffn_layout
            .gate_weight
            .as_ref()
            .context("SwiGLU requires gate")?;

        let gate_t =
            GpuTensor::from_model_weights(&context, weights, &name(gate_name), target_dt, "gate")?;

        let up_t = GpuTensor::from_model_weights(
            &context,
            weights,
            &name(&ffn_layout.up_weight),
            target_dt,
            "up",
        )?;
        let down_t = GpuTensor::from_model_weights(
            &context,
            weights,
            &name(&ffn_layout.down_weight),
            target_dt,
            "down",
        )?;

        log::info!("gate_t shape: {:?}", gate_t.shape());
        log::info!("up_t shape: {:?}", up_t.shape());
        log::info!("down_t shape: {:?}", down_t.shape());

        let ff_weights =
            GpuFeedForwardWeights::SwiGLU(GpuSwiGLUFFNWeights::new(gate_t, up_t, down_t)?);

        let ffn_norm_w = GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(
            GpuTensor::from_model_weights(
                &context,
                weights,
                &name(&ffn_layout.norm_weight),
                target_dt,
                "ffn_norm",
            )?,
        )?);

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
        input: ModelInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        match input {
            ModelInput::TokensCpu(ids) => {
                // Convert view to owned for embedding lookup
                let ids_owned = ids.to_owned();
                self.embeddings
                    .forward(encoder, pool, &ids_owned, None, position_offset)
            }
            ModelInput::TokensGpu(ids_tensor) => {
                self.embeddings
                    .forward_gpu(encoder, pool, ids_tensor, None, position_offset)
                    .await
            }
            ModelInput::HiddenGpu(t) => Ok(t.clone()),
            ModelInput::HiddenCpu(t) => GpuTensor::from_ndarray(&self.context, &t.to_owned()),
        }
    }

    async fn embed_and_normalize(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
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
        let mut current_state: GpuTensor = hidden_states.clone();

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
        input: ModelInput<'_>,
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
    // pub self_attn: GpuAttention,
    pub self_attn: GpuRoPEAttention,
    pub self_attn_weights: GpuAttentionWeights,
    pub self_attn_norm: GpuNormalization,
    pub self_attn_norm_weights: GpuNormalizationWeights,
    pub feedforward: GpuFeedForward,
    pub ff_weights: GpuFeedForwardWeights,
    pub ffn_norm: GpuNormalization,
    pub ffn_norm_weights: GpuNormalizationWeights,
    pub add: GpuAdd,
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
        let self_attn = GpuRoPEAttention::new(
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
        rope: &GpuRoPE,
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

        // Get cached KV if available (returns Option<(&GpuTensor, &GpuTensor)>)
        let cached_tensors = gpu_cache.as_ref().and_then(|c| c.get(layer_idx));

        let cached_kv: Option<(&GpuTensor, &GpuTensor)> =
            cached_tensors.as_ref().map(|(k, v)| (k, v));

        // Single forward call handles: Q/K/V projection, RoPE, GQA, attention, output projection
        let attn_output = self.self_attn.forward(
            encoder,
            &ln1_out,
            &self.self_attn_weights,
            rope,
            attention_mask,
            cached_kv,
            position_offset,
            pool,
        )?;

        // Update cache with new K/V (need mutable borrow now)
        if let Some(cache) = gpu_cache {
            cache.update(
                encoder,
                layer_idx,
                &attn_output.new_k,
                &attn_output.new_v,
                position_offset,
            )?;
        }

        // Residual add
        let attn_block_output = pool.get(hidden_states.shape().to_vec());
        self.add.encode(
            encoder,
            &[residual, &attn_output.hidden_states],
            &attn_block_output,
        );

        // --- 2. Feed-Forward Block (Pre-Norm) ---
        let residual_2 = &attn_block_output;
        let ln2_out = pool.get(residual_2.shape().to_vec());
        self.ffn_norm
            .encode(encoder, &self.ffn_norm_weights, residual_2, &ln2_out);

        // FFN (needs 2D input)
        let (b, s, h) = ln2_out.dims3();
        let ln2_out_2d = ln2_out.view(vec![b * s, h]);
        let ffn_out_2d = pool.get(vec![b * s, h]);

        self.feedforward
            .encode(encoder, &self.ff_weights, &ln2_out_2d, &ffn_out_2d, pool);
        let ffn_out = ffn_out_2d.view(vec![b, s, h]);

        // Residual add
        let final_output = pool.get(residual_2.shape().to_vec());
        self.add
            .encode(encoder, &[residual_2, &ffn_out], &final_output);

        Ok(final_output)
    }
    // pub fn forward(
    //     &self,
    //     encoder: &mut wgpu::CommandEncoder,
    //     hidden_states: &GpuTensor,
    //     attention_mask: &GpuTensor,
    //     layer_idx: usize,
    //     position_offset: usize,
    //     gpu_cache: Option<&mut GpuKVCache>,
    //     pool: &mut GpuTensorPool,
    //     rope: &GpuRoPE, // Llama always uses RoPE
    // ) -> Result<GpuTensor> {
    //     // --- 1. Self-Attention Block (Pre-Norm) ---
    //     let residual = hidden_states;
    //     let ln1_out = pool.get(hidden_states.shape().to_vec());
    //     self.self_attn_norm.encode(
    //         encoder,
    //         &self.self_attn_norm_weights,
    //         hidden_states,
    //         &ln1_out,
    //     );

    //     // Project Q, K, V
    //     let q_proj = self.self_attn.project(
    //         encoder,
    //         &ln1_out,
    //         &self.self_attn_weights.q_weight,
    //         &self.self_attn_weights.q_bias,
    //         pool,
    //     );
    //     let k_proj = self.self_attn.project(
    //         encoder,
    //         &ln1_out,
    //         &self.self_attn_weights.k_weight,
    //         &self.self_attn_weights.k_bias,
    //         pool,
    //     );
    //     let v_proj = self.self_attn.project(
    //         encoder,
    //         &ln1_out,
    //         &self.self_attn_weights.v_weight,
    //         &self.self_attn_weights.v_bias,
    //         pool,
    //     );

    //     // Split heads
    //     let q_split = self.self_attn.split_heads(encoder, &q_proj, pool);
    //     let k_split = self.self_attn.split_heads(encoder, &k_proj, pool);
    //     let v_split = self.self_attn.split_heads(encoder, &v_proj, pool);

    //     // Apply RoPE
    //     let q_rotated = pool.get(q_split.shape().to_vec());
    //     let k_rotated = pool.get(k_split.shape().to_vec());
    //     rope.encode(encoder, &q_split, &q_rotated, position_offset);
    //     rope.encode(encoder, &k_split, &k_rotated, position_offset);

    //     // Attention with Cache
    //     let attn_out = if let Some(cache) = gpu_cache {
    //         let k_rotated_3d = self.self_attn.merge_heads(encoder, &k_rotated, pool);
    //         cache.update(encoder, layer_idx, &k_rotated_3d, &v_proj, position_offset)?;

    //         let (cached_k, cached_v) = cache.get(layer_idx).unwrap();
    //         self.self_attn.llama_attention(
    //             encoder,
    //             &q_rotated,
    //             &cached_k,
    //             &cached_v,
    //             attention_mask,
    //             position_offset,
    //             pool,
    //             &self.self_attn_weights,
    //         )
    //     } else {
    //         self.self_attn.llama_attention(
    //             encoder,
    //             &q_rotated,
    //             &k_rotated,
    //             &v_split,
    //             attention_mask,
    //             position_offset,
    //             pool,
    //             &self.self_attn_weights,
    //         )
    //     };

    //     let attn_block_output = pool.get(hidden_states.shape().to_vec());
    //     self.add
    //         .encode(encoder, &[residual, &attn_out], &attn_block_output);

    //     // --- 2. Feed-Forward Block (Pre-Norm) ---
    //     let residual_2 = &attn_block_output;
    //     let ln2_out = pool.get(residual_2.shape().to_vec());
    //     self.ffn_norm
    //         .encode(encoder, &self.ffn_norm_weights, residual_2, &ln2_out);

    //     // FFN Reshape (Llama needs 2D FFN)
    //     let (b, s, h) = ln2_out.dims3();
    //     let ln2_out_2d = ln2_out.view(vec![b * s, h]);
    //     let ffn_out_2d = pool.get(vec![b * s, h]);

    //     self.feedforward
    //         .encode(encoder, &self.ff_weights, &ln2_out_2d, &ffn_out_2d, pool);
    //     let ffn_out = ffn_out_2d.view(vec![b, s, h]);

    //     let final_output = pool.get(residual_2.shape().to_vec());
    //     self.add
    //         .encode(encoder, &[residual_2, &ffn_out], &final_output);

    //     Ok(final_output)
    // }
}
