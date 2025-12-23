use crate::models::bart::config::BartConfig;
use anyhow::anyhow;
// IMPORTANT: You will need to create this generic layer component.
// Its structure is implied by the usage here.
use anyhow::Result;
use async_trait::async_trait;
use kjarni_transformers::activations::Activation;
use kjarni_transformers::cache::Cache;
use kjarni_transformers::cache::GpuBeamKVCache;
use kjarni_transformers::encoder_decoder::traits::{
    GpuCrossAttentionKVCache, GpuCrossDecoder, GpuCrossDecoderOutput,
};

use kjarni_transformers::WgpuContext;
use kjarni_transformers::gpu_ops::blocks::attention::GpuAttentionWeights;
use kjarni_transformers::gpu_ops::blocks::decoder_cross_attention::GpuCrossAttentionDecoderLayer;
use kjarni_transformers::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use kjarni_transformers::gpu_ops::blocks::{
    GpuFeedForward, GpuFeedForwardStd, GpuFeedForwardWeights, GpuFeedForwardWeightsStd,
    GpuNormalization, GpuNormalizationWeights,
    layer_norm::{GpuLayerNorm, GpuLayerNormWeights},
};
use kjarni_transformers::gpu_ops::{GpuTensor, GpuTensorPool};
use kjarni_transformers::models::base::ModelInput;
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::traits::ModelConfig;
use kjarni_transformers::weights::ModelWeights;
use ndarray::Array2;
use std::sync::Arc;
use wgpu::CommandEncoder;

pub struct BartGpuDecoder {
    context: Arc<WgpuContext>,
    config: Arc<BartConfig>,

    // Kernels
    pub embeddings: GpuEmbeddings,
    pub embed_layer_norm: GpuNormalization,

    // Weights
    pub embedding_weights: GpuEmbeddingWeights,
    pub embed_ln_weights: GpuNormalizationWeights,

    // Layers
    pub layers: Vec<GpuCrossAttentionDecoderLayer>,
}

impl BartGpuDecoder {
    pub fn new(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        config: Arc<BartConfig>,
        load_config: ModelLoadConfig,
    ) -> Result<Self> {
        // 1. Get Metadata and Layout
        let meta = config.metadata();
        let layout = config.layout();
        let target_dt = load_config.target_dtype;

        // --- 1. Embeddings - MANUALLY load with decoder-specific paths (Preserved Logic) ---
        let word_emb = GpuTensor::from_raw(
            context,
            &weights.get_raw(&layout.token_embedding)?,
            "decoder_word_emb",
        )?;

        // BART uses a specific separate position table for the decoder
        let pos_emb = GpuTensor::from_raw(
            context,
            &weights.get_raw("model.decoder.embed_positions.weight")?,
            "decoder_pos_emb",
        )?;

        let embedding_weights = GpuEmbeddingWeights {
            word_embeddings: word_emb,
            position_embeddings: Some(pos_emb),
            token_type_embeddings: None,
        };

        // --- 2. Initial LayerNorm (Preserved Logic) ---
        // Using specific BART decoder naming convention
        let norm_w_name = "model.decoder.layernorm_embedding.weight";
        let norm_b_name = "model.decoder.layernorm_embedding.bias";

        let embed_ln_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
            GpuTensor::from_raw(context, &weights.get_raw(norm_w_name)?, norm_w_name)?,
            GpuTensor::from_raw(context, &weights.get_raw(norm_b_name)?, norm_b_name)?,
        )?);

        // --- 3. Decoder Layers Loop ---
        let mut layers = Vec::with_capacity(config.decoder_layers);
        for i in 0..config.decoder_layers {
            let idx = i.to_string();
            let name = |t: &String| t.replace("{}", &idx);
            let opt_name = |t: &Option<String>| t.as_ref().map(|s| s.replace("{}", &idx)).unwrap();

            // --- 3a. Self Attention (Using Layout templates) ---
            let self_attn_weights = GpuAttentionWeights::new(
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw_resolved(&name(&layout.attn_q), target_dt)?,
                    "q",
                )?,
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw_resolved(&opt_name(&layout.attn_q_bias), target_dt)?,
                    "q_b",
                )?,
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw_resolved(&name(&layout.attn_k), target_dt)?,
                    "k",
                )?,
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw_resolved(&opt_name(&layout.attn_k_bias), target_dt)?,
                    "k_b",
                )?,
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw_resolved(&name(&layout.attn_v), target_dt)?,
                    "v",
                )?,
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw_resolved(&opt_name(&layout.attn_v_bias), target_dt)?,
                    "v_b",
                )?,
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw_resolved(&name(&layout.attn_o), target_dt)?,
                    "o",
                )?,
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw_resolved(&opt_name(&layout.attn_o_bias), target_dt)?,
                    "o_b",
                )?,
            )?;

            let self_attn_norm =
                GpuNormalization::LayerNorm(GpuLayerNorm::new(context, meta.norm_eps));

            let self_attn_ln_weights =
                GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_raw(
                        context,
                        &weights.get_raw(&name(&layout.attn_norm))?,
                        "sa_ln_w",
                    )?,
                    GpuTensor::from_raw(
                        context,
                        &weights.get_raw(&opt_name(&layout.attn_norm_bias))?,
                        "sa_ln_b",
                    )?,
                )?);

            // --- 3b. Cross Attention (Using Layout templates) ---
            let cross_attn_weights = GpuAttentionWeights::new(
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw_resolved(
                        &name(layout.cross_attn_q.as_ref().unwrap()),
                        target_dt,
                    )?,
                    "cq",
                )?,
                GpuTensor::from_raw(
                    context,
                    &weights
                        .get_raw_resolved(&name(&layout.cross_attn_q_bias.clone().unwrap()), target_dt)?,
                    "cq_b",
                )?, // Reusing bias template
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw_resolved(
                        &name(layout.cross_attn_k.as_ref().unwrap()),
                        target_dt,
                    )?,
                    "ck",
                )?,
                GpuTensor::from_raw(
                    context,
                    &weights
                        .get_raw_resolved(&name(&layout.cross_attn_k_bias.clone().unwrap()), target_dt)?,
                    "ck_b",
                )?,
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw_resolved(
                        &name(layout.cross_attn_v.as_ref().unwrap()),
                        target_dt,
                    )?,
                    "cv",
                )?,
                GpuTensor::from_raw(
                    context,
                    &weights
                        .get_raw_resolved(&name(&layout.cross_attn_v_bias.clone().unwrap()), target_dt)?,
                    "cv_b",
                )?,
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw_resolved(
                        &name(layout.cross_attn_o.as_ref().unwrap()),
                        target_dt,
                    )?,
                    "co",
                )?,
                GpuTensor::from_raw(
                    context,
                    &weights
                        .get_raw_resolved(&name(&layout.cross_attn_o_bias.clone().unwrap()), target_dt)?,
                    "co_b",
                )?,
            )?;

            let cross_attn_norm =
                GpuNormalization::LayerNorm(GpuLayerNorm::new(context, meta.norm_eps));

            let cross_attn_ln_weights =
                GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_raw(
                        context,
                        &weights.get_raw(&name(layout.cross_attn_norm.as_ref().unwrap()))?,
                        "ca_ln_w",
                    )?,
                    GpuTensor::from_raw(
                        context,
                        &weights.get_raw(&opt_name(&layout.cross_attn_norm_bias))?,
                        "ca_ln_b",
                    )?,
                )?);

            // --- 3c. FFN (Preserved Logic: GpuFeedForwardWeightsStd) ---
            let feedforward =
                GpuFeedForward::Standard(GpuFeedForwardStd::new(context, meta.activation)?);

            let ff_weights = GpuFeedForwardWeights::Standard(GpuFeedForwardWeightsStd::new(
                GpuTensor::from_raw(context, &weights.get_raw(&name(&layout.ffn_up))?, "ff1_w")?,
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw(&opt_name(&layout.ffn_up_bias))?,
                    "ff1_b",
                )?,
                GpuTensor::from_raw(context, &weights.get_raw(&name(&layout.ffn_down))?, "ff2_w")?,
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw(&opt_name(&layout.ffn_down_bias))?,
                    "ff2_b",
                )?,
            )?);

            let ffn_norm = GpuNormalization::LayerNorm(GpuLayerNorm::new(context, meta.norm_eps));

            let ffn_ln_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw(&name(&layout.ffn_norm))?,
                    "ffn_ln_w",
                )?,
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw(&opt_name(&layout.ffn_norm_bias))?,
                    "ffn_ln_b",
                )?,
            )?);

            let layer = GpuCrossAttentionDecoderLayer::new(
                context,
                self_attn_weights,
                self_attn_norm,
                self_attn_ln_weights,
                cross_attn_weights,
                cross_attn_norm,
                cross_attn_ln_weights,
                feedforward,
                ff_weights,
                ffn_norm,
                ffn_ln_weights,
                &meta, // Pass metadata struct
            )?;
            layers.push(layer);
        }

        // --- 4. Instantiate Kernels ---
        let embeddings = GpuEmbeddings::new(context)?;
        let embed_layer_norm =
            GpuNormalization::LayerNorm(GpuLayerNorm::new(context, meta.norm_eps));

        Ok(Self {
            context: context.clone(),
            config,
            embeddings,
            embedding_weights,
            embed_layer_norm,
            embed_ln_weights,
            layers,
        })
    }
}

#[async_trait(?Send)]
impl GpuCrossDecoder for BartGpuDecoder {
    async fn embed(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        let meta = self.config.metadata();
        let cpu_embeddings = false; // todo modelLoadConfig
        match input {
            ModelInput::TokensCpu(ids) => {
                // If BART has CPU embeddings offloaded
                let ids = ids.as_standard_layout().to_owned();
                let ids_gpu = GpuTensor::from_ndarray(&self.context, &ids)?;
                self.embeddings.encode(
                    encoder, &self.embedding_weights, &ids_gpu, None,
                    position_offset, meta.hidden_size, meta.extra_pos_embeddings, meta.scale_embeddings, pool
                )
                // if let Some(cpu_embeds) = &self.cpu_embeddings {
                //     let input_array = Array2::from_shape_vec((1, ids.len()), ids.to_vec())?;
                //     let hidden_cpu = cpu_embeds.forward(
                //         &input_array, None, position_offset, meta.scale_embeddings
                //     );
                //     GpuTensor::from_ndarray(&self.context, &hidden_cpu)
                // } else {
                //     let ids_gpu = GpuTensor::from_ndarray(&self.context, &Array2::from_shape_vec((1, ids.len()), ids.to_vec())?)?;
                //     self.embeddings.encode(
                //         encoder, &self.embedding_weights, &ids_gpu, None,
                //         position_offset, meta.hidden_size, meta.extra_pos_embeddings, meta.scale_embeddings, pool
                //     )
                // }
            }
            ModelInput::TokensGpu(ids_gpu) => {
                self.embeddings.encode(
                    encoder, &self.embedding_weights, ids_gpu, None,
                    position_offset, meta.hidden_size, meta.extra_pos_embeddings, meta.scale_embeddings, pool
                )
            }
            ModelInput::HiddenGpu(t) => Ok(t.clone()),
            ModelInput::HiddenCpu(t) => GpuTensor::from_ndarray(&self.context, &t.as_standard_layout().to_owned()),
        }
    }

    async fn embed_and_normalize(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        let hidden = self.embed(encoder, pool, input, position_offset).await?;
        let ln_output = pool.get(hidden.shape().to_vec());
        self.embed_layer_norm.encode(encoder, &self.embed_ln_weights, &hidden, &ln_output);
        Ok(ln_output)
    }

    fn forward_layers(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        encoder_hidden_states: &GpuTensor,
        decoder_attention_mask: &GpuTensor,
        position_offset: usize,
        mut cache: Option<&mut dyn Cache>,
        cross_kv_cache: Option<&GpuCrossAttentionKVCache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<GpuCrossDecoderOutput> {
        let gpu_cache = cache.as_deref_mut().and_then(|c| c.as_any_mut().downcast_mut::<GpuBeamKVCache>());
        let mut current_hidden = hidden_states.clone();
        let mut new_self_attn_kvs = Vec::with_capacity(self.layers.len());

        for i in start_layer..end_layer {
            let layer = &self.layers[i];
            let cross_kv_for_layer = cross_kv_cache.and_then(|c| c.0.get(i));
            let self_attn_past_kv = gpu_cache.as_ref().and_then(|c| c.get_layer_tensors(i));

            let (new_hidden, new_k, new_v) = layer.forward(
                encoder,
                &current_hidden,
                Some(encoder_hidden_states),
                decoder_attention_mask,
                None, // Encoder mask
                self_attn_past_kv,
                cross_kv_for_layer,
                position_offset,
                pool,
            )?;

            current_hidden = new_hidden;
            new_self_attn_kvs.push((new_k, new_v));
        }

        Ok(GpuCrossDecoderOutput {
            last_hidden_state: current_hidden,
            new_self_attn_kv: new_self_attn_kvs,
        })
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_size(&self) -> usize {
        self.config.metadata().hidden_size
    }

    fn precompute_cross_attention_kv(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        encoder_hidden_states: &GpuTensor,
    ) -> Result<GpuCrossAttentionKVCache> {
        let mut cache = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            cache.push(layer.precompute_cross_kv(encoder, encoder_hidden_states, pool));
        }
        Ok(GpuCrossAttentionKVCache(cache))
    }

    
    // fn forward(
    //     &self,
    //     encoder: &mut CommandEncoder,
    //     pool: &mut GpuTensorPool,
    //     decoder_input: ModelInput<'_>,
    //     encoder_hidden_states: &GpuTensor,
    //     decoder_attention_mask: &GpuTensor,
    //     cache: Option<&mut dyn Cache>,
    //     cross_kv_cache: Option<&GpuCrossAttentionKVCache>,
    // ) -> Result<GpuCrossDecoderOutput> {
    //     // --- Get position offset and GPU cache ---
    //     let position_offset = cache.as_ref().map_or(0, |c| c.get_seq_length());
    //     let gpu_cache = cache.and_then(|c| c.as_any().downcast_ref::<GpuBeamKVCache>());
    //     let metadata = self.config.metadata();
    //     // --- 1. Embeddings + Initial LayerNorm ---
    //     let embedded_tokens = match decoder_input {
    //         ModelInput::TokensCpu(ids) => {
    //             // This is an inefficient path for GPU, but necessary for correctness.
    //             // It should ideally be handled at a higher level (e.g., in the backend).
    //             // For now, we upload the tokens here.
    //             let ids_gpu = GpuTensor::from_ndarray(
    //                 &self.context,
    //                 &Array2::from_shape_vec((1, ids.len()), ids.to_vec())?,
    //             )?;
    //             self.embeddings.encode(
    //                 encoder,
    //                 &self.embedding_weights,
    //                 &ids_gpu, // Pass the GpuTensor
    //                 None,
    //                 position_offset,
    //                 metadata.hidden_size,
    //                 metadata.extra_pos_embeddings,
    //                 metadata.scale_embeddings,
    //                 pool,
    //             )?
    //         }
    //         ModelInput::TokensGpu(ids_gpu) => {
    //             self.embeddings.encode(
    //                 encoder,
    //                 &self.embedding_weights,
    //                 ids_gpu, // Pass the GpuTensor
    //                 None,
    //                 position_offset,
    //                 metadata.hidden_size,
    //                 metadata.extra_pos_embeddings,
    //                 metadata.scale_embeddings,
    //                 pool,
    //             )?
    //         }
    //         ModelInput::HiddenGpu(hidden) => hidden.clone(),
    //         _ => {
    //             return Err(anyhow!(
    //                 "Unsupported ModelInput variant for GpuCrossDecoder"
    //             ));
    //         }
    //     };
    //     let mut hidden_states = embedded_tokens;
    //     let ln_output = pool.get(hidden_states.shape().to_vec());
    //     self.embed_layer_norm
    //         .encode(encoder, &self.embed_ln_weights, &hidden_states, &ln_output);
    //     hidden_states = ln_output;

    //     // --- 2. Decoder Layers ---
    //     let mut new_self_attn_kvs = Vec::with_capacity(self.layers.len());

    //     for (i, layer) in self.layers.iter().enumerate() {
    //         // Get the specific cross-attention K/V for this layer from the pre-computed cache
    //         let cross_kv_for_layer = cross_kv_cache.and_then(|c| c.0.get(i));

    //         // Get the past self-attention K/V for this layer from the mutable cache
    //         let self_attn_past_kv = gpu_cache.and_then(|c| c.get_layer_tensors(i));

    //         let (new_hidden, new_k, new_v) = layer.forward(
    //             encoder,
    //             &hidden_states,
    //             Some(encoder_hidden_states),
    //             decoder_attention_mask,
    //             None, // Encoder mask is usually handled by attention kernel
    //             self_attn_past_kv,
    //             cross_kv_for_layer,
    //             position_offset,
    //             pool,
    //         )?;

    //         hidden_states = new_hidden;
    //         new_self_attn_kvs.push((new_k, new_v));
    //     }

    //     // --- 3. Return the complete output struct ---
    //     Ok(GpuCrossDecoderOutput {
    //         last_hidden_state: hidden_states,
    //         new_self_attn_kv: new_self_attn_kvs,
    //     })
    // }
}
