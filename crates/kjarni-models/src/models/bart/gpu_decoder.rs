use crate::models::bart::config::BartConfig;
use anyhow::anyhow;
// IMPORTANT: You will need to create this generic layer component.
// Its structure is implied by the usage here.
use anyhow::Result;
use async_trait::async_trait;
use kjarni_transformers::activations::Activation;
use kjarni_transformers::cache::Cache;
use kjarni_transformers::cache::GpuBeamKVCache;
use kjarni_transformers::decoder::traits::DecoderInput;
use kjarni_transformers::encoder_decoder::traits::{
    GpuCrossAttentionKVCache, GpuCrossDecoder, GpuCrossDecoderOutput,
};

use kjarni_transformers::gpu_ops::blocks::attention::GpuAttentionWeights;
use kjarni_transformers::gpu_ops::blocks::decoder_cross_attention::GpuCrossAttentionDecoderLayer;
use kjarni_transformers::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use kjarni_transformers::gpu_ops::blocks::{
    layer_norm::{GpuLayerNorm, GpuLayerNormWeights}, GpuFeedForward, GpuFeedForwardStd, GpuFeedForwardWeights,
    GpuFeedForwardWeightsStd, GpuNormalization,
    GpuNormalizationWeights,
};
use kjarni_transformers::gpu_ops::{GpuTensor, GpuTensorPool};
use kjarni_transformers::traits::{EncoderDecoderArchitecture, TransformerConfig};
use kjarni_transformers::weights::ModelWeights;
use kjarni_transformers::WgpuContext;
use ndarray::Array2;
use std::sync::Arc;
use wgpu::CommandEncoder;

pub struct BartGpuDecoder {
    context: Arc<WgpuContext>,
    config: Arc<BartConfig>,

    // Kernels
    embeddings: GpuEmbeddings,
    embed_layer_norm: GpuNormalization,

    // Weights
    embedding_weights: GpuEmbeddingWeights,
    embed_ln_weights: GpuNormalizationWeights,

    // Layers
    layers: Vec<GpuCrossAttentionDecoderLayer>,
}

impl BartGpuDecoder {
    pub fn debug_embeddings(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        decoder_input_ids: &GpuTensor,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        unimplemented!()
        // self.embeddings.encode(
        //     encoder,
        //     &self.embedding_weights,
        //     decoder_input_ids,
        //     None,
        //     position_offset,
        //     self.config.as_ref(),
        //     pool,
        // )
    }

    pub fn debug_embed_with_ln(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        decoder_input_ids: &GpuTensor,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        let hidden = self.debug_embeddings(encoder, pool, decoder_input_ids, position_offset)?;
        let ln_output = pool.get(hidden.shape().to_vec());
        self.embed_layer_norm
            .encode(encoder, &self.embed_ln_weights, &hidden, &ln_output);
        Ok(ln_output)
    }
    pub fn new(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        config: Arc<BartConfig>,
    ) -> Result<Self> {
        // 1. Embeddings
        // let embedding_weights = GpuEmbeddingWeights::new(context, weights, config.as_ref())?;

        // 1. Embeddings - MANUALLY load with decoder-specific paths
        let word_emb = GpuTensor::from_raw(
            context,
            &weights.get_raw(config.get_shared_embedding_weight_name())?,
            "decoder_word_emb",
        )?;
        let pos_emb = GpuTensor::from_raw(
            context,
            &weights.get_raw("model.decoder.embed_positions.weight")?, // DECODER positions!
            "decoder_pos_emb",
        )?;

        let embedding_weights = GpuEmbeddingWeights {
            word_embeddings: word_emb,
            position_embeddings: Some(pos_emb),
            token_type_embeddings: None,
        };
        // 2. Initial LayerNorm
        let (norm_w_name, norm_b_name) = config.get_decoder_embedding_ln_names();
        let embed_ln_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
            GpuTensor::from_raw(context, &weights.get_raw(norm_w_name)?, norm_w_name)?,
            GpuTensor::from_raw(context, &weights.get_raw(norm_b_name)?, norm_b_name)?,
        )?);

        // 3. Decoder Layers
        let mut layers = Vec::with_capacity(config.decoder_layers);
        for i in 0..config.decoder_layers {
            let sa_names = config.get_decoder_self_attention_names(i);
            let self_attn_weights =
                GpuAttentionWeights::from_config_names(context, weights, &sa_names)?;
            let self_attn_norm =
                GpuNormalization::LayerNorm(GpuLayerNorm::new(context, config.layer_norm_eps()));
            let self_attn_ln_weights =
                GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_raw(
                        context,
                        &weights.get_raw(&sa_names.norm_weight)?,
                        "sa_ln_w",
                    )?,
                    GpuTensor::from_raw(
                        context,
                        &weights.get_raw(&sa_names.norm_bias)?,
                        "sa_ln_b",
                    )?,
                )?);

            let ca_names = config.get_decoder_cross_attention_names(i);
            let cross_attn_weights =
                GpuAttentionWeights::from_config_names(context, weights, &ca_names)?;
            let cross_attn_norm =
                GpuNormalization::LayerNorm(GpuLayerNorm::new(context, config.layer_norm_eps()));
            let cross_attn_ln_weights =
                GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_raw(
                        context,
                        &weights.get_raw(&ca_names.norm_weight)?,
                        "ca_ln_w",
                    )?,
                    GpuTensor::from_raw(
                        context,
                        &weights.get_raw(&ca_names.norm_bias)?,
                        "ca_ln_b",
                    )?,
                )?);

            let ff_names = config.get_decoder_feed_forward_names(i);
            let feedforward =
                GpuFeedForward::Standard(GpuFeedForwardStd::new(context, Activation::Gelu)?);
            let ff_weights = GpuFeedForwardWeights::Standard(
                GpuFeedForwardWeightsStd::from_config_names(context, weights, &ff_names)?,
            );
            let ffn_norm =
                GpuNormalization::LayerNorm(GpuLayerNorm::new(context, config.layer_norm_eps()));
            let ffn_ln_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                GpuTensor::from_raw(
                    context,
                    &weights.get_raw(&ff_names.norm_weight)?,
                    "ffn_ln_w",
                )?,
                GpuTensor::from_raw(context, &weights.get_raw(&ff_names.norm_bias)?, "ffn_ln_b")?,
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
                config.as_ref(),
            )?;
            layers.push(layer);
        }

        // 4. Instantiate Kernels
        let embeddings = GpuEmbeddings::new(context)?;
        let embed_layer_norm =
            GpuNormalization::LayerNorm(GpuLayerNorm::new(context, config.layer_norm_eps()));

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
    /// Records the GPU commands for a single forward pass of the cross-attention decoder.
    fn forward(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        decoder_input: DecoderInput<'_>,
        encoder_hidden_states: &GpuTensor,
        decoder_attention_mask: &GpuTensor,
        cache: Option<&mut dyn Cache>,
        cross_kv_cache: Option<&GpuCrossAttentionKVCache>,
    ) -> Result<GpuCrossDecoderOutput> {
        // --- Get position offset and GPU cache ---
        let position_offset = cache.as_ref().map_or(0, |c| c.get_seq_length());
        let gpu_cache = cache.and_then(|c| c.as_any().downcast_ref::<GpuBeamKVCache>());

        // --- 1. Embeddings + Initial LayerNorm ---
        let embedded_tokens = match decoder_input {
            DecoderInput::TokensCpu(ids) => {
                // This is an inefficient path for GPU, but necessary for correctness.
                // It should ideally be handled at a higher level (e.g., in the backend).
                // For now, we upload the tokens here.
                let ids_gpu = GpuTensor::from_ndarray(
                    &self.context,
                    &Array2::from_shape_vec((1, ids.len()), ids.to_vec())?,
                )?;
                unimplemented!()
                // self.embeddings.encode(
                //     encoder,
                //     &self.embedding_weights,
                //     &ids_gpu, // Pass the GpuTensor
                //     None,
                //     position_offset,
                //     self.config.as_ref(),
                //     pool,
                // )?
            }
            DecoderInput::TokensGpu(ids_gpu) => {
                // This is the efficient path.
                unimplemented!()
                // self.embeddings.encode(
                //     encoder,
                //     &self.embedding_weights,
                //     ids_gpu, // Pass the GpuTensor
                //     None,
                //     position_offset,
                //     self.config.as_ref(),
                //     pool,
                // )?
            }
            DecoderInput::HiddenGpu(hidden) => hidden.clone(),
            _ => {
                return Err(anyhow!(
                    "Unsupported DecoderInput variant for GpuCrossDecoder"
                ));
            }
        };
        let mut hidden_states = embedded_tokens;
        let ln_output = pool.get(hidden_states.shape().to_vec());
        self.embed_layer_norm
            .encode(encoder, &self.embed_ln_weights, &hidden_states, &ln_output);
        hidden_states = ln_output;

        // --- 2. Decoder Layers ---
        let mut new_self_attn_kvs = Vec::with_capacity(self.layers.len());

        for (i, layer) in self.layers.iter().enumerate() {
            // Get the specific cross-attention K/V for this layer from the pre-computed cache
            let cross_kv_for_layer = cross_kv_cache.and_then(|c| c.0.get(i));

            // Get the past self-attention K/V for this layer from the mutable cache
            let self_attn_past_kv = gpu_cache.and_then(|c| c.get_layer_tensors(i));

            let (new_hidden, new_k, new_v) = layer.forward(
                encoder,
                &hidden_states,
                Some(encoder_hidden_states),
                decoder_attention_mask,
                None, // Encoder mask is usually handled by attention kernel
                self_attn_past_kv,
                cross_kv_for_layer,
                position_offset,
                pool,
            )?;

            hidden_states = new_hidden;
            new_self_attn_kvs.push((new_k, new_v));
        }

        // --- 3. Return the complete output struct ---
        Ok(GpuCrossDecoderOutput {
            last_hidden_state: hidden_states,
            new_self_attn_kv: new_self_attn_kvs,
        })
    }
}
