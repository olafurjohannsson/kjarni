use crate::models::bart::config::BartConfig;
// IMPORTANT: You will need to create this generic layer component.
// Its structure is implied by the usage here.
use anyhow::anyhow;
use anyhow::Result;
use async_trait::async_trait;
use edgetransformers::activations::Activation;
use edgetransformers::cache::Cache;
use edgetransformers::cache::GpuBeamKVCache;
use edgetransformers::encoder_decoder::traits::GpuCrossAttentionDecoder;
use edgetransformers::gpu_context::WgpuContext;
use edgetransformers::gpu_ops::blocks::attention::GpuAttentionWeights;
use edgetransformers::gpu_ops::blocks::decoder_cross_attention::GpuCrossAttentionDecoderLayer;
use edgetransformers::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use edgetransformers::gpu_ops::blocks::{
    layer_norm::{GpuLayerNorm, GpuLayerNormWeights}, GpuFeedForward, GpuFeedForwardStd, GpuFeedForwardWeights,
    GpuFeedForwardWeightsStd, GpuNormalization,
    GpuNormalizationWeights,
};
use edgetransformers::gpu_ops::{GpuTensor, GpuTensorPool};
use edgetransformers::traits::{EncoderDecoderArchitecture, TransformerConfig};
use edgetransformers::weights::ModelWeights;
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
        self.embeddings.encode(
            encoder,
            &self.embedding_weights,
            decoder_input_ids,
            None,
            position_offset,
            self.config.as_ref(),
            pool,
        )
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
        self.embed_layer_norm.encode(encoder, &self.embed_ln_weights, &hidden, &ln_output);
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
            &weights.get_raw("model.shared.weight")?,
            "decoder_word_emb",
        )?;
        let pos_emb = GpuTensor::from_raw(
            context,
            &weights.get_raw("model.decoder.embed_positions.weight")?,  // DECODER positions!
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
impl GpuCrossAttentionDecoder for BartGpuDecoder {
    fn precompute_cross_attention_kv(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        encoder_hidden_states: &GpuTensor,
    ) -> Result<Vec<(GpuTensor, GpuTensor)>> {
        let mut cache = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            cache.push(layer.precompute_cross_kv(encoder, encoder_hidden_states, pool));
        }
        Ok(cache)
    }

    fn forward(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        decoder_input_ids: &GpuTensor,
        encoder_hidden_states: Option<&GpuTensor>,
        cross_attention_kv_cache: Option<&Vec<(GpuTensor, GpuTensor)>>,
        decoder_attention_mask: &GpuTensor,
        cache: &mut dyn Cache,
    ) -> Result<GpuTensor> {
        let gpu_cache = cache
            .as_any_mut()
            .downcast_mut::<GpuBeamKVCache>()
            .ok_or_else(|| anyhow!("BartGpuDecoder requires a GpuBeamKVCache"))?;

        let position_offset = gpu_cache.get_seq_length();

        // 1. Embeddings (BART-specific offset of 2)
        let mut hidden_states = self.embeddings.encode(
            encoder,
            &self.embedding_weights,
            decoder_input_ids,
            None,
            position_offset,
            self.config.as_ref(),
            pool,
        )?;

        // 2. Initial LayerNorm
        let ln_output = pool.get(hidden_states.shape().to_vec());
        self.embed_layer_norm
            .encode(encoder, &self.embed_ln_weights, &hidden_states, &ln_output);
        hidden_states = ln_output;

        // 3. Decoder Layers
        for (i, layer) in self.layers.iter().enumerate() {
            let cross_kv = cross_attention_kv_cache.map(|c| &c[i]);
            let self_attn_past_kv = gpu_cache.get_layer_tensors(i);

            // CORRECTED: Calling layer.forward with the correct arguments in the correct order.
            let (new_hidden, new_k, new_v) = layer.forward(
                encoder,
                &hidden_states,         // decoder_hidden_states
                encoder_hidden_states,  // encoder_hidden_states (optional)
                decoder_attention_mask, // decoder_attn_mask
                None,                   // encoder_attn_mask
                self_attn_past_kv,      // cached_kv (for self-attention)
                cross_kv,               // precomputed_cross_kv
                position_offset,        // cache_len
                pool,                   // pool
            )?;

            hidden_states = new_hidden;

            // CRUCIAL: Update the self-attention cache for the next iteration.
            gpu_cache.update(encoder, i, &new_k, &new_v)?;
        }

        Ok(hidden_states)
    }
}
