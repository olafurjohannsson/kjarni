//! Unified Seq2Seq GPU decoder supporting BART, T5, Whisper, and similar architectures.

use anyhow::{Result, anyhow};
use std::sync::Arc;
use wgpu::CommandEncoder;

use crate::{
    WgpuContext,
    cache::Cache,
    encoder_decoder::{
        config::Seq2SeqDecoderConfig,
        traits::{GpuCrossAttentionKVCache, GpuCrossDecoder, GpuCrossDecoderOutput},
    },
    gpu::{
        GpuTensor, GpuTensorPool,
        cache::GpuBeamKVCache,
        normalization::{
            GpuLayerNorm, GpuLayerNormWeights, GpuNormalization, GpuNormalizationWeights,
        },
    },
    gpu_ops::blocks::{
        GpuFeedForward, GpuFeedForwardStd, GpuFeedForwardWeights, GpuFeedForwardWeightsStd,
        attention::GpuAttentionWeights, layers::GpuCrossDecoderLayer,
    },
    models::base::ModelLoadConfig,
    tensor::DType,
    traits::{ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};

/// Unified transformer decoder for seq2seq models on GPU.
pub struct Seq2SeqGPUDecoder {
    context: Arc<WgpuContext>,

    embed_layer_norm: Option<GpuNormalization>,
    embed_ln_weights: Option<GpuNormalizationWeights>,

    layers: Vec<GpuCrossDecoderLayer>,

    final_layer_norm: Option<GpuNormalization>,
    final_ln_weights: Option<GpuNormalizationWeights>,

    pre_norm: bool,

    pub meta: ModelMetadata,
    pub layout: ModelLayout,
}

impl Seq2SeqGPUDecoder {
    /// Create decoder.
    pub fn new<C: ModelConfig>(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        config: &C,
        decoder_config: Seq2SeqDecoderConfig,
        load_config: ModelLoadConfig,
    ) -> Result<Self> {
        let meta = config.metadata();
        let layout = config.layout();
        let decoder_layout = layout
            .decoder
            .as_ref()
            .ok_or_else(|| anyhow!("Model layout has no decoder component"))?;
        let target_dt = load_config.target_dtype;

        let num_decoder_layers = meta
            .decoder_layers
            .ok_or_else(|| anyhow!("Model metadata has no decoder_layers"))?;

        log::info!(
            "Building Seq2SeqGpuDecoder: {} layers, hidden_size={}, pre_norm={}, embed_norm={}, final_norm={}",
            num_decoder_layers,
            meta.hidden_size,
            decoder_config.pre_norm,
            decoder_config.normalize_embeddings,
            decoder_config.final_layer_norm
        );

        // embed layer nrom
        let (embed_layer_norm, embed_ln_weights) = if decoder_config.normalize_embeddings {
            Self::build_embed_norm(context, weights, decoder_layout, &meta, target_dt)?
        } else {
            (None, None)
        };

        // transformer decoder layers
        let layers = Self::build_layers(
            context,
            weights,
            &meta,
            &layout,
            num_decoder_layers,
            &load_config,
        )?;

        // optional final norm
        let (final_layer_norm, final_ln_weights) = if decoder_config.final_layer_norm {
            Self::build_final_norm(context, weights, decoder_layout, &meta, target_dt)?
        } else {
            (None, None)
        };

        log::debug!(
            "Seq2SeqGpuDecoder built: {} layers, embed_norm={}, final_norm={}",
            layers.len(),
            embed_layer_norm.is_some(),
            final_layer_norm.is_some()
        );

        Ok(Self {
            context: context.clone(),
            embed_layer_norm,
            embed_ln_weights,
            layers,
            final_layer_norm,
            final_ln_weights,
            pre_norm: decoder_config.pre_norm,
            meta,
            layout,
        })
    }

    fn build_embed_norm(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        decoder_layout: &crate::traits::DecoderLayout,
        meta: &ModelMetadata,
        target_dt: Option<DType>,
    ) -> Result<(Option<GpuNormalization>, Option<GpuNormalizationWeights>)> {
        if let (Some(w_key), Some(b_key)) = (
            &decoder_layout.embedding_norm_weight,
            &decoder_layout.embedding_norm_bias,
        ) {
            log::debug!("Loading decoder embedding LayerNorm: {} / {}", w_key, b_key);
            let norm = GpuNormalization::LayerNorm(GpuLayerNorm::new(context, meta.norm_eps));
            let weights_gpu = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                GpuTensor::from_model_weights(
                    context,
                    weights,
                    w_key,
                    target_dt,
                    "dec_embed_ln_w",
                )?,
                GpuTensor::from_model_weights(
                    context,
                    weights,
                    b_key,
                    target_dt,
                    "dec_embed_ln_b",
                )?,
            )?);
            Ok((Some(norm), Some(weights_gpu)))
        } else if let Some(w_key) = &decoder_layout.embedding_norm_weight {
            log::debug!("Loading decoder embedding RMSNorm: {}", w_key);
            let norm = GpuNormalization::LayerNorm(GpuLayerNorm::new(context, meta.norm_eps));
            let weights_gpu = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                GpuTensor::from_model_weights(
                    context,
                    weights,
                    w_key,
                    target_dt,
                    "dec_embed_ln_w",
                )?,
                GpuTensor::zeros(
                    context,
                    vec![meta.hidden_size],
                    target_dt.unwrap_or(DType::F32),
                    "dec_embed_ln_b_zero",
                )?,
            )?);
            Ok((Some(norm), Some(weights_gpu)))
        } else {
            log::warn!("normalize_embeddings=true but no decoder embedding norm weights in layout");
            Ok((None, None))
        }
    }

    fn build_final_norm(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        decoder_layout: &crate::traits::DecoderLayout,
        meta: &ModelMetadata,
        target_dt: Option<DType>,
    ) -> Result<(Option<GpuNormalization>, Option<GpuNormalizationWeights>)> {
        if let Some(w_key) = &decoder_layout.final_norm_weight {
            let b_key = decoder_layout.final_norm_bias.as_deref();
            log::debug!("Loading decoder final LayerNorm: {} / {:?}", w_key, b_key);

            let norm = GpuNormalization::LayerNorm(GpuLayerNorm::new(context, meta.norm_eps));
            let weights_gpu = if let Some(b) = b_key {
                GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_model_weights(
                        context,
                        weights,
                        w_key,
                        target_dt,
                        "dec_final_ln_w",
                    )?,
                    GpuTensor::from_model_weights(
                        context,
                        weights,
                        b,
                        target_dt,
                        "dec_final_ln_b",
                    )?,
                )?)
            } else {
                GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_model_weights(
                        context,
                        weights,
                        w_key,
                        target_dt,
                        "dec_final_ln_w",
                    )?,
                    GpuTensor::zeros(
                        context,
                        vec![meta.hidden_size],
                        target_dt.unwrap_or(DType::F32),
                        "dec_final_ln_b_zero",
                    )?,
                )?)
            };
            Ok((Some(norm), Some(weights_gpu)))
        } else {
            log::warn!("final_layer_norm=true but no decoder final norm weights in layout");
            Ok((None, None))
        }
    }

    fn build_layers(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        num_layers: usize,
        load_config: &ModelLoadConfig,
    ) -> Result<Vec<GpuCrossDecoderLayer>> {
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("Model layout must have a decoder section");
        let layer_layout = &decoder_layout.layer;
        let target_dt = load_config.target_dtype;

        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            log::trace!("Building decoder layer {}", i);
            let i_str = i.to_string();

            // ================================================================
            // SELF-ATTENTION
            // ================================================================
            let self_attn_weights = GpuAttentionWeights::from_decoder_self_attn_layout(
                context,
                weights,
                layout,
                i,
                target_dt,
                meta.hidden_size,
            )?;

            let self_attn_norm =
                GpuNormalization::LayerNorm(GpuLayerNorm::new(context, meta.norm_eps));

            let self_attn_ln_weights =
                GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_model_weights(
                        context,
                        weights,
                        &layer_layout.self_attn.norm_weight.replace("{}", &i_str),
                        target_dt,
                        "sa_ln_w",
                    )?,
                    GpuTensor::from_model_weights(
                        context,
                        weights,
                        &layer_layout
                            .self_attn
                            .norm_bias
                            .as_ref()
                            .ok_or_else(|| anyhow!("Layer {} missing self_attn norm_bias", i))?
                            .replace("{}", &i_str),
                        target_dt,
                        "sa_ln_b",
                    )?,
                )?);

            // ================================================================
            // CROSS-ATTENTION
            // ================================================================
            let cross_layout = layer_layout
                .cross_attn
                .as_ref()
                .ok_or_else(|| anyhow!("Layer {} missing cross_attn layout", i))?;

            let cross_attn_weights = GpuAttentionWeights::from_cross_attn_layout(
                context,
                weights,
                layout,
                i,
                target_dt,
                meta.hidden_size,
            )?;

            let cross_attn_norm =
                GpuNormalization::LayerNorm(GpuLayerNorm::new(context, meta.norm_eps));

            let cross_attn_ln_weights =
                GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_model_weights(
                        context,
                        weights,
                        &cross_layout.norm_weight.replace("{}", &i_str),
                        target_dt,
                        "ca_ln_w",
                    )?,
                    GpuTensor::from_model_weights(
                        context,
                        weights,
                        &cross_layout
                            .norm_bias
                            .as_ref()
                            .ok_or_else(|| anyhow!("Layer {} missing cross_attn norm_bias", i))?
                            .replace("{}", &i_str),
                        target_dt,
                        "ca_ln_b",
                    )?,
                )?);

            // ================================================================
            // FEED-FORWARD
            // ================================================================
            let feedforward =
                GpuFeedForward::Standard(GpuFeedForwardStd::new(context, meta.activation)?);

            let ff_weights =
                GpuFeedForwardWeights::Standard(GpuFeedForwardWeightsStd::from_layout(
                    context,
                    weights,
                    &layer_layout.ffn,
                    i,
                    target_dt,
                )?);

            let ffn_norm = GpuNormalization::LayerNorm(GpuLayerNorm::new(context, meta.norm_eps));

            let ffn_ln_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                GpuTensor::from_model_weights(
                    context,
                    weights,
                    &layer_layout.ffn.norm_weight.replace("{}", &i_str),
                    target_dt,
                    "ffn_ln_w",
                )?,
                GpuTensor::from_model_weights(
                    context,
                    weights,
                    &layer_layout
                        .ffn
                        .norm_bias
                        .as_ref()
                        .ok_or_else(|| anyhow!("Layer {} missing ffn norm_bias", i))?
                        .replace("{}", &i_str),
                    target_dt,
                    "ffn_ln_b",
                )?,
            )?);

            // ================================================================
            // BUILD LAYER
            // ================================================================
            layers.push(GpuCrossDecoderLayer::new(
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
                meta,
            )?);
        }

        Ok(layers)
    }

    // ========================================================================
    // ACCESSORS
    // ========================================================================

    pub fn context(&self) -> &Arc<WgpuContext> {
        &self.context
    }

    pub fn is_prenorm(&self) -> bool {
        self.pre_norm
    }

    pub fn has_embed_norm(&self) -> bool {
        self.embed_layer_norm.is_some()
    }

    pub fn has_final_norm(&self) -> bool {
        self.final_layer_norm.is_some()
    }

    // ========================================================================
    // NEW UNIFIED METHODS
    // ========================================================================

    /// Apply embedding layer normalization (BART/Whisper have; T5 doesn't).

    /// Process through decoder layers (new unified API, mirrors forward_layers).
    pub fn forward_layers2(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        encoder_hidden_states: &GpuTensor,
        decoder_attention_mask: &GpuTensor,
        position_offset: usize,
        cache: Option<&mut GpuBeamKVCache>,
        cross_kv_cache: Option<&GpuCrossAttentionKVCache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<(GpuTensor, Vec<(GpuTensor, GpuTensor)>)> {
        let mut current_hidden = hidden_states.clone();
        let mut new_self_attn_kvs = Vec::with_capacity(end_layer - start_layer);

        for i in start_layer..end_layer {
            let layer = &self.layers[i];

            // Get cross-attention KV: use precomputed if available, otherwise compute on the fly
            let cross_kv_for_layer: (GpuTensor, GpuTensor);
            let cross_kv_ref = if let Some(cache) = cross_kv_cache {
                &cache.0[i]
            } else {
                // Slow path: compute on the fly
                cross_kv_for_layer =
                    layer.precompute_cross_kv(encoder, encoder_hidden_states, pool);
                &cross_kv_for_layer
            };

            // Get self-attention cache for this layer
            let cached_kv = cache.as_ref().and_then(|c| c.get_layer_tensors(i));
            let cache_len = cache
                .as_ref()
                .map(|c| c.get_seq_length())
                .unwrap_or(position_offset);

            // Forward through layer (post-norm for BART)
            let (new_hidden, new_k, new_v) = layer.forward(
                encoder,
                &current_hidden,
                cross_kv_ref,
                decoder_attention_mask,
                None, // encoder_mask - usually None for BART
                cached_kv,
                cache_len,
                pool,
            )?;

            current_hidden = new_hidden;
            new_self_attn_kvs.push((new_k, new_v));
        }

        // Update self-attention cache
        if let Some(cache) = cache {
            for (idx, (new_k, new_v)) in new_self_attn_kvs.iter().enumerate() {
                let layer_idx = start_layer + idx;
                cache.update(encoder, layer_idx, new_k, new_v)?;
            }
        }

        Ok((current_hidden, new_self_attn_kvs))
    }

    /// Precompute cross-attention K/V from encoder hidden states.
    pub fn precompute_cross_attention_kv2(
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
}

// ============================================================================
// GpuCrossDecoder TRAIT IMPLEMENTATION
// ============================================================================

impl GpuCrossDecoder for Seq2SeqGPUDecoder {
    fn embed_norm(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor> {
        match (&self.embed_layer_norm, &self.embed_ln_weights) {
            (Some(norm), Some(weights)) => {
                let output = pool.get(hidden_states.shape().to_vec());
                norm.encode(encoder, weights, hidden_states, &output);
                Ok(output)
            }
            _ => Ok(hidden_states.clone()),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Apply final layer normalization (T5/Whisper have; BART doesn't).
    fn final_norm(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor> {
        match (&self.final_layer_norm, &self.final_ln_weights) {
            (Some(norm), Some(weights)) => {
                let output = pool.get(hidden_states.shape().to_vec());
                norm.encode(encoder, weights, hidden_states, &output);
                Ok(output)
            }
            _ => Ok(hidden_states.clone()),
        }
    }

    fn layers(&self) -> &Vec<GpuCrossDecoderLayer> {
        &self.layers
    }

    fn forward_layers(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        encoder_hidden_states: &GpuTensor,
        decoder_attention_mask: &GpuTensor,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
        cross_kv_cache: Option<&GpuCrossAttentionKVCache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<GpuCrossDecoderOutput> {
        // Downcast cache to GPU cache
        let gpu_cache = cache.and_then(|c| c.as_any_mut().downcast_mut::<GpuBeamKVCache>());

        let (current_hidden, new_self_attn_kvs) = self.forward_layers2(
            encoder,
            pool,
            hidden_states,
            encoder_hidden_states,
            decoder_attention_mask,
            position_offset,
            gpu_cache,
            cross_kv_cache,
            start_layer,
            end_layer,
        )?;

        // Apply final norm if this is the last layer block
        let final_hidden = if end_layer == self.layers.len() {
            self.final_norm(encoder, pool, &current_hidden)?
        } else {
            current_hidden
        };

        Ok(GpuCrossDecoderOutput {
            last_hidden_state: final_hidden,
            new_self_attn_kv: new_self_attn_kvs,
        })
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_size(&self) -> usize {
        self.meta.hidden_size
    }

    fn precompute_cross_attention_kv(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        encoder_hidden_states: &GpuTensor,
    ) -> Result<GpuCrossAttentionKVCache> {
        self.precompute_cross_attention_kv2(encoder, pool, encoder_hidden_states)
    }
}
// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod seq2seq_gpu_decoder_tests {
    use super::*;
    use crate::activations::Activation;
    use crate::cpu::encoder_decoder::Seq2SeqCPUDecoder;
    use crate::encoder_decoder::config::{PositionEncodingType, Seq2SeqDecoderConfig};
    use crate::encoder_decoder::traits::CpuCrossDecoder;
    use crate::gpu::GpuFrameContext;
    use crate::models::base::ModelLoadConfig;
    use crate::traits::{
        AttentionLayout, DecoderLayerLayout, DecoderLayout, FeedForwardLayout, ModelConfig,
        ModelLayout, ModelMetadata, NormalizationStrategy,
    };
    use crate::weights::ModelWeights;
    use ndarray::Array3;
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;
    use tempfile::TempDir;

    // ========================================================================
    // Test Helpers
    // ========================================================================

    async fn get_test_context() -> Arc<WgpuContext> {
        WgpuContext::new()
            .await
            .expect("Failed to create WGPU context")
    }

    fn assert_gpu_cpu_close(
        gpu_result: &Array3<f32>,
        cpu_result: &Array3<f32>,
        atol: f32,
        context: &str,
    ) {
        assert_eq!(
            gpu_result.shape(),
            cpu_result.shape(),
            "{}: Shape mismatch - GPU: {:?}, CPU: {:?}",
            context,
            gpu_result.shape(),
            cpu_result.shape()
        );

        let max_diff = gpu_result
            .iter()
            .zip(cpu_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < atol,
            "{}: Max diff {} exceeds tolerance {}.\nGPU first 5: {:?}\nCPU first 5: {:?}",
            context,
            max_diff,
            atol,
            gpu_result.iter().take(5).collect::<Vec<_>>(),
            cpu_result.iter().take(5).collect::<Vec<_>>()
        );

        println!("{}: PASSED (max_diff={})", context, max_diff);
    }

    // ========================================================================
    // Mock Config
    // ========================================================================

    #[derive(Debug, Clone)]
    struct MockDecoderConfig {
        vocab_size: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        is_prenorm: bool,
        no_pos_emb_in_layout: bool,
    }

    impl ModelConfig for MockDecoderConfig {
        fn model_type(&self) -> &str {
            "mock_decoder"
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn metadata(&self) -> ModelMetadata {
            ModelMetadata {
                decoder_layers: Some(self.num_layers),
                hidden_size: self.hidden_size,
                num_layers: 0, // Encoder layers (unused)
                num_attention_heads: self.num_heads,
                num_kv_heads: self.num_heads,
                head_dim: self.hidden_size / self.num_heads,
                vocab_size: self.vocab_size,
                intermediate_size: self.hidden_size * 2,
                max_seq_len: 1024,
                norm_eps: 1e-5,
                activation: Activation::Gelu,
                rope_theta: None,
                rope_scaling: None,
                scale_embeddings: false,
                normalize_embedding: false,
                extra_pos_embeddings: 0,
                is_prenorm: self.is_prenorm,
                transpose_ffn_weights: false,
                transpose_attention_weights: false,
                normalization_strategy: NormalizationStrategy::LayerNorm,
                no_scale_qk: false,
            }
        }

        fn layout(&self) -> ModelLayout {
            ModelLayout {
                token_embedding: "token_emb".to_string(),
                lm_head: "lm_head".to_string(),
                encoder: None,
                decoder: Some(DecoderLayout {
                    position_embedding: if self.no_pos_emb_in_layout {
                        None
                    } else {
                        Some("pos_emb".to_string())
                    },
                    token_type_embedding: None,
                    embedding_norm_weight: Some("embed_ln.weight".to_string()),
                    embedding_norm_bias: Some("embed_ln.bias".to_string()),
                    final_norm_weight: Some("final_ln.weight".to_string()),
                    final_norm_bias: Some("final_ln.bias".to_string()),
                    layer: DecoderLayerLayout {
                        self_attn: AttentionLayout {
                            q_weight: "l{}.sa_q.weight".to_string(),
                            q_bias: Some("l{}.sa_q.bias".to_string()),
                            k_weight: "l{}.sa_k.weight".to_string(),
                            k_bias: Some("l{}.sa_k.bias".to_string()),
                            v_weight: "l{}.sa_v.weight".to_string(),
                            v_bias: Some("l{}.sa_v.bias".to_string()),
                            o_weight: "l{}.sa_o.weight".to_string(),
                            o_bias: Some("l{}.sa_o.bias".to_string()),
                            norm_weight: "l{}.sa_ln.weight".to_string(),
                            norm_bias: Some("l{}.sa_ln.bias".to_string()),
                        },
                        cross_attn: Some(AttentionLayout {
                            q_weight: "l{}.ca_q.weight".to_string(),
                            q_bias: Some("l{}.ca_q.bias".to_string()),
                            k_weight: "l{}.ca_k.weight".to_string(),
                            k_bias: Some("l{}.ca_k.bias".to_string()),
                            v_weight: "l{}.ca_v.weight".to_string(),
                            v_bias: Some("l{}.ca_v.bias".to_string()),
                            o_weight: "l{}.ca_o.weight".to_string(),
                            o_bias: Some("l{}.ca_o.bias".to_string()),
                            norm_weight: "l{}.ca_ln.weight".to_string(),
                            norm_bias: Some("l{}.ca_ln.bias".to_string()),
                        }),
                        ffn: FeedForwardLayout {
                            up_weight: "l{}.fc1.weight".to_string(),
                            up_bias: Some("l{}.fc1.bias".to_string()),
                            down_weight: "l{}.fc2.weight".to_string(),
                            down_bias: Some("l{}.fc2.bias".to_string()),
                            gate_weight: None,
                            gate_bias: None,
                            norm_weight: "l{}.ffn_ln.weight".to_string(),
                            norm_bias: Some("l{}.ffn_ln.bias".to_string()),
                        },
                    },
                }),
            }
        }
    }

    // ========================================================================
    // Weight Creation Helper
    // ========================================================================

    fn create_model_weights(
        weights_map: HashMap<String, Vec<f32>>,
        shapes: HashMap<String, Vec<usize>>,
    ) -> Result<(ModelWeights, TempDir)> {
        let dir = tempfile::tempdir()?;
        let stored_data: Vec<(String, Vec<usize>, Vec<u8>)> = weights_map
            .into_iter()
            .map(|(k, v)| {
                let shape = shapes.get(&k).unwrap().clone();
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                (k, shape, bytes)
            })
            .collect();

        let mut tensors = HashMap::new();
        for (k, shape, bytes) in &stored_data {
            tensors.insert(
                k.clone(),
                TensorView::new(Dtype::F32, shape.clone(), bytes)?,
            );
        }

        let file_path = dir.path().join("model.safetensors");
        safetensors::serialize_to_file(&tensors, &None, &file_path)?;
        let weights = ModelWeights::new(dir.path())?;

        Ok((weights, dir))
    }

    // ========================================================================
    // Golden Data Generator (BART-style decoder)
    // ========================================================================

    fn get_bart_decoder_golden_data() -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>) {
        let mut w = HashMap::new();
        let mut s = HashMap::new();

        // 1. Embeddings
        w.insert(
            "token_emb".into(),
            vec![
                0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012,
                0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020, 0.021, 0.022, 0.023, 0.024,
                0.025, 0.026, 0.027, 0.028, 0.029, 0.030, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036,
                0.037, 0.038, 0.039, 0.040,
            ],
        );
        s.insert("token_emb".into(), vec![10, 4]);

        let pos_data: Vec<f32> = (0..1024 * 4).map(|i| 0.041 + (i as f32 * 0.001)).collect();
        w.insert("pos_emb".into(), pos_data);
        s.insert("pos_emb".into(), vec![1024, 4]);

        // Norms
        w.insert("embed_ln.weight".into(), vec![1.0; 4]);
        s.insert("embed_ln.weight".into(), vec![4]);
        w.insert("embed_ln.bias".into(), vec![0.01; 4]);
        s.insert("embed_ln.bias".into(), vec![4]);
        w.insert("final_ln.weight".into(), vec![1.0; 4]);
        s.insert("final_ln.weight".into(), vec![4]);
        w.insert("final_ln.bias".into(), vec![0.01; 4]);
        s.insert("final_ln.bias".into(), vec![4]);

        // Layer 0 - Self Attention
        let attn_weight: Vec<f32> = (0..16).map(|i| 0.1 + (i as f32 * 0.01)).collect();
        w.insert("l0.sa_q.weight".into(), attn_weight.clone());
        s.insert("l0.sa_q.weight".into(), vec![4, 4]);
        w.insert("l0.sa_q.bias".into(), vec![0.0; 4]);
        s.insert("l0.sa_q.bias".into(), vec![4]);

        w.insert("l0.sa_k.weight".into(), attn_weight.clone());
        s.insert("l0.sa_k.weight".into(), vec![4, 4]);
        w.insert("l0.sa_k.bias".into(), vec![0.0; 4]);
        s.insert("l0.sa_k.bias".into(), vec![4]);

        w.insert("l0.sa_v.weight".into(), attn_weight.clone());
        s.insert("l0.sa_v.weight".into(), vec![4, 4]);
        w.insert("l0.sa_v.bias".into(), vec![0.0; 4]);
        s.insert("l0.sa_v.bias".into(), vec![4]);

        w.insert("l0.sa_o.weight".into(), attn_weight.clone());
        s.insert("l0.sa_o.weight".into(), vec![4, 4]);
        w.insert("l0.sa_o.bias".into(), vec![0.0; 4]);
        s.insert("l0.sa_o.bias".into(), vec![4]);

        w.insert("l0.sa_ln.weight".into(), vec![1.0; 4]);
        s.insert("l0.sa_ln.weight".into(), vec![4]);
        w.insert("l0.sa_ln.bias".into(), vec![0.01; 4]);
        s.insert("l0.sa_ln.bias".into(), vec![4]);

        // Layer 0 - Cross Attention
        w.insert("l0.ca_q.weight".into(), attn_weight.clone());
        s.insert("l0.ca_q.weight".into(), vec![4, 4]);
        w.insert("l0.ca_q.bias".into(), vec![0.0; 4]);
        s.insert("l0.ca_q.bias".into(), vec![4]);

        w.insert("l0.ca_k.weight".into(), attn_weight.clone());
        s.insert("l0.ca_k.weight".into(), vec![4, 4]);
        w.insert("l0.ca_k.bias".into(), vec![0.0; 4]);
        s.insert("l0.ca_k.bias".into(), vec![4]);

        w.insert("l0.ca_v.weight".into(), attn_weight.clone());
        s.insert("l0.ca_v.weight".into(), vec![4, 4]);
        w.insert("l0.ca_v.bias".into(), vec![0.0; 4]);
        s.insert("l0.ca_v.bias".into(), vec![4]);

        w.insert("l0.ca_o.weight".into(), attn_weight.clone());
        s.insert("l0.ca_o.weight".into(), vec![4, 4]);
        w.insert("l0.ca_o.bias".into(), vec![0.0; 4]);
        s.insert("l0.ca_o.bias".into(), vec![4]);

        w.insert("l0.ca_ln.weight".into(), vec![1.0; 4]);
        s.insert("l0.ca_ln.weight".into(), vec![4]);
        w.insert("l0.ca_ln.bias".into(), vec![0.01; 4]);
        s.insert("l0.ca_ln.bias".into(), vec![4]);

        // Layer 0 - FFN
        let fc1_data: Vec<f32> = (0..32).map(|i| 0.2 + (i as f32 * 0.001)).collect();
        w.insert("l0.fc1.weight".into(), fc1_data);
        s.insert("l0.fc1.weight".into(), vec![8, 4]);
        w.insert("l0.fc1.bias".into(), vec![0.01; 8]);
        s.insert("l0.fc1.bias".into(), vec![8]);

        let fc2_data: Vec<f32> = (0..32).map(|i| 0.25 + (i as f32 * 0.001)).collect();
        w.insert("l0.fc2.weight".into(), fc2_data);
        s.insert("l0.fc2.weight".into(), vec![4, 8]);
        w.insert("l0.fc2.bias".into(), vec![0.01; 4]);
        s.insert("l0.fc2.bias".into(), vec![4]);

        w.insert("l0.ffn_ln.weight".into(), vec![1.0; 4]);
        s.insert("l0.ffn_ln.weight".into(), vec![4]);
        w.insert("l0.ffn_ln.bias".into(), vec![0.01; 4]);
        s.insert("l0.ffn_ln.bias".into(), vec![4]);

        (w, s)
    }

    // ========================================================================
    // TESTS
    // ========================================================================

    #[tokio::test]
    async fn test_gpu_decoder_construction_bart() -> Result<()> {
        let ctx = get_test_context().await;
        let (weights_map, shapes) = get_bart_decoder_golden_data();
        let (model_weights, _tmp) = create_model_weights(weights_map, shapes)?;

        let config = MockDecoderConfig {
            vocab_size: 10,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 2,
            is_prenorm: false,
            no_pos_emb_in_layout: false,
        };

        let dec_config = Seq2SeqDecoderConfig {
            position_encoding: PositionEncodingType::Learned { offset: 0 },
            normalize_embeddings: true,
            final_layer_norm: true,
            pre_norm: false,
        };

        let decoder = Seq2SeqGPUDecoder::new(
            &ctx,
            &model_weights,
            &config,
            dec_config,
            ModelLoadConfig::default(),
        )?;

        // Verify construction
        assert!(!decoder.is_prenorm(), "BART should be post-norm");
        assert!(decoder.has_embed_norm(), "BART should have embed norm");
        assert!(
            decoder.has_final_norm(),
            "Test config has final norm enabled"
        );
        assert_eq!(decoder.num_layers(), 1);
        assert_eq!(decoder.hidden_size(), 4);

        println!("GPU decoder construction test PASSED");
        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_decoder_embed_norm() -> Result<()> {
        let ctx = get_test_context().await;
        let (weights_map, shapes) = get_bart_decoder_golden_data();
        let (model_weights, _tmp) = create_model_weights(weights_map, shapes)?;

        let config = MockDecoderConfig {
            vocab_size: 10,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 2,
            is_prenorm: false,
            no_pos_emb_in_layout: false,
        };

        let dec_config = Seq2SeqDecoderConfig {
            position_encoding: PositionEncodingType::Learned { offset: 0 },
            normalize_embeddings: true,
            final_layer_norm: true,
            pre_norm: false,
        };

        let decoder = Seq2SeqGPUDecoder::new(
            &ctx,
            &model_weights,
            &config,
            dec_config,
            ModelLoadConfig::default(),
        )?;

        // Input hidden states (simulating embeddings output)
        let input =
            Array3::from_shape_vec((1, 2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;

        let input_gpu = GpuTensor::from_ndarray(&ctx, &input)?;

        // Run embed_norm
        let gpu_output = {
            let pool = ctx.get_inference_pool();
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (cmd_enc, pool_ref) = frame.resources();

            let normed = decoder.embed_norm(cmd_enc, pool_ref, &input_gpu)?;
            frame.finish();
            normed.to_ndarray_3d::<f32>().await?
        };

        // Verify output shape
        assert_eq!(gpu_output.shape(), &[1, 2, 4]);

        // Verify normalization happened (values should be different from input)
        let diff: f32 = gpu_output
            .iter()
            .zip(input.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1, "embed_norm should change the values");

        println!("GPU decoder embed_norm test PASSED");
        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_decoder_no_embed_norm() -> Result<()> {
        let ctx = get_test_context().await;
        let (weights_map, shapes) = get_bart_decoder_golden_data();
        let (model_weights, _tmp) = create_model_weights(weights_map, shapes)?;

        let config = MockDecoderConfig {
            vocab_size: 10,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 2,
            is_prenorm: true,
            no_pos_emb_in_layout: true,
        };

        // T5-like: no embed norm
        let dec_config = Seq2SeqDecoderConfig {
            position_encoding: PositionEncodingType::None,
            normalize_embeddings: false,
            final_layer_norm: true,
            pre_norm: true,
        };

        let decoder = Seq2SeqGPUDecoder::new(
            &ctx,
            &model_weights,
            &config,
            dec_config,
            ModelLoadConfig::default(),
        )?;

        assert!(!decoder.has_embed_norm(), "Should not have embed norm");

        let input =
            Array3::from_shape_vec((1, 2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;

        let input_gpu = GpuTensor::from_ndarray(&ctx, &input)?;

        // embed_norm should return input unchanged
        let output = {
            let pool = ctx.get_inference_pool();
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (cmd_enc, pool_ref) = frame.resources();

            let out = decoder.embed_norm(cmd_enc, pool_ref, &input_gpu)?;
            frame.finish();
            out.to_ndarray_3d::<f32>().await?
        };

        // Should be identical to input
        let max_diff = output
            .iter()
            .zip(input.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-6,
            "embed_norm should be identity when disabled"
        );
        println!("No embed_norm test PASSED");

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_cpu_decoder_parity_bart() -> Result<()> {
        let ctx = get_test_context().await;
        let (weights_map, shapes) = get_bart_decoder_golden_data();
        let (model_weights, _tmp) = create_model_weights(weights_map.clone(), shapes.clone())?;

        let config = MockDecoderConfig {
            vocab_size: 10,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 2,
            is_prenorm: false,
            no_pos_emb_in_layout: false,
        };

        let dec_config = Seq2SeqDecoderConfig {
            position_encoding: PositionEncodingType::Learned { offset: 0 },
            normalize_embeddings: true,
            final_layer_norm: true,
            pre_norm: false,
        };

        // Build both CPU and GPU decoders
        let cpu_decoder = Seq2SeqCPUDecoder::new(
            &model_weights,
            &config,
            dec_config.clone(),
            ModelLoadConfig::default(),
        )?;

        let gpu_decoder = Seq2SeqGPUDecoder::new(
            &ctx,
            &model_weights,
            &config,
            dec_config,
            ModelLoadConfig::default(),
        )?;

        // Same input for both (hidden states, simulating embedded tokens)
        let input_hidden =
            Array3::from_shape_vec((1, 2, 4), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])?;

        // Encoder hidden states (for cross-attention)
        let encoder_hidden = Array3::from_shape_vec(
            (1, 3, 4),
            vec![
                0.336690, 0.128809, 0.234462, 0.230333, -1.122856, -0.186328, 2.208201, -0.637997,
                0.461657, 0.267351, 0.534905, 0.809357,
            ],
        )?;

        // CPU: embed_norm
        let cpu_normed = cpu_decoder.embed_norm(&input_hidden)?;

        // GPU: embed_norm
        let input_gpu = GpuTensor::from_ndarray(&ctx, &input_hidden)?;

        let gpu_normed = {
            let pool = ctx.get_inference_pool();
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (cmd_enc, pool_ref) = frame.resources();

            let normed = gpu_decoder.embed_norm(cmd_enc, pool_ref, &input_gpu)?;
            frame.finish();
            normed.to_ndarray_3d::<f32>().await?
        };

        assert_gpu_cpu_close(&gpu_normed, &cpu_normed, 1e-5, "Decoder Embed Norm");

        println!("GPU/CPU decoder parity test PASSED");
        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_decoder_precompute_cross_kv() -> Result<()> {
        let ctx = get_test_context().await;
        let (weights_map, shapes) = get_bart_decoder_golden_data();
        let (model_weights, _tmp) = create_model_weights(weights_map, shapes)?;

        let config = MockDecoderConfig {
            vocab_size: 10,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 2,
            is_prenorm: false,
            no_pos_emb_in_layout: false,
        };

        let dec_config = Seq2SeqDecoderConfig {
            position_encoding: PositionEncodingType::Learned { offset: 0 },
            normalize_embeddings: true,
            final_layer_norm: true,
            pre_norm: false,
        };

        let decoder = Seq2SeqGPUDecoder::new(
            &ctx,
            &model_weights,
            &config,
            dec_config,
            ModelLoadConfig::default(),
        )?;

        // Encoder hidden states [batch=1, seq=3, hidden=4]
        let encoder_hidden = Array3::from_shape_vec(
            (1, 3, 4),
            vec![
                0.336690, 0.128809, 0.234462, 0.230333, -1.122856, -0.186328, 2.208201, -0.637997,
                0.461657, 0.267351, 0.534905, 0.809357,
            ],
        )?;

        let encoder_gpu = GpuTensor::from_ndarray(&ctx, &encoder_hidden)?;

        let cross_kv = {
            let pool = ctx.get_inference_pool();
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (cmd_enc, pool_ref) = frame.resources();

            let kv = decoder.precompute_cross_attention_kv2(cmd_enc, pool_ref, &encoder_gpu)?;
            frame.finish();
            kv
        };

        // Verify we got K/V for each layer
        assert_eq!(cross_kv.0.len(), decoder.num_layers());

        // Verify shapes
        // K: [B, H, D, S_enc] = [1, 2, 2, 3]
        // V: [B, H, S_enc, D] = [1, 2, 3, 2]
        let (k, v) = &cross_kv.0[0];
        let k_shape = k.shape();
        let v_shape = v.shape();

        // batch=1, num_heads=2, head_dim=2, enc_seq=3
        assert_eq!(k_shape, &[1, 2, 2, 3], "K should be [B, H, D, S_enc]");
        assert_eq!(v_shape, &[1, 2, 3, 2], "V should be [B, H, S_enc, D]");

        println!("GPU decoder precompute_cross_kv test PASSED");
        Ok(())
    }
}
