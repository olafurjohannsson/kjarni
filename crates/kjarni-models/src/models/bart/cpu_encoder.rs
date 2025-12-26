use crate::models::bart::config::BartConfig;
use anyhow::Result;
use async_trait::async_trait;
use kjarni_transformers::{
    WgpuContext,
    activations::Activation,
    embeddings::Embeddings,
    encoder::{encoder_layer::EncoderLayer, prelude::*},
    encoder_decoder::traits::CpuCrossDecoder,
    feedforward::{FeedForward, StdFeedForward},
    linear_layer::LinearLayer,
    models::base::ModelLoadConfig,
    normalization::LayerNorm,
    traits::{Device, InferenceModel, ModelConfig, ModelLayout, ModelMetadata},
    utils::linear_algebra::{apply_attention_mask, matmul_4d},
    weights::ModelWeights,
};
use ndarray::{Array2, Array3};
use std::sync::Arc;

pub struct BartCpuEncoder {
    embeddings: Embeddings,
    embed_layer_norm: LayerNorm,
    pub layers: Vec<EncoderLayer>,
    config: Arc<BartConfig>,
    pub meta: ModelMetadata,
    pub layout: ModelLayout,
}

impl BartCpuEncoder {
    pub fn new(
        weights: &ModelWeights,
        config: Arc<BartConfig>,
        load_config: ModelLoadConfig,
    ) -> Result<Self> {
        let meta = config.metadata();
        let layout = config.layout();

        // 1. Embeddings

        let word_embeddings = weights.get_array2(&layout.token_embedding)?;
        let embed = kjarni_transformers::embeddings::EmbeddingData::F32(word_embeddings);
        let embeddings = Embeddings::new(
            embed,
            Some(weights.get_array2("model.encoder.embed_positions.weight")?),
            None,
        );

        // 2. LayerNorm
        let embed_layer_norm = LayerNorm::new(
            weights.get_array1("model.encoder.layernorm_embedding.weight")?,
            weights.get_array1("model.encoder.layernorm_embedding.bias")?,
            config.layer_norm_eps,
        );

        // 3. Layers
        let mut layers = Vec::with_capacity(config.encoder_layers);
        for i in 0..config.encoder_layers {
            let prefix = format!("model.encoder.layers.{}", i);

            // 1. Load LinearLayers externally
            let q_proj = LinearLayer::from_weights(
                weights,
                &format!("{}.self_attn.q_proj.weight", prefix),
                Some(&format!("{}.self_attn.q_proj.bias", prefix)),
                None,
                None,
            )?;
            let k_proj = LinearLayer::from_weights(
                weights,
                &format!("{}.self_attn.k_proj.weight", prefix),
                Some(&format!("{}.self_attn.k_proj.bias", prefix)),
                None,
                None,
            )?;
            let v_proj = LinearLayer::from_weights(
                weights,
                &format!("{}.self_attn.v_proj.weight", prefix),
                Some(&format!("{}.self_attn.v_proj.bias", prefix)),
                None,
                None,
            )?;
            let out_proj = LinearLayer::from_weights(
                weights,
                &format!("{}.self_attn.out_proj.weight", prefix),
                Some(&format!("{}.self_attn.out_proj.bias", prefix)),
                None,
                None,
            )?;

            assert!(q_proj.has_bias());
            assert!(k_proj.has_bias());
            assert!(v_proj.has_bias());
            assert!(out_proj.has_bias());

            // 2. Pass them to the constructor
            let self_attn = EncoderSelfAttention::new(
                config.d_model,
                config.encoder_attention_heads,
                q_proj,
                k_proj,
                v_proj,
                out_proj,
            );

            let self_attn_layer_norm = LayerNorm::new(
                weights.get_array1(&format!("{}.self_attn_layer_norm.weight", prefix))?,
                weights.get_array1(&format!("{}.self_attn_layer_norm.bias", prefix))?,
                config.layer_norm_eps,
            );

            let fc1 = weights.get_array2(&format!("{}.fc1.weight", prefix))?;
            let fc2 = weights.get_array2(&format!("{}.fc2.weight", prefix))?;

            let feedforward = FeedForward::Standard(StdFeedForward::new(
                fc1, // Keep as [Out, In] - no transpose
                weights.get_array1(&format!("{}.fc1.bias", prefix))?,
                fc2, // Keep as [Out, In] - no transpose
                weights.get_array1(&format!("{}.fc2.bias", prefix))?,
                Activation::Gelu,
            ));

            let ffn_layer_norm = LayerNorm::new(
                weights.get_array1(&format!("{}.final_layer_norm.weight", prefix))?,
                weights.get_array1(&format!("{}.final_layer_norm.bias", prefix))?,
                config.layer_norm_eps,
            );

            layers.push(EncoderLayer {
                self_attn,
                self_attn_layer_norm,
                feedforward,
                ffn_layer_norm,
            });
        }

        Ok(Self {
            embeddings,
            embed_layer_norm,
            layers,
            config,
            meta,
            layout,
        })
    }
}

impl InferenceModel for BartCpuEncoder {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        None
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl CpuEncoder for BartCpuEncoder {
    // type Input = Array2<u32>;
    // type Output = EncoderOutput;

    /// Compute embeddings only (word + position + token_type)
    fn embed(&self, input_ids: &Array2<u32>, token_type_ids: Option<&Array2<u32>>) -> Array3<f32> {
        self.embeddings.forward(input_ids, token_type_ids, 2, false)
    }

    /// Compute embeddings + initial normalization
    fn embed_and_normalize(
        &self,
        input_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
    ) -> Array3<f32> {
        let hidden = self.embed(input_ids, token_type_ids);
        if self.config.normalize_embedding {
            self.embed_layer_norm.forward_3d(&hidden)
        }
        else {
            hidden
        }
    }

    /// Run layers [start_layer, end_layer) on pre-computed hidden states
    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>> {
        let mut hidden = hidden_states.clone();

        for layer in self.layers.iter().take(end_layer).skip(start_layer) {
            hidden = layer.forward(hidden, attention_mask, None, false)?;
        }

        Ok(hidden)
    }

    /// Number of encoder layers
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Hidden size (needed for projection heads)
    fn hidden_size(&self) -> usize {
        self.meta.hidden_size
    }
}
