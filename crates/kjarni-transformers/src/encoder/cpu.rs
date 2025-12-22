use anyhow::{Context, Result};
use ndarray::{Array2, Array3};

use crate::encoder::encoder_self_attention::EncoderSelfAttention;
use crate::encoder::CpuEncoder;
use crate::feedforward::LegacyFeedForward;
use crate::linear_layer::LinearLayer;
use crate::models::base::ModelLoadConfig;
use crate::traits::{Device, InferenceModel, ModelLayout, ModelMetadata};
use crate::weights::ModelWeights;
use crate::{
    encoder::encoder_layer::EncoderLayer, normalization::LayerNorm, Embeddings, FeedForward,
};

pub struct CpuTransformerEncoder {
    embeddings: Embeddings,
    embeddings_layer_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
    pub metadata: ModelMetadata,
}

impl CpuTransformerEncoder {
    pub fn new(
        weights: &ModelWeights,
        meta: ModelMetadata,
        layout: ModelLayout,
        load_cfg: ModelLoadConfig,
    ) -> Result<Self> {
        let dtype = load_cfg.target_dtype;

        // 1. Embeddings (Generic Loader handles Word/Pos/Type)
        let embeddings = Embeddings::from_weights(
            weights,
            &layout.token_embedding,
            layout.position_embedding.as_deref(),
            layout.token_type_embedding.as_deref(),
        )?;

        // 2. Embedding LayerNorm
        let emb_norm_w = layout
            .embedding_norm
            .as_ref()
            .context("Encoder requires embedding_norm")?;
        let emb_norm_b = layout
            .embedding_norm_bias
            .as_ref()
            .context("Encoder requires embedding_norm_bias")?;
        let embeddings_layer_norm = LayerNorm::new(
            weights.get_array1(emb_norm_w)?,
            weights.get_array1(emb_norm_b)?,
            meta.norm_eps,
        );
        let q_bias = &layout.attn_q_bias.unwrap();
        let k_bias = &layout.attn_k_bias.unwrap();
        let v_bias = &layout.attn_v_bias.unwrap();
        let o_bias = &layout.attn_o_bias.unwrap();
        let ffn_up_bias = &layout.ffn_up_bias.unwrap();
        let ffn_down_bias = &layout.ffn_down_bias.unwrap();
        let attn_norm_bias = &layout.attn_norm_bias.unwrap();
        let ffn_norm_bias = &layout.ffn_norm_bias.unwrap();

        // 3. Build Layers
        let mut layers = Vec::with_capacity(meta.num_layers);
        for i in 0..meta.num_layers {
            let idx = i.to_string();
            let name = |template: &String| template.replace("{}", &idx);

            let self_attn = EncoderSelfAttention::new(
                meta.hidden_size,
                meta.num_attention_heads,
                LinearLayer::from_weights(
                    weights,
                    &name(&layout.attn_q),
                    Some(&name(q_bias)),
                    dtype,
                    None,
                )?,
                LinearLayer::from_weights(
                    weights,
                    &name(&layout.attn_k),
                    Some(&name(k_bias)),
                    dtype,
                    None,
                )?,
                LinearLayer::from_weights(
                    weights,
                    &name(&layout.attn_v),
                    Some(&name(v_bias)),
                    dtype,
                    None,
                )?,
                LinearLayer::from_weights(
                    weights,
                    &name(&layout.attn_o),
                    Some(&name(o_bias)),
                    dtype,
                    None,
                )?,
            );

            // 4. Load FFN weights as Array2 (as required by LegacyFeedForward)
            let raw_up_w = weights.get_array2(&name(&layout.ffn_up))?;
            let raw_down_w = weights.get_array2(&name(&layout.ffn_down))?;

            let up_w = if meta.transpose_ffn_weights {
                raw_up_w.t().as_standard_layout().to_owned()
            } else {
                raw_up_w
            };

            let down_w = if meta.transpose_ffn_weights {
                raw_down_w.t().as_standard_layout().to_owned()
            } else {
                raw_down_w
            };

            let feedforward = FeedForward::Legacy(LegacyFeedForward::new(
                up_w,
                weights.get_array1(&name(ffn_up_bias))?,
                down_w,
                weights.get_array1(&name(ffn_down_bias))?,
                meta.activation,
            ));

            // 5. Layer Norms
            let self_attn_layer_norm = LayerNorm::new(
                weights.get_array1(&name(&layout.attn_norm))?,
                weights.get_array1(&name(attn_norm_bias))?,
                meta.norm_eps,
            );

            let ffn_layer_norm = LayerNorm::new(
                weights.get_array1(&name(&layout.ffn_norm))?,
                weights.get_array1(&name(ffn_norm_bias))?,
                meta.norm_eps,
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
            embeddings_layer_norm,
            layers,
            metadata: meta,
        })
    }
}

impl InferenceModel for CpuTransformerEncoder {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CpuEncoder for CpuTransformerEncoder {
    fn embed(&self, input_ids: &Array2<u32>, token_type_ids: Option<&Array2<u32>>) -> Array3<f32> {
        self.embeddings.forward(
            input_ids,
            token_type_ids,
            self.metadata.extra_pos_embeddings,
            self.metadata.scale_embeddings,
        )
    }

    fn embed_and_normalize(
        &self,
        input_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
    ) -> Array3<f32> {
        let hidden = self.embed(input_ids, token_type_ids);
        self.embeddings_layer_norm.forward_3d(&hidden)
    }

    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>> {
        let mut hidden = hidden_states.clone();
        let is_prenorm = self.metadata.is_prenorm;
        for layer in &self.layers[start_layer..end_layer] {
            hidden = layer.forward(hidden, attention_mask, None, is_prenorm)?;
        }
        Ok(hidden)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_size(&self) -> usize {
        self.metadata.hidden_size
    }
}
