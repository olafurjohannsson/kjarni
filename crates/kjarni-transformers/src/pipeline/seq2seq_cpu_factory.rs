//! Factory for building Seq2Seq components from weights + layout.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use ndarray::Array1;

use crate::cpu::encoder_decoder::{
    decoder_cross_attn::DecoderCrossAttention, decoder_cross_attn_layer::CrossDecoderLayer,
};
use crate::encoder_decoder::DecoderSelfAttention;

use crate::feedforward::{FeedForward, LegacyFeedForward, StdFeedForward, SwiGluFeedForward};
use crate::{
    Normalization,
    activations::Activation,
    cpu::encoder::{encoder_layer::EncoderLayer, encoder_self_attention::EncoderSelfAttention},
    {EmbeddingData, Embeddings},
    linear_layer::{F32MatmulStrategy, LinearLayer},
    models::base::ModelLoadConfig,
    cpu::normalization::{LayerNorm, RMSNorm},
    tensor::DType,
    traits::{
        AttentionLayout, EncoderLayout, FeedForwardLayout, ModelMetadata, NormalizationStrategy,
    },
    weights::ModelWeights,
};

/// Factory for building encoder/decoder components using ModelLayout.
pub struct Seq2SeqFactory<'a> {
    weights: &'a ModelWeights,
    target_dtype: Option<DType>,
    f32_strategy: Option<F32MatmulStrategy>,
}

impl<'a> Seq2SeqFactory<'a> {
    pub fn new(weights: &'a ModelWeights) -> Self {
        Self {
            weights,
            target_dtype: None,
            f32_strategy: Some(F32MatmulStrategy::CustomSimd),
        }
    }

    pub fn with_load_config(mut self, config: &ModelLoadConfig) -> Self {
        self.target_dtype = config.target_dtype;
        self
    }

    pub fn with_target_dtype(mut self, dtype: Option<DType>) -> Self {
        self.target_dtype = dtype;
        self
    }

    pub fn build_norm(
        &self,
        weight_template: &str,
        bias_template: Option<&str>,
        strategy: NormalizationStrategy,
        eps: f32,
        layer_idx: usize,
    ) -> Result<Normalization> {
        let weight_name = Self::resolve(weight_template, layer_idx);
        let weight = self.weights.get_array1(&weight_name)?;

        match strategy {
            NormalizationStrategy::RMSNorm => Ok(Normalization::RMSNorm(RMSNorm::new(weight, eps))),
            NormalizationStrategy::LayerNorm => {
                let bias_name = bias_template
                    .map(|t| Self::resolve(t, layer_idx))
                    .ok_or_else(|| anyhow!("LayerNorm requires bias"))?;
                let bias = self.weights.get_array1(&bias_name)?;
                Ok(Normalization::LayerNorm(LayerNorm::new(weight, bias, eps)))
            }
        }
    }

    pub fn build_norm_from_layout(
        &self,
        norm_weight: &str,
        norm_bias: Option<&String>,
        strategy: NormalizationStrategy,
        eps: f32,
        layer_idx: usize,
    ) -> Result<Normalization> {
        self.build_norm(
            norm_weight,
            norm_bias.map(|s| s.as_str()),
            strategy,
            eps,
            layer_idx,
        )
    }

    /// Build a LinearLayer from template strings.
    pub fn build_linear(
        &self,
        weight_template: &str,
        bias_template: Option<&str>,
        layer_idx: usize,
    ) -> Result<LinearLayer> {
        let weight_name = Self::resolve(weight_template, layer_idx);
        let bias_name = bias_template.map(|t| Self::resolve(t, layer_idx));

        LinearLayer::builder(self.weights, &weight_name)
            .with_optional_bias(bias_name.as_deref())
            .with_target_dtype(self.target_dtype)
            .with_f32_strategy(self.f32_strategy)
            .build()
    }

    /// Build encoder self-attention from AttentionLayout.
    pub fn build_encoder_self_attention(
        &self,
        layout: &AttentionLayout,
        hidden_size: usize,
        num_heads: usize,
        layer_idx: usize,
    ) -> Result<EncoderSelfAttention> {
        let q = self.build_linear(&layout.q_weight, layout.q_bias.as_deref(), layer_idx)?;
        let k = self.build_linear(&layout.k_weight, layout.k_bias.as_deref(), layer_idx)?;
        let v = self.build_linear(&layout.v_weight, layout.v_bias.as_deref(), layer_idx)?;
        let o = self.build_linear(&layout.o_weight, layout.o_bias.as_deref(), layer_idx)?;

        Ok(EncoderSelfAttention::new(
            hidden_size,
            num_heads,
            q,
            k,
            v,
            o,
        ))
    }

    /// Build decoder self-attention from AttentionLayout.
    pub fn build_decoder_self_attention(
        &self,
        layout: &AttentionLayout,
        hidden_size: usize,
        num_heads: usize,
        layer_idx: usize,
    ) -> Result<DecoderSelfAttention> {
        let q = self.build_linear(&layout.q_weight, layout.q_bias.as_deref(), layer_idx)?;
        let k = self.build_linear(&layout.k_weight, layout.k_bias.as_deref(), layer_idx)?;
        let v = self.build_linear(&layout.v_weight, layout.v_bias.as_deref(), layer_idx)?;
        let o = self.build_linear(&layout.o_weight, layout.o_bias.as_deref(), layer_idx)?;

        Ok(DecoderSelfAttention::new(
            hidden_size,
            num_heads,
            q,
            k,
            v,
            o,
        ))
    }

    /// Build decoder cross-attention from AttentionLayout.
    pub fn build_decoder_cross_attention(
        &self,
        layout: &AttentionLayout,
        hidden_size: usize,
        num_heads: usize,
        layer_idx: usize,
    ) -> Result<DecoderCrossAttention> {
        let q = self.build_linear(&layout.q_weight, layout.q_bias.as_deref(), layer_idx)?;
        let k = self.build_linear(&layout.k_weight, layout.k_bias.as_deref(), layer_idx)?;
        let v = self.build_linear(&layout.v_weight, layout.v_bias.as_deref(), layer_idx)?;
        let o = self.build_linear(&layout.o_weight, layout.o_bias.as_deref(), layer_idx)?;

        Ok(DecoderCrossAttention::new(
            hidden_size,
            num_heads,
            q,
            k,
            v,
            o,
        ))
    }

    /// Build LayerNorm from layout fields.
    pub fn build_layer_norm(
        &self,
        weight_template: &str,
        bias_template: &str,
        eps: f32,
        layer_idx: usize,
    ) -> Result<LayerNorm> {
        let weight_name = Self::resolve(weight_template, layer_idx);
        let bias_name = Self::resolve(bias_template, layer_idx);

        let weight = self.weights.get_array1(&weight_name)?;
        let bias = self.weights.get_array1(&bias_name)?;

        Ok(LayerNorm::new(weight, bias, eps))
    }
   
    /// Build standard FFN from FeedForwardLayout.
    pub fn build_standard_ffn(
        &self,
        layout: &FeedForwardLayout,
        activation: Activation,
        layer_idx: usize,
    ) -> Result<FeedForward> {
        let up_name = Self::resolve(&layout.up_weight, layer_idx);
        let down_name = Self::resolve(&layout.down_weight, layer_idx);

        let fc1 = self.weights.get_array2(&up_name)?;
        let fc2 = self.weights.get_array2(&down_name)?;

        let fc1_bias = layout
            .up_bias
            .as_ref()
            .and_then(|t| self.weights.get_array1(&Self::resolve(t, layer_idx)).ok())
            .unwrap_or_else(|| Array1::zeros(fc1.nrows()));

        let fc2_bias = layout
            .down_bias
            .as_ref()
            .and_then(|t| self.weights.get_array1(&Self::resolve(t, layer_idx)).ok())
            .unwrap_or_else(|| Array1::zeros(fc2.nrows()));

        Ok(FeedForward::Standard(StdFeedForward::new(
            fc1, fc1_bias, fc2, fc2_bias, activation,
        )))
    }

    /// Build SwiGLU FFN from FeedForwardLayout.
    pub fn build_swiglu_ffn(
        &self,
        layout: &FeedForwardLayout,
        activation: Activation,
        layer_idx: usize,
    ) -> Result<SwiGluFeedForward> {
        let gate_template = layout
            .gate_weight
            .as_ref()
            .ok_or_else(|| anyhow!("SwiGLU requires gate_weight"))?;

        let gate = self.build_linear(gate_template, layout.gate_bias.as_deref(), layer_idx)?;
        let up = self.build_linear(&layout.up_weight, layout.up_bias.as_deref(), layer_idx)?;
        let down =
            self.build_linear(&layout.down_weight, layout.down_bias.as_deref(), layer_idx)?;

        Ok(SwiGluFeedForward::new(gate, up, down, activation))
    }

    /// Build FFN, auto-selecting SwiGLU vs Standard based on layout.
    pub fn build_ffn(
        &self,
        layout: &FeedForwardLayout,
        meta: &ModelMetadata,
        layer_idx: usize,
    ) -> Result<FeedForward> {
        if let Some(gate_template) = &layout.gate_weight {
            let gate = self.build_linear(gate_template, layout.gate_bias.as_deref(), layer_idx)?;
            let up = self.build_linear(&layout.up_weight, layout.up_bias.as_deref(), layer_idx)?;
            let down =
                self.build_linear(&layout.down_weight, layout.down_bias.as_deref(), layer_idx)?;

            return Ok(FeedForward::SwiGLU(SwiGluFeedForward::new(
                gate,
                up,
                down,
                meta.activation,
            )));
        }

        let up_template = &layout.up_weight;
        let up_name = Self::resolve(up_template, layer_idx);

        let t5_gate_name = format!("{}_0", up_name);
        let t5_up_name = format!("{}_1", up_name);

        if self.weights.contains(&t5_gate_name) && self.weights.contains(&t5_up_name) {
            log::debug!("Detected implicit T5 gated FFN for layer {}", layer_idx);
            let gate = self.build_linear(&t5_gate_name, layout.gate_bias.as_deref(), layer_idx)?;
            let up = self.build_linear(&t5_up_name, layout.up_bias.as_deref(), layer_idx)?;
            let down =
                self.build_linear(&layout.down_weight, layout.down_bias.as_deref(), layer_idx)?;

            return Ok(FeedForward::SwiGLU(SwiGluFeedForward::new(
                gate,
                up,
                down,
                meta.activation,
            )));
        }
        self.build_standard_ffn(layout, meta.activation, layer_idx)
    }

    pub fn build_legacy_ffn(
        &self,
        layout: &FeedForwardLayout,
        activation: Activation,
        layer_idx: usize,
    ) -> Result<FeedForward> {
        let up_name = Self::resolve(&layout.up_weight, layer_idx);
        let down_name = Self::resolve(&layout.down_weight, layer_idx);
        let fc1 = self.weights.get_array2(&up_name)?;
        let fc2 = self.weights.get_array2(&down_name)?;

        let fc1_transposed = fc1.t().as_standard_layout().to_owned();
        let fc2_transposed = fc2.t().as_standard_layout().to_owned();

        let fc1_bias = layout
            .up_bias
            .as_ref()
            .map(|t| self.weights.get_array1(&Self::resolve(t, layer_idx)))
            .transpose()?
            .unwrap_or_else(|| Array1::zeros(fc1_transposed.ncols()));

        let fc2_bias = layout
            .down_bias
            .as_ref()
            .map(|t| self.weights.get_array1(&Self::resolve(t, layer_idx)))
            .transpose()?
            .unwrap_or_else(|| Array1::zeros(fc2_transposed.ncols()));

        Ok(FeedForward::Legacy(LegacyFeedForward::new(
            fc1_transposed,
            fc1_bias,
            fc2_transposed,
            fc2_bias,
            activation,
        )))
    }

    /// Build a complete encoder layer from EncoderLayerLayout.
    pub fn build_encoder_layer(
        &self,
        encoder_layout: &EncoderLayout,
        meta: &ModelMetadata,
        layer_idx: usize,
    ) -> Result<EncoderLayer> {
        let layer_layout = &encoder_layout.layer;

        // Self attention
        let mut self_attn = self.build_encoder_self_attention(
            &layer_layout.self_attn,
            meta.hidden_size,
            meta.num_attention_heads,
            layer_idx,
        )?;

        if meta.no_scale_qk {
            self_attn = self_attn.with_no_qk_scaling();
        }

        // Self attention norm
        let self_attn_layer_norm = self.build_norm_from_layout(
            &layer_layout.self_attn.norm_weight,
            layer_layout.self_attn.norm_bias.as_ref(),
            meta.normalization_strategy.clone(),
            meta.norm_eps,
            layer_idx,
        )?;

        // FFN
        let feedforward = self.build_ffn(&layer_layout.ffn, meta, layer_idx)?;

        // FFN norm
        let ffn_layer_norm = self.build_norm_from_layout(
            &layer_layout.ffn.norm_weight,
            layer_layout.ffn.norm_bias.as_ref(),
            meta.normalization_strategy.clone(),
            meta.norm_eps,
            layer_idx,
        )?;

        Ok(EncoderLayer {
            self_attn,
            self_attn_layer_norm,
            feedforward,
            ffn_layer_norm,
        })
    }

    /// Build a complete decoder layer with cross-attention.
    pub fn build_decoder_layer(
        &self,
        decoder_layout: &crate::traits::DecoderLayout,
        meta: &ModelMetadata,
        layer_idx: usize,
        is_prenorm: bool,
    ) -> Result<CrossDecoderLayer> {
        let layer_layout = &decoder_layout.layer;
        let mut self_attn = self.build_decoder_self_attention(
            &layer_layout.self_attn,
            meta.hidden_size,
            meta.num_attention_heads,
            layer_idx,
        )?;

        if meta.no_scale_qk {
            self_attn = self_attn.with_no_qk_scaling();
        }

        let self_attn_layer_norm = self.build_norm_from_layout(
            &layer_layout.self_attn.norm_weight,
            layer_layout.self_attn.norm_bias.as_ref(),
            meta.normalization_strategy.clone(),
            meta.norm_eps,
            layer_idx,
        )?;
        let cross_attn_layout = layer_layout
            .cross_attn
            .as_ref()
            .ok_or_else(|| anyhow!("Seq2Seq decoder requires cross_attn in layout"))?;

        let mut cross_attn = self.build_decoder_cross_attention(
            cross_attn_layout,
            meta.hidden_size,
            meta.num_attention_heads,
            layer_idx,
        )?;

        if meta.no_scale_qk {
            cross_attn = cross_attn.with_no_qk_scaling();
        }

        let cross_attn_layer_norm = self.build_norm_from_layout(
            &cross_attn_layout.norm_weight,
            cross_attn_layout.norm_bias.as_ref(),
            meta.normalization_strategy.clone(),
            meta.norm_eps,
            layer_idx,
        )?;

        let feedforward = self.build_ffn(&layer_layout.ffn, meta, layer_idx)?;

        let ffn_layer_norm = self.build_norm_from_layout(
            &layer_layout.ffn.norm_weight,
            layer_layout.ffn.norm_bias.as_ref(),
            meta.normalization_strategy.clone(),
            meta.norm_eps,
            layer_idx,
        )?;

        Ok(CrossDecoderLayer {
            self_attn,
            self_attn_layer_norm,
            cross_attn,
            cross_attn_layer_norm,
            feedforward,
            ffn_layer_norm,
            pre_norm: is_prenorm,
        })
    }

    /// Build token embeddings
    pub fn build_embeddings(
        &self,
        token_embedding_name: &str,
        position_embedding_name: Option<&str>,
    ) -> Result<Embeddings> {
        let word_emb = self.weights.get_array2(token_embedding_name)?;
        let pos_emb = position_embedding_name
            .map(|name| self.weights.get_array2(name))
            .transpose()?;

        Ok(Embeddings::new(
            EmbeddingData::F32(Arc::new(word_emb)),
            pos_emb,
            None, // token_type_embeddings
        ))
    }

    fn resolve(template: &str, layer_idx: usize) -> String {
        template.replace("{}", &layer_idx.to_string())
    }
}
