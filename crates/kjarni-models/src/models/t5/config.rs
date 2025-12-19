//! T5 configuration

use kjarni_transformers::{
    activations::Activation,
    traits::{
        EncoderDecoderArchitecture, LanguageModelConfig, LayerAttentionNames,
        LayerFeedForwardNames, TransformerConfig,
    },
};
use serde::Deserialize;
use std::any::Any;

fn default_layer_norm_eps() -> f32 {
    1e-6
}

fn default_relative_attention_num_buckets() -> usize {
    32
}

fn default_relative_attention_max_distance() -> usize {
    128
}

#[derive(Debug, Clone, Deserialize)]
pub struct T5Config {
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_decoder_layers: Option<usize>,
    pub num_heads: usize,
    pub vocab_size: usize,

    #[serde(default = "default_relative_attention_num_buckets")]
    pub relative_attention_num_buckets: usize,

    #[serde(default = "default_relative_attention_max_distance")]
    pub relative_attention_max_distance: usize,

    #[serde(default = "default_layer_norm_eps", alias = "layer_norm_epsilon")]
    pub layer_norm_epsilon: f32,

    pub eos_token_id: u32,
    pub decoder_start_token_id: u32,
    pub pad_token_id: u32,

    #[serde(default)]
    pub is_encoder_decoder: bool,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(alias = "dense_act_fn", alias = "feed_forward_proj")]
    pub feed_forward_proj: Option<String>,

    pub model_type: Option<String>,
}

impl T5Config {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn num_decoder_layers(&self) -> usize {
        self.num_decoder_layers.unwrap_or(self.num_layers)
    }

    /// T5 uses "gated-gelu" activation in newer versions
    pub fn is_gated_activation(&self) -> bool {
        self.feed_forward_proj
            .as_ref()
            .map(|s| s.contains("gated"))
            .unwrap_or(false)
    }
}

impl TransformerConfig for T5Config {
    fn hidden_size(&self) -> usize {
        self.d_model
    }

    fn num_attention_heads(&self) -> usize {
        self.num_heads
    }

    fn num_hidden_layers(&self) -> usize {
        self.num_layers
    }

    fn layer_norm_eps(&self) -> f32 {
        self.layer_norm_epsilon
    }

    fn is_causal(&self) -> bool {
        false
    }

    fn is_prenorm(&self) -> bool {
        true // T5 is pre-norm (layer norm before attention/ff)
    }

    fn extra_pos_embeddings(&self) -> usize {
        0 // T5 uses relative position bias, not absolute embeddings
    }
}

impl LanguageModelConfig for T5Config {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn decoder_start_token_id(&self) -> u32 {
        self.decoder_start_token_id
    }

    fn max_position_embeddings(&self) -> usize {
        // T5 uses relative positions, but we need a practical limit
        2048
    }

    fn activation_function(&self) -> Activation {
        match self.feed_forward_proj.as_deref() {
            Some("gated-gelu") => Activation::GeluNew,
            Some("relu") => Activation::Relu,
            _ => Activation::Relu, // T5 1.0 default
        }
    }

    fn intermediate_size(&self) -> usize {
        self.d_ff
    }

    fn transpose_ffn_weights(&self) -> bool {
        false
    }

    fn transpose_attention_weights(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(self.eos_token_id)
    }

    fn bos_token_id(&self) -> Option<u32> {
        None // T5 doesn't use BOS
    }

    fn pad_token_id(&self) -> Option<u32> {
        Some(self.pad_token_id)
    }

    fn is_encoder_decoder(&self) -> Option<bool> {
        Some(self.is_encoder_decoder)
    }

    fn model_type(&self) -> Option<String> {
        self.model_type.clone()
    }

    fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>) {
        ("shared.weight", "", None) // T5 has no position embeddings
    }

    fn scale_embeddings(&self) -> bool {
        false
    }

    fn forced_bos_token_id(&self) -> Option<u32> {
        None
    }
}

impl EncoderDecoderArchitecture for T5Config {
    fn get_shared_embedding_weight_name(&self) -> &str {
        "shared.weight"
    }

    fn get_lm_head_name(&self) -> &str {
        if self.tie_word_embeddings {
            "shared.weight"
        } else {
            "lm_head.weight"
        }
    }

    fn get_final_logits_bias_name(&self) -> Option<&str> {
        None // T5 doesn't have final logits bias
    }

    fn num_encoder_layers(&self) -> usize {
        self.num_layers
    }

    fn num_decoder_layers(&self) -> usize {
        self.num_decoder_layers()
    }

    // T5 weight naming convention
    fn get_encoder_embedding_names(&self) -> (&str, &str, Option<&str>) {
        ("shared.weight", "", None)
    }

    fn get_encoder_embedding_ln_names(&self) -> (&str, &str) {
        ("encoder.final_layer_norm.weight", "") // T5 uses RMSNorm, no bias
    }

    fn get_encoder_attention_names(&self, i: usize) -> LayerAttentionNames {
        LayerAttentionNames {
            q_weight: format!("encoder.block.{}.layer.0.SelfAttention.q.weight", i),
            q_bias: String::new(), // T5 attention has no bias
            k_weight: format!("encoder.block.{}.layer.0.SelfAttention.k.weight", i),
            k_bias: String::new(),
            v_weight: format!("encoder.block.{}.layer.0.SelfAttention.v.weight", i),
            v_bias: String::new(),
            output_weight: format!("encoder.block.{}.layer.0.SelfAttention.o.weight", i),
            output_bias: String::new(),
            norm_weight: format!("encoder.block.{}.layer.0.layer_norm.weight", i),
            norm_bias: String::new(),
        }
    }

    fn get_encoder_feed_forward_names(&self, i: usize) -> LayerFeedForwardNames {
        if self.is_gated_activation() {
            LayerFeedForwardNames {
                intermediate_weight: format!("encoder.block.{}.layer.1.DenseReluDense.wi_0.weight", i),
                intermediate_bias: String::new(),
                output_weight: format!("encoder.block.{}.layer.1.DenseReluDense.wo.weight", i),
                output_bias: String::new(),
                norm_weight: format!("encoder.block.{}.layer.1.layer_norm.weight", i),
                norm_bias: String::new(),
                gate_weight: Some(format!("encoder.block.{}.layer.1.DenseReluDense.wi_1.weight", i)),
            }
        } else {
            LayerFeedForwardNames {
                intermediate_weight: format!("encoder.block.{}.layer.1.DenseReluDense.wi.weight", i),
                intermediate_bias: String::new(),
                output_weight: format!("encoder.block.{}.layer.1.DenseReluDense.wo.weight", i),
                output_bias: String::new(),
                norm_weight: format!("encoder.block.{}.layer.1.layer_norm.weight", i),
                norm_bias: String::new(),
                gate_weight: None,
            }
        }
    }

    fn get_decoder_embedding_names(&self) -> (&str, &str, Option<&str>) {
        ("shared.weight", "", None)
    }

    fn get_decoder_embedding_ln_names(&self) -> (&str, &str) {
        ("decoder.final_layer_norm.weight", "")
    }

    fn get_decoder_self_attention_names(&self, i: usize) -> LayerAttentionNames {
        LayerAttentionNames {
            q_weight: format!("decoder.block.{}.layer.0.SelfAttention.q.weight", i),
            q_bias: String::new(),
            k_weight: format!("decoder.block.{}.layer.0.SelfAttention.k.weight", i),
            k_bias: String::new(),
            v_weight: format!("decoder.block.{}.layer.0.SelfAttention.v.weight", i),
            v_bias: String::new(),
            output_weight: format!("decoder.block.{}.layer.0.SelfAttention.o.weight", i),
            output_bias: String::new(),
            norm_weight: format!("decoder.block.{}.layer.0.layer_norm.weight", i),
            norm_bias: String::new(),
        }
    }

    fn get_decoder_cross_attention_names(&self, i: usize) -> LayerAttentionNames {
        LayerAttentionNames {
            q_weight: format!("decoder.block.{}.layer.1.EncDecAttention.q.weight", i),
            q_bias: String::new(),
            k_weight: format!("decoder.block.{}.layer.1.EncDecAttention.k.weight", i),
            k_bias: String::new(),
            v_weight: format!("decoder.block.{}.layer.1.EncDecAttention.v.weight", i),
            v_bias: String::new(),
            output_weight: format!("decoder.block.{}.layer.1.EncDecAttention.o.weight", i),
            output_bias: String::new(),
            norm_weight: format!("decoder.block.{}.layer.1.layer_norm.weight", i),
            norm_bias: String::new(),
        }
    }

    fn get_decoder_feed_forward_names(&self, i: usize) -> LayerFeedForwardNames {
        if self.is_gated_activation() {
            LayerFeedForwardNames {
                intermediate_weight: format!("decoder.block.{}.layer.2.DenseReluDense.wi_0.weight", i),
                intermediate_bias: String::new(),
                output_weight: format!("decoder.block.{}.layer.2.DenseReluDense.wo.weight", i),
                output_bias: String::new(),
                norm_weight: format!("decoder.block.{}.layer.2.layer_norm.weight", i),
                norm_bias: String::new(),
                gate_weight: Some(format!("decoder.block.{}.layer.2.DenseReluDense.wi_1.weight", i)),
            }
        } else {
            LayerFeedForwardNames {
                intermediate_weight: format!("decoder.block.{}.layer.2.DenseReluDense.wi.weight", i),
                intermediate_bias: String::new(),
                output_weight: format!("decoder.block.{}.layer.2.DenseReluDense.wo.weight", i),
                output_bias: String::new(),
                norm_weight: format!("decoder.block.{}.layer.2.layer_norm.weight", i),
                norm_bias: String::new(),
                gate_weight: None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLAN_T5_BASE_CONFIG: &str = r#"{
        "d_ff": 2048,
        "d_kv": 64,
        "d_model": 768,
        "decoder_start_token_id": 0,
        "dense_act_fn": "gelu_new",
        "dropout_rate": 0.1,
        "eos_token_id": 1,
        "feed_forward_proj": "gated-gelu",
        "initializer_factor": 1.0,
        "is_encoder_decoder": true,
        "is_gated_act": true,
        "layer_norm_epsilon": 1e-06,
        "model_type": "t5",
        "num_decoder_layers": 12,
        "num_heads": 12,
        "num_layers": 12,
        "output_past": true,
        "pad_token_id": 0,
        "relative_attention_max_distance": 128,
        "relative_attention_num_buckets": 32,
        "tie_word_embeddings": true,
        "vocab_size": 32128
    }"#;

    #[test]
    fn test_flan_t5_deserialization() {
        let config: T5Config = serde_json::from_str(FLAN_T5_BASE_CONFIG).unwrap();

        assert_eq!(config.d_model, 768);
        assert_eq!(config.d_ff, 2048);
        assert_eq!(config.d_kv, 64);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.vocab_size, 32128);
        assert!(config.is_encoder_decoder);
        assert!(config.tie_word_embeddings);
        assert!(config.is_gated_activation());
    }

    #[test]
    fn test_t5_weight_names() {
        let config: T5Config = serde_json::from_str(FLAN_T5_BASE_CONFIG).unwrap();

        // Encoder attention
        let attn = config.get_encoder_attention_names(0);
        assert_eq!(attn.q_weight, "encoder.block.0.layer.0.SelfAttention.q.weight");
        assert!(attn.q_bias.is_empty()); // T5 has no attention bias

        // Encoder FFN (gated)
        let ffn = config.get_encoder_feed_forward_names(2);
        assert_eq!(ffn.intermediate_weight, "encoder.block.2.layer.1.DenseReluDense.wi_0.weight");
        assert!(ffn.gate_weight.is_some());

        // Decoder cross attention
        let cross = config.get_decoder_cross_attention_names(1);
        assert_eq!(cross.q_weight, "decoder.block.1.layer.1.EncDecAttention.q.weight");
    }
}