use kjarni_transformers::{
    activations::Activation,
    encoder_decoder::TaskSpecificParams,
    traits::{
        AttentionLayout, DecoderLayerLayout, DecoderLayout, EncoderLayerLayout, EncoderLayout,
        FeedForwardLayout, ModelConfig, ModelLayout, ModelMetadata, NormalizationStrategy,
    },
};
use serde::{Deserialize, Serialize};

fn default_layer_norm_eps() -> f32 {
    1e-6
}

fn default_relative_attention_num_buckets() -> usize {
    32
}

fn default_relative_attention_max_distance() -> usize {
    128
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct T5Config {
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_decoder_layers: Option<usize>,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub model_type: String,

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

    // This field determines if we use standard Relu FFN (T5) or Gated Gelu (Flan-T5)
    #[serde(alias = "dense_act_fn", alias = "feed_forward_proj")]
    pub feed_forward_proj: Option<String>,

    pub task_specific_params: Option<TaskSpecificParams>,
}

impl T5Config {
    pub fn is_gated(&self) -> bool {
        match self.feed_forward_proj.as_deref() {
            Some("gated-gelu") | Some("gated-silu") => true,
            _ => false,
        }
    }
}

impl ModelConfig for T5Config {
    fn model_type(&self) -> &str {
        &self.model_type
    }

    fn metadata(&self) -> ModelMetadata {
        let (activation, is_gated) = match self.feed_forward_proj.as_deref() {
            Some("gated-gelu") => (Activation::GeluNew, true), // GeGLU
            Some("relu") => (Activation::Relu, false),
            Some("gelu") => (Activation::Gelu, false),
            _ => (Activation::Relu, false), // Default to T5 1.0 Relu
        };

        ModelMetadata {
            hidden_size: self.d_model,
            num_layers: self.num_layers,
            num_attention_heads: self.num_heads,
            num_kv_heads: self.num_heads,
            head_dim: self.d_kv, // T5 explicit head dim
            vocab_size: self.vocab_size,
            max_seq_len: 2048, // T5 uses relative positions, theoretically infinite, but practical limit needed
            norm_eps: self.layer_norm_epsilon,
            activation,
            rope_theta: None,
            rope_scaling: None,
            scale_embeddings: false, // T5 does not scale embeddings
            normalize_embedding: false,
            extra_pos_embeddings: 0,
            is_prenorm: true, // T5 is Pre-Norm
            transpose_ffn_weights: false, // T5 weights are [out, in] in HF, usually
            transpose_attention_weights: false,
            normalization_strategy: NormalizationStrategy::RMSNorm, // T5LayerNorm is effectively RMSNorm
            no_scale_qk: true,
        }
    }

    fn layout(&self) -> ModelLayout {
        let is_gated = self.is_gated();

        // 1. Encoder FFN
        let enc_ffn = if is_gated {
            // Flan-T5: wi_0 (up/act), wi_1 (gate), wo (down)
            FeedForwardLayout {
                up_weight: "encoder.block.{}.layer.1.DenseReluDense.wi_0.weight".to_string(),
                up_bias: None,
                gate_weight: Some("encoder.block.{}.layer.1.DenseReluDense.wi_1.weight".to_string()),
                gate_bias: None,
                down_weight: "encoder.block.{}.layer.1.DenseReluDense.wo.weight".to_string(),
                down_bias: None,
                norm_weight: "encoder.block.{}.layer.1.layer_norm.weight".to_string(),
                norm_bias: None,
            }
        } else {
            // T5 1.0: wi (up), wo (down)
            FeedForwardLayout {
                up_weight: "encoder.block.{}.layer.1.DenseReluDense.wi.weight".to_string(),
                up_bias: None,
                gate_weight: None,
                gate_bias: None,
                down_weight: "encoder.block.{}.layer.1.DenseReluDense.wo.weight".to_string(),
                down_bias: None,
                norm_weight: "encoder.block.{}.layer.1.layer_norm.weight".to_string(),
                norm_bias: None,
            }
        };

        // 2. Decoder FFN (Layer 2 in Decoder)
        let dec_ffn = if is_gated {
             FeedForwardLayout {
                up_weight: "decoder.block.{}.layer.2.DenseReluDense.wi_0.weight".to_string(),
                up_bias: None,
                gate_weight: Some("decoder.block.{}.layer.2.DenseReluDense.wi_1.weight".to_string()),
                gate_bias: None,
                down_weight: "decoder.block.{}.layer.2.DenseReluDense.wo.weight".to_string(),
                down_bias: None,
                norm_weight: "decoder.block.{}.layer.2.layer_norm.weight".to_string(),
                norm_bias: None,
            }
        } else {
             FeedForwardLayout {
                up_weight: "decoder.block.{}.layer.2.DenseReluDense.wi.weight".to_string(),
                up_bias: None,
                gate_weight: None,
                gate_bias: None,
                down_weight: "decoder.block.{}.layer.2.DenseReluDense.wo.weight".to_string(),
                down_bias: None,
                norm_weight: "decoder.block.{}.layer.2.layer_norm.weight".to_string(),
                norm_bias: None,
            }
        };

        ModelLayout {
            token_embedding: "shared.weight".to_string(),
            lm_head: if self.tie_word_embeddings {
                "shared.weight".to_string()
            } else {
                "lm_head.weight".to_string()
            },
            encoder: Some(EncoderLayout {
                position_embedding: None, // T5 uses relative bias
                token_type_embedding: None,
                embedding_norm_weight: Some("encoder.final_layer_norm.weight".to_string()),
                embedding_norm_bias: None, // RMSNorm
                final_norm_weight: None,
                final_norm_bias: None,
                layer: EncoderLayerLayout {
                    self_attn: AttentionLayout {
                        q_weight: "encoder.block.{}.layer.0.SelfAttention.q.weight".to_string(),
                        q_bias: None,
                        k_weight: "encoder.block.{}.layer.0.SelfAttention.k.weight".to_string(),
                        k_bias: None,
                        v_weight: "encoder.block.{}.layer.0.SelfAttention.v.weight".to_string(),
                        v_bias: None,
                        o_weight: "encoder.block.{}.layer.0.SelfAttention.o.weight".to_string(),
                        o_bias: None,
                        norm_weight: "encoder.block.{}.layer.0.layer_norm.weight".to_string(),
                        norm_bias: None,
                    },
                    ffn: enc_ffn,
                },
            }),
            decoder: Some(DecoderLayout {
                position_embedding: None,
                token_type_embedding: None,
                embedding_norm_weight: Some("decoder.final_layer_norm.weight".to_string()),
                embedding_norm_bias: None,
                final_norm_weight: None,
                final_norm_bias: None,
                layer: DecoderLayerLayout {
                    self_attn: AttentionLayout {
                        q_weight: "decoder.block.{}.layer.0.SelfAttention.q.weight".to_string(),
                        q_bias: None,
                        k_weight: "decoder.block.{}.layer.0.SelfAttention.k.weight".to_string(),
                        k_bias: None,
                        v_weight: "decoder.block.{}.layer.0.SelfAttention.v.weight".to_string(),
                        v_bias: None,
                        o_weight: "decoder.block.{}.layer.0.SelfAttention.o.weight".to_string(),
                        o_bias: None,
                        norm_weight: "decoder.block.{}.layer.0.layer_norm.weight".to_string(),
                        norm_bias: None,
                    },
                    cross_attn: Some(AttentionLayout {
                        q_weight: "decoder.block.{}.layer.1.EncDecAttention.q.weight".to_string(),
                        q_bias: None,
                        k_weight: "decoder.block.{}.layer.1.EncDecAttention.k.weight".to_string(),
                        k_bias: None,
                        v_weight: "decoder.block.{}.layer.1.EncDecAttention.v.weight".to_string(),
                        v_bias: None,
                        o_weight: "decoder.block.{}.layer.1.EncDecAttention.o.weight".to_string(),
                        o_bias: None,
                        norm_weight: "decoder.block.{}.layer.1.layer_norm.weight".to_string(),
                        norm_bias: None,
                    }),
                    ffn: dec_ffn,
                },
            }),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    const FLAN_T5_BASE_JSON: &str = r#"{
      "architectures": [ "T5ForConditionalGeneration" ],
      "d_ff": 2048,
      "d_kv": 64,
      "d_model": 768,
      "decoder_start_token_id": 0,
      "dropout_rate": 0.1,
      "eos_token_id": 1,
      "feed_forward_proj": "gated-gelu",
      "initializer_factor": 1.0,
      "is_encoder_decoder": true,
      "layer_norm_epsilon": 1e-06,
      "model_type": "t5",
      "n_positions": 512,
      "num_decoder_layers": 12,
      "num_heads": 12,
      "num_layers": 12,
      "output_past": true,
      "pad_token_id": 0,
      "relative_attention_max_distance": 128,
      "relative_attention_num_buckets": 32,
      "tie_word_embeddings": false,
      "vocab_size": 32128
    }"#;

    const T5_BASE_JSON: &str = r#"{
      "architectures": [ "T5ForConditionalGeneration" ],
      "d_ff": 3072,
      "d_kv": 64,
      "d_model": 768,
      "decoder_start_token_id": 0,
      "eos_token_id": 1,
      "feed_forward_proj": null, 
      "layer_norm_epsilon": 1e-06,
      "model_type": "t5",
      "num_heads": 12,
      "num_layers": 12,
      "pad_token_id": 0,
      "tie_word_embeddings": true, 
      "vocab_size": 32128
    }"#;

    #[test]
    fn test_flan_t5_metadata() {
        let config: T5Config = serde_json::from_str(FLAN_T5_BASE_JSON).unwrap();
        let meta = config.metadata();
        let layout = config.layout();

        // Check Metadata
        assert_eq!(meta.activation, Activation::GeluNew); // gated-gelu -> GeluNew
        assert_eq!(meta.normalization_strategy, NormalizationStrategy::RMSNorm);
        assert!(!meta.scale_embeddings);

        // Check Layout (Gated)
        let ffn = layout.encoder.unwrap().layer.ffn;
        assert!(ffn.gate_weight.is_some(), "Flan-T5 should have a gate weight");
        assert!(ffn.up_weight.contains("wi_0"));
        assert!(ffn.gate_weight.unwrap().contains("wi_1"));
    }

    #[test]
    fn test_t5_base_metadata() {
        let config: T5Config = serde_json::from_str(T5_BASE_JSON).unwrap();
        let meta = config.metadata();
        let layout = config.layout();

        // Check Metadata
        assert_eq!(meta.activation, Activation::Relu); // Default -> Relu
        assert_eq!(meta.normalization_strategy, NormalizationStrategy::RMSNorm);
        
        // Check Layout (Standard)
        let ffn = layout.encoder.unwrap().layer.ffn;
        assert!(ffn.gate_weight.is_none(), "Standard T5 should NOT have a gate weight");
        assert!(ffn.up_weight.contains(".wi.weight"));
    }

}