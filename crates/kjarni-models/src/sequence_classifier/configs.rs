//! Model configurations for cross-encoders

use anyhow::Result;
use kjarni_transformers::activations::Activation;
use kjarni_transformers::traits::{ModelConfig, ModelLayout, ModelMetadata};
use serde::Deserialize;

/// Configuration for MiniLM cross-encoder (ms-marco-MiniLM-L-6-v2)
#[derive(Debug, Clone, Deserialize)]
pub struct MiniLMCrossEncoderConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub vocab_size: usize,
    pub layer_norm_eps: f32,
    #[serde(default = "default_num_labels")]
    pub num_labels: usize, // Typically 1 for ranking
}

fn default_num_labels() -> usize {
    1
}

impl MiniLMCrossEncoderConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

impl ModelConfig for MiniLMCrossEncoderConfig {
    fn model_type(&self) -> &str {
        "bert_cross_encoder"
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            hidden_size: self.hidden_size,
            num_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_attention_heads, // Standard Encoder
            head_dim: self.hidden_size / self.num_attention_heads,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_position_embeddings,
            norm_eps: self.layer_norm_eps,
            activation: match self.hidden_act.as_str() {
                "gelu" => Activation::Gelu,
                "gelu_new" => Activation::GeluNew,
                _ => Activation::Gelu, // BERT default
            },
            rope_theta: None,
            rope_scaling: None,
            scale_embeddings: false,
            extra_pos_embeddings: 0,
            is_prenorm: false, // BERT uses Post-Norm
            transpose_ffn_weights: true, // MiniLM quirk
            transpose_attention_weights: false,
        }
    }

    fn layout(&self) -> ModelLayout {
        ModelLayout {
            // Root Level Weights
            token_embedding: "bert.embeddings.word_embeddings.weight".to_string(),
            position_embedding: Some("bert.embeddings.position_embeddings.weight".to_string()),
            token_type_embedding: Some("bert.embeddings.token_type_embeddings.weight".to_string()),
            embedding_norm: Some("bert.embeddings.LayerNorm.weight".to_string()),
            embedding_norm_bias: Some("bert.embeddings.LayerNorm.bias".to_string()),
            
            // For Cross-Encoders, the 'final_norm' is often the Pooler
            final_norm: "bert.pooler.dense.weight".to_string(),
            final_norm_bias: None,
            // The classifier head
            lm_head: "classifier.weight".to_string(),

            // Attention Templates
            attn_q: "bert.encoder.layer.{}.attention.self.query.weight".to_string(),
            attn_q_bias: Some("bert.encoder.layer.{}.attention.self.query.bias".to_string()),
            attn_k: "bert.encoder.layer.{}.attention.self.key.weight".to_string(),
            attn_k_bias: Some("bert.encoder.layer.{}.attention.self.key.bias".to_string()),
            attn_v: "bert.encoder.layer.{}.attention.self.value.weight".to_string(),
            attn_v_bias: Some("bert.encoder.layer.{}.attention.self.value.bias".to_string()),
            attn_o: "bert.encoder.layer.{}.attention.output.dense.weight".to_string(),
            attn_o_bias: Some("bert.encoder.layer.{}.attention.output.dense.bias".to_string()),
            attn_norm: "bert.encoder.layer.{}.attention.output.LayerNorm.weight".to_string(),
            attn_norm_bias: Some("bert.encoder.layer.{}.attention.output.LayerNorm.bias".to_string()),

            // FFN Templates
            ffn_gate: None, // No SwiGLU in MiniLM
            ffn_up: "bert.encoder.layer.{}.intermediate.dense.weight".to_string(),
            ffn_up_bias: Some("bert.encoder.layer.{}.intermediate.dense.bias".to_string()),
            ffn_down: "bert.encoder.layer.{}.output.dense.weight".to_string(),
            ffn_down_bias: Some("bert.encoder.layer.{}.output.dense.bias".to_string()),
            ffn_norm: "bert.encoder.layer.{}.output.LayerNorm.weight".to_string(),
            ffn_norm_bias: Some("bert.encoder.layer.{}.output.LayerNorm.bias".to_string()),

            // Cross-attention not used in encoder-only MiniLM
            cross_attn_q: None,
            cross_attn_k: None,
            cross_attn_v: None,
            cross_attn_o: None,
            cross_attn_norm: None,
            cross_attn_q_bias: None,
                cross_attn_k_bias: None,
                cross_attn_v_bias: None,
                cross_attn_o_bias: None,
                cross_attn_norm_bias: None,
        }
    }
}