//! Model configurations for cross-encoders

use anyhow::Result;
use serde::Deserialize;
use edgetransformers::traits::{EncoderArchitecture, LanguageModelConfig, LayerAttentionNames, LayerFeedForwardNames, TransformerConfig};

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
    pub num_labels: usize,  // Typically 1 for ranking
}

fn default_num_labels() -> usize {
    1
}

impl MiniLMCrossEncoderConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

impl LanguageModelConfig for MiniLMCrossEncoderConfig {
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn transpose_ffn_weights(&self) -> bool {
        true
    }
}

impl TransformerConfig for MiniLMCrossEncoderConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    fn layer_norm_eps(&self) -> f32 {
        self.layer_norm_eps
    }

    fn is_causal(&self) -> bool {
        false
    }

    fn is_prenorm(&self) -> bool {
        false
    }

}



impl EncoderArchitecture for MiniLMCrossEncoderConfig {
    fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>) {
        (
            "bert.embeddings.word_embeddings.weight",
            "bert.embeddings.position_embeddings.weight",
            None,
        )
    }

    fn get_embedding_layer_norm_names(&self) -> (&str, &str) {
        (
            "bert.embeddings.LayerNorm.weight",
            "bert.embeddings.LayerNorm.bias",
        )
    }

    fn get_attention_names(&self, layer: usize) -> LayerAttentionNames {
        LayerAttentionNames {
            q_weight: format!("bert.encoder.layer.{}.attention.self.query.weight", layer),
            q_bias: format!("bert.encoder.layer.{}.attention.self.query.bias", layer),
            k_weight: format!("bert.encoder.layer.{}.attention.self.key.weight", layer),
            k_bias: format!("bert.encoder.layer.{}.attention.self.key.bias", layer),
            v_weight: format!("bert.encoder.layer.{}.attention.self.value.weight", layer),
            v_bias: format!("bert.encoder.layer.{}.attention.self.value.bias", layer),
            output_weight: format!("bert.encoder.layer.{}.attention.output.dense.weight", layer),
            output_bias: format!("bert.encoder.layer.{}.attention.output.dense.bias", layer),
            norm_weight: format!("bert.encoder.layer.{}.attention.output.LayerNorm.weight", layer),
            norm_bias: format!("bert.encoder.layer.{}.attention.output.LayerNorm.bias", layer),
        }
    }

    fn get_feed_forward_names(&self, layer: usize) -> LayerFeedForwardNames {
        LayerFeedForwardNames {
            intermediate_weight: format!("bert.encoder.layer.{}.intermediate.dense.weight", layer),
            intermediate_bias: format!("bert.encoder.layer.{}.intermediate.dense.bias", layer),
            output_weight: format!("bert.encoder.layer.{}.output.dense.weight", layer),
            output_bias: format!("bert.encoder.layer.{}.output.dense.bias", layer),
            norm_weight: format!("bert.encoder.layer.{}.output.LayerNorm.weight", layer),
            norm_bias: format!("bert.encoder.layer.{}.output.LayerNorm.bias", layer),
        }
    }
}