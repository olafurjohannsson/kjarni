//! Model-specific configurations for sentence encoders

use anyhow::Result;
use edgetransformers::traits::{
    EncoderArchitecture, LanguageModelConfig, LayerAttentionNames, LayerFeedForwardNames,
    TransformerConfig,
};
use std::any::Any;
use serde::Deserialize;

/// Configuration for MiniLM models (sentence-transformers/all-MiniLM-L6-v2)
#[derive(Debug, Clone, Deserialize)]
pub struct MiniLMConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub vocab_size: usize,
    pub layer_norm_eps: f32,
}

impl MiniLMConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

impl LanguageModelConfig for MiniLMConfig {
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn as_any(&self) -> &dyn Any {
        self // Simply return a reference to self as a `&dyn Any`
    }
    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
    fn transpose_attention_weights(&self) -> bool {
        false
    }
    fn transpose_ffn_weights(&self) -> bool {
        true
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

impl TransformerConfig for MiniLMConfig {
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
        false // Encoders are not causal
    }

    fn is_prenorm(&self) -> bool {
        false // BERT-style is post-norm
    }
}

impl EncoderArchitecture for MiniLMConfig {
    fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>) {
        (
            "embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight",
            Some("embeddings.token_type_embeddings.weight"),
        )
    }

    fn get_embedding_layer_norm_names(&self) -> (&str, &str) {
        ("embeddings.LayerNorm.weight", "embeddings.LayerNorm.bias")
    }

    fn get_attention_names(&self, layer: usize) -> LayerAttentionNames {
        LayerAttentionNames {
            q_weight: format!("encoder.layer.{}.attention.self.query.weight", layer),
            q_bias: format!("encoder.layer.{}.attention.self.query.bias", layer),
            k_weight: format!("encoder.layer.{}.attention.self.key.weight", layer),
            k_bias: format!("encoder.layer.{}.attention.self.key.bias", layer),
            v_weight: format!("encoder.layer.{}.attention.self.value.weight", layer),
            v_bias: format!("encoder.layer.{}.attention.self.value.bias", layer),
            output_weight: format!("encoder.layer.{}.attention.output.dense.weight", layer),
            output_bias: format!("encoder.layer.{}.attention.output.dense.bias", layer),
            norm_weight: format!("encoder.layer.{}.attention.output.LayerNorm.weight", layer),
            norm_bias: format!("encoder.layer.{}.attention.output.LayerNorm.bias", layer),
        }
    }

    fn get_feed_forward_names(&self, layer: usize) -> LayerFeedForwardNames {
        LayerFeedForwardNames {
            intermediate_weight: format!("encoder.layer.{}.intermediate.dense.weight", layer),
            intermediate_bias: format!("encoder.layer.{}.intermediate.dense.bias", layer),
            output_weight: format!("encoder.layer.{}.output.dense.weight", layer),
            output_bias: format!("encoder.layer.{}.output.dense.bias", layer),
            norm_weight: format!("encoder.layer.{}.output.LayerNorm.weight", layer),
            norm_bias: format!("encoder.layer.{}.output.LayerNorm.bias", layer),
            gate_weight: None,
        }
    }
}

/// Configuration for MPNet models (sentence-transformers/all-mpnet-base-v2)
#[derive(Debug, Clone, Deserialize)]
pub struct MPNetConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub layer_norm_eps: f32,
}

impl MPNetConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

impl TransformerConfig for MPNetConfig {
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

impl LanguageModelConfig for MPNetConfig {
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
    fn transpose_attention_weights(&self) -> bool {
        false
    }
    fn transpose_ffn_weights(&self) -> bool {
        false
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn as_any(&self) -> &dyn Any {
        self // Simply return a reference to self as a `&dyn Any`
    }
}

impl EncoderArchitecture for MPNetConfig {
    fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>) {
        (
            "embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight",
            Some("embeddings.token_type_embeddings.weight"), // MPNet doesn't use this but we return it
        )
    }

    fn get_embedding_layer_norm_names(&self) -> (&str, &str) {
        ("embeddings.LayerNorm.weight", "embeddings.LayerNorm.bias")
    }

    fn get_attention_names(&self, layer: usize) -> LayerAttentionNames {
        // MPNet uses "mpnet" prefix instead of "encoder"
        LayerAttentionNames {
            q_weight: format!("mpnet.encoder.layer.{}.attention.attn.q.weight", layer),
            q_bias: format!("mpnet.encoder.layer.{}.attention.attn.q.bias", layer),
            k_weight: format!("mpnet.encoder.layer.{}.attention.attn.k.weight", layer),
            k_bias: format!("mpnet.encoder.layer.{}.attention.attn.k.bias", layer),
            v_weight: format!("mpnet.encoder.layer.{}.attention.attn.v.weight", layer),
            v_bias: format!("mpnet.encoder.layer.{}.attention.attn.v.bias", layer),
            output_weight: format!("mpnet.encoder.layer.{}.attention.attn.o.weight", layer),
            output_bias: format!("mpnet.encoder.layer.{}.attention.attn.o.bias", layer),
            norm_weight: format!("mpnet.encoder.layer.{}.attention.LayerNorm.weight", layer),
            norm_bias: format!("mpnet.encoder.layer.{}.attention.LayerNorm.bias", layer),
        }
    }

    fn get_feed_forward_names(&self, layer: usize) -> LayerFeedForwardNames {
        LayerFeedForwardNames {
            intermediate_weight: format!("mpnet.encoder.layer.{}.ffn.intermediate.weight", layer),
            intermediate_bias: format!("mpnet.encoder.layer.{}.ffn.intermediate.bias", layer),
            output_weight: format!("mpnet.encoder.layer.{}.ffn.output.weight", layer),
            output_bias: format!("mpnet.encoder.layer.{}.ffn.output.bias", layer),
            norm_weight: format!("mpnet.encoder.layer.{}.LayerNorm.weight", layer),
            norm_bias: format!("mpnet.encoder.layer.{}.LayerNorm.bias", layer),
            gate_weight: None,
        }
    }
}

/// Configuration for DistilBERT models
#[derive(Debug, Clone, Deserialize)]
pub struct DistilBERTConfig {
    pub dim: usize, // DistilBERT uses 'dim' instead of 'hidden_size'
    pub n_layers: usize,
    pub n_heads: usize,
    pub hidden_dim: usize, // intermediate_size
    pub activation: String,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
}

impl DistilBERTConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

impl TransformerConfig for DistilBERTConfig {
    fn hidden_size(&self) -> usize {
        self.dim
    }

    fn num_hidden_layers(&self) -> usize {
        self.n_layers
    }

    fn num_attention_heads(&self) -> usize {
        self.n_heads
    }

    fn layer_norm_eps(&self) -> f32 {
        1e-12
    }

    fn is_causal(&self) -> bool {
        false
    }

    fn is_prenorm(&self) -> bool {
        false
    }
}

impl LanguageModelConfig for DistilBERTConfig {
    fn intermediate_size(&self) -> usize {
        self.hidden_dim
    }
    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
    fn transpose_attention_weights(&self) -> bool {
        false
    }
    fn transpose_ffn_weights(&self) -> bool {
        true
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn as_any(&self) -> &dyn Any {
        self // Simply return a reference to self as a `&dyn Any`
    }
}

impl EncoderArchitecture for DistilBERTConfig {
    fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>) {
        (
            "distilbert.embeddings.word_embeddings.weight",
            "distilbert.embeddings.position_embeddings.weight",
            None, // DistilBERT doesn't have token_type_embeddings
        )
    }

    fn get_embedding_layer_norm_names(&self) -> (&str, &str) {
        (
            "distilbert.embeddings.LayerNorm.weight",
            "distilbert.embeddings.LayerNorm.bias",
        )
    }

    fn get_attention_names(&self, layer: usize) -> LayerAttentionNames {
        LayerAttentionNames {
            q_weight: format!(
                "distilbert.transformer.layer.{}.attention.q_lin.weight",
                layer
            ),
            q_bias: format!(
                "distilbert.transformer.layer.{}.attention.q_lin.bias",
                layer
            ),
            k_weight: format!(
                "distilbert.transformer.layer.{}.attention.k_lin.weight",
                layer
            ),
            k_bias: format!(
                "distilbert.transformer.layer.{}.attention.k_lin.bias",
                layer
            ),
            v_weight: format!(
                "distilbert.transformer.layer.{}.attention.v_lin.weight",
                layer
            ),
            v_bias: format!(
                "distilbert.transformer.layer.{}.attention.v_lin.bias",
                layer
            ),
            output_weight: format!(
                "distilbert.transformer.layer.{}.attention.out_lin.weight",
                layer
            ),
            output_bias: format!(
                "distilbert.transformer.layer.{}.attention.out_lin.bias",
                layer
            ),
            norm_weight: format!(
                "distilbert.transformer.layer.{}.sa_layer_norm.weight",
                layer
            ),
            norm_bias: format!("distilbert.transformer.layer.{}.sa_layer_norm.bias", layer),
        }
    }

    fn get_feed_forward_names(&self, layer: usize) -> LayerFeedForwardNames {
        LayerFeedForwardNames {
            intermediate_weight: format!("distilbert.transformer.layer.{}.ffn.lin1.weight", layer),
            intermediate_bias: format!("distilbert.transformer.layer.{}.ffn.lin1.bias", layer),
            output_weight: format!("distilbert.transformer.layer.{}.ffn.lin2.weight", layer),
            output_bias: format!("distilbert.transformer.layer.{}.ffn.lin2.bias", layer),
            norm_weight: format!(
                "distilbert.transformer.layer.{}.output_layer_norm.weight",
                layer
            ),
            norm_bias: format!(
                "distilbert.transformer.layer.{}.output_layer_norm.bias",
                layer
            ),
            gate_weight: None,
        }
    }
}
