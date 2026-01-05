//! Model configurations for cross-encoders

use anyhow::Result;
use kjarni_transformers::activations::Activation;
use kjarni_transformers::traits::{
    AttentionLayout, EncoderLayerLayout, EncoderLayout, FeedForwardLayout, ModelConfig,
    ModelLayout, ModelMetadata, NormalizationStrategy,
};
use serde::Deserialize;
use std::collections::HashMap;

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
    #[serde(default)]
    pub id2label: Option<HashMap<String, String>>,

    #[serde(skip)]
    labels_vec: Option<Vec<String>>,
}

fn default_num_labels() -> usize {
    1
}

impl MiniLMCrossEncoderConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        let mut config: Self = serde_json::from_str(json)?;

        // Convert HashMap to sorted Vec
        if let Some(ref map) = config.id2label {
            let mut labels: Vec<(usize, String)> = map
                .iter()
                .filter_map(|(k, v)| k.parse::<usize>().ok().map(|idx| (idx, v.clone())))
                .collect();
            labels.sort_by_key(|(idx, _)| *idx);
            config.labels_vec = Some(labels.into_iter().map(|(_, v)| v).collect());
        }

        Ok(config)
    }
}

impl ModelConfig for MiniLMCrossEncoderConfig {
    fn model_type(&self) -> &str {
        "bert_cross_encoder"
    }
    fn id2label(&self) -> Option<&[String]> {
        self.labels_vec.as_deref()
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
            normalize_embedding: false,
            extra_pos_embeddings: 0,
            is_prenorm: false,            // BERT uses Post-Norm
            transpose_ffn_weights: false, // true, // MiniLM quirk
            transpose_attention_weights: false,
            normalization_strategy: NormalizationStrategy::LayerNorm,
            no_scale_qk: false,
        }
    }

    fn layout(&self) -> ModelLayout {
        // --- Define the Encoder's Layer Structure ---
        let encoder_layer = EncoderLayerLayout {
            self_attn: AttentionLayout {
                q_weight: "bert.encoder.layer.{}.attention.self.query.weight".to_string(),
                q_bias: Some("bert.encoder.layer.{}.attention.self.query.bias".to_string()),
                k_weight: "bert.encoder.layer.{}.attention.self.key.weight".to_string(),
                k_bias: Some("bert.encoder.layer.{}.attention.self.key.bias".to_string()),
                v_weight: "bert.encoder.layer.{}.attention.self.value.weight".to_string(),
                v_bias: Some("bert.encoder.layer.{}.attention.self.value.bias".to_string()),
                o_weight: "bert.encoder.layer.{}.attention.output.dense.weight".to_string(),
                o_bias: Some("bert.encoder.layer.{}.attention.output.dense.bias".to_string()),
                norm_weight: "bert.encoder.layer.{}.attention.output.LayerNorm.weight".to_string(),
                norm_bias: Some(
                    "bert.encoder.layer.{}.attention.output.LayerNorm.bias".to_string(),
                ),
            },
            ffn: FeedForwardLayout {
                up_weight: "bert.encoder.layer.{}.intermediate.dense.weight".to_string(),
                up_bias: Some("bert.encoder.layer.{}.intermediate.dense.bias".to_string()),
                down_weight: "bert.encoder.layer.{}.output.dense.weight".to_string(),
                down_bias: Some("bert.encoder.layer.{}.output.dense.bias".to_string()),
                gate_weight: None, // No SwiGLU in BERT/MiniLM
                gate_bias: None,
                norm_weight: "bert.encoder.layer.{}.output.LayerNorm.weight".to_string(),
                norm_bias: Some("bert.encoder.layer.{}.output.LayerNorm.bias".to_string()),
            },
        };

        // --- Assemble the final ModelLayout ---
        ModelLayout {
            token_embedding: "bert.embeddings.word_embeddings.weight".to_string(),
            // For encoder-only models, lm_head is often a separate classifier or pooler.
            lm_head: "classifier.weight".to_string(),
            encoder: Some(EncoderLayout {
                position_embedding: Some("bert.embeddings.position_embeddings.weight".to_string()),
                token_type_embedding: Some(
                    "bert.embeddings.token_type_embeddings.weight".to_string(),
                ),
                embedding_norm_weight: Some("bert.embeddings.LayerNorm.weight".to_string()),
                embedding_norm_bias: Some("bert.embeddings.LayerNorm.bias".to_string()),
                // For BERT, the "final_norm" is often considered the pooler layer.
                // If you have a model with a true final norm, you would populate this.
                final_norm_weight: Some("bert.pooler.dense.weight".to_string()),
                final_norm_bias: Some("bert.pooler.dense.bias".to_string()), // Pooler has a bias
                layer: encoder_layer,
            }),
            decoder: None, // This is an encoder-only model
        }
    }
}
