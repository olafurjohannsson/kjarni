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
    fn num_labels(&self) -> Option<usize> {
        Some(self.num_labels)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
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
            decoder_layers: None,
            rope_theta: None,
            rope_scaling: None,
            intermediate_size: 0,
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




// ============================================================================
// 4. RoBERTa Configuration
// ============================================================================

/// RoBERTa configuration for sequence classification.
#[derive(Debug, Clone, Deserialize)]
pub struct RobertaConfig {
    // Core architecture
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub layer_norm_eps: f32,
    pub type_vocab_size: usize,
    
    // Position embedding type (e.g., "absolute")
    #[serde(default)]
    pub position_embedding_type: String,

    // Classification head config
    #[serde(default)]
    pub id2label: Option<HashMap<String, String>>,
    #[serde(default)]
    pub label2id: Option<HashMap<String, usize>>,
    #[serde(default)]
    pub num_labels: Option<usize>,

    // Computed field (not from JSON)
    #[serde(skip)]
    pub labels_vec: Option<Vec<String>>,
}

impl RobertaConfig {
    /// Deserializes the config from a JSON string and processes the labels.
    pub fn from_json(json: &str) -> Result<Self> {
        let mut config: Self = serde_json::from_str(json)?;

        // Convert id2label HashMap to a deterministically sorted Vec
        if let Some(ref map) = config.id2label {
            let mut labels: Vec<(usize, String)> = map
                .iter()
                .filter_map(|(k, v)| k.parse::<usize>().ok().map(|idx| (idx, v.clone())))
                .collect();
            labels.sort_by_key(|(idx, _)| *idx);
            config.labels_vec = Some(labels.into_iter().map(|(_, v)| v).collect());

            // Infer num_labels if not explicitly set
            if config.num_labels.is_none() {
                config.num_labels = Some(config.labels_vec.as_ref().unwrap().len());
            }
        }

        Ok(config)
    }
}

impl ModelConfig for RobertaConfig {
    fn model_type(&self) -> &str {
        "roberta"
    }
fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn id2label(&self) -> Option<&[String]> {
        self.labels_vec.as_deref()
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            hidden_size: self.hidden_size,
            num_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_attention_heads, // RoBERTa uses standard MHA
            head_dim: self.hidden_size / self.num_attention_heads,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_position_embeddings,
            norm_eps: self.layer_norm_eps,
            activation: match self.hidden_act.as_str() {
                "gelu" => Activation::Gelu,
                "gelu_new" => Activation::GeluNew,
                "relu" => Activation::Relu,
                _ => Activation::Gelu, // Default to Gelu
            },
            // RoBERTa has a padding offset for position embeddings.
            // max_position_embeddings is often 514 (512 + 2 for padding).
            extra_pos_embeddings: 2, 
            
            // RoBERTa does not use these features
            rope_theta: None,
            rope_scaling: None,
            decoder_layers: None,
            intermediate_size: 0,
            scale_embeddings: false,
            normalize_embedding: false,
            is_prenorm: false, 
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            normalization_strategy: NormalizationStrategy::LayerNorm,
            no_scale_qk: false,
        }
    }

    fn layout(&self) -> ModelLayout {
        // This layout is derived directly from the tensor names you provided.
        // The key is the "roberta." prefix on all encoder and embedding layers.
        let encoder_layer = EncoderLayerLayout {
            self_attn: AttentionLayout {
                q_weight: "roberta.encoder.layer.{}.attention.self.query.weight".to_string(),
                q_bias: Some("roberta.encoder.layer.{}.attention.self.query.bias".to_string()),
                k_weight: "roberta.encoder.layer.{}.attention.self.key.weight".to_string(),
                k_bias: Some("roberta.encoder.layer.{}.attention.self.key.bias".to_string()),
                v_weight: "roberta.encoder.layer.{}.attention.self.value.weight".to_string(),
                v_bias: Some("roberta.encoder.layer.{}.attention.self.value.bias".to_string()),
                o_weight: "roberta.encoder.layer.{}.attention.output.dense.weight".to_string(),
                o_bias: Some("roberta.encoder.layer.{}.attention.output.dense.bias".to_string()),
                norm_weight: "roberta.encoder.layer.{}.attention.output.LayerNorm.weight".to_string(),
                norm_bias: Some("roberta.encoder.layer.{}.attention.output.LayerNorm.bias".to_string()),
            },
            ffn: FeedForwardLayout {
                up_weight: "roberta.encoder.layer.{}.intermediate.dense.weight".to_string(),
                up_bias: Some("roberta.encoder.layer.{}.intermediate.dense.bias".to_string()),
                down_weight: "roberta.encoder.layer.{}.output.dense.weight".to_string(),
                down_bias: Some("roberta.encoder.layer.{}.output.dense.bias".to_string()),
                gate_weight: None,
                gate_bias: None,
                norm_weight: "roberta.encoder.layer.{}.output.LayerNorm.weight".to_string(),
                norm_bias: Some("roberta.encoder.layer.{}.output.LayerNorm.bias".to_string()),
            },
        };

        ModelLayout {
            token_embedding: "roberta.embeddings.word_embeddings.weight".to_string(),
            // For sequence classification, the "head" consists of the classifier layers.
            // Pointing lm_head to a terminal weight like this is a convention.
            lm_head: "classifier.out_proj.weight".to_string(),
            encoder: Some(EncoderLayout {
                position_embedding: Some("roberta.embeddings.position_embeddings.weight".to_string()),
                token_type_embedding: Some("roberta.embeddings.token_type_embeddings.weight".to_string()),
                embedding_norm_weight: Some("roberta.embeddings.LayerNorm.weight".to_string()),
                embedding_norm_bias: Some("roberta.embeddings.LayerNorm.bias".to_string()),
                final_norm_weight: None, // RoBERTa does not have a final norm before the classifier
                final_norm_bias: None,
                layer: encoder_layer,
            }),
            decoder: None, // This is an encoder-only model
        }
    }
}

