//! Model-specific configurations for sentence encoders
use anyhow::Result;
use kjarni_transformers::activations::Activation;
use kjarni_transformers::traits::{ModelConfig, ModelLayout, ModelMetadata};
use serde::Deserialize;

// ============================================================================
// 1. MiniLM Configuration
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
pub struct MiniLMConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    #[serde(alias = "hidden_act", alias = "activation_function")]
    pub activation_function: Option<String>,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub layer_norm_eps: f32,
}

impl MiniLMConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

impl ModelConfig for MiniLMConfig {
    fn model_type(&self) -> &str {
        "minilm"
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            hidden_size: self.hidden_size,
            num_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_attention_heads,
            head_dim: self.hidden_size / self.num_attention_heads,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_position_embeddings,
            norm_eps: self.layer_norm_eps,
            activation: self
                .activation_function
                .as_ref()
                .and_then(|s| s.parse().ok())
                .unwrap_or(Activation::Gelu),
            rope_theta: None,
            rope_scaling: None,
            scale_embeddings: false,
            extra_pos_embeddings: 0,
            is_prenorm: false, // BERT-style
            transpose_ffn_weights: true,
        }
    }

    fn layout(&self) -> ModelLayout {
        ModelLayout {
            token_embedding: "embeddings.word_embeddings.weight".to_string(),
            position_embedding: Some("embeddings.position_embeddings.weight".to_string()),
            token_type_embedding: Some("embeddings.token_type_embeddings.weight".to_string()),
            embedding_norm: Some("embeddings.LayerNorm.weight".to_string()),
            embedding_norm_bias: Some("embeddings.LayerNorm.bias".to_string()),
            final_norm: "encoder.layer.5.output.LayerNorm.weight".to_string(), // Placeholder
            lm_head: "cls.predictions.decoder.weight".to_string(),

            attn_q: "encoder.layer.{}.attention.self.query.weight".to_string(),
            attn_q_bias: Some("encoder.layer.{}.attention.self.query.bias".to_string()),
            attn_k: "encoder.layer.{}.attention.self.key.weight".to_string(),
            attn_k_bias: Some("encoder.layer.{}.attention.self.key.bias".to_string()),
            attn_v: "encoder.layer.{}.attention.self.value.weight".to_string(),
            attn_v_bias: Some("encoder.layer.{}.attention.self.value.bias".to_string()),
            attn_o: "encoder.layer.{}.attention.output.dense.weight".to_string(),
            attn_o_bias: Some("encoder.layer.{}.attention.output.dense.bias".to_string()),
            attn_norm: "encoder.layer.{}.attention.output.LayerNorm.weight".to_string(),
            attn_norm_bias: Some("encoder.layer.{}.attention.output.LayerNorm.bias".to_string()),

            ffn_gate: None,
            ffn_up: "encoder.layer.{}.intermediate.dense.weight".to_string(),
            ffn_up_bias: Some("encoder.layer.{}.intermediate.dense.bias".to_string()),
            ffn_down: "encoder.layer.{}.output.dense.weight".to_string(),
            ffn_down_bias: Some("encoder.layer.{}.output.dense.bias".to_string()),
            ffn_norm: "encoder.layer.{}.output.LayerNorm.weight".to_string(),
            ffn_norm_bias: Some("encoder.layer.{}.output.LayerNorm.bias".to_string()),

            cross_attn_q: None,
            cross_attn_k: None,
            cross_attn_v: None,
            cross_attn_o: None,
            cross_attn_norm: None,
        }
    }
}

// ============================================================================
// 2. MPNet Configuration
// ============================================================================
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

impl ModelConfig for MPNetConfig {
    fn model_type(&self) -> &str {
        "mpnet"
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            hidden_size: self.hidden_size,
            num_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_attention_heads,
            head_dim: self.hidden_size / self.num_attention_heads,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_position_embeddings,
            norm_eps: self.layer_norm_eps,
            activation: Activation::GeluNew,
            rope_theta: None,
            rope_scaling: None,
            scale_embeddings: false,
            extra_pos_embeddings: 0,
            is_prenorm: false,
            transpose_ffn_weights: false,
        }
    }

    fn layout(&self) -> ModelLayout {
        ModelLayout {
            token_embedding: "embeddings.word_embeddings.weight".to_string(),
            position_embedding: Some("embeddings.position_embeddings.weight".to_string()),
            token_type_embedding: Some("embeddings.token_type_embeddings.weight".to_string()),
            embedding_norm: Some("embeddings.LayerNorm.weight".to_string()),
            embedding_norm_bias: Some("embeddings.LayerNorm.bias".to_string()),
            final_norm: "mpnet.pooler.dense.weight".to_string(),
            lm_head: "classifier.weight".to_string(),

            attn_q: "mpnet.encoder.layer.{}.attention.attn.q.weight".to_string(),
            attn_q_bias: Some("mpnet.encoder.layer.{}.attention.attn.q.bias".to_string()),
            attn_k: "mpnet.encoder.layer.{}.attention.attn.k.weight".to_string(),
            attn_k_bias: Some("mpnet.encoder.layer.{}.attention.attn.k.bias".to_string()),
            attn_v: "mpnet.encoder.layer.{}.attention.attn.v.weight".to_string(),
            attn_v_bias: Some("mpnet.encoder.layer.{}.attention.attn.v.bias".to_string()),
            attn_o: "mpnet.encoder.layer.{}.attention.attn.o.weight".to_string(),
            attn_o_bias: Some("mpnet.encoder.layer.{}.attention.attn.o.bias".to_string()),
            attn_norm: "mpnet.encoder.layer.{}.attention.LayerNorm.weight".to_string(),
            attn_norm_bias: Some("mpnet.encoder.layer.{}.attention.LayerNorm.bias".to_string()),

            ffn_gate: None,
            ffn_up: "mpnet.encoder.layer.{}.ffn.intermediate.weight".to_string(),
            ffn_up_bias: Some("mpnet.encoder.layer.{}.ffn.intermediate.bias".to_string()),
            ffn_down: "mpnet.encoder.layer.{}.ffn.output.weight".to_string(),
            ffn_down_bias: Some("mpnet.encoder.layer.{}.ffn.output.bias".to_string()),
            ffn_norm: "mpnet.encoder.layer.{}.LayerNorm.weight".to_string(),
            ffn_norm_bias: Some("mpnet.encoder.layer.{}.LayerNorm.bias".to_string()),

            cross_attn_q: None,
            cross_attn_k: None,
            cross_attn_v: None,
            cross_attn_o: None,
            cross_attn_norm: None,
        }
    }
}

// ============================================================================
// 3. DistilBERT Configuration
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
pub struct DistilBERTConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub hidden_dim: usize,
    pub activation: String,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
}

impl DistilBERTConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

impl ModelConfig for DistilBERTConfig {
    fn model_type(&self) -> &str {
        "distilbert"
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            hidden_size: self.dim,
            num_layers: self.n_layers,
            num_attention_heads: self.n_heads,
            num_kv_heads: self.n_heads,
            head_dim: self.dim / self.n_heads,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_position_embeddings,
            norm_eps: 1e-12,
            activation: Activation::GeluNew,
            rope_theta: None,
            rope_scaling: None,
            scale_embeddings: false,
            extra_pos_embeddings: 0,
            is_prenorm: false,
            transpose_ffn_weights: true,
        }
    }

    fn layout(&self) -> ModelLayout {
        ModelLayout {
            token_embedding: "distilbert.embeddings.word_embeddings.weight".to_string(),
            position_embedding: Some(
                "distilbert.embeddings.position_embeddings.weight".to_string(),
            ),
            token_type_embedding: None,
            embedding_norm: Some("distilbert.embeddings.LayerNorm.weight".to_string()),
            embedding_norm_bias: Some("distilbert.embeddings.LayerNorm.bias".to_string()),
            final_norm: "".to_string(),
            lm_head: "vocab_projector.weight".to_string(),

            attn_q: "distilbert.transformer.layer.{}.attention.q_lin.weight".to_string(),
            attn_q_bias: Some("distilbert.transformer.layer.{}.attention.q_lin.bias".to_string()),
            attn_k: "distilbert.transformer.layer.{}.attention.k_lin.weight".to_string(),
            attn_k_bias: Some("distilbert.transformer.layer.{}.attention.k_lin.bias".to_string()),
            attn_v: "distilbert.transformer.layer.{}.attention.v_lin.weight".to_string(),
            attn_v_bias: Some("distilbert.transformer.layer.{}.attention.v_lin.bias".to_string()),
            attn_o: "distilbert.transformer.layer.{}.attention.out_lin.weight".to_string(),
            attn_o_bias: Some("distilbert.transformer.layer.{}.attention.out_lin.bias".to_string()),
            attn_norm: "distilbert.transformer.layer.{}.sa_layer_norm.weight".to_string(),
            attn_norm_bias: Some("distilbert.transformer.layer.{}.sa_layer_norm.bias".to_string()),

            ffn_gate: None,
            ffn_up: "distilbert.transformer.layer.{}.ffn.lin1.weight".to_string(),
            ffn_up_bias: Some("distilbert.transformer.layer.{}.ffn.lin1.bias".to_string()),
            ffn_down: "distilbert.transformer.layer.{}.ffn.lin2.weight".to_string(),
            ffn_down_bias: Some("distilbert.transformer.layer.{}.ffn.lin2.bias".to_string()),
            ffn_norm: "distilbert.transformer.layer.{}.output_layer_norm.weight".to_string(),
            ffn_norm_bias: Some(
                "distilbert.transformer.layer.{}.output_layer_norm.bias".to_string(),
            ),

            cross_attn_q: None,
            cross_attn_k: None,
            cross_attn_v: None,
            cross_attn_o: None,
            cross_attn_norm: None,
        }
    }
}
