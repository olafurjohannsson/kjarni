use anyhow::{Context, Result};
use kjarni_transformers::{
    activations::Activation,
    traits::{
        AttentionLayout, EncoderLayerLayout, EncoderLayout, FeedForwardLayout, ModelConfig,
        ModelLayout, ModelMetadata,
    },
    weights::WeightLoader,
};
use serde::Deserialize;
use std::sync::Arc;

// ============================================================================
// 1. BERT Configuration (MiniLM, Nomic, etc.)
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
pub struct BertConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    #[serde(alias = "hidden_act")]
    pub activation_function: Option<String>,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    #[serde(alias = "layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default)]
    pub type_vocab_size: usize,
    
    // For Nomic/Modern BERTs
    pub rotary_embedding_fraction: Option<f32>, 
}

impl BertConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn from_loader(loader: &dyn WeightLoader, config_json: Option<&str>) -> Result<Arc<Self>> {
        if loader.has_metadata() {
            // GGUF Logic
            let arch = loader.get_string("general.architecture").unwrap_or("bert");
            let get_u32 = |k: &str| loader.get_u32(&format!("{}.{}", arch, k));
            let get_f32 = |k: &str| loader.get_f32(&format!("{}.{}", arch, k));

            Ok(Arc::new(Self {
                hidden_size: get_u32("embedding_length").context("no embedding_length")? as usize,
                num_hidden_layers: get_u32("block_count").context("no block_count")? as usize,
                num_attention_heads: get_u32("attention.head_count").context("no head_count")? as usize,
                intermediate_size: get_u32("feed_forward_length").unwrap_or(0) as usize,
                activation_function: Some(loader.get_string(&format!("{}.feed_forward_activation", arch)).unwrap_or("gelu").to_string()),
                max_position_embeddings: get_u32("context_length").unwrap_or(512) as usize,
                vocab_size: loader.get_u32("general.vocabulary_size").unwrap_or(30522) as usize,
                layer_norm_eps: get_f32("attention.layer_norm_epsilon").unwrap_or(1e-12),
                type_vocab_size: 2, // Standard BERT default
                rotary_embedding_fraction: None, // GGUF usually maps this differently if present
            }))
        } else {
            let json_str = config_json.context("Config JSON required for SafeTensors")?;
            Ok(Arc::new(Self::from_json(json_str)?))
        }
    }
}

impl ModelConfig for BertConfig {
    fn model_type(&self) -> &str { "bert" }

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
            activation: match self.activation_function.as_deref() {
                Some("gelu") => Activation::Gelu,
                Some("gelu_new") => Activation::GeluNew, // GeLU Fast/Tanh
                Some("relu") => Activation::Relu,
                _ => Activation::Gelu,
            },
            // Nomic Support: Check if rotary is enabled
            rope_theta: None, 
            rope_scaling: None,
            scale_embeddings: false,
            normalize_embedding: false,
            extra_pos_embeddings: 0,
            is_prenorm: false, // Standard BERT is Post-Norm
            transpose_ffn_weights: true, // BERT Linear layers usually need transpose in some engines
            transpose_attention_weights: false,
        }
    }

    fn layout(&self) -> ModelLayout {
        let encoder_layer = EncoderLayerLayout {
            self_attn: AttentionLayout {
                q_weight: "encoder.layer.{}.attention.self.query.weight".to_string(),
                q_bias: Some("encoder.layer.{}.attention.self.query.bias".to_string()),
                k_weight: "encoder.layer.{}.attention.self.key.weight".to_string(),
                k_bias: Some("encoder.layer.{}.attention.self.key.bias".to_string()),
                v_weight: "encoder.layer.{}.attention.self.value.weight".to_string(),
                v_bias: Some("encoder.layer.{}.attention.self.value.bias".to_string()),
                o_weight: "encoder.layer.{}.attention.output.dense.weight".to_string(),
                o_bias: Some("encoder.layer.{}.attention.output.dense.bias".to_string()),
                norm_weight: "encoder.layer.{}.attention.output.LayerNorm.weight".to_string(),
                norm_bias: Some("encoder.layer.{}.attention.output.LayerNorm.bias".to_string()),
            },
            ffn: FeedForwardLayout {
                up_weight: "encoder.layer.{}.intermediate.dense.weight".to_string(),
                up_bias: Some("encoder.layer.{}.intermediate.dense.bias".to_string()),
                down_weight: "encoder.layer.{}.output.dense.weight".to_string(),
                down_bias: Some("encoder.layer.{}.output.dense.bias".to_string()),
                gate_weight: None, // BERT uses GeLU(Up) -> Down, no Gate
                norm_weight: "encoder.layer.{}.output.LayerNorm.weight".to_string(),
                norm_bias: Some("encoder.layer.{}.output.LayerNorm.bias".to_string()),
            },
        };

        ModelLayout {
            token_embedding: "embeddings.word_embeddings.weight".to_string(),
            lm_head: "cls.predictions.decoder.weight".to_string(), // For MLM, or classifier head
            encoder: Some(EncoderLayout {
                position_embedding: Some("embeddings.position_embeddings.weight".to_string()),
                token_type_embedding: Some("embeddings.token_type_embeddings.weight".to_string()),
                embedding_norm_weight: Some("embeddings.LayerNorm.weight".to_string()),
                embedding_norm_bias: Some("embeddings.LayerNorm.bias".to_string()),
                final_norm_weight: None, // BERT usually has no final norm after last layer
                final_norm_bias: None,
                layer: encoder_layer,
            }),
            decoder: None,
        }
    }
}

// ============================================================================
// 2. MPNet Configuration
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
pub struct MpnetConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub layer_norm_eps: f32,
}

impl MpnetConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn from_loader(_loader: &dyn WeightLoader, config_json: Option<&str>) -> Result<Arc<Self>> {
        // MPNet GGUF support is rare, prioritizing JSON path
        let json_str = config_json.context("Config JSON required for MPNet")?;
        Ok(Arc::new(Self::from_json(json_str)?))
    }
}

impl ModelConfig for MpnetConfig {
    fn model_type(&self) -> &str { "mpnet" }

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
            normalize_embedding: false,
            extra_pos_embeddings: 0,
            is_prenorm: false,
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
        }
    }

    fn layout(&self) -> ModelLayout {
        // MPNet uses 'mpnet.encoder' prefix
        let encoder_layer = EncoderLayerLayout {
            self_attn: AttentionLayout {
                q_weight: "mpnet.encoder.layer.{}.attention.attn.q.weight".to_string(),
                q_bias: Some("mpnet.encoder.layer.{}.attention.attn.q.bias".to_string()),
                k_weight: "mpnet.encoder.layer.{}.attention.attn.k.weight".to_string(),
                k_bias: Some("mpnet.encoder.layer.{}.attention.attn.k.bias".to_string()),
                v_weight: "mpnet.encoder.layer.{}.attention.attn.v.weight".to_string(),
                v_bias: Some("mpnet.encoder.layer.{}.attention.attn.v.bias".to_string()),
                o_weight: "mpnet.encoder.layer.{}.attention.attn.o.weight".to_string(),
                o_bias: Some("mpnet.encoder.layer.{}.attention.attn.o.bias".to_string()),
                norm_weight: "mpnet.encoder.layer.{}.attention.LayerNorm.weight".to_string(),
                norm_bias: Some("mpnet.encoder.layer.{}.attention.LayerNorm.bias".to_string()),
            },
            ffn: FeedForwardLayout {
                up_weight: "mpnet.encoder.layer.{}.ffn.intermediate.weight".to_string(),
                up_bias: Some("mpnet.encoder.layer.{}.ffn.intermediate.bias".to_string()),
                down_weight: "mpnet.encoder.layer.{}.ffn.output.weight".to_string(),
                down_bias: Some("mpnet.encoder.layer.{}.ffn.output.bias".to_string()),
                gate_weight: None,
                norm_weight: "mpnet.encoder.layer.{}.LayerNorm.weight".to_string(),
                norm_bias: Some("mpnet.encoder.layer.{}.LayerNorm.bias".to_string()),
            },
        };

        ModelLayout {
            token_embedding: "embeddings.word_embeddings.weight".to_string(),
            lm_head: "classifier.weight".to_string(),
            encoder: Some(EncoderLayout {
                position_embedding: Some("embeddings.position_embeddings.weight".to_string()),
                token_type_embedding: Some("embeddings.token_type_embeddings.weight".to_string()),
                embedding_norm_weight: Some("embeddings.LayerNorm.weight".to_string()),
                embedding_norm_bias: Some("embeddings.LayerNorm.bias".to_string()),
                final_norm_weight: Some("mpnet.pooler.dense.weight".to_string()),
                final_norm_bias: Some("mpnet.pooler.dense.bias".to_string()),
                layer: encoder_layer,
            }),
            decoder: None,
        }
    }
}

// ============================================================================
// 3. DistilBERT Configuration
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
pub struct DistilBertConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub hidden_dim: usize,
    pub activation: String,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
}

impl DistilBertConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn from_loader(loader: &dyn WeightLoader, config_json: Option<&str>) -> Result<Arc<Self>> {
        if loader.has_metadata() {
            let arch = "distilbert"; // GGUF arch name
            let get_u32 = |k: &str| loader.get_u32(&format!("{}.{}", arch, k));
            
            // Note: DistilBERT GGUF mapping is less standardized, strict fallback recommended
            Ok(Arc::new(Self {
                dim: get_u32("embedding_length").context("no embedding_length")? as usize,
                n_layers: get_u32("block_count").context("no block_count")? as usize,
                n_heads: get_u32("attention.head_count").context("no head_count")? as usize,
                hidden_dim: get_u32("feed_forward_length").unwrap_or(0) as usize,
                activation: "gelu".to_string(),
                max_position_embeddings: get_u32("context_length").unwrap_or(512) as usize,
                vocab_size: loader.get_u32("general.vocabulary_size").unwrap_or(30522) as usize,
            }))
        } else {
            let json_str = config_json.context("Config JSON required for DistilBERT")?;
            Ok(Arc::new(Self::from_json(json_str)?))
        }
    }
}

impl ModelConfig for DistilBertConfig {
    fn model_type(&self) -> &str { "distilbert" }

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
            activation: Activation::Gelu,
            rope_theta: None,
            rope_scaling: None,
            scale_embeddings: false,
            normalize_embedding: false,
            extra_pos_embeddings: 0,
            is_prenorm: false,
            transpose_ffn_weights: true,
            transpose_attention_weights: false,
        }
    }

    fn layout(&self) -> ModelLayout {
        // DistilBERT uses 'distilbert.transformer' prefix and 'lin' suffixes
        let encoder_layer = EncoderLayerLayout {
            self_attn: AttentionLayout {
                q_weight: "distilbert.transformer.layer.{}.attention.q_lin.weight".to_string(),
                q_bias: Some("distilbert.transformer.layer.{}.attention.q_lin.bias".to_string()),
                k_weight: "distilbert.transformer.layer.{}.attention.k_lin.weight".to_string(),
                k_bias: Some("distilbert.transformer.layer.{}.attention.k_lin.bias".to_string()),
                v_weight: "distilbert.transformer.layer.{}.attention.v_lin.weight".to_string(),
                v_bias: Some("distilbert.transformer.layer.{}.attention.v_lin.bias".to_string()),
                o_weight: "distilbert.transformer.layer.{}.attention.out_lin.weight".to_string(),
                o_bias: Some("distilbert.transformer.layer.{}.attention.out_lin.bias".to_string()),
                norm_weight: "distilbert.transformer.layer.{}.sa_layer_norm.weight".to_string(),
                norm_bias: Some("distilbert.transformer.layer.{}.sa_layer_norm.bias".to_string()),
            },
            ffn: FeedForwardLayout {
                up_weight: "distilbert.transformer.layer.{}.ffn.lin1.weight".to_string(),
                up_bias: Some("distilbert.transformer.layer.{}.ffn.lin1.bias".to_string()),
                down_weight: "distilbert.transformer.layer.{}.ffn.lin2.weight".to_string(),
                down_bias: Some("distilbert.transformer.layer.{}.ffn.lin2.bias".to_string()),
                gate_weight: None,
                norm_weight: "distilbert.transformer.layer.{}.output_layer_norm.weight".to_string(),
                norm_bias: Some("distilbert.transformer.layer.{}.output_layer_norm.bias".to_string()),
            },
        };

        ModelLayout {
            token_embedding: "distilbert.embeddings.word_embeddings.weight".to_string(),
            lm_head: "qa_outputs.weight".to_string(), // Fallback head
            encoder: Some(EncoderLayout {
                position_embedding: Some("distilbert.embeddings.position_embeddings.weight".to_string()),
                token_type_embedding: None,
                embedding_norm_weight: Some("distilbert.embeddings.LayerNorm.weight".to_string()),
                embedding_norm_bias: Some("distilbert.embeddings.LayerNorm.bias".to_string()),
                final_norm_weight: None,
                final_norm_bias: None,
                layer: encoder_layer,
            }),
            decoder: None,
        }
    }
}