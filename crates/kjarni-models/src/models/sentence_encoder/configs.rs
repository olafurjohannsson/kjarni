use anyhow::{Context, Result};
use kjarni_transformers::{
    activations::Activation,
    cpu::encoder::classifier::ClassificationHeadLayout,
    traits::{
        AttentionLayout, EncoderLayerLayout, EncoderLayout, FeedForwardLayout, ModelConfig,
        ModelLayout, ModelMetadata, NormalizationStrategy,
    },
    weights::WeightLoader,
};
use serde::Deserialize;
use std::{collections::HashMap, sync::Arc};
// ============================================================================
// 1. BERT Configuration (MiniLM, Nomic, etc.)
// ============================================================================
#[derive(Debug, Clone, Deserialize)]
pub struct BertConfig {
    // --- Aliases handle Nomic-BERT's variable names ---
    #[serde(alias = "n_embd")]
    pub hidden_size: usize,

    #[serde(alias = "n_layer")]
    pub num_hidden_layers: usize,

    #[serde(alias = "n_head")]
    pub num_attention_heads: usize,

    #[serde(alias = "n_inner")]
    pub intermediate_size: usize,
    // Classification labels from config.json
    #[serde(default)]
    pub id2label: Option<HashMap<String, String>>,
    #[serde(default)]
    pub label2id: Option<HashMap<String, usize>>,
    #[serde(default)]
    pub num_labels: Option<usize>,
    #[serde(alias = "hidden_act")]
    pub activation_function: Option<String>,
    // Computed: ordered labels (NOT from JSON, computed in from_json)
    #[serde(skip)]
    pub labels_vec: Option<Vec<String>>,
    // --- FIX: Separate fields to avoid "duplicate field" error ---
    // Standard BERT
    pub max_position_embeddings: Option<usize>,
    // Nomic / Mosaic BERT
    pub n_positions: Option<usize>,

    pub vocab_size: usize,

    #[serde(alias = "layer_norm_epsilon")]
    pub layer_norm_eps: f32,

    #[serde(default)]
    pub type_vocab_size: usize,

    // --- Nomic Specifics ---
    pub model_type: Option<String>, // "bert" or "nomic_bert"

    #[serde(alias = "rotary_emb_fraction")]
    pub rotary_embedding_fraction: Option<f32>,
    #[serde(alias = "rotary_emb_base")]
    pub rotary_embedding_base: Option<f32>,

    #[serde(default = "default_true")]
    pub qkv_proj_bias: bool,
    #[serde(default = "default_true")]
    pub mlp_fc1_bias: bool,
    #[serde(default = "default_true")]
    pub mlp_fc2_bias: bool,
}

fn default_true() -> bool {
    true
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
                num_attention_heads: get_u32("attention.head_count").context("no head_count")?
                    as usize,
                intermediate_size: get_u32("feed_forward_length").unwrap_or(0) as usize,
                activation_function: Some(
                    loader
                        .get_string(&format!("{}.feed_forward_activation", arch))
                        .unwrap_or("gelu")
                        .to_string(),
                ),
                // GGUF provides 'context_length', map it to standard field
                max_position_embeddings: Some(get_u32("context_length").unwrap_or(2048) as usize),
                n_positions: None,
                vocab_size: loader.get_u32("general.vocabulary_size").unwrap_or(30522) as usize,
                layer_norm_eps: get_f32("attention.layer_norm_epsilon").unwrap_or(1e-12),
                type_vocab_size: 2,
                model_type: Some(arch.to_string()),
                rotary_embedding_fraction: get_f32("rope.freq_scale").or(None),
                rotary_embedding_base: get_f32("rope.freq_base").or(None),
                qkv_proj_bias: get_f32("attention.qkv_bias").unwrap_or(1.0) > 0.0,
                mlp_fc1_bias: true,
                mlp_fc2_bias: true,
                id2label: None,
                label2id: None,
                labels_vec: None,
                num_labels: None,
            }))
        } else {
            let json_str = config_json.context("Config JSON required for SafeTensors")?;
            Ok(Arc::new(Self::from_json(json_str)?))
        }
    }

    fn is_nomic(&self) -> bool {
        self.model_type.as_deref() == Some("nomic_bert")
    }

    /// Helper to resolve the context length from conflicting fields
    fn get_max_seq_len(&self) -> usize {
        self.n_positions
            .or(self.max_position_embeddings)
            .unwrap_or(512)
    }
}

impl ModelConfig for BertConfig {
    fn model_type(&self) -> &str {
        if self.is_nomic() {
            "nomic_bert"
        } else {
            "bert"
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn id2label(&self) -> Option<&[String]> {
        self.labels_vec.as_deref()
    }
    fn metadata(&self) -> ModelMetadata {
        // Nomic sets rotary_emb_fraction (usually 1.0) if enabled
        let uses_rope =
            self.rotary_embedding_fraction.is_some() || self.rotary_embedding_base.is_some();

        let rope_theta = if uses_rope {
            Some(self.rotary_embedding_base.unwrap_or(10000.0))
        } else {
            None
        };

        ModelMetadata {
            hidden_size: self.hidden_size,
            num_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_attention_heads,
            head_dim: self.hidden_size / self.num_attention_heads,
            vocab_size: self.vocab_size,
            // Use helper to resolve max len
            max_seq_len: self.get_max_seq_len(),
            norm_eps: self.layer_norm_eps,
            activation: match self.activation_function.as_deref() {
                Some("swiglu") => Activation::SilU, // Nomic
                Some("gelu") => Activation::Gelu,
                Some("gelu_new") => Activation::GeluNew,
                Some("relu") => Activation::Relu,
                _ => Activation::Gelu,
            },
            decoder_layers: None,
            rope_theta,
            rope_scaling: None,

            scale_embeddings: false,
            normalize_embedding: false,
            extra_pos_embeddings: 0,
            is_prenorm: false,
            // Nomic (SwiGlu) weights usually match LinearLayer expectation better
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            normalization_strategy: NormalizationStrategy::LayerNorm,
            no_scale_qk: false,
        }
    }

    fn layout(&self) -> ModelLayout {
        if self.is_nomic() {
            let prefix = "encoder.layers.{}";

            let encoder_layer = EncoderLayerLayout {
                self_attn: AttentionLayout {
                    // GPT-2 Style: Point all to the fused tensor.
                    // The loader will detect this and slice.
                    q_weight: format!("{}.attn.Wqkv.weight", prefix),
                    // Pointing K/V to the same file signals "It's fused"
                    k_weight: format!("{}.attn.Wqkv.weight", prefix),
                    v_weight: format!("{}.attn.Wqkv.weight", prefix),

                    // Nomic has NO bias for QKV
                    q_bias: None,
                    k_bias: None,
                    v_bias: None,

                    o_weight: format!("{}.attn.out_proj.weight", prefix),
                    o_bias: None, // Verified from your list: no bias for out_proj

                    norm_weight: format!("{}.norm1.weight", prefix),
                    norm_bias: Some(format!("{}.norm1.bias", prefix)),
                },
                ffn: FeedForwardLayout {
                    // SwiGLU: fc11 = Gate, fc12 = Up (Standard split)
                    gate_weight: Some(format!("{}.mlp.fc11.weight", prefix)),
                    up_weight: format!("{}.mlp.fc12.weight", prefix),
                    down_weight: format!("{}.mlp.fc2.weight", prefix),

                    up_bias: None,
                    down_bias: None,
                    gate_bias: None,

                    norm_weight: format!("{}.norm2.weight", prefix),
                    norm_bias: Some(format!("{}.norm2.bias", prefix)),
                },
            };

            ModelLayout {
                token_embedding: "embeddings.word_embeddings.weight".to_string(),
                lm_head: "embeddings.word_embeddings.weight".to_string(),
                encoder: Some(EncoderLayout {
                    position_embedding: None,
                    token_type_embedding: Some(
                        "embeddings.token_type_embeddings.weight".to_string(),
                    ),
                    embedding_norm_weight: Some("emb_ln.weight".to_string()), // Updated key
                    embedding_norm_bias: Some("emb_ln.bias".to_string()),     // Updated key
                    final_norm_weight: None,
                    final_norm_bias: None,
                    layer: encoder_layer,
                }),
                decoder: None,
            }
        } else {
            // === STANDARD BERT LAYOUT (MiniLM) ===
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
                    gate_weight: None,
                    gate_bias: None,
                    norm_weight: "encoder.layer.{}.output.LayerNorm.weight".to_string(),
                    norm_bias: Some("encoder.layer.{}.output.LayerNorm.bias".to_string()),
                },
            };

            ModelLayout {
                token_embedding: "embeddings.word_embeddings.weight".to_string(),
                lm_head: "cls.predictions.decoder.weight".to_string(),
                encoder: Some(EncoderLayout {
                    position_embedding: Some("embeddings.position_embeddings.weight".to_string()),
                    token_type_embedding: Some(
                        "embeddings.token_type_embeddings.weight".to_string(),
                    ),
                    embedding_norm_weight: Some("embeddings.LayerNorm.weight".to_string()),
                    embedding_norm_bias: Some("embeddings.LayerNorm.bias".to_string()),
                    final_norm_weight: None,
                    final_norm_bias: None,
                    layer: encoder_layer,
                }),
                decoder: None,
            }
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
    fn model_type(&self) -> &str {
        "mpnet"
    }
fn as_any(&self) -> &dyn std::any::Any {
        self
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
            decoder_layers: None,
            rope_scaling: None,
            scale_embeddings: false,
            normalize_embedding: false,
            extra_pos_embeddings: 0,
            is_prenorm: false,
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            normalization_strategy: NormalizationStrategy::LayerNorm,
            no_scale_qk: false,
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
                gate_bias: None,
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

/// DistilBERT configuration supporting both base and sequence classification variants.
#[derive(Debug, Clone, Deserialize)]
pub struct DistilBertConfig {
    // Core architecture
    pub activation: String,
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,

    // Dropout (optional, for reference)
    #[serde(default)]
    pub attention_dropout: Option<f32>,
    #[serde(default)]
    pub dropout: Option<f32>,
    #[serde(default)]
    pub seq_classif_dropout: Option<f32>,

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

    // Architecture detection
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
    #[serde(default)]
    pub finetuning_task: Option<String>,

    // Other optional fields
    #[serde(default)]
    pub pad_token_id: Option<usize>,
    #[serde(default)]
    pub sinusoidal_pos_embds: Option<bool>,
    #[serde(default)]
    pub tie_weights_: Option<bool>,
}

impl DistilBertConfig {
    /// Get the classification head layout for DistilBERT sequence classification.
    pub fn classification_head_layout(&self) -> ClassificationHeadLayout {
        ClassificationHeadLayout {
            // DistilBERT has a pre_classifier layer before the final classifier
            pre_classifier_weight: Some("pre_classifier.weight".to_string()),
            pre_classifier_bias: Some("pre_classifier.bias".to_string()),
            classifier_weight: "classifier.weight".to_string(),
            classifier_bias: Some("classifier.bias".to_string()),
        }
    }
    pub fn from_json(json: &str) -> Result<Self> {
        let mut config: Self = serde_json::from_str(json)?;

        // Convert id2label HashMap to sorted Vec
        if let Some(ref map) = config.id2label {
            let mut labels: Vec<(usize, String)> = map
                .iter()
                .filter_map(|(k, v)| k.parse::<usize>().ok().map(|idx| (idx, v.clone())))
                .collect();
            labels.sort_by_key(|(idx, _)| *idx);
            config.labels_vec = Some(labels.into_iter().map(|(_, v)| v).collect());

            // Infer num_labels if not set
            if config.num_labels.is_none() {
                config.num_labels = Some(config.labels_vec.as_ref().unwrap().len());
            }
        }

        Ok(config)
    }

    pub fn from_loader(loader: &dyn WeightLoader, config_json: Option<&str>) -> Result<Arc<Self>> {
        if loader.has_metadata() {
            let arch = "distilbert";
            let get_u32 = |k: &str| loader.get_u32(&format!("{}.{}", arch, k));

            Ok(Arc::new(Self {
                dim: get_u32("embedding_length").context("no embedding_length")? as usize,
                n_layers: get_u32("block_count").context("no block_count")? as usize,
                n_heads: get_u32("attention.head_count").context("no head_count")? as usize,
                hidden_dim: get_u32("feed_forward_length").unwrap_or(3072) as usize,
                activation: "gelu".to_string(),
                max_position_embeddings: get_u32("context_length").unwrap_or(512) as usize,
                vocab_size: loader.get_u32("general.vocabulary_size").unwrap_or(30522) as usize,
                attention_dropout: None,
                dropout: None,
                seq_classif_dropout: None,
                id2label: None,
                label2id: None,
                num_labels: None,
                labels_vec: None,
                architectures: None,
                finetuning_task: None,
                pad_token_id: Some(0),
                sinusoidal_pos_embds: Some(false),
                tie_weights_: None,
            }))
        } else {
            let json_str = config_json.context("Config JSON required for DistilBERT")?;
            Ok(Arc::new(Self::from_json(json_str)?))
        }
    }

    /// Check if this is a sequence classification model.
    pub fn is_sequence_classifier(&self) -> bool {
        self.architectures
            .as_ref()
            .map(|archs| archs.iter().any(|a| a.contains("SequenceClassification")))
            .unwrap_or(false)
            || self.id2label.is_some()
            || self.num_labels.is_some()
    }

    /// Get the number of classification labels.
    pub fn get_num_labels(&self) -> Option<usize> {
        self.num_labels
            .or_else(|| self.labels_vec.as_ref().map(|v| v.len()))
            .or_else(|| self.id2label.as_ref().map(|m| m.len()))
    }

    /// Get ordered label names.
    pub fn get_labels(&self) -> Option<&[String]> {
        self.labels_vec.as_deref()
    }
}

impl ModelConfig for DistilBertConfig {
    fn model_type(&self) -> &str {
        "distilbert"
    }
fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn id2label(&self) -> Option<&[String]> {
        self.labels_vec.as_deref()
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
            activation: Activation::Gelu,
            rope_theta: None,
            rope_scaling: None,
            decoder_layers: None,
            scale_embeddings: false,
            normalize_embedding: false,
            extra_pos_embeddings: 0,
            is_prenorm: false,
            transpose_ffn_weights: true,
            transpose_attention_weights: false,
            normalization_strategy: NormalizationStrategy::LayerNorm,
            no_scale_qk: false,
        }
    }

    fn layout(&self) -> ModelLayout {
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
                gate_bias: None,
                norm_weight: "distilbert.transformer.layer.{}.output_layer_norm.weight".to_string(),
                norm_bias: Some(
                    "distilbert.transformer.layer.{}.output_layer_norm.bias".to_string(),
                ),
            },
        };

        ModelLayout {
            token_embedding: "distilbert.embeddings.word_embeddings.weight".to_string(),
            // For sequence classification, classifier.weight is the "head"
            lm_head: if self.is_sequence_classifier() {
                "classifier.weight".to_string()
            } else {
                "qa_outputs.weight".to_string() // QA fallback
            },
            encoder: Some(EncoderLayout {
                position_embedding: Some(
                    "distilbert.embeddings.position_embeddings.weight".to_string(),
                ),
                token_type_embedding: None, // DistilBERT doesn't use token type embeddings
                embedding_norm_weight: Some("distilbert.embeddings.LayerNorm.weight".to_string()),
                embedding_norm_bias: Some("distilbert.embeddings.LayerNorm.bias".to_string()),
                final_norm_weight: None, // No final norm in DistilBERT
                final_norm_bias: None,
                layer: encoder_layer,
            }),
            decoder: None,
        }
    }
}
