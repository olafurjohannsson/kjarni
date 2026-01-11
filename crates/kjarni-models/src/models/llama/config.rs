use anyhow::{Context, Result};
use kjarni_transformers::models::base::RopeScalingConfig;
use kjarni_transformers::traits::NormalizationStrategy;
use kjarni_transformers::{
    activations::Activation,
    traits::{AttentionLayout, DecoderLayerLayout, DecoderLayout, FeedForwardLayout, ModelConfig, ModelLayout, ModelMetadata},
    weights::WeightLoader,
};
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::Deserialize;
use std::sync::Arc;

fn deserialize_token_id<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: Deserializer<'de>,
{
    struct TokenIdVisitor;

    impl<'de> Visitor<'de> for TokenIdVisitor {
        type Value = u32;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("u32 or array of u32")
        }

        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
            Ok(v as u32)
        }

        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
            Ok(v as u32)
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let first = seq.next_element::<u32>()?
                .ok_or_else(|| de::Error::custom("empty token_id array"))?;
            while seq.next_element::<u32>()?.is_some() {}
            Ok(first)
        }
    }

    deserializer.deserialize_any(TokenIdVisitor)
}

// New deserializer that returns ALL token IDs
fn deserialize_token_ids<'de, D>(deserializer: D) -> Result<Vec<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    struct TokenIdsVisitor;

    impl<'de> Visitor<'de> for TokenIdsVisitor {
        type Value = Vec<u32>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("u32 or array of u32")
        }

        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
            Ok(vec![v as u32])
        }

        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
            Ok(vec![v as u32])
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut ids = Vec::new();
            while let Some(id) = seq.next_element::<u32>()? {
                ids.push(id);
            }
            if ids.is_empty() {
                return Err(de::Error::custom("empty token_id array"));
            }
            Ok(ids)
        }
    }

    deserializer.deserialize_any(TokenIdsVisitor)
}


#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum TokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

impl TokenId {
    pub fn first(&self) -> u32 {
        match self {
            TokenId::Single(id) => *id,
            TokenId::Multiple(ids) => ids.first().copied().unwrap_or(0),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    // --- RoPE (Rotary Position Embedding) ---
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScalingConfig>,
    pub head_dim: Option<usize>,

    // --- Token IDs ---
    #[serde(deserialize_with = "deserialize_token_id")]
    pub bos_token_id: u32,
    #[serde(deserialize_with = "deserialize_token_ids")]
    pub eos_token_id: Vec<u32>,

    #[serde(default)]
    pub pad_token_id: Option<u32>,

    // --- Critical Logic Flags ---
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    // --- Identity & Metadata ---
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub torch_dtype: Option<String>,

    // --- Execution Hints ---
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f32,
    #[serde(default = "default_true")]
    pub use_cache: bool,
}

// Default functions for Serde
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    500000.0
}
fn default_hidden_act() -> String {
    "silu".to_string()
}
fn default_tie_word_embeddings() -> bool {
    true
}
fn default_true() -> bool {
    true
}

impl LlamaConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
    pub fn get_kv_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
            * 8
    }

    /// Helper to get the head dimension, falling back to calculation if not explicitly set
    pub fn get_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn from_loader(loader: &dyn WeightLoader, config_json: Option<&str>) -> Result<Arc<Self>> {
        if loader.has_metadata() {
            let arch = loader.get_string("general.architecture").unwrap_or("llama");
            let get_u32 = |k: &str| loader.get_u32(&format!("{}.{}", arch, k));
            let get_f32 = |k: &str| loader.get_f32(&format!("{}.{}", arch, k));

            let hidden_size =
                get_u32("embedding_length").context("Missing embedding_length")? as usize;
            let n_heads = get_u32("attention.head_count").context("Missing head_count")? as usize;

            // Handle Llama 3.2 Scaling
            let rope_type = loader.get_string(&format!("{}.rope.scaling.type", arch));
            let rope_scaling = rope_type.map(|rtype| RopeScalingConfig {
                rope_type: rtype.to_string(),
                factor: get_f32("rope.scaling.factor").unwrap_or(1.0),
                low_freq_factor: get_f32("rope.scaling.low_freq_factor").unwrap_or(1.0),
                high_freq_factor: get_f32("rope.scaling.high_freq_factor").unwrap_or(1.0),
                original_max_position_embeddings: get_u32("rope.scaling.orig_ctx_len")
                    .map(|v| v as usize)
                    .unwrap_or(8192),
            });

            Ok(Arc::new(Self {
                hidden_size,
                num_attention_heads: n_heads,
                num_hidden_layers: get_u32("block_count").context("Missing block_count")? as usize,
                num_key_value_heads: get_u32("attention.head_count_kv")
                    .unwrap_or(n_heads as u32 / 4) as usize,
                intermediate_size: get_u32("feed_forward_length").unwrap_or(hidden_size as u32 * 4)
                    as usize,
                vocab_size: loader.get_u32("general.vocabulary_size").unwrap_or(128256) as usize,
                max_position_embeddings: get_u32("context_length").unwrap_or(2048) as usize,
                rms_norm_eps: get_f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
                hidden_act: loader
                    .get_string(&format!("{}.feed_forward_activation", arch))
                    .unwrap_or("silu")
                    .to_string(),
                rope_theta: get_f32("rope.freq_base").unwrap_or(500000.0),
                bos_token_id: 128000,
                eos_token_id: vec![128001, 128008, 128009],
                pad_token_id: None,
                tie_word_embeddings: !loader.contains("output.weight"),
                rope_scaling,
                head_dim: get_u32("attention.head_dim").map(|v| v as usize),
                architectures: vec!["LlamaForCausalLM".to_string()],
                model_type: arch.to_string(),
                torch_dtype: Some("bfloat16".to_string()),
                attention_bias: false,
                attention_dropout: 0.0,
                use_cache: true,
            }))
        } else {
            let json_str = config_json.context("Safetensors requires config.json")?;
            Ok(Arc::new(Self::from_json(json_str)?))
        }
    }
}

impl ModelConfig for LlamaConfig {
    fn model_type(&self) -> &str {
        "llama"
    }
fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            // --- Basic Dimensions ---
            hidden_size: self.hidden_size,
            num_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            decoder_layers: None,
            // Calculate head_dim if not explicitly provided (Standard Llama logic)
            head_dim: self
                .head_dim
                .unwrap_or(self.hidden_size / self.num_attention_heads),

            vocab_size: self.vocab_size,
            max_seq_len: self.max_position_embeddings,

            // --- Math Constants ---
            norm_eps: self.rms_norm_eps,
            activation: match self.hidden_act.as_str() {
                "relu" => Activation::Relu,
                "gelu" => Activation::Gelu,
                "gelu_new" => Activation::GeluNew,
                "tanh" => Activation::Tanh,
                "silu" => Activation::SilU,
                _ => Activation::SilU, // Llama default is SiLU
            },

            // --- Positional Math (RoPE) ---
            rope_theta: Some(self.rope_theta),
            rope_scaling: self.rope_scaling.clone(),

            // --- Style Flags ---
            scale_embeddings: false, // Llama does not use sqrt(d) scaling
            normalize_embedding: false,
            extra_pos_embeddings: 0, // Llama has no position offset
            is_prenorm: true,        // Llama uses Pre-Normalization
            transpose_ffn_weights: false, // Standard Llama weights are [Out, In]
            transpose_attention_weights: false,
            normalization_strategy: NormalizationStrategy::RMSNorm,
            no_scale_qk: false,
        }
    }

    fn layout(&self) -> ModelLayout {
        // --- Define the Decoder's Layer Structure ---
        let decoder_layer = DecoderLayerLayout {
            self_attn: AttentionLayout {
                q_weight: "model.layers.{}.self_attn.q_proj.weight".to_string(),
                q_bias: None,
                k_weight: "model.layers.{}.self_attn.k_proj.weight".to_string(),
                k_bias: None,
                v_weight: "model.layers.{}.self_attn.v_proj.weight".to_string(),
                v_bias: None,
                o_weight: "model.layers.{}.self_attn.o_proj.weight".to_string(),
                o_bias: None,
                norm_weight: "model.layers.{}.input_layernorm.weight".to_string(),
                norm_bias: None,
            },
            cross_attn: None, // Llama is decoder-only
            ffn: FeedForwardLayout {
                up_weight: "model.layers.{}.mlp.up_proj.weight".to_string(),
                up_bias: None,

                down_weight: "model.layers.{}.mlp.down_proj.weight".to_string(),
                down_bias: None,
                gate_weight: Some("model.layers.{}.mlp.gate_proj.weight".to_string()), // Llama uses SwiGLU
                gate_bias: None,
                norm_weight: "model.layers.{}.post_attention_layernorm.weight".to_string(),
                norm_bias: None,
            },
        };

        // --- Assemble the final ModelLayout ---
        ModelLayout {
            token_embedding: "model.embed_tokens.weight".to_string(),
            lm_head: if self.tie_word_embeddings {
                "model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            }
                .to_string(),
            encoder: None, // Llama is decoder-only
            decoder: Some(DecoderLayout {
                position_embedding: None, // Llama uses RoPE, not learned positional embeddings
                token_type_embedding: None,
                embedding_norm_weight: None, // Llama has no embedding norm
                embedding_norm_bias: None,
                final_norm_weight: Some("model.norm.weight".to_string()),
                final_norm_bias: None,
                layer: decoder_layer,
            }),
        }
    }
}
