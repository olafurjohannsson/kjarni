use anyhow::{Context, Result};
use kjarni_transformers::models::base::RopeScalingConfig;
use kjarni_transformers::traits::NormalizationStrategy;
use kjarni_transformers::{
    activations::Activation,
    traits::{
        AttentionLayout, DecoderLayerLayout, DecoderLayout, FeedForwardLayout, ModelConfig,
        ModelLayout, ModelMetadata,
    },
    weights::WeightLoader,
};
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use serde::Deserialize;
use std::sync::Arc;

// --- Helper Deserializers (Standard) ---
fn deserialize_token_id<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: Deserializer<'de>,
{
    struct TokenIdVisitor;
    impl<'de> Visitor<'de> for TokenIdVisitor {
        type Value = u32;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("u32/array") }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> { Ok(v as u32) }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> { Ok(v as u32) }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let first = seq.next_element::<u32>()?.ok_or_else(|| de::Error::custom("empty"))?;
            while seq.next_element::<u32>()?.is_some() {}
            Ok(first)
        }
    }
    deserializer.deserialize_any(TokenIdVisitor)
}

fn deserialize_token_ids<'de, D>(deserializer: D) -> Result<Vec<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    struct TokenIdsVisitor;
    impl<'de> Visitor<'de> for TokenIdsVisitor {
        type Value = Vec<u32>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("u32/array") }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> { Ok(vec![v as u32]) }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> { Ok(vec![v as u32]) }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut ids = Vec::new();
            while let Some(id) = seq.next_element::<u32>()? { ids.push(id); }
            Ok(ids)
        }
    }
    deserializer.deserialize_any(TokenIdsVisitor)
}

// --- Defaults ---
fn default_rms_norm_eps() -> f32 { 1e-5 }
fn default_hidden_act() -> String { "silu".to_string() }
fn default_true() -> bool { true }

#[derive(Debug, Clone, Deserialize)]
pub struct MistralConfig {
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

    pub rope_theta: Option<f32>,
    pub rope_scaling: Option<RopeScalingConfig>,
    pub head_dim: Option<usize>,
    pub sliding_window: Option<usize>, // Specific to Mistral

    #[serde(deserialize_with = "deserialize_token_id")]
    pub bos_token_id: u32,
    #[serde(deserialize_with = "deserialize_token_ids")]
    pub eos_token_id: Vec<u32>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,

    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: String,

    #[serde(default)] // Mistral usually No Bias
    pub attention_bias: bool,
    #[serde(default = "default_true")]
    pub use_cache: bool,
}

impl MistralConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn from_loader(loader: &dyn WeightLoader, config_json: Option<&str>) -> Result<Arc<Self>> {
        if loader.has_metadata() {
            // Mistral GGUF uses "llama" architecture keys usually
            let arch = loader.get_string("general.architecture").unwrap_or("llama");

            let get_u32 = |k: &str| loader.get_u32(&format!("{}.{}", arch, k));
            let get_f32 = |k: &str| loader.get_f32(&format!("{}.{}", arch, k));

            let hidden_size = get_u32("embedding_length").context("Missing embedding_length")? as usize;
            let n_heads = get_u32("attention.head_count").context("Missing head_count")? as usize;

            let rope_type = loader.get_string(&format!("{}.rope.scaling.type", arch));
            let rope_scaling = rope_type.map(|rtype| RopeScalingConfig {
                rope_type: rtype.to_string(),
                factor: get_f32("rope.scaling.factor").unwrap_or(1.0),
                low_freq_factor: get_f32("rope.scaling.low_freq_factor").unwrap_or(1.0),
                high_freq_factor: get_f32("rope.scaling.high_freq_factor").unwrap_or(1.0),
                original_max_position_embeddings: get_u32("rope.scaling.orig_ctx_len")
                    .map(|v| v as usize)
                    .unwrap_or(32768),
            });

            Ok(Arc::new(Self {
                hidden_size,
                num_attention_heads: n_heads,
                num_hidden_layers: get_u32("block_count").context("Missing block_count")? as usize,
                num_key_value_heads: get_u32("attention.head_count_kv").unwrap_or(n_heads as u32) as usize,
                intermediate_size: get_u32("feed_forward_length").unwrap_or(14336) as usize, // Mistral 7B default
                vocab_size: loader.get_u32("general.vocabulary_size").unwrap_or(32000) as usize,
                // Mistral 0.3 is 32k, 0.1 is 32k/4k
                max_position_embeddings: get_u32("context_length").unwrap_or(32768) as usize,
                rms_norm_eps: get_f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
                hidden_act: loader.get_string(&format!("{}.feed_forward_activation", arch)).unwrap_or("silu").to_string(),

                // Mistral v0.1 was 10000.0, v0.3 often 1000000.0
                rope_theta: Some(get_f32("rope.freq_base").unwrap_or(1000000.0)), // Default for v0.3
                rope_scaling,

                sliding_window: get_u32("attention.sliding_window").map(|v| v as usize),
                head_dim: get_u32("attention.head_dim").map(|v| v as usize),

                bos_token_id: 1, // Mistral standard
                eos_token_id: vec![2],
                pad_token_id: None,

                architectures: vec!["MistralForCausalLM".to_string()],
                model_type: "mistral".to_string(),
                attention_bias: false,
                use_cache: true,
            }))
        } else {
            let json_str = config_json.context("Safetensors requires config.json")?;
            Ok(Arc::new(Self::from_json(json_str)?))
        }
    }
}

impl ModelConfig for MistralConfig {
    fn model_type(&self) -> &str { "mistral" }
fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            hidden_size: self.hidden_size,
            num_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads),
            vocab_size: self.vocab_size,
            max_seq_len: self.max_position_embeddings,
            norm_eps: self.rms_norm_eps,
            activation: Activation::SilU,
            rope_theta: self.rope_theta.or(Some(10000.0)),
            rope_scaling: self.rope_scaling.clone(),
            scale_embeddings: false,
            normalize_embedding: false,
            extra_pos_embeddings: 0,
            is_prenorm: true,
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            normalization_strategy: NormalizationStrategy::RMSNorm,
            no_scale_qk: false,
            decoder_layers: None,
        }
    }

    fn layout(&self) -> ModelLayout {
        // Mistral layout is identical to Llama
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
            cross_attn: None,
            ffn: FeedForwardLayout {
                up_weight: "model.layers.{}.mlp.up_proj.weight".to_string(),
                up_bias: None,

                down_weight: "model.layers.{}.mlp.down_proj.weight".to_string(),
                down_bias: None,
                gate_weight: Some("model.layers.{}.mlp.gate_proj.weight".to_string()),
                gate_bias: None,
                norm_weight: "model.layers.{}.post_attention_layernorm.weight".to_string(),
                norm_bias: None,
            },
        };

        ModelLayout {
            token_embedding: "model.embed_tokens.weight".to_string(),
            lm_head: "lm_head.weight".to_string(),
            encoder: None,
            decoder: Some(DecoderLayout {
                position_embedding: None,
                token_type_embedding: None,
                embedding_norm_weight: None,
                embedding_norm_bias: None,
                final_norm_weight: Some("model.norm.weight".to_string()),
                final_norm_bias: None,
                layer: decoder_layer,
            }),
        }
    }
}