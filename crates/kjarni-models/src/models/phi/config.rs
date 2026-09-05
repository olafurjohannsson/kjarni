use anyhow::{Context, Result};
use kjarni_transformers::models::base::RopeScalingConfig;
use kjarni_transformers::{
    activations::Activation,
    traits::{
        AttentionLayout, DecoderLayerLayout, DecoderLayout, FeedForwardLayout, ModelConfig,
        ModelLayout, ModelMetadata,
    },
    weights::WeightLoader,
};
use serde::Deserialize;
use serde::de::{self, Deserializer, SeqAccess, Visitor};
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
            let first = seq
                .next_element::<u32>()?
                .ok_or_else(|| de::Error::custom("empty token_id array"))?;
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

fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    1000000.0
}
fn default_hidden_act() -> String {
    "silu".to_string()
}
fn default_tie_word_embeddings() -> bool {
    false
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct PhiConfig {
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

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScalingConfig>,
    pub head_dim: Option<usize>,

    #[serde(deserialize_with = "deserialize_token_id")]
    pub bos_token_id: u32,
    #[serde(deserialize_with = "deserialize_token_ids")]
    pub eos_token_id: Vec<u32>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,

    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub torch_dtype: Option<String>,

    #[serde(default = "default_true")] // Qwen HAS bias
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f32,
    #[serde(default = "default_true")]
    pub use_cache: bool,
}

impl PhiConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn from_loader(loader: &dyn WeightLoader, config_json: Option<&str>) -> Result<Arc<Self>> {
        if loader.has_metadata() {
            // GGUF usually stores Phi3 metadata under "phi3"
            let arch = loader.get_string("general.architecture").unwrap_or("phi3");

            // Helper to handle both "phi3.key" and "llama.key" (some converters map it)
            let get_val = |k: &str| {
                loader
                    .get_u32(&format!("{}.{}", arch, k))
                    .or_else(|| loader.get_u32(&format!("llama.{}", k)))
            };
            let get_f32_val = |k: &str| {
                loader
                    .get_f32(&format!("{}.{}", arch, k))
                    .or_else(|| loader.get_f32(&format!("llama.{}", k)))
            };

            let hidden_size =
                get_val("embedding_length").context("Missing embedding_length")? as usize;
            let n_heads = get_val("attention.head_count").context("Missing head_count")? as usize;

            // Handle RoPE Scaling
            let rope_type = loader.get_string(&format!("{}.rope.scaling.type", arch));
            let rope_scaling = rope_type.map(|rtype| RopeScalingConfig {
                rope_type: rtype.to_string(),
                factor: get_f32_val("rope.scaling.factor").unwrap_or(1.0),
                low_freq_factor: get_f32_val("rope.scaling.low_freq_factor").unwrap_or(1.0),
                high_freq_factor: get_f32_val("rope.scaling.high_freq_factor").unwrap_or(1.0),
                original_max_position_embeddings: get_val("rope.scaling.orig_ctx_len")
                    .map(|v| v as usize)
                    .unwrap_or(32768),
                long_factor: None,
                short_factor: None,
            });

            Ok(Arc::new(Self {
                hidden_size,
                num_attention_heads: n_heads,
                num_hidden_layers: get_val("block_count").context("Missing block_count")? as usize,
                num_key_value_heads: get_val("attention.head_count_kv").unwrap_or(n_heads as u32)
                    as usize,
                intermediate_size: get_val("feed_forward_length").unwrap_or(hidden_size as u32 * 4)
                    as usize,
                vocab_size: loader.get_u32("general.vocabulary_size").unwrap_or(151936) as usize,
                max_position_embeddings: get_val("context_length").unwrap_or(32768) as usize,
                rms_norm_eps: get_f32_val("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
                hidden_act: loader
                    .get_string(&format!("{}.feed_forward_activation", arch))
                    .unwrap_or("silu")
                    .to_string(),
                rope_theta: get_f32_val("rope.freq_base").unwrap_or(1000000.0),
                // Phi specific tokens (todo change to phi)
                bos_token_id: 151643, // <|endoftext|> usually serves as BOS/EOS
                eos_token_id: vec![151643, 151645], // <|endoftext|>, <|im_end|>
                pad_token_id: Some(151643),
                tie_word_embeddings: !loader.contains("output.weight"),
                rope_scaling,
                head_dim: get_val("attention.head_dim").map(|v| v as usize),
                architectures: vec!["Phi3ForCausalLM".to_string()],
                model_type: arch.to_string(),
                torch_dtype: Some("bfloat16".to_string()),
                attention_bias: true, // Phi usually has bias
                attention_dropout: 0.0,
                use_cache: true,
            }))
        } else {
            let json_str = config_json.context("Safetensors requires config.json")?;
            Ok(Arc::new(Self::from_json(json_str)?))
        }
    }
}

impl ModelConfig for PhiConfig {
    fn model_type(&self) -> &str {
        "phi3"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            hidden_size: self.hidden_size,
            num_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self
                .head_dim
                .unwrap_or(self.hidden_size / self.num_attention_heads),
            vocab_size: self.vocab_size,
            max_seq_len: self.max_position_embeddings,
            norm_eps: self.rms_norm_eps,
            activation: Activation::SilU, // Phi-3 uses SwiGLU, like Llama and Qwen
            rope_theta: Some(self.rope_theta),
            rope_scaling: self.rope_scaling.clone(),
            scale_embeddings: false,
            normalize_embedding: false,
            extra_pos_embeddings: 0,
            is_prenorm: true,
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            normalization_strategy: kjarni_transformers::traits::NormalizationStrategy::RMSNorm,
            no_scale_qk: false,
            problem_type: None,
            decoder_layers: None,
            intermediate_size: self.intermediate_size(),
        }
    }

    fn layout(&self) -> ModelLayout {
        // Defines where to find tensors in the GGUF file
        let decoder_layer = DecoderLayerLayout {
            // Phi-3 stores one fused `qkv_proj` of shape [3 * hidden, hidden] where
            // Llama and Qwen store three tensors. The `[start:end]` suffix asks the
            // loader for that row range, and rows of a [out, in] weight are output
            // features, so this is the same split the reference implementation does.
            //
            // Phi-3 is full multi-head attention: num_key_value_heads equals
            // num_attention_heads, so all three ranges are `hidden` rows wide.
            self_attn: AttentionLayout {
                q_weight: format!(
                    "model.layers.{{}}.self_attn.qkv_proj.weight[0:{}]",
                    self.hidden_size
                ),
                k_weight: format!(
                    "model.layers.{{}}.self_attn.qkv_proj.weight[{}:{}]",
                    self.hidden_size,
                    2 * self.hidden_size
                ),
                v_weight: format!(
                    "model.layers.{{}}.self_attn.qkv_proj.weight[{}:{}]",
                    2 * self.hidden_size,
                    3 * self.hidden_size
                ),
                // attention_bias is false for Phi-3; there are no bias tensors.
                q_bias: None,
                k_bias: None,
                v_bias: None,
                o_weight: "model.layers.{}.self_attn.o_proj.weight".to_string(),
                o_bias: None,
                norm_weight: "model.layers.{}.input_layernorm.weight".to_string(),
                norm_bias: None,
            },
            cross_attn: None,
            // Likewise `gate_up_proj` is one tensor of [2 * intermediate, hidden],
            // gate first and up second.
            ffn: FeedForwardLayout {
                gate_weight: Some(format!(
                    "model.layers.{{}}.mlp.gate_up_proj.weight[0:{}]",
                    self.intermediate_size
                )),
                gate_bias: None,
                up_weight: format!(
                    "model.layers.{{}}.mlp.gate_up_proj.weight[{}:{}]",
                    self.intermediate_size,
                    2 * self.intermediate_size
                ),
                up_bias: None,
                down_weight: "model.layers.{}.mlp.down_proj.weight".to_string(),
                down_bias: None,
                norm_weight: "model.layers.{}.post_attention_layernorm.weight".to_string(),
                norm_bias: None,
            },
        };

        ModelLayout {
            token_embedding: "model.embed_tokens.weight".to_string(),
            lm_head: if self.tie_word_embeddings {
                "model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            }
            .to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The parts of Phi-3.5-mini's config.json that decide the layout.
    fn phi35_json() -> String {
        r#"{
            "architectures": ["Phi3ForCausalLM"],
            "model_type": "phi3",
            "hidden_size": 3072,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 8192,
            "vocab_size": 32064,
            "max_position_embeddings": 131072,
            "original_max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu",
            "rope_theta": 10000.0,
            "rope_scaling": {
                "long_factor": [1.08, 1.11],
                "short_factor": [1.0, 1.0],
                "type": "longrope"
            },
            "bos_token_id": 1,
            "eos_token_id": 32000,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "attention_dropout": 0.0,
            "torch_dtype": "bfloat16",
            "use_cache": true
        }"#
        .to_string()
    }

    #[test]
    fn parses_phi35_config_including_longrope() {
        let cfg = PhiConfig::from_json(&phi35_json()).expect("phi config should parse");
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_hidden_layers, 32);
        // Phi-3 is full multi-head attention, not grouped-query.
        assert_eq!(cfg.num_attention_heads, cfg.num_key_value_heads);
        let scaling = cfg.rope_scaling.as_ref().expect("longrope scaling");
        assert_eq!(scaling.rope_type, "longrope");
        assert!(scaling.long_factor.is_some());
    }

    #[test]
    fn reports_itself_as_phi_not_qwen() {
        // This file began as a copy of the Qwen config and reported "qwen2" for a while,
        // which is the kind of thing nothing downstream complains about.
        let cfg = PhiConfig::from_json(&phi35_json()).unwrap();
        assert_eq!(cfg.model_type(), "phi3");
    }

    #[test]
    fn head_dim_is_derived_and_is_not_128() {
        // 3072 over 32 heads is 96. Llama and Qwen are 128, so a copied constant here
        // would be wrong in a way that still runs.
        let cfg = PhiConfig::from_json(&phi35_json()).unwrap();
        assert_eq!(cfg.metadata().head_dim, 96);
    }

    #[test]
    fn attention_layout_names_row_ranges_of_the_fused_qkv() {
        // Phi-3 ships one qkv_proj of [3 * hidden, hidden]. The three ranges must
        // partition it in q, k, v order and cover it exactly.
        let cfg = PhiConfig::from_json(&phi35_json()).unwrap();
        let layout = cfg.layout();
        let attn = &layout.decoder.as_ref().unwrap().layer.self_attn;

        assert!(attn.q_weight.contains("qkv_proj"), "{}", attn.q_weight);
        assert!(attn.q_weight.ends_with("[0:3072]"), "{}", attn.q_weight);
        assert!(attn.k_weight.ends_with("[3072:6144]"), "{}", attn.k_weight);
        assert!(attn.v_weight.ends_with("[6144:9216]"), "{}", attn.v_weight);

        // Phi-3 has attention_bias false, so naming a bias tensor would look for
        // something that does not exist.
        assert!(attn.q_bias.is_none());
        assert!(attn.k_bias.is_none());
        assert!(attn.v_bias.is_none());
    }

    #[test]
    fn ffn_layout_splits_the_fused_gate_up() {
        // gate_up_proj is [2 * intermediate, hidden], gate first.
        let cfg = PhiConfig::from_json(&phi35_json()).unwrap();
        let layout = cfg.layout();
        let ffn = &layout.decoder.as_ref().unwrap().layer.ffn;

        let gate = ffn.gate_weight.as_ref().expect("phi is gated");
        assert!(gate.contains("gate_up_proj"), "{gate}");
        assert!(gate.ends_with("[0:8192]"), "{gate}");
        assert!(ffn.up_weight.ends_with("[8192:16384]"), "{}", ffn.up_weight);
        assert!(ffn.down_weight.contains("down_proj"), "{}", ffn.down_weight);
    }

    #[test]
    fn every_layout_range_parses_back_out() {
        // The loader reads these with `parse_row_range`, so a name it cannot parse
        // would be looked up literally and fail at load time.
        use kjarni_transformers::weights::parse_row_range;
        let cfg = PhiConfig::from_json(&phi35_json()).unwrap();
        let layout = cfg.layout();
        let layer = &layout.decoder.as_ref().unwrap().layer;

        for name in [
            &layer.self_attn.q_weight,
            &layer.self_attn.k_weight,
            &layer.self_attn.v_weight,
            layer.ffn.gate_weight.as_ref().unwrap(),
            &layer.ffn.up_weight,
        ] {
            let resolved = name.replace("{}", "0");
            assert!(
                parse_row_range(&resolved).is_some(),
                "layout emitted a range the loader cannot parse: {resolved}"
            );
        }
    }

    #[test]
    fn an_untied_model_points_the_head_at_its_own_tensor() {
        let cfg = PhiConfig::from_json(&phi35_json()).unwrap();
        assert_eq!(cfg.layout().lm_head, "lm_head.weight");
    }
}
