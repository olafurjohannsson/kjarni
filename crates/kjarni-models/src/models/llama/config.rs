use anyhow::{Context, Result};
use kjarni_transformers::models::base::RopeScalingConfig;
use kjarni_transformers::{
    activations::Activation,
    traits::{ModelConfig, ModelLayout, ModelMetadata, DecoderLayout, DecoderLayerLayout, AttentionLayout, FeedForwardLayout},
    weights::WeightLoader,
};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    // --- Core Architecture ---
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,

    // --- Normalization & Activation ---
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
    pub bos_token_id: u32,
    pub eos_token_id: u32,
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
                eos_token_id: 128001,
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

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            // --- Basic Dimensions ---
            hidden_size: self.hidden_size,
            num_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,

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
            extra_pos_embeddings: 0, // Llama has no position offset
            is_prenorm: true,        // Llama uses Pre-Normalization
            transpose_ffn_weights: false, // Standard Llama weights are [Out, In]
            transpose_attention_weights: false,
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

// use anyhow::{anyhow, Context, Result};
// use kjarni_transformers::activations::Activation;
// use kjarni_transformers::models::base::RopeScalingConfig;
// use kjarni_transformers::traits::{
//     DecoderArchitecture, LanguageModelConfig, LayerAttentionNames, LayerDecoderAttentionNames,
//     LayerFeedForwardNames, TransformerConfig,
// };
// use kjarni_transformers::weights::WeightLoader;
// use serde::Deserialize;
// use std::any::Any;
// use std::sync::Arc;
//
// /// A comprehensive configuration for LLaMA models (1, 2, 3, and 3.2).
// ///
// /// This struct is designed to be deserialized directly from a model's `config.json` file,
// /// making it robust to variations between different model versions.
// #[derive(Debug, Clone, Deserialize)]
// pub struct LlamaConfig {
//     // --- Core Architecture ---
//     pub hidden_size: usize,
//     pub num_hidden_layers: usize,
//     pub num_attention_heads: usize,
//     pub num_key_value_heads: usize,
//     pub intermediate_size: usize,
//     pub vocab_size: usize,
//     pub max_position_embeddings: usize,
//
//     // --- Normalization & Activation ---
//     #[serde(default = "default_rms_norm_eps")]
//     pub rms_norm_eps: f32,
//     #[serde(default = "default_hidden_act")]
//     pub hidden_act: String,
//
//     // --- RoPE (Rotary Position Embedding) ---
//     #[serde(default = "default_rope_theta")]
//     pub rope_theta: f32,
//     pub rope_scaling: Option<RopeScalingConfig>,
//
//     // --- Token IDs ---
//     pub bos_token_id: u32,
//     pub eos_token_id: u32,
//     #[serde(default)] // Handles `null` or missing field
//     pub pad_token_id: Option<u32>,
//
//     // --- Critical Logic Flags ---
//     #[serde(default = "default_tie_word_embeddings")]
//     pub tie_word_embeddings: bool,
//
//     // --- Optional Metadata & Other Flags (for robust parsing) ---
//     #[serde(default)]
//     pub architectures: Vec<String>,
//     #[serde(default)]
//     pub attention_bias: bool,
//     #[serde(default)]
//     pub attention_dropout: f32,
//     #[serde(default)]
//     pub head_dim: Option<usize>,
//     #[serde(default = "default_model_type")]
//     pub model_type: String,
//     #[serde(default)]
//     pub torch_dtype: String,
//     #[serde(default = "default_use_cache")]
//     pub use_cache: bool,
// }
//
// fn default_rms_norm_eps() -> f32 {
//     1e-5
// }
// fn default_rope_theta() -> f32 {
//     500000.0 // LLaMA 3 default
// }
// fn default_hidden_act() -> String {
//     "silu".to_string()
// }
// fn default_tie_word_embeddings() -> bool {
//     // Some older models might not have this key. If so, assume true.
//     true
// }
// fn default_model_type() -> String {
//     "llama".to_string()
// }
// fn default_use_cache() -> bool {
//     true
// }
//
// impl LlamaConfig {
//     /// Create config from a JSON string (from a config.json file)
//     pub fn from_json(json: &str) -> Result<Self> {
//         Ok(serde_json::from_str(json)?)
//     }
//
//     /// Get the dimensionality of each attention head.
//     /// Prefers the explicit `head_dim` from config, otherwise calculates it.
//     pub fn head_dim(&self) -> usize {
//         self.head_dim
//             .unwrap_or(self.hidden_size / self.num_attention_heads)
//     }
//     pub fn from_loader(loader: &dyn WeightLoader, config_json: Option<&str>) -> Result<Arc<Self>> {
//         if loader.has_metadata() {
//             // --- GGUF Metadata Extraction ---
//             let arch = loader.get_string("general.architecture").unwrap_or("llama");
//             let get_u32 = |k: &str| loader.get_u32(&format!("{}.{}", arch, k));
//             let get_f32 = |k: &str| loader.get_f32(&format!("{}.{}", arch, k));
//
//             // 1. Extract base dimensions into local variables first
//             let hidden_size = get_u32("embedding_length")
//                 .context("GGUF metadata missing 'embedding_length'")?
//                 as usize;
//
//             let num_attention_heads = get_u32("attention.head_count")
//                 .context("GGUF metadata missing 'head_count'")?
//                 as usize;
//
//             // 2. Use those local variables to calculate defaults for other fields
//             let num_key_value_heads = get_u32("attention.head_count_kv")
//                 .map(|v| v as usize)
//                 .unwrap_or(num_attention_heads / 4); // Default to GQA 4:1 if missing
//
//             let intermediate_size = get_u32("feed_forward_length")
//                 .map(|v| v as usize)
//                 .unwrap_or(hidden_size * 4); // Standard heuristic if missing
//
//             let vocab_size = loader
//                 .get_u32("general.vocabulary_size")
//                 .map(|v| v as usize)
//                 .unwrap_or(128256);
//
//             Ok(Arc::new(Self {
//                 hidden_size,
//                 num_hidden_layers: get_u32("block_count").context("Missing block_count")? as usize,
//                 num_attention_heads,
//                 num_key_value_heads,
//                 intermediate_size,
//                 vocab_size,
//                 max_position_embeddings: get_u32("context_length").unwrap_or(2048) as usize,
//
//                 rms_norm_eps: get_f32("attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
//                 hidden_act: loader
//                     .get_string(&format!("{}.feed_forward_activation", arch))
//                     .unwrap_or("silu")
//                     .to_string(),
//
//                 rope_theta: get_f32("rope.freq_base").unwrap_or(500000.0),
//                 rope_scaling: None, // Complex GGUF mapping, default to None
//
//                 bos_token_id: 128000,
//                 eos_token_id: 128001,
//                 pad_token_id: None,
//
//                 // GGUF Logic: if it doesn't have an output weight, it's tied.
//                 tie_word_embeddings: !loader.contains("output.weight"),
//
//                 architectures: vec!["LlamaForCausalLM".to_string()],
//                 attention_bias: false,
//                 attention_dropout: 0.0,
//                 head_dim: Some(hidden_size / num_attention_heads),
//                 model_type: arch.to_string(),
//                 torch_dtype: "bfloat16".to_string(),
//                 use_cache: true,
//             }))
//         } else {
//             // --- Safetensors / JSON Fallback ---
//             let json_str =
//                 config_json.ok_or_else(|| anyhow!("Safetensors requires config.json"))?;
//             Ok(Arc::new(Self::from_json(json_str)?))
//         }
//     }
// }
//
// // --- Trait Implementations ---
//
// impl LanguageModelConfig for LlamaConfig {
//     fn intermediate_size(&self) -> usize {
//         self.intermediate_size
//     }
//
//     fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>) {
//         ("model.embed_tokens.weight", "", None) // RoPE has no position embedding table
//     }
//
//     fn max_position_embeddings(&self) -> usize {
//         self.max_position_embeddings
//     }
//
//     fn vocab_size(&self) -> usize {
//         self.vocab_size
//     }
//
//     fn num_key_value_heads(&self) -> usize {
//         self.num_key_value_heads
//     }
//
//     fn bos_token_id(&self) -> Option<u32> {
//         Some(self.bos_token_id)
//     }
//
//     fn eos_token_id(&self) -> Option<u32> {
//         Some(self.eos_token_id)
//     }
//
//     fn pad_token_id(&self) -> Option<u32> {
//         self.pad_token_id
//     }
//
//     fn activation_function(&self) -> Activation {
//         match self.hidden_act.as_str() {
//             "silu" => Activation::SilU,
//             "relu" => Activation::Relu,
//             "gelu" => Activation::Gelu,
//             "gelu_new" => Activation::GeluNew,
//             _ => Activation::SilU, // Default to SiLU for Llama
//         }
//     }
//
//     // Required trait methods
//     fn as_any(&self) -> &dyn Any {
//         self
//     }
//     fn decoder_start_token_id(&self) -> u32 {
//         self.bos_token_id
//     }
// }
//
// impl TransformerConfig for LlamaConfig {
//     fn hidden_size(&self) -> usize {
//         self.hidden_size
//     }
//
//     fn num_hidden_layers(&self) -> usize {
//         self.num_hidden_layers
//     }
//
//     fn num_attention_heads(&self) -> usize {
//         self.num_attention_heads
//     }
//
//     fn layer_norm_eps(&self) -> f32 {
//         self.rms_norm_eps
//     }
//
//     fn is_causal(&self) -> bool {
//         true // Decoder-only models are always causal
//     }
//
//     fn is_prenorm(&self) -> bool {
//         true // Llama uses Pre-Normalization
//     }
// }
//
// impl DecoderArchitecture for LlamaConfig {
//     fn get_lm_head_name(&self) -> &str {
//         // --- THIS IS THE CRITICAL FIX ---
//         if self.tie_word_embeddings {
//             // For models like Llama-3.2, the LM head shares weights with the token embeddings.
//             "model.embed_tokens.weight"
//         } else {
//             // For models like Llama-3-8B-Instruct, there is a separate, dedicated LM head tensor.
//             "lm_head.weight"
//         }
//     }
//
//     fn get_layer_attention_names(&self, layer: usize) -> LayerAttentionNames {
//         LayerAttentionNames {
//             q_weight: format!("model.layers.{}.self_attn.q_proj.weight", layer),
//             k_weight: format!("model.layers.{}.self_attn.k_proj.weight", layer),
//             v_weight: format!("model.layers.{}.self_attn.v_proj.weight", layer),
//             output_weight: format!("model.layers.{}.self_attn.o_proj.weight", layer),
//             norm_weight: format!("model.layers.{}.input_layernorm.weight", layer),
//             // Llama models do not use biases
//             q_bias: String::new(),
//             k_bias: String::new(),
//             v_bias: String::new(),
//             output_bias: String::new(),
//             norm_bias: String::new(),
//         }
//     }
//
//     fn get_feed_forward_names(&self, layer: usize) -> LayerFeedForwardNames {
//         LayerFeedForwardNames {
//             gate_weight: Some(format!("model.layers.{}.mlp.gate_proj.weight", layer)),
//             intermediate_weight: format!("model.layers.{}.mlp.up_proj.weight", layer),
//             output_weight: format!("model.layers.{}.mlp.down_proj.weight", layer),
//             norm_weight: format!("model.layers.{}.post_attention_layernorm.weight", layer),
//             // Llama models do not use biases
//             intermediate_bias: String::new(),
//             output_bias: String::new(),
//             norm_bias: String::new(),
//         }
//     }
//
//     fn get_final_layer_norm_names(&self) -> (&str, &str) {
//         ("model.norm.weight", "") // Final RMSNorm, no bias
//     }
//
//     // This method is not used by Llama which has separate Q,K,V projections.
//     // We implement it to satisfy the trait, but it should not be called.
//     fn get_attention_names(&self, _layer_index: usize) -> LayerDecoderAttentionNames {
//         unimplemented!("Llama uses get_layer_attention_names with separate Q/K/V projections")
//     }
// }
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn test_parse_and_validate_llama_3_8b_instruct() {
//         let json = r#"{
//           "architectures": [ "LlamaForCausalLM" ],
//           "attention_bias": false, "attention_dropout": 0.0, "bos_token_id": 128000,
//           "eos_token_id": 128009, "hidden_act": "silu", "hidden_size": 4096,
//           "initializer_range": 0.02, "intermediate_size": 14336, "max_position_embeddings": 8192,
//           "model_type": "llama", "num_attention_heads": 32, "num_hidden_layers": 32,
//           "num_key_value_heads": 8, "pretraining_tp": 1, "rms_norm_eps": 1e-05,
//           "rope_scaling": null, "rope_theta": 500000.0, "tie_word_embeddings": false,
//           "torch_dtype": "bfloat16", "transformers_version": "4.40.0.dev0",
//           "use_cache": true, "vocab_size": 128256
//         }"#;
//
//         let config = LlamaConfig::from_json(json).unwrap();
//
//         // Validate key parameters
//         assert_eq!(config.hidden_size, 4096);
//         assert_eq!(config.num_key_value_heads, 8);
//         assert_eq!(config.intermediate_size, 14336);
//         assert_eq!(config.eos_token_id, 128009);
//         assert_eq!(config.tie_word_embeddings, false);
//         assert!(config.rope_scaling.is_none());
//
//         // --- CRITICAL TEST: Validate the LM Head tensor name ---
//         assert_eq!(
//             config.get_lm_head_name(),
//             "lm_head.weight",
//             "8B Instruct must use the separate 'lm_head.weight' tensor"
//         );
//     }
//
//     #[test]
//     fn test_parse_and_validate_llama_3_2_3b() {
//         let json = r#"{
//           "architectures": [ "LlamaForCausalLM" ], "attention_bias": false,
//           "attention_dropout": 0.0, "bos_token_id": 128000, "eos_token_id": 128001,
//           "head_dim": 128, "hidden_act": "silu", "hidden_size": 3072,
//           "initializer_range": 0.02, "intermediate_size": 8192, "max_position_embeddings": 131072,
//           "mlp_bias": false, "model_type": "llama", "num_attention_heads": 24,
//           "num_hidden_layers": 28, "num_key_value_heads": 8, "pretraining_tp": 1,
//           "rms_norm_eps": 1e-05,
//           "rope_scaling": {
//             "factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
//             "original_max_position_embeddings": 8192, "rope_type": "llama3"
//           },
//           "rope_theta": 500000.0, "tie_word_embeddings": true, "torch_dtype": "bfloat16",
//           "transformers_version": "4.45.0.dev0", "use_cache": true, "vocab_size": 128256
//         }"#;
//
//         let config = LlamaConfig::from_json(json).unwrap();
//
//         // Validate key parameters
//         assert_eq!(config.hidden_size, 3072);
//         assert_eq!(config.num_attention_heads, 24);
//         assert_eq!(config.intermediate_size, 8192);
//         assert_eq!(config.eos_token_id, 128001);
//         assert_eq!(config.tie_word_embeddings, true);
//         assert!(config.rope_scaling.is_some());
//         assert_eq!(config.head_dim(), 128); // Check head_dim is parsed correctly
//
//         // --- CRITICAL TEST: Validate the LM Head tensor name ---
//         assert_eq!(
//             config.get_lm_head_name(),
//             "model.embed_tokens.weight",
//             "3.2B must use the tied 'model.embed_tokens.weight' tensor"
//         );
//     }
//
//     #[test]
//     fn test_parse_and_validate_llama_3_2_1b() {
//         let json = r#"{
//           "architectures": [ "LlamaForCausalLM" ], "attention_bias": false,
//           "attention_dropout": 0.0, "bos_token_id": 128000, "eos_token_id": 128001,
//           "head_dim": 64, "hidden_act": "silu", "hidden_size": 2048,
//           "initializer_range": 0.02, "intermediate_size": 8192, "max_position_embeddings": 131072,
//           "mlp_bias": false, "model_type": "llama", "num_attention_heads": 32,
//           "num_hidden_layers": 16, "num_key_value_heads": 8, "pretraining_tp": 1,
//           "rms_norm_eps": 1e-05,
//           "rope_scaling": {
//             "factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
//             "original_max_position_embeddings": 8192, "rope_type": "llama3"
//           },
//           "rope_theta": 500000.0, "tie_word_embeddings": true, "torch_dtype": "bfloat16",
//           "transformers_version": "4.45.0.dev0", "use_cache": true, "vocab_size": 128256
//         }"#;
//
//         let config = LlamaConfig::from_json(json).unwrap();
//
//         // Validate key parameters
//         assert_eq!(config.hidden_size, 2048);
//         assert_eq!(config.num_attention_heads, 32);
//         assert_eq!(config.num_hidden_layers, 16);
//         assert_eq!(config.eos_token_id, 128001);
//         assert_eq!(config.tie_word_embeddings, true);
//         assert!(config.rope_scaling.is_some());
//
//         // --- CRITICAL TEST: Validate the LM Head tensor name ---
//         assert_eq!(
//             config.get_lm_head_name(),
//             "model.embed_tokens.weight",
//             "3.2B must use the tied 'model.embed_tokens.weight' tensor"
//         );
//     }
//
//     #[test]
//     fn test_weight_name_generation() {
//         let json = r#"{
//           "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
//           "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256,
//           "max_position_embeddings": 8192, "bos_token_id": 128000, "eos_token_id": 128009,
//           "tie_word_embeddings": false
//         }"#;
//         let config = LlamaConfig::from_json(json).unwrap();
//
//         let (embed, pos_embed, _) = config.get_embedding_weight_names();
//         assert_eq!(embed, "model.embed_tokens.weight");
//         assert_eq!(pos_embed, ""); // Llama uses RoPE, not learned position embeddings
//
//         let attn_names = config.get_layer_attention_names(5); // Test an arbitrary layer index
//         assert_eq!(
//             attn_names.q_weight,
//             "model.layers.5.self_attn.q_proj.weight"
//         );
//         assert_eq!(
//             attn_names.k_weight,
//             "model.layers.5.self_attn.k_proj.weight"
//         );
//         assert_eq!(
//             attn_names.v_weight,
//             "model.layers.5.self_attn.v_proj.weight"
//         );
//         assert_eq!(
//             attn_names.output_weight,
//             "model.layers.5.self_attn.o_proj.weight"
//         );
//         assert_eq!(
//             attn_names.norm_weight,
//             "model.layers.5.input_layernorm.weight"
//         );
//         assert_eq!(attn_names.q_bias, ""); // Llama has no biases
//
//         let ffn_names = config.get_feed_forward_names(5);
//         assert_eq!(
//             ffn_names.intermediate_weight,
//             "model.layers.5.mlp.up_proj.weight"
//         );
//         assert_eq!(
//             ffn_names.gate_weight.unwrap(),
//             "model.layers.5.mlp.gate_proj.weight"
//         );
//         assert_eq!(
//             ffn_names.output_weight,
//             "model.layers.5.mlp.down_proj.weight"
//         );
//         assert_eq!(
//             ffn_names.norm_weight,
//             "model.layers.5.post_attention_layernorm.weight"
//         );
//
//         let (final_norm, final_norm_bias) = config.get_final_layer_norm_names();
//         assert_eq!(final_norm, "model.norm.weight");
//         assert_eq!(final_norm_bias, "");
//     }
// }
