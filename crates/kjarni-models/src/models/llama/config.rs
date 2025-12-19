use anyhow::Result;
use kjarni_transformers::activations::Activation;
use kjarni_transformers::models::base::RopeScalingConfig;
use kjarni_transformers::traits::{
    DecoderArchitecture, LanguageModelConfig, LayerAttentionNames, LayerDecoderAttentionNames,
    LayerFeedForwardNames, TransformerConfig,
};
use serde::Deserialize;
use std::any::Any;

/// A comprehensive configuration for LLaMA models (1, 2, 3, and 3.2).
///
/// This struct is designed to be deserialized directly from a model's `config.json` file,
/// making it robust to variations between different model versions.
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

    // --- Token IDs ---
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    #[serde(default)] // Handles `null` or missing field
    pub pad_token_id: Option<u32>,

    // --- Critical Logic Flags ---
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    // --- Optional Metadata & Other Flags (for robust parsing) ---
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f32,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default)]
    pub torch_dtype: String,
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
}

// --- Default value functions for serde ---

fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    500000.0 // LLaMA 3 default
}
fn default_hidden_act() -> String {
    "silu".to_string()
}
fn default_tie_word_embeddings() -> bool {
    // Some older models might not have this key. If so, assume true.
    true
}
fn default_model_type() -> String {
    "llama".to_string()
}
fn default_use_cache() -> bool {
    true
}

impl LlamaConfig {
    /// Create config from a JSON string (from a config.json file)
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// Get the dimensionality of each attention head.
    /// Prefers the explicit `head_dim` from config, otherwise calculates it.
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

// --- Trait Implementations ---

impl LanguageModelConfig for LlamaConfig {
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>) {
        ("model.embed_tokens.weight", "", None) // RoPE has no position embedding table
    }

    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    fn bos_token_id(&self) -> Option<u32> {
        Some(self.bos_token_id)
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(self.eos_token_id)
    }

    fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }

    fn activation_function(&self) -> Activation {
        match self.hidden_act.as_str() {
            "silu" => Activation::SilU,
            "relu" => Activation::Relu,
            "gelu" => Activation::Gelu,
            "gelu_new" => Activation::GeluNew,
            _ => Activation::SilU, // Default to SiLU for Llama
        }
    }

    // Required trait methods
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn decoder_start_token_id(&self) -> u32 {
        self.bos_token_id
    }
}

impl TransformerConfig for LlamaConfig {
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
        self.rms_norm_eps
    }

    fn is_causal(&self) -> bool {
        true // Decoder-only models are always causal
    }

    fn is_prenorm(&self) -> bool {
        true // Llama uses Pre-Normalization
    }
}

impl DecoderArchitecture for LlamaConfig {
    fn get_lm_head_name(&self) -> &str {
        // --- THIS IS THE CRITICAL FIX ---
        if self.tie_word_embeddings {
            // For models like Llama-3.2, the LM head shares weights with the token embeddings.
            "model.embed_tokens.weight"
        } else {
            // For models like Llama-3-8B-Instruct, there is a separate, dedicated LM head tensor.
            "lm_head.weight"
        }
    }

    fn get_layer_attention_names(&self, layer: usize) -> LayerAttentionNames {
        LayerAttentionNames {
            q_weight: format!("model.layers.{}.self_attn.q_proj.weight", layer),
            k_weight: format!("model.layers.{}.self_attn.k_proj.weight", layer),
            v_weight: format!("model.layers.{}.self_attn.v_proj.weight", layer),
            output_weight: format!("model.layers.{}.self_attn.o_proj.weight", layer),
            norm_weight: format!("model.layers.{}.input_layernorm.weight", layer),
            // Llama models do not use biases
            q_bias: String::new(),
            k_bias: String::new(),
            v_bias: String::new(),
            output_bias: String::new(),
            norm_bias: String::new(),
        }
    }

    fn get_feed_forward_names(&self, layer: usize) -> LayerFeedForwardNames {
        LayerFeedForwardNames {
            gate_weight: Some(format!("model.layers.{}.mlp.gate_proj.weight", layer)),
            intermediate_weight: format!("model.layers.{}.mlp.up_proj.weight", layer),
            output_weight: format!("model.layers.{}.mlp.down_proj.weight", layer),
            norm_weight: format!("model.layers.{}.post_attention_layernorm.weight", layer),
            // Llama models do not use biases
            intermediate_bias: String::new(),
            output_bias: String::new(),
            norm_bias: String::new(),
        }
    }

    fn get_final_layer_norm_names(&self) -> (&str, &str) {
        ("model.norm.weight", "") // Final RMSNorm, no bias
    }

    // This method is not used by Llama which has separate Q,K,V projections.
    // We implement it to satisfy the trait, but it should not be called.
    fn get_attention_names(&self, _layer_index: usize) -> LayerDecoderAttentionNames {
        unimplemented!("Llama uses get_layer_attention_names with separate Q/K/V projections")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_and_validate_llama_3_8b_instruct() {
        let json = r#"{
          "architectures": [ "LlamaForCausalLM" ],
          "attention_bias": false, "attention_dropout": 0.0, "bos_token_id": 128000,
          "eos_token_id": 128009, "hidden_act": "silu", "hidden_size": 4096,
          "initializer_range": 0.02, "intermediate_size": 14336, "max_position_embeddings": 8192,
          "model_type": "llama", "num_attention_heads": 32, "num_hidden_layers": 32,
          "num_key_value_heads": 8, "pretraining_tp": 1, "rms_norm_eps": 1e-05,
          "rope_scaling": null, "rope_theta": 500000.0, "tie_word_embeddings": false,
          "torch_dtype": "bfloat16", "transformers_version": "4.40.0.dev0",
          "use_cache": true, "vocab_size": 128256
        }"#;

        let config = LlamaConfig::from_json(json).unwrap();

        // Validate key parameters
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.intermediate_size, 14336);
        assert_eq!(config.eos_token_id, 128009);
        assert_eq!(config.tie_word_embeddings, false);
        assert!(config.rope_scaling.is_none());

        // --- CRITICAL TEST: Validate the LM Head tensor name ---
        assert_eq!(
            config.get_lm_head_name(),
            "lm_head.weight",
            "8B Instruct must use the separate 'lm_head.weight' tensor"
        );
    }

    #[test]
    fn test_parse_and_validate_llama_3_2_3b() {
        let json = r#"{
          "architectures": [ "LlamaForCausalLM" ], "attention_bias": false,
          "attention_dropout": 0.0, "bos_token_id": 128000, "eos_token_id": 128001,
          "head_dim": 128, "hidden_act": "silu", "hidden_size": 3072,
          "initializer_range": 0.02, "intermediate_size": 8192, "max_position_embeddings": 131072,
          "mlp_bias": false, "model_type": "llama", "num_attention_heads": 24,
          "num_hidden_layers": 28, "num_key_value_heads": 8, "pretraining_tp": 1,
          "rms_norm_eps": 1e-05,
          "rope_scaling": {
            "factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192, "rope_type": "llama3"
          },
          "rope_theta": 500000.0, "tie_word_embeddings": true, "torch_dtype": "bfloat16",
          "transformers_version": "4.45.0.dev0", "use_cache": true, "vocab_size": 128256
        }"#;

        let config = LlamaConfig::from_json(json).unwrap();

        // Validate key parameters
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_attention_heads, 24);
        assert_eq!(config.intermediate_size, 8192);
        assert_eq!(config.eos_token_id, 128001);
        assert_eq!(config.tie_word_embeddings, true);
        assert!(config.rope_scaling.is_some());
        assert_eq!(config.head_dim(), 128); // Check head_dim is parsed correctly

        // --- CRITICAL TEST: Validate the LM Head tensor name ---
        assert_eq!(
            config.get_lm_head_name(),
            "model.embed_tokens.weight",
            "3.2B must use the tied 'model.embed_tokens.weight' tensor"
        );
    }

    #[test]
    fn test_parse_and_validate_llama_3_2_1b() {
        let json = r#"{
          "architectures": [ "LlamaForCausalLM" ], "attention_bias": false,
          "attention_dropout": 0.0, "bos_token_id": 128000, "eos_token_id": 128001,
          "head_dim": 64, "hidden_act": "silu", "hidden_size": 2048,
          "initializer_range": 0.02, "intermediate_size": 8192, "max_position_embeddings": 131072,
          "mlp_bias": false, "model_type": "llama", "num_attention_heads": 32,
          "num_hidden_layers": 16, "num_key_value_heads": 8, "pretraining_tp": 1,
          "rms_norm_eps": 1e-05,
          "rope_scaling": {
            "factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192, "rope_type": "llama3"
          },
          "rope_theta": 500000.0, "tie_word_embeddings": true, "torch_dtype": "bfloat16",
          "transformers_version": "4.45.0.dev0", "use_cache": true, "vocab_size": 128256
        }"#;

        let config = LlamaConfig::from_json(json).unwrap();

        // Validate key parameters
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_hidden_layers, 16);
        assert_eq!(config.eos_token_id, 128001);
        assert_eq!(config.tie_word_embeddings, true);
        assert!(config.rope_scaling.is_some());

        // --- CRITICAL TEST: Validate the LM Head tensor name ---
        assert_eq!(
            config.get_lm_head_name(),
            "model.embed_tokens.weight",
            "3.2B must use the tied 'model.embed_tokens.weight' tensor"
        );
    }

    #[test]
    fn test_weight_name_generation() {
        let json = r#"{
          "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
          "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256,
          "max_position_embeddings": 8192, "bos_token_id": 128000, "eos_token_id": 128009,
          "tie_word_embeddings": false
        }"#;
        let config = LlamaConfig::from_json(json).unwrap();

        let (embed, pos_embed, _) = config.get_embedding_weight_names();
        assert_eq!(embed, "model.embed_tokens.weight");
        assert_eq!(pos_embed, ""); // Llama uses RoPE, not learned position embeddings

        let attn_names = config.get_layer_attention_names(5); // Test an arbitrary layer index
        assert_eq!(attn_names.q_weight, "model.layers.5.self_attn.q_proj.weight");
        assert_eq!(attn_names.k_weight, "model.layers.5.self_attn.k_proj.weight");
        assert_eq!(attn_names.v_weight, "model.layers.5.self_attn.v_proj.weight");
        assert_eq!(attn_names.output_weight, "model.layers.5.self_attn.o_proj.weight");
        assert_eq!(attn_names.norm_weight, "model.layers.5.input_layernorm.weight");
        assert_eq!(attn_names.q_bias, ""); // Llama has no biases

        let ffn_names = config.get_feed_forward_names(5);
        assert_eq!(ffn_names.intermediate_weight, "model.layers.5.mlp.up_proj.weight");
        assert_eq!(ffn_names.gate_weight.unwrap(), "model.layers.5.mlp.gate_proj.weight");
        assert_eq!(ffn_names.output_weight, "model.layers.5.mlp.down_proj.weight");
        assert_eq!(ffn_names.norm_weight, "model.layers.5.post_attention_layernorm.weight");

        let (final_norm, final_norm_bias) = config.get_final_layer_norm_names();
        assert_eq!(final_norm, "model.norm.weight");
        assert_eq!(final_norm_bias, "");
    }
}