use std::arch::x86_64::_MM_SET_ROUNDING_MODE;

use anyhow::Result;
use edgetransformers::models::base::RopeScalingConfig;
use edgetransformers::traits::{
    DecoderArchitecture, LanguageModelConfig, LayerAttentionNames, LayerDecoderAttentionNames,
    LayerFeedForwardNames, TransformerConfig,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use edgetransformers::activations::Activation;

/// Configuration for LLaMA models
///
/// This config supports LLaMA 1, 2, and 3 variants.
/// Default values are for LLaMA-3.2 1B (smallest production LLaMA model).
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    /// Dimensionality of hidden states
    pub hidden_size: usize,

    /// Number of transformer decoder layers
    pub num_hidden_layers: usize,

    /// Number of attention heads
    pub num_attention_heads: usize,

    /// Number of key-value heads (for Grouped Query Attention)
    /// If equal to num_attention_heads, uses standard Multi-Head Attention
    /// If less, uses GQA (more efficient for large models)
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,

    /// Dimensionality of the FFN intermediate layer (SwiGLU)
    pub intermediate_size: usize,

    /// Vocabulary size
    pub vocab_size: usize,

    /// Maximum sequence length
    pub max_position_embeddings: usize,

    /// RMSNorm epsilon for numerical stability
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    /// Base value for RoPE frequencies
    /// 10000.0 for LLaMA 1/2, 500000.0 for LLaMA 3 (enables longer context)
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    /// Activation function (always "silu" for LLaMA, used in SwiGLU)
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    /// Beginning of sequence token ID
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: u32,

    /// End of sequence token ID
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,

    /// Padding token ID (often same as EOS for LLaMA)
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: Option<u32>,

    pub rope_scaling: Option<RopeScalingConfig>,
}

// Default values for optional fields (LLaMA-3.2 1B defaults)
fn default_num_key_value_heads() -> usize {
    8 // GQA with 8 KV heads for 1B model
}

fn default_rms_norm_eps() -> f32 {
    1e-5
}

fn default_rope_theta() -> f32 {
    500000.0 // LLaMA 3 default (long context)
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_bos_token_id() -> u32 {
    128000 // LLaMA 3 default
}

fn default_eos_token_id() -> u32 {
    128001 // LLaMA 3 default
}

fn default_pad_token_id() -> Option<u32> {
    None // LLaMA typically doesn't use padding token
}

impl LlamaConfig {
    /// Create config from JSON string (from config.json file)
    pub fn from_json(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// LLaMA-3.2 1B configuration (smallest production LLaMA)
    pub fn llama_3_2_1b() -> Self {
        Self {
            hidden_size: 2048,
            num_hidden_layers: 16,
            num_attention_heads: 32,
            num_key_value_heads: 8, // GQA: 4 heads per KV head
            intermediate_size: 8192,
            vocab_size: 128256,
            max_position_embeddings: 131072, // 128k context
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            hidden_act: "silu".to_string(),
            bos_token_id: 128000,
            eos_token_id: 128001,
            pad_token_id: None,
            rope_scaling: None,
        }
    }

    /// LLaMA-3.2 3B configuration
    pub fn llama_3_2_3b() -> Self {
        Self {
            hidden_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 24,
            num_key_value_heads: 8,
            intermediate_size: 8192,
            vocab_size: 128256,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            hidden_act: "silu".to_string(),
            bos_token_id: 128000,
            eos_token_id: 128001,
            pad_token_id: None,
            rope_scaling: None,
        }
    }

    /// LLaMA-3 8B configuration (most popular)
    pub fn llama_3_8b() -> Self {
        Self {
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 14336,
            vocab_size: 128256,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            hidden_act: "silu".to_string(),
            bos_token_id: 128000,
            eos_token_id: 128001,
            pad_token_id: None,
            rope_scaling: None,
        }
    }

    /// LLaMA 2 7B configuration (original, widely used)
    pub fn llama_2_7b() -> Self {
        Self {
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32, // No GQA in LLaMA 2
            intermediate_size: 11008,
            vocab_size: 32000,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0, // Original RoPE theta
            hidden_act: "silu".to_string(),
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: Some(0),
            rope_scaling: None,
        }
    }
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    // pub fn uses_gqa(&self) -> bool {
    //     self.num_attention_heads != self.num_key_value_heads
    // }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get KV head dimension (for GQA)
    pub fn kv_head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Check if using Grouped Query Attention
    pub fn uses_gqa(&self) -> bool {
        self.num_key_value_heads < self.num_attention_heads
    }
    fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }
}

impl LanguageModelConfig for LlamaConfig {
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>) {
        (
            "model.embed_tokens.weight",
            "", // No position embeddings (uses RoPE instead)
            None,
        )
    }
fn as_any(&self) -> &dyn Any {
        self // Simply return a reference to self as a `&dyn Any`
    }
    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn activation_function(&self) -> Activation {
        Activation::GeluNew
    }
    fn transpose_ffn_weights(&self) -> bool {
        true // LLaMA uses same convention as other transformers
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
        // LLaMA uses RMSNorm, but we return this for compatibility
        self.rms_norm_eps
    }

    fn is_causal(&self) -> bool {
        true // LLaMA is decoder-only, always causal
    }

    fn is_prenorm(&self) -> bool {
        true // LLaMA uses pre-normalization (norm before attention/FFN)
    }
}

impl DecoderArchitecture for LlamaConfig {
    // fn get_embedding_weight_names(&self) -> (&str, &str) {
    //     (
    //         "model.embed_tokens.weight",
    //         "", // No position embeddings (uses RoPE instead)
    //     )
    // }
    // fn as_any(&self) -> &dyn Any {
    //     self // Simply return a reference to self as a `&dyn Any`
    // }
    // fn num_key_value_heads(&self) -> usize {
    //     self.num_key_value_heads // 8 for LLaMA 3.2 1B
    // }
    fn get_attention_names(&self, _layer_index: usize) -> LayerDecoderAttentionNames {
        LayerDecoderAttentionNames {
            qkv_weight: String::new(),
            norm_bias: String::new(),
            norm_weight: String::new(),
            output_bias: String::new(),
            output_weight: String::new(),
            qkv_bias: String::new(),
        }
    }
    fn get_layer_attention_names(&self, layer: usize) -> LayerAttentionNames {
        LayerAttentionNames {
            // LLaMA uses separate Q, K, V projections (not combined)
            q_weight: format!("model.layers.{}.self_attn.q_proj.weight", layer),
            q_bias: String::new(), // LLaMA has NO biases
            k_weight: format!("model.layers.{}.self_attn.k_proj.weight", layer),
            k_bias: String::new(),
            v_weight: format!("model.layers.{}.self_attn.v_proj.weight", layer),
            v_bias: String::new(),
            output_weight: format!("model.layers.{}.self_attn.o_proj.weight", layer),
            output_bias: String::new(),
            // Pre-norm: normalization happens BEFORE attention
            norm_weight: format!("model.layers.{}.input_layernorm.weight", layer),
            norm_bias: String::new(), // RMSNorm has no bias
        }
    }

    fn get_feed_forward_names(&self, layer: usize) -> LayerFeedForwardNames {
        LayerFeedForwardNames {
            // SwiGLU has 3 projections: gate, up, down
            // We map them to the standard names for compatibility
            intermediate_weight: format!("model.layers.{}.mlp.up_proj.weight", layer),
            intermediate_bias: String::new(),
            // The gate projection (used in SwiGLU)
            // You'll need to handle this specially in your FFN implementation
            // gate_proj: format!("model.layers.{}.mlp.gate_proj.weight", layer),
            output_weight: format!("model.layers.{}.mlp.down_proj.weight", layer),
            output_bias: String::new(),
            // Post-FFN normalization
            norm_weight: format!("model.layers.{}.post_attention_layernorm.weight", layer),
            norm_bias: String::new(),
            gate_weight: Some(format!("model.layers.{}.mlp.gate_proj.weight", layer)),
        }
    }

    fn get_final_layer_norm_names(&self) -> (&str, &str) {
        ("model.norm.weight", "") // Final RMSNorm, no bias
    }

    fn get_lm_head_name(&self) -> &str {
        "model.embed_tokens.weight" // "lm_head.weight"
    }
    // should use    fn get_embedding_weight_names(&self) -> (&str, &str) {
    // (
    //     "model.embed_tokens.weight", LLaMA has "tie_word_embeddings": true, which means if lm_head.weight doesn't exist as a separate tensor,
    //     "", // No position embeddings (uses RoPE instead)
    // )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_3_2_1b_config() {
        let config = LlamaConfig::llama_3_2_1b();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 16);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim(), 64);
        assert!(config.uses_gqa());
        assert!(config.is_causal());
        assert!(config.is_prenorm());
    }

    #[test]
    fn test_llama_2_7b_config() {
        let config = LlamaConfig::llama_2_7b();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 32);
        assert!(!config.uses_gqa()); // LLaMA 2 doesn't use GQA
        assert_eq!(config.rope_theta, 10000.0);
    }

    #[test]
    fn test_weight_names() {
        let config = LlamaConfig::llama_3_2_1b();

        let (embed, pos_embed, _) = config.get_embedding_weight_names();
        assert_eq!(embed, "model.embed_tokens.weight");
        assert_eq!(pos_embed, ""); // No position embeddings

        let attn_names = config.get_layer_attention_names(0);
        assert_eq!(
            attn_names.q_weight,
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(attn_names.q_bias, ""); // No biases
        assert_eq!(
            attn_names.norm_weight,
            "model.layers.0.input_layernorm.weight"
        );
        assert_eq!(attn_names.norm_bias, ""); // RMSNorm no bias

        let ffn_names = config.get_feed_forward_names(0);
        assert_eq!(
            ffn_names.intermediate_weight,
            "model.layers.0.mlp.up_proj.weight"
        );
        assert_eq!(
            ffn_names.norm_weight,
            "model.layers.0.post_attention_layernorm.weight"
        );

        let (final_norm, final_norm_bias) = config.get_final_layer_norm_names();
        assert_eq!(final_norm, "model.norm.weight");
        assert_eq!(final_norm_bias, "");

        let lm_head = config.get_lm_head_name();
        assert_eq!(lm_head, "model.embed_tokens.weight");
    }

    #[test]
    fn test_from_json() {
        let json = r#"{
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "bos_token_id": 128000,
            "eos_token_id": 128001
        }"#;

        let config = LlamaConfig::from_json(json).unwrap();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.rope_theta, 500000.0);
    }
}
