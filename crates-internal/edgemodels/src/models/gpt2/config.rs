use std::arch::x86_64::_MM_SET_ROUNDING_MODE;

use anyhow::Result;
use edgetransformers::traits::{
    DecoderArchitecture, LanguageModelConfig, LayerAttentionNames, LayerDecoderAttentionNames,
    LayerFeedForwardNames, TransformerConfig,
};
use serde::Deserialize;
use std::any::Any;
use edgetransformers::activations::Activation;
// This config is for GPT-2 style models like DistilGPT2, GPT-2, etc.
#[derive(Debug, Clone, Deserialize)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub n_ctx: usize,   // max sequence length
    pub n_embd: usize,  // hidden size
    pub n_layer: usize, // number of layers
    pub n_head: usize,  // number of attention heads
    pub layer_norm_epsilon: f32,

    #[serde(alias = "activation_function")]
    pub activation_function: Option<String>,

    #[serde(default = "default_model_type")]
    pub model_type: String,
}

// Helper function for the serde default
fn default_model_type() -> String {
    "gpt2".to_string()
}

impl Gpt2Config {
    fn is_distil(&self) -> bool {
        self.model_type == "distilgpt2"
    }
    pub fn set_model_type(&mut self, model_type: String) {
        self.model_type = model_type
    }
}

impl TransformerConfig for Gpt2Config {
    fn hidden_size(&self) -> usize {
        self.n_embd
    }
    fn num_attention_heads(&self) -> usize {
        self.n_head
    }
    fn num_hidden_layers(&self) -> usize {
        self.n_layer
    }
    fn layer_norm_eps(&self) -> f32 {
        self.layer_norm_epsilon
    }
    fn is_causal(&self) -> bool {
        true
    }
    fn is_prenorm(&self) -> bool {
        true
    } // GPT-2 is a Pre-Norm model
}

impl LanguageModelConfig for Gpt2Config {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn decoder_start_token_id(&self) -> u32 {
        self.bos_token_id().unwrap()
    }
    fn as_any(&self) -> &dyn Any {
        self // Simply return a reference to self as a `&dyn Any`
    }
    fn max_position_embeddings(&self) -> usize {
        self.n_ctx
    }
    fn intermediate_size(&self) -> usize {
        self.n_embd * 4
    } 
    
    fn legacy_ffn_weights(&self) -> bool {
        true
    }
    // Keep In, Out and use Legacy FFN
    fn transpose_ffn_weights(&self) -> bool {
        false
    } // GPT-2 weights are not transposed
    fn transpose_attention_weights(&self) -> bool {
        false
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(50256)
    }
    fn bos_token_id(&self) -> Option<u32> {
        Some(50256)
    }
    fn pad_token_id(&self) -> Option<u32> {
        Some(50256)
    }
     fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>) {
        if self.is_distil() {
            ("transformer.wte.weight", "transformer.wpe.weight", None)
        } else {
            ("wte.weight", "wpe.weight", None)
        }
    }
    fn activation_function(&self) -> Activation {
        self.activation_function
            .as_ref()
            .and_then(|s| s.parse().ok())
            .unwrap_or(Activation::GeluNew) // GPT-2 default
    }
}

impl DecoderArchitecture for Gpt2Config {
    // fn get_embedding_weight_names(&self) -> (&str, &str) {
    //     if self.is_distil() {
    //         ("transformer.wte.weight", "transformer.wpe.weight")
    //     } else {
    //         ("wte.weight", "wpe.weight")
    //     }
    // }
    // fn as_any(&self) -> &dyn Any {
    //     self // Simply return a reference to self as a `&dyn Any`
    // }
    fn get_final_layer_norm_names(&self) -> (&str, &str) {
        if self.is_distil() {
            ("transformer.ln_f.weight", "transformer.ln_f.bias")
        } else {
            ("ln_f.weight", "ln_f.bias")
        }
    }
    fn get_lm_head_name(&self) -> &str {
        if self.is_distil() {
            "transformer.wte.weight" // DistilGPT2 shares weights but with the prefix
        } else {
            "wte.weight" // Standard GPT2 shares weights without the prefix
        }
    } // Shares weights with word embeddings
    fn get_layer_attention_names(&self, layer_index: usize) -> LayerAttentionNames {
        unimplemented!("get_layer_attention_names not implemented")
    }
    fn get_attention_names(&self, i: usize) -> LayerDecoderAttentionNames {
        let prefix = if self.is_distil() {
            "transformer.h"
        } else {
            "h"
        };
        LayerDecoderAttentionNames {
            qkv_weight: format!("{}.{}.attn.c_attn.weight", prefix, i),
            qkv_bias: format!("{}.{}.attn.c_attn.bias", prefix, i),
            output_weight: format!("{}.{}.attn.c_proj.weight", prefix, i),
            output_bias: format!("{}.{}.attn.c_proj.bias", prefix, i),
            norm_weight: format!("{}.{}.ln_1.weight", prefix, i),
            norm_bias: format!("{}.{}.ln_1.bias", prefix, i),
        }
    }

    fn get_feed_forward_names(&self, i: usize) -> LayerFeedForwardNames {
        let prefix = if self.is_distil() {
            "transformer.h"
        } else {
            "h"
        };
        LayerFeedForwardNames {
            intermediate_weight: format!("{}.{}.mlp.c_fc.weight", prefix, i),
            intermediate_bias: format!("{}.{}.mlp.c_fc.bias", prefix, i),
            output_weight: format!("{}.{}.mlp.c_proj.weight", prefix, i),
            output_bias: format!("{}.{}.mlp.c_proj.bias", prefix, i),
            norm_weight: format!("{}.{}.ln_2.weight", prefix, i),
            norm_bias: format!("{}.{}.ln_2.bias", prefix, i),
            gate_weight: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GPT2_CONFIG_JSON: &str = r#"{
        "vocab_size": 50257,
        "n_ctx": 1024,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "layer_norm_epsilon": 1e-5
    }"#;

    const DISTILGPT2_CONFIG_JSON: &str = r#"{
        "model_type": "distilgpt2",
        "vocab_size": 50257,
        "n_ctx": 1024,
        "n_embd": 768,
        "n_layer": 6,
        "n_head": 12,
        "layer_norm_epsilon": 1e-5
    }"#;

    #[test]
    fn test_gpt2_deserialization_and_names() {
        let config: Gpt2Config = serde_json::from_str(GPT2_CONFIG_JSON).unwrap();

        assert!(!config.is_distil());
        assert_eq!(config.hidden_size(), 768);

        let (embed, pos_embed, _) = config.get_embedding_weight_names();
        assert_eq!(embed, "wte.weight");
        assert_eq!(pos_embed, "wpe.weight");

        let attn_names = config.get_attention_names(5);
        assert_eq!(attn_names.qkv_weight, "h.5.attn.c_attn.weight");

        let lm_head = config.get_lm_head_name();
        assert_eq!(lm_head, "wte.weight");
    }

    #[test]
    fn test_distilgpt2_deserialization_and_names() {
        let mut config: Gpt2Config = serde_json::from_str(DISTILGPT2_CONFIG_JSON).unwrap();
        // In a real scenario, the Gpt2Model loader would call this:
        config.set_model_type("distilgpt2".to_string());

        assert!(config.is_distil());
        assert_eq!(config.num_hidden_layers(), 6);

        let (embed, pos_embed, _) = config.get_embedding_weight_names();
        assert_eq!(embed, "transformer.wte.weight");
        assert_eq!(pos_embed, "transformer.wpe.weight");

        let ffn_names = config.get_feed_forward_names(0);
        assert_eq!(
            ffn_names.intermediate_weight,
            "transformer.h.0.mlp.c_fc.weight"
        );

        let lm_head = config.get_lm_head_name();
        assert_eq!(lm_head, "transformer.wte.weight");
    }
}
