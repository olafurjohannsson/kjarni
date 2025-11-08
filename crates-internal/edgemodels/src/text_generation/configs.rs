use std::arch::x86_64::_MM_SET_ROUNDING_MODE;

use edgetransformers::traits::{
    DecoderArchitecture, LanguageModelConfig, LayerDecoderAttentionNames, LayerFeedForwardNames,
    TransformerConfig,
};
use serde::Deserialize;

// This config is for GPT-2 style models like DistilGPT2, GPT-2, etc.
#[derive(Debug, Clone, Deserialize)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub n_ctx: usize,   // max sequence length
    pub n_embd: usize,  // hidden size
    pub n_layer: usize, // number of layers
    pub n_head: usize,  // number of attention heads
    pub layer_norm_epsilon: f32,

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
    fn max_position_embeddings(&self) -> usize {
        self.n_ctx
    }
    fn intermediate_size(&self) -> usize {
        self.n_embd * 4
    } // Standard for GPT-2
    fn transpose_ffn_weights(&self) -> bool {
        false
    } // GPT-2 weights are not transposed
    fn transpose_attention_weights(&self) -> bool {
        false
    }
}

impl DecoderArchitecture for Gpt2Config {
    fn get_embedding_weight_names(&self) -> (&str, &str) {
        if self.is_distil() {
            ("transformer.wte.weight", "transformer.wpe.weight")
        } else {
            ("wte.weight", "wpe.weight")
        }
    }
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
        }
    }
}
