use serde::Deserialize;
use edgetransformers::traits::{
    DecoderArchitecture, LanguageModelConfig, LayerFeedForwardNames, TransformerConfig, LayerDecoderAttentionNames
};

// This config is for GPT-2 style models like DistilGPT2, GPT-2, etc.
#[derive(Debug, Clone, Deserialize)]
pub struct Gpt2Config {
    pub vocab_size: usize,
    pub n_ctx: usize,      // max sequence length
    pub n_embd: usize,     // hidden size
    pub n_layer: usize,    // number of layers
    pub n_head: usize,     // number of attention heads
    pub layer_norm_epsilon: f32,
}

impl TransformerConfig for Gpt2Config {
    fn hidden_size(&self) -> usize { self.n_embd }
    fn num_attention_heads(&self) -> usize { self.n_head }
    fn num_hidden_layers(&self) -> usize { self.n_layer }
    fn layer_norm_eps(&self) -> f32 { self.layer_norm_epsilon }
    fn is_causal(&self) -> bool { true }
    fn is_prenorm(&self) -> bool { true } // GPT-2 is a Pre-Norm model
}

impl LanguageModelConfig for Gpt2Config {
    fn vocab_size(&self) -> usize { self.vocab_size }
    fn max_position_embeddings(&self) -> usize { self.n_ctx }
    fn intermediate_size(&self) -> usize { self.n_embd * 4 } // Standard for GPT-2
    fn transpose_ffn_weights(&self) -> bool { false } // GPT-2 weights are not transposed
    fn transpose_attention_weights(&self) -> bool { false }
}

impl DecoderArchitecture for Gpt2Config {
    // These tensor names are taken directly from your old `GPTBase` implementation
    fn get_embedding_weight_names(&self) -> (&str, &str) { ("transformer.wte.weight", "transformer.wpe.weight") }
    fn get_final_layer_norm_names(&self) -> (&str, &str) { ("transformer.ln_f.weight", "transformer.ln_f.bias") }
    fn get_lm_head_name(&self) -> &str { "transformer.wte.weight" } // Shares weights with word embeddings

    fn get_attention_names(&self, i: usize) -> LayerDecoderAttentionNames {
        LayerDecoderAttentionNames {
            qkv_weight: format!("transformer.h.{}.attn.c_attn.weight", i),
            qkv_bias: format!("transformer.h.{}.attn.c_attn.bias", i),
            output_weight: format!("transformer.h.{}.attn.c_proj.weight", i),
            output_bias: format!("transformer.h.{}.attn.c_proj.bias", i),
            norm_weight: format!("transformer.h.{}.ln_1.weight", i),
            norm_bias: format!("transformer.h.{}.ln_1.bias", i),
        }
    }

    fn get_feed_forward_names(&self, i: usize) -> LayerFeedForwardNames {
        LayerFeedForwardNames {
            intermediate_weight: format!("transformer.h.{}.mlp.c_fc.weight", i),
            intermediate_bias: format!("transformer.h.{}.mlp.c_fc.bias", i),
            output_weight: format!("transformer.h.{}.mlp.c_proj.weight", i),
            output_bias: format!("transformer.h.{}.mlp.c_proj.bias", i),
            norm_weight: format!("transformer.h.{}.ln_2.weight", i),
            norm_bias: format!("transformer.h.{}.ln_2.bias", i),
        }
    }
}