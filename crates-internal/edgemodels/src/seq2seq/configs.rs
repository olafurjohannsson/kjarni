use edgetransformers::traits::{
    EncoderDecoderArchitecture, LanguageModelConfig, LayerAttentionNames,
    LayerFeedForwardNames, TransformerConfig,
};
use serde::Deserialize;
use std::any::Any;
fn default_layer_norm_eps() -> f32 {
    1e-5
}

#[derive(Debug, Clone, Deserialize)]
#[allow(non_snake_case)] // To allow serde to match the camelCase keys
pub struct SummarizationParams {
    pub early_stopping: bool,
    pub length_penalty: f32,
    pub max_length: usize,
    pub min_length: usize,
    pub no_repeat_ngram_size: usize,
    pub num_beams: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(non_snake_case)]
pub struct TaskSpecificParams {
    pub summarization: SummarizationParams,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BartConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,

    pub task_specific_params: Option<TaskSpecificParams>,

    #[serde(default)]
    pub scale_embedding: bool,
    #[serde(default = "default_layer_norm_eps", alias = "layer_norm_epsilon")]
    pub layer_norm_eps: f32,
    pub eos_token_id: u32,
    pub decoder_start_token_id: u32,
}

impl TransformerConfig for BartConfig {
    fn hidden_size(&self) -> usize {
        self.d_model
    }
    fn num_attention_heads(&self) -> usize {
        self.encoder_attention_heads
    } // Use encoder's as default
    fn num_hidden_layers(&self) -> usize {
        self.encoder_layers
    } // Use encoder's as default
    fn layer_norm_eps(&self) -> f32 {
        self.layer_norm_eps
    }
    fn is_causal(&self) -> bool {
        false
    } // Not relevant for the top-level config
    fn is_prenorm(&self) -> bool {
        false
    } // BART is post-norm
}

impl LanguageModelConfig for BartConfig {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
    fn intermediate_size(&self) -> usize {
        self.encoder_ffn_dim
    } // Use encoder's as default
    fn transpose_ffn_weights(&self) -> bool {
        true
    } // PyTorch linear layers need transposing
    fn transpose_attention_weights(&self) -> bool {
        true
    }
}

impl EncoderDecoderArchitecture for BartConfig {
    // --- Shared ---
    fn get_shared_embedding_weight_name(&self) -> &str {
        "model.shared.weight"
    }
    fn get_lm_head_name(&self) -> &str {
        "model.shared.weight"
    } // BART shares embedding and LM head
    fn get_final_logits_bias_name(&self) -> Option<&str> {
        Some("final_logits_bias")
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
    fn decoder_start_token_id(&self) -> u32 {
        self.decoder_start_token_id
    }

    fn num_encoder_layers(&self) -> usize {
        self.encoder_layers
    }
    fn num_decoder_layers(&self) -> usize {
        self.decoder_layers
    }
    fn as_any(&self) -> &dyn Any {
        self // Simply return a reference to self as a `&dyn Any`
    }

    // --- Encoder Methods ---
    fn get_encoder_embedding_names(&self) -> (&str, &str, Option<&str>) {
        (
            "model.shared.weight",
            "model.encoder.embed_positions.weight",
            None,
        )
    }
    fn get_encoder_embedding_ln_names(&self) -> (&str, &str) {
        (
            "model.encoder.layernorm_embedding.weight",
            "model.encoder.layernorm_embedding.bias",
        )
    }
    fn get_encoder_attention_names(&self, i: usize) -> LayerAttentionNames {
        LayerAttentionNames {
            q_weight: format!("model.encoder.layers.{}.self_attn.q_proj.weight", i),
            q_bias: format!("model.encoder.layers.{}.self_attn.q_proj.bias", i),
            k_weight: format!("model.encoder.layers.{}.self_attn.k_proj.weight", i),
            k_bias: format!("model.encoder.layers.{}.self_attn.k_proj.bias", i),
            v_weight: format!("model.encoder.layers.{}.self_attn.v_proj.weight", i),
            v_bias: format!("model.encoder.layers.{}.self_attn.v_proj.bias", i),
            output_weight: format!("model.encoder.layers.{}.self_attn.out_proj.weight", i),
            output_bias: format!("model.encoder.layers.{}.self_attn.out_proj.bias", i),
            norm_weight: format!("model.encoder.layers.{}.self_attn_layer_norm.weight", i),
            norm_bias: format!("model.encoder.layers.{}.self_attn_layer_norm.bias", i),
        }
    }
    fn get_encoder_feed_forward_names(&self, i: usize) -> LayerFeedForwardNames {
        LayerFeedForwardNames {
            intermediate_weight: format!("model.encoder.layers.{}.fc1.weight", i),
            intermediate_bias: format!("model.encoder.layers.{}.fc1.bias", i),
            output_weight: format!("model.encoder.layers.{}.fc2.weight", i),
            output_bias: format!("model.encoder.layers.{}.fc2.bias", i),
            norm_weight: format!("model.encoder.layers.{}.final_layer_norm.weight", i),
            norm_bias: format!("model.encoder.layers.{}.final_layer_norm.bias", i),
            gate_weight: None,
        }
    }

    // --- Decoder Methods ---
    fn get_decoder_embedding_names(&self) -> (&str, &str) {
        (
            "model.shared.weight",
            "model.decoder.embed_positions.weight",
        )
    }
    fn get_decoder_embedding_ln_names(&self) -> (&str, &str) {
        (
            "model.decoder.layernorm_embedding.weight",
            "model.decoder.layernorm_embedding.bias",
        )
    }
    fn get_decoder_self_attention_names(&self, i: usize) -> LayerAttentionNames {
        LayerAttentionNames {
            q_weight: format!("model.decoder.layers.{}.self_attn.q_proj.weight", i),
            q_bias: format!("model.decoder.layers.{}.self_attn.q_proj.bias", i),
            k_weight: format!("model.decoder.layers.{}.self_attn.k_proj.weight", i),
            k_bias: format!("model.decoder.layers.{}.self_attn.k_proj.bias", i),
            v_weight: format!("model.decoder.layers.{}.self_attn.v_proj.weight", i),
            v_bias: format!("model.decoder.layers.{}.self_attn.v_proj.bias", i),
            output_weight: format!("model.decoder.layers.{}.self_attn.out_proj.weight", i),
            output_bias: format!("model.decoder.layers.{}.self_attn.out_proj.bias", i),
            norm_weight: format!("model.decoder.layers.{}.self_attn_layer_norm.weight", i),
            norm_bias: format!("model.decoder.layers.{}.self_attn_layer_norm.bias", i),
        }
    }
    fn get_decoder_cross_attention_names(&self, i: usize) -> LayerAttentionNames {
        LayerAttentionNames {
            q_weight: format!("model.decoder.layers.{}.encoder_attn.q_proj.weight", i),
            q_bias: format!("model.decoder.layers.{}.encoder_attn.q_proj.bias", i),
            k_weight: format!("model.decoder.layers.{}.encoder_attn.k_proj.weight", i),
            k_bias: format!("model.decoder.layers.{}.encoder_attn.k_proj.bias", i),
            v_weight: format!("model.decoder.layers.{}.encoder_attn.v_proj.weight", i),
            v_bias: format!("model.decoder.layers.{}.encoder_attn.v_proj.bias", i),
            output_weight: format!("model.decoder.layers.{}.encoder_attn.out_proj.weight", i),
            output_bias: format!("model.decoder.layers.{}.encoder_attn.out_proj.bias", i),
            norm_weight: format!("model.decoder.layers.{}.encoder_attn_layer_norm.weight", i),
            norm_bias: format!("model.decoder.layers.{}.encoder_attn_layer_norm.bias", i),
        }
    }
    fn get_decoder_feed_forward_names(&self, i: usize) -> LayerFeedForwardNames {
        LayerFeedForwardNames {
            intermediate_weight: format!("model.decoder.layers.{}.fc1.weight", i),
            intermediate_bias: format!("model.decoder.layers.{}.fc1.bias", i),
            output_weight: format!("model.decoder.layers.{}.fc2.weight", i),
            output_bias: format!("model.decoder.layers.{}.fc2.bias", i),
            norm_weight: format!("model.decoder.layers.{}.final_layer_norm.weight", i),
            norm_bias: format!("model.decoder.layers.{}.final_layer_norm.bias", i),
            gate_weight: None,
        }
    }
}
