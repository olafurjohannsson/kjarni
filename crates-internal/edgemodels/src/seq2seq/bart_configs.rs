use edgetransformers::traits::{
    EncoderDecoderArchitecture, LanguageModelConfig, LayerAttentionNames, LayerFeedForwardNames,
    TransformerConfig,
};
use serde::Deserialize;
use std::any::Any;
fn default_layer_norm_eps() -> f32 {
    1e-5
}

pub trait BartLikeConfig {
    fn task_specific_params(&self) -> Option<&TaskSpecificParams>;
}

impl BartLikeConfig for BartConfig {
    fn task_specific_params(&self) -> Option<&TaskSpecificParams> {
        self.task_specific_params.as_ref()
    }
}

#[derive(Debug, Clone, Deserialize, Copy)]
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
    pub bos_token_id: u32,
    pub pad_token_id: u32,
    pub decoder_start_token_id: u32,

    pub extra_pos_embeddings: u32,
    pub is_encoder_decoder: bool,
    pub model_type: String,
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
    fn as_any(&self) -> &dyn Any {
        self // Simply return a reference to self as a `&dyn Any`
    }
    fn eos_token_id(&self) -> Option<u32> {
        Some(self.eos_token_id)
    }
    fn bos_token_id(&self) -> Option<u32> {
        Some(self.bos_token_id)
    }
    fn pad_token_id(&self) -> Option<u32> {
        Some(self.pad_token_id)
    }
    fn extra_pos_embeddings(&self) -> Option<u32> {
        Some(self.extra_pos_embeddings)
    }
    fn is_encoder_decoder(&self) -> Option<bool> {
        Some(self.is_encoder_decoder)
    } 
    fn model_type(&self) -> Option<String> {
        Some(self.model_type.clone())
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

    // fn eos_token_id(&self) -> u32 {
    //     self.eos_token_id
    // }
    fn decoder_start_token_id(&self) -> u32 {
        self.decoder_start_token_id
    }

    fn num_encoder_layers(&self) -> usize {
        self.encoder_layers
    }
    fn num_decoder_layers(&self) -> usize {
        self.decoder_layers
    }
    // fn as_any(&self) -> &dyn Any {
    //     self // Simply return a reference to self as a `&dyn Any`
    // }

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

mod tests {
    use super::*;

    const DISTILBART_CNN_CONFIG_JSON: &str = r#"{
      "d_model": 1024,
      "decoder_attention_heads": 16,
      "decoder_ffn_dim": 4096,
      "decoder_layers": 6,
      "decoder_start_token_id": 2,
      "encoder_attention_heads": 16,
      "encoder_ffn_dim": 4096,
      "encoder_layers": 12,
      "eos_token_id": 2,
      "bos_token_id": 0,
      "pad_token_id": 1,
      "is_encoder_decoder": true,
      "extra_pos_embeddings": 2,
      "max_position_embeddings": 1024,
      "model_type": "bart",
      "scale_embedding": false,
      "vocab_size": 50264,
      "task_specific_params": {
        "summarization": {
          "early_stopping": true,
          "length_penalty": 2.0,
          "max_length": 142,
          "min_length": 56,
          "no_repeat_ngram_size": 3,
          "num_beams": 4
        }
      }
    }"#;

    #[test]
    fn test_distilbart_deserialization() -> Result<(), serde_json::Error> {
        let config: BartConfig = serde_json::from_str(DISTILBART_CNN_CONFIG_JSON)?;

        // Test top-level fields
        assert_eq!(config.d_model, 1024);
        assert_eq!(config.hidden_size(), 1024); // Test trait method
        assert_eq!(config.encoder_layers, 12);
        assert_eq!(config.decoder_layers, 6);
        assert_eq!(config.vocab_size, 50264);
        assert_eq!(config.decoder_start_token_id, 2);
        assert_eq!(config.eos_token_id, 2);
        assert!(!config.scale_embedding);

        // Test nested task-specific params (very important!)
        let params = config
            .task_specific_params
            .expect("Task specific params should be present");
        assert_eq!(params.summarization.num_beams, 4);
        assert_eq!(params.summarization.max_length, 142);
        assert_eq!(params.summarization.min_length, 56);
        assert_eq!(params.summarization.length_penalty, 2.0);
        assert!(params.summarization.early_stopping);

        Ok(())
    }

    #[test]
    fn test_bart_weight_names() {
        // We can just create a default config for this test
        let config: BartConfig = serde_json::from_str(DISTILBART_CNN_CONFIG_JSON).unwrap();

        // --- Shared ---
        assert_eq!(
            config.get_shared_embedding_weight_name(),
            "model.shared.weight"
        );
        assert_eq!(config.get_lm_head_name(), "model.shared.weight");
        assert_eq!(
            config.get_final_logits_bias_name(),
            Some("final_logits_bias")
        );

        // --- Encoder ---
        let (embed, pos_embed, _) = config.get_encoder_embedding_names();
        assert_eq!(embed, "model.shared.weight");
        assert_eq!(pos_embed, "model.encoder.embed_positions.weight");

        let attn_names = config.get_encoder_attention_names(5); // Test an arbitrary layer index
        assert_eq!(
            attn_names.q_weight,
            "model.encoder.layers.5.self_attn.q_proj.weight"
        );
        assert_eq!(
            attn_names.output_bias,
            "model.encoder.layers.5.self_attn.out_proj.bias"
        );
        assert_eq!(
            attn_names.norm_weight,
            "model.encoder.layers.5.self_attn_layer_norm.weight"
        );

        // --- Decoder ---
        let (embed, pos_embed) = config.get_decoder_embedding_names();
        assert_eq!(embed, "model.shared.weight");
        assert_eq!(pos_embed, "model.decoder.embed_positions.weight");

        let self_attn_names = config.get_decoder_self_attention_names(3);
        assert_eq!(
            self_attn_names.k_weight,
            "model.decoder.layers.3.self_attn.k_proj.weight"
        );
        assert_eq!(
            self_attn_names.v_bias,
            "model.decoder.layers.3.self_attn.v_proj.bias"
        );

        let cross_attn_names = config.get_decoder_cross_attention_names(1);
        assert_eq!(
            cross_attn_names.q_weight,
            "model.decoder.layers.1.encoder_attn.q_proj.weight"
        );
        assert_eq!(
            cross_attn_names.output_weight,
            "model.decoder.layers.1.encoder_attn.out_proj.weight"
        );

        let ffn_names = config.get_decoder_feed_forward_names(0);
        assert_eq!(
            ffn_names.intermediate_weight,
            "model.decoder.layers.0.fc1.weight"
        );
        assert_eq!(
            ffn_names.norm_bias,
            "model.decoder.layers.0.final_layer_norm.bias"
        );
    }
}
