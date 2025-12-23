use kjarni_transformers::{
    activations::Activation,
    encoder_decoder::TaskSpecificParams,
    traits::{ModelConfig, ModelLayout, ModelMetadata},
};
use serde::Deserialize;

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
    pub forced_bos_token_id: Option<u32>,

    #[serde(alias = "hidden_act", alias = "activation_function")]
    pub activation_function: Option<String>,

    #[serde(default)]
    pub extra_pos_embeddings: u32,
    pub is_encoder_decoder: bool,
    pub model_type: String,

    pub force_bos_token_to_be_generated: Option<bool>,

    #[serde(skip)]
    pub shared_embedding_key: Option<String>,
}

impl BartConfig {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    fn get_shared_weight_name(&self) -> String {
        self.shared_embedding_key
            .clone()
            .unwrap_or_else(|| "model.shared.weight".to_string())
    }
}

impl ModelConfig for BartConfig {
    fn model_type(&self) -> &str {
        &self.model_type
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            hidden_size: self.d_model,
            num_layers: self.encoder_layers, // Defaults to encoder layers for backbone
            num_attention_heads: self.encoder_attention_heads,
            num_kv_heads: self.encoder_attention_heads, // BART is not GQA
            head_dim: self.d_model / self.encoder_attention_heads,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_position_embeddings,
            norm_eps: self.layer_norm_eps,
            activation: match self.activation_function.as_deref() {
                Some("gelu") => Activation::Gelu,
                Some("gelu_new") => Activation::GeluNew,
                Some("silu") => Activation::SilU,
                _ => Activation::Gelu, // BART default
            },
            rope_theta: None, // BART uses learned absolute positions
            rope_scaling: None,
            scale_embeddings: self.scale_embedding,
            extra_pos_embeddings: self.extra_pos_embeddings as usize,
            is_prenorm: false, // BART is Post-Norm
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
        }
    }

    fn layout(&self) -> ModelLayout {
        let shared = self.get_shared_weight_name();

        ModelLayout {
            // --- Shared Embeddings ---
            token_embedding: shared.clone(),
            position_embedding: Some("model.encoder.embed_positions.weight".to_string()),
            token_type_embedding: None,
            embedding_norm: Some("model.encoder.layernorm_embedding.weight".to_string()),
            embedding_norm_bias: Some("model.encoder.layernorm_embedding.bias".to_string()),

            // BART ties LM Head and Shared Weight
            lm_head: shared,
            final_norm: "model.encoder.layernorm_embedding.weight".to_string(), // Simplified
            final_norm_bias: None,

            // --- Encoder Layer Templates ---
            attn_q: "model.encoder.layers.{}.self_attn.q_proj.weight".to_string(),
            attn_q_bias: Some("model.encoder.layers.{}.self_attn.q_proj.bias".to_string()),
            attn_k: "model.encoder.layers.{}.self_attn.k_proj.weight".to_string(),
            attn_k_bias: Some("model.encoder.layers.{}.self_attn.k_proj.bias".to_string()),
            attn_v: "model.encoder.layers.{}.self_attn.v_proj.weight".to_string(),
            attn_v_bias: Some("model.encoder.layers.{}.self_attn.v_proj.bias".to_string()),
            attn_o: "model.encoder.layers.{}.self_attn.out_proj.weight".to_string(),
            attn_o_bias: Some("model.encoder.layers.{}.self_attn.out_proj.bias".to_string()),
            attn_norm: "model.encoder.layers.{}.self_attn_layer_norm.weight".to_string(),
            attn_norm_bias: Some("model.encoder.layers.{}.self_attn_layer_norm.bias".to_string()),

            ffn_gate: None,
            ffn_up: "model.encoder.layers.{}.fc1.weight".to_string(),
            ffn_up_bias: Some("model.encoder.layers.{}.fc1.bias".to_string()),
            ffn_down: "model.encoder.layers.{}.fc2.weight".to_string(),
            ffn_down_bias: Some("model.encoder.layers.{}.fc2.bias".to_string()),
            ffn_norm: "model.encoder.layers.{}.final_layer_norm.weight".to_string(),
            ffn_norm_bias: Some("model.encoder.layers.{}.final_layer_norm.bias".to_string()),

            // --- Decoder Cross-Attention Templates ---
            cross_attn_q: Some("model.decoder.layers.{}.encoder_attn.q_proj.weight".to_string()),
            cross_attn_q_bias: Some("model.decoder.layers.{}.encoder_attn.q_proj.bias".to_string()),

            cross_attn_k: Some("model.decoder.layers.{}.encoder_attn.k_proj.weight".to_string()),
            cross_attn_k_bias: Some("model.decoder.layers.{}.encoder_attn.k_proj.bias".to_string()),

            cross_attn_v: Some("model.decoder.layers.{}.encoder_attn.v_proj.weight".to_string()),
            cross_attn_v_bias: Some("model.decoder.layers.{}.encoder_attn.v_proj.bias".to_string()),

            cross_attn_o: Some("model.decoder.layers.{}.encoder_attn.out_proj.weight".to_string()),
            cross_attn_o_bias: Some(
                "model.decoder.layers.{}.encoder_attn.out_proj.bias".to_string(),
            ),

            cross_attn_norm: Some(
                "model.decoder.layers.{}.encoder_attn_layer_norm.weight".to_string(),
            ),
            cross_attn_norm_bias: Some(
                "model.decoder.layers.{}.encoder_attn_layer_norm.bias".to_string(),
            ),
        }
    }
}

#[cfg(test)]
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
    fn test_distilbart_metadata() {
        let config: BartConfig = serde_json::from_str(DISTILBART_CNN_CONFIG_JSON).unwrap();
        let meta = config.metadata();

        assert_eq!(meta.hidden_size, 1024);
        assert_eq!(meta.num_layers, 12);
        assert_eq!(meta.vocab_size, 50264);
        assert_eq!(meta.extra_pos_embeddings, 2);
        assert!(!meta.scale_embeddings);
    }

    #[test]
    fn test_bart_layout_templates() {
        let config: BartConfig = serde_json::from_str(DISTILBART_CNN_CONFIG_JSON).unwrap();
        let layout = config.layout();

        assert_eq!(layout.token_embedding, "model.shared.weight");
        assert_eq!(
            layout.attn_q.replace("{}", "5"),
            "model.encoder.layers.5.self_attn.q_proj.weight"
        );
        assert_eq!(
            layout.ffn_up.replace("{}", "0"),
            "model.encoder.layers.0.fc1.weight"
        );

        let cross_q = layout.cross_attn_q.unwrap();
        assert_eq!(
            cross_q.replace("{}", "1"),
            "model.decoder.layers.1.encoder_attn.q_proj.weight"
        );
    }
}
