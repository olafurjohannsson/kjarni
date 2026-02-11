use kjarni_transformers::{
    activations::Activation,
    encoder_decoder::TaskSpecificParams,
    traits::{
        AttentionLayout, DecoderLayerLayout, DecoderLayout, EncoderLayerLayout, EncoderLayout,
        FeedForwardLayout, ModelConfig, ModelLayout, ModelMetadata, NormalizationStrategy,
    },
};
use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, Deserialize, Serialize)]
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
    pub forced_eos_token_id: Option<u32>,

    #[serde(alias = "hidden_act", alias = "activation_function")]
    pub activation_function: Option<String>,

    #[serde(default)]
    pub extra_pos_embeddings: u32,
    pub is_encoder_decoder: bool,
    pub model_type: String,

    pub force_bos_token_to_be_generated: Option<bool>,

    #[serde(skip)]
    pub shared_embedding_key: Option<String>,

    #[serde(default)]
    pub normalize_embedding: bool,

    #[serde(default)]
    pub normalize_before: bool,

    /// Static position embeddings (BART uses learned, not static)
    #[serde(default)]
    pub static_position_embeddings: bool,

    // zero shot
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
    #[serde(default)]
    pub id2label: Option<std::collections::HashMap<String, String>>,
    #[serde(skip)]
    pub labels_vec: Option<Vec<String>>,
}

impl BartConfig {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let mut config: Self = serde_json::from_str(json)?;
    //     if config.extra_pos_embeddings == 0 {
    //     config.extra_pos_embeddings = 2;
    // }
        // Add label processing logic, just like in your other classifier configs.
        if let Some(ref map) = config.id2label {
            let mut labels: Vec<(usize, String)> = map
                .iter()
                .filter_map(|(k, v)| k.parse::<usize>().ok().map(|idx| (idx, v.clone())))
                .collect();
            labels.sort_by_key(|(idx, _)| *idx);
            config.labels_vec = Some(labels.into_iter().map(|(_, v)| v).collect());
        }

        Ok(config)
    }
    pub fn get_forced_eos_token_id(&self) -> Option<u32> {
        self.forced_eos_token_id
    }
    fn get_shared_weight_name(&self) -> String {
        self.shared_embedding_key
            .clone()
            .unwrap_or_else(|| "model.shared.weight".to_string())
    }
    /// Checks if the model architecture is for sequence classification.
    fn is_sequence_classifier(&self) -> bool {
        self.architectures
            .as_ref()
            .map(|archs| archs.iter().any(|a| a == "BartForSequenceClassification"))
            .unwrap_or(false)
    }
    pub fn to_str(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

impl ModelConfig for BartConfig {
    fn model_type(&self) -> &str {
        &self.model_type
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn id2label(&self) -> Option<&[String]> {
        self.labels_vec.as_deref()
    }
    fn eos_token_id(&self) -> Option<u32> {
        Some(self.eos_token_id)
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
            normalize_embedding: self.normalize_embedding,
            extra_pos_embeddings: self.extra_pos_embeddings as usize,
            is_prenorm: false, // BART is Post-Norm
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            problem_type: None,
            normalization_strategy: NormalizationStrategy::LayerNorm,
            no_scale_qk: false,
            decoder_layers: Some(self.decoder_layers),
            intermediate_size: self.intermediate_size(),
        }
    }

    fn intermediate_size(&self) -> usize {
        self.encoder_ffn_dim
    }

    fn layout(&self) -> ModelLayout {
        let shared = self.get_shared_weight_name();

        let encoder_layer = EncoderLayerLayout {
            self_attn: AttentionLayout {
                q_weight: "model.encoder.layers.{}.self_attn.q_proj.weight".to_string(),
                q_bias: Some("model.encoder.layers.{}.self_attn.q_proj.bias".to_string()),
                k_weight: "model.encoder.layers.{}.self_attn.k_proj.weight".to_string(),
                k_bias: Some("model.encoder.layers.{}.self_attn.k_proj.bias".to_string()),
                v_weight: "model.encoder.layers.{}.self_attn.v_proj.weight".to_string(),
                v_bias: Some("model.encoder.layers.{}.self_attn.v_proj.bias".to_string()),
                o_weight: "model.encoder.layers.{}.self_attn.out_proj.weight".to_string(),
                o_bias: Some("model.encoder.layers.{}.self_attn.out_proj.bias".to_string()),
                norm_weight: "model.encoder.layers.{}.self_attn_layer_norm.weight".to_string(),
                norm_bias: Some("model.encoder.layers.{}.self_attn_layer_norm.bias".to_string()),
            },
            ffn: FeedForwardLayout {
                up_weight: "model.encoder.layers.{}.fc1.weight".to_string(),
                up_bias: Some("model.encoder.layers.{}.fc1.bias".to_string()),
                down_weight: "model.encoder.layers.{}.fc2.weight".to_string(),
                down_bias: Some("model.encoder.layers.{}.fc2.bias".to_string()),
                gate_weight: None,
                gate_bias: None,
                norm_weight: "model.encoder.layers.{}.final_layer_norm.weight".to_string(),
                norm_bias: Some("model.encoder.layers.{}.final_layer_norm.bias".to_string()),
            },
        };

        let decoder_layer = DecoderLayerLayout {
            self_attn: AttentionLayout {
                q_weight: "model.decoder.layers.{}.self_attn.q_proj.weight".to_string(),
                q_bias: Some("model.decoder.layers.{}.self_attn.q_proj.bias".to_string()),
                k_weight: "model.decoder.layers.{}.self_attn.k_proj.weight".to_string(),
                k_bias: Some("model.decoder.layers.{}.self_attn.k_proj.bias".to_string()),
                v_weight: "model.decoder.layers.{}.self_attn.v_proj.weight".to_string(),
                v_bias: Some("model.decoder.layers.{}.self_attn.v_proj.bias".to_string()),
                o_weight: "model.decoder.layers.{}.self_attn.out_proj.weight".to_string(),
                o_bias: Some("model.decoder.layers.{}.self_attn.out_proj.bias".to_string()),
                norm_weight: "model.decoder.layers.{}.self_attn_layer_norm.weight".to_string(),
                norm_bias: Some("model.decoder.layers.{}.self_attn_layer_norm.bias".to_string()),
            },
            cross_attn: Some(AttentionLayout {
                q_weight: "model.decoder.layers.{}.encoder_attn.q_proj.weight".to_string(),
                q_bias: Some("model.decoder.layers.{}.encoder_attn.q_proj.bias".to_string()),
                k_weight: "model.decoder.layers.{}.encoder_attn.k_proj.weight".to_string(),
                k_bias: Some("model.decoder.layers.{}.encoder_attn.k_proj.bias".to_string()),
                v_weight: "model.decoder.layers.{}.encoder_attn.v_proj.weight".to_string(),
                v_bias: Some("model.decoder.layers.{}.encoder_attn.v_proj.bias".to_string()),
                o_weight: "model.decoder.layers.{}.encoder_attn.out_proj.weight".to_string(),
                o_bias: Some("model.decoder.layers.{}.encoder_attn.out_proj.bias".to_string()),
                norm_weight: "model.decoder.layers.{}.encoder_attn_layer_norm.weight".to_string(),
                norm_bias: Some("model.decoder.layers.{}.encoder_attn_layer_norm.bias".to_string()),
            }),
            ffn: FeedForwardLayout {
                up_weight: "model.decoder.layers.{}.fc1.weight".to_string(),
                up_bias: Some("model.decoder.layers.{}.fc1.bias".to_string()),
                down_weight: "model.decoder.layers.{}.fc2.weight".to_string(),
                down_bias: Some("model.decoder.layers.{}.fc2.bias".to_string()),
                gate_weight: None,
                gate_bias: None,
                norm_weight: "model.decoder.layers.{}.final_layer_norm.weight".to_string(),
                norm_bias: Some("model.decoder.layers.{}.final_layer_norm.bias".to_string()),
            },
        };

        ModelLayout {
            token_embedding: shared.clone(),
            lm_head: shared,
            encoder: Some(EncoderLayout {
                position_embedding: Some("model.encoder.embed_positions.weight".to_string()),
                token_type_embedding: None,
                embedding_norm_weight: Some("model.encoder.layernorm_embedding.weight".to_string()),
                embedding_norm_bias: Some("model.encoder.layernorm_embedding.bias".to_string()),
                final_norm_weight: None, // BART encoder doesn't have a final norm
                final_norm_bias: None,
                layer: encoder_layer,
            }),
            decoder: Some(DecoderLayout {
                position_embedding: Some("model.decoder.embed_positions.weight".to_string()),
                token_type_embedding: None,
                embedding_norm_weight: Some("model.decoder.layernorm_embedding.weight".to_string()),
                embedding_norm_bias: Some("model.decoder.layernorm_embedding.bias".to_string()),
                final_norm_weight: None, // BART decoder doesn't have a final norm before the LM head
                final_norm_bias: None,
                layer: decoder_layer,
            }),
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

        let encoder_layout = layout
            .encoder
            .as_ref()
            .expect("BART should have an encoder layout");
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("BART should have a decoder layout");

        assert_eq!(layout.token_embedding, "model.shared.weight");
        assert_eq!(layout.lm_head, "model.shared.weight");

        assert_eq!(
            encoder_layout.layer.self_attn.q_weight.replace("{}", "5"),
            "model.encoder.layers.5.self_attn.q_proj.weight"
        );
        assert_eq!(
            encoder_layout.layer.ffn.up_weight.replace("{}", "0"),
            "model.encoder.layers.0.fc1.weight"
        );
        assert_eq!(
            encoder_layout.embedding_norm_weight.as_ref().unwrap(),
            "model.encoder.layernorm_embedding.weight"
        );

        assert_eq!(
            decoder_layout.layer.self_attn.q_weight.replace("{}", "3"),
            "model.decoder.layers.3.self_attn.q_proj.weight"
        );

        let cross_attn_layout = decoder_layout.layer.cross_attn.as_ref().unwrap();
        assert_eq!(
            cross_attn_layout.q_weight.replace("{}", "1"),
            "model.decoder.layers.1.encoder_attn.q_proj.weight"
        );
        assert_eq!(
            cross_attn_layout
                .norm_bias
                .as_ref()
                .unwrap()
                .replace("{}", "4"),
            "model.decoder.layers.4.encoder_attn_layer_norm.bias"
        );

        assert_eq!(
            decoder_layout
                .layer
                .ffn
                .down_bias
                .as_ref()
                .unwrap()
                .replace("{}", "2"),
            "model.decoder.layers.2.fc2.bias"
        );
    }
}
