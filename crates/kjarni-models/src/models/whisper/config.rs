use kjarni_transformers::{
    activations::Activation,
    encoder_decoder::TaskSpecificParams,
    traits::{
        AttentionLayout, DecoderLayerLayout, DecoderLayout, EncoderLayerLayout, EncoderLayout,
        FeedForwardLayout, ModelConfig, ModelLayout, ModelMetadata, NormalizationStrategy,
    },
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WhisperConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    pub vocab_size: usize,

    pub max_source_positions: usize,
    pub max_target_positions: usize,

    pub decoder_start_token_id: u32,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    pub pad_token_id: u32,

    pub activation_function: String,

    #[serde(default)]
    pub scale_embedding: bool,

    pub model_type: String,
    pub num_mel_bins: usize,

    pub task_specific_params: Option<TaskSpecificParams>,
}

impl ModelConfig for WhisperConfig {
    fn model_type(&self) -> &str {
        &self.model_type
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    // In WhisperConfig::metadata()
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            hidden_size: self.d_model,
            num_layers: self.encoder_layers,
            num_attention_heads: self.encoder_attention_heads,
            num_kv_heads: self.encoder_attention_heads,
            head_dim: self.d_model / self.encoder_attention_heads,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_target_positions,
            norm_eps: 1e-5,
            activation: match self.activation_function.as_str() {
                "gelu" => Activation::Gelu,
                "silu" => Activation::SilU,
                _ => Activation::Gelu,
            },
            rope_theta: None,
            rope_scaling: None,
            scale_embeddings: self.scale_embedding,
            normalize_embedding: false,
            extra_pos_embeddings: 0,
            is_prenorm: true,
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            normalization_strategy: NormalizationStrategy::LayerNorm,
            no_scale_qk: false,
            decoder_layers: Some(self.decoder_layers),
            intermediate_size: self.decoder_ffn_dim,
        }
    }

    fn layout(&self) -> ModelLayout {
        let shared = "model.decoder.embed_tokens.weight".to_string();

        ModelLayout {
            token_embedding: shared.clone(),
            lm_head: shared.clone(),

            encoder: Some(EncoderLayout {
                position_embedding: Some("model.encoder.embed_positions.weight".to_string()),
                token_type_embedding: None,
                embedding_norm_weight: None,
                embedding_norm_bias: None,
                final_norm_weight: Some("model.encoder.layer_norm.weight".to_string()),
                final_norm_bias: Some("model.encoder.layer_norm.bias".to_string()),
                layer: EncoderLayerLayout {
                    self_attn: AttentionLayout {
                        q_weight: "model.encoder.layers.{}.self_attn.q_proj.weight".to_string(),
                        q_bias: Some("model.encoder.layers.{}.self_attn.q_proj.bias".to_string()),
                        k_weight: "model.encoder.layers.{}.self_attn.k_proj.weight".to_string(),
                        k_bias: Some("model.encoder.layers.{}.self_attn.k_proj.bias".to_string()),
                        v_weight: "model.encoder.layers.{}.self_attn.v_proj.weight".to_string(),
                        v_bias: Some("model.encoder.layers.{}.self_attn.v_proj.bias".to_string()),
                        o_weight: "model.encoder.layers.{}.self_attn.out_proj.weight".to_string(),
                        o_bias: Some("model.encoder.layers.{}.self_attn.out_proj.bias".to_string()),
                        norm_weight: "model.encoder.layers.{}.self_attn_layer_norm.weight"
                            .to_string(),
                        norm_bias: Some(
                            "model.encoder.layers.{}.self_attn_layer_norm.bias".to_string(),
                        ),
                    },
                    ffn: FeedForwardLayout {
                        up_weight: "model.encoder.layers.{}.fc1.weight".to_string(),
                        up_bias: Some("model.encoder.layers.{}.fc1.bias".to_string()),
                        down_weight: "model.encoder.layers.{}.fc2.weight".to_string(),
                        down_bias: Some("model.encoder.layers.{}.fc2.bias".to_string()),
                        gate_weight: None,
                        gate_bias: None,
                        norm_weight: "model.encoder.layers.{}.final_layer_norm.weight".to_string(),
                        norm_bias: Some(
                            "model.encoder.layers.{}.final_layer_norm.bias".to_string(),
                        ),
                    },
                },
            }),

            decoder: Some(DecoderLayout {
                // Whisper Decoder uses learned positions
                position_embedding: Some("model.decoder.embed_positions.weight".to_string()),
                token_type_embedding: None,
                embedding_norm_weight: None, // No embedding norm!
                embedding_norm_bias: None,
                final_norm_weight: Some("model.decoder.layer_norm.weight".to_string()),
                final_norm_bias: Some("model.decoder.layer_norm.bias".to_string()),
                layer: DecoderLayerLayout {
                    self_attn: AttentionLayout {
                        q_weight: "model.decoder.layers.{}.self_attn.q_proj.weight".to_string(),
                        q_bias: Some("model.decoder.layers.{}.self_attn.q_proj.bias".to_string()),
                        k_weight: "model.decoder.layers.{}.self_attn.k_proj.weight".to_string(),
                        k_bias: Some("model.decoder.layers.{}.self_attn.k_proj.bias".to_string()),
                        v_weight: "model.decoder.layers.{}.self_attn.v_proj.weight".to_string(),
                        v_bias: Some("model.decoder.layers.{}.self_attn.v_proj.bias".to_string()),
                        o_weight: "model.decoder.layers.{}.self_attn.out_proj.weight".to_string(),
                        o_bias: Some("model.decoder.layers.{}.self_attn.out_proj.bias".to_string()),
                        norm_weight: "model.decoder.layers.{}.self_attn_layer_norm.weight"
                            .to_string(),
                        norm_bias: Some(
                            "model.decoder.layers.{}.self_attn_layer_norm.bias".to_string(),
                        ),
                    },
                    cross_attn: Some(AttentionLayout {
                        q_weight: "model.decoder.layers.{}.encoder_attn.q_proj.weight".to_string(),
                        q_bias: Some(
                            "model.decoder.layers.{}.encoder_attn.q_proj.bias".to_string(),
                        ),
                        k_weight: "model.decoder.layers.{}.encoder_attn.k_proj.weight".to_string(),
                        k_bias: Some(
                            "model.decoder.layers.{}.encoder_attn.k_proj.bias".to_string(),
                        ),
                        v_weight: "model.decoder.layers.{}.encoder_attn.v_proj.weight".to_string(),
                        v_bias: Some(
                            "model.decoder.layers.{}.encoder_attn.v_proj.bias".to_string(),
                        ),
                        o_weight: "model.decoder.layers.{}.encoder_attn.out_proj.weight"
                            .to_string(),
                        o_bias: Some(
                            "model.decoder.layers.{}.encoder_attn.out_proj.bias".to_string(),
                        ),
                        norm_weight: "model.decoder.layers.{}.encoder_attn_layer_norm.weight"
                            .to_string(),
                        norm_bias: Some(
                            "model.decoder.layers.{}.encoder_attn_layer_norm.bias".to_string(),
                        ),
                    }),
                    ffn: FeedForwardLayout {
                        up_weight: "model.decoder.layers.{}.fc1.weight".to_string(),
                        up_bias: Some("model.decoder.layers.{}.fc1.bias".to_string()),
                        down_weight: "model.decoder.layers.{}.fc2.weight".to_string(),
                        down_bias: Some("model.decoder.layers.{}.fc2.bias".to_string()),
                        gate_weight: None,
                        gate_bias: None,
                        norm_weight: "model.decoder.layers.{}.final_layer_norm.weight".to_string(),
                        norm_bias: Some(
                            "model.decoder.layers.{}.final_layer_norm.bias".to_string(),
                        ),
                    },
                },
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_config() {
        let json = r#"{
            "model_type": "whisper",
            "d_model": 1280,
            "encoder_layers": 32,
            "decoder_layers": 32,
            "encoder_attention_heads": 20,
            "decoder_attention_heads": 20,
            "encoder_ffn_dim": 5120,
            "decoder_ffn_dim": 5120,
            "vocab_size": 51866,
            "max_source_positions": 1500,
            "max_target_positions": 448,
            "decoder_start_token_id": 50258,
            "eos_token_id": 50257,
            "bos_token_id": 50257,
            "pad_token_id": 50256,
            "activation_function": "gelu",
            "scale_embedding": false,
            "num_mel_bins": 128
        }"#;

        let config: WhisperConfig = serde_json::from_str(json).unwrap();
        let meta = config.metadata();
        let layout = config.layout();

        // Verify Whisper-specifics
        assert_eq!(meta.is_prenorm, true);
        assert_eq!(
            meta.normalization_strategy,
            NormalizationStrategy::LayerNorm
        );
        assert!(!meta.scale_embeddings);

        // Verify Encoder Attention Layout (Biased)
        let attn = layout.encoder.unwrap().layer.self_attn;
        assert!(attn.q_bias.is_some());
        assert!(attn.norm_bias.is_some());
    }
}
