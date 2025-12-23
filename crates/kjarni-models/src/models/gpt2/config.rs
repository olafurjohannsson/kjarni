use kjarni_transformers::activations::Activation;
use kjarni_transformers::traits::{
    AttentionLayout, DecoderLayerLayout, DecoderLayout, FeedForwardLayout, ModelConfig,
    ModelLayout, ModelMetadata,
};
use serde::Deserialize;

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

fn default_model_type() -> String {
    "gpt2".to_string()
}

impl Gpt2Config {
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    fn is_distil(&self) -> bool {
        self.model_type == "distilgpt2"
    }

    pub fn set_model_type(&mut self, model_type: String) {
        self.model_type = model_type
    }
}

impl ModelConfig for Gpt2Config {
    fn model_type(&self) -> &str {
        &self.model_type
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            hidden_size: self.n_embd,
            num_layers: self.n_layer,
            num_attention_heads: self.n_head,
            num_kv_heads: self.n_head, // GPT-2 does not use GQA
            head_dim: self.n_embd / self.n_head,
            vocab_size: self.vocab_size,
            max_seq_len: self.n_ctx,
            norm_eps: self.layer_norm_epsilon,
            activation: self
                .activation_function
                .as_ref()
                .and_then(|s| s.parse().ok())
                .unwrap_or(Activation::GeluNew),
            rope_theta: None, // GPT-2 uses absolute learned position embeddings
            rope_scaling: None,
            scale_embeddings: false,
            extra_pos_embeddings: 0,
            is_prenorm: true, // GPT-2 is a Pre-Norm model
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
        }
    }

    fn layout(&self) -> ModelLayout {
        // Handle the "transformer." prefix used by DistilGPT2
        let p = if self.is_distil() { "transformer." } else { "" };
        let lp = if self.is_distil() {
            "transformer.h.{}"
        } else {
            "h.{}"
        };

        // --- Define the Decoder's Layer Structure ---
        let decoder_layer = DecoderLayerLayout {
            self_attn: AttentionLayout {
                // NOTE: GPT-2 uses a fused QKV weight matrix. We point q_weight to it
                // and leave k/v empty to signal to the loader that it's a fused operation.
                // The loader for the GPT-2 layer will need to handle this special case.
                q_weight: format!("{}.attn.c_attn.weight", lp),
                q_bias: Some(format!("{}.attn.c_attn.bias", lp)),
                k_weight: String::new(), // Empty indicates fused with Q
                k_bias: None,
                v_weight: String::new(), // Empty indicates fused with Q
                v_bias: None,
                o_weight: format!("{}.attn.c_proj.weight", lp),
                o_bias: Some(format!("{}.attn.c_proj.bias", lp)),
                norm_weight: format!("{}.ln_1.weight", lp),
                norm_bias: Some(format!("{}.ln_1.bias", lp)),
            },
            cross_attn: None, // GPT-2 is decoder-only
            ffn: FeedForwardLayout {
                up_weight: format!("{}.mlp.c_fc.weight", lp),
                up_bias: Some(format!("{}.mlp.c_fc.bias", lp)),
                down_weight: format!("{}.mlp.c_proj.weight", lp),
                down_bias: Some(format!("{}.mlp.c_proj.bias", lp)),
                gate_weight: None, // GPT-2 uses GELU, not SwiGLU
                norm_weight: format!("{}.ln_2.weight", lp),
                norm_bias: Some(format!("{}.ln_2.bias", lp)),
            },
        };

        // --- Assemble the final ModelLayout ---
        ModelLayout {
            token_embedding: format!("{}wte.weight", p),
            lm_head: format!("{}wte.weight", p), // GPT-2 ties weights
            encoder: None,                       // GPT-2 is decoder-only
            decoder: Some(DecoderLayout {
                position_embedding: Some(format!("{}wpe.weight", p)),
                token_type_embedding: None,
                embedding_norm_weight: None, // GPT-2 has no embedding norm
                embedding_norm_bias: None,
                final_norm_weight: Some(format!("{}ln_f.weight", p)),
                final_norm_bias: Some(format!("{}ln_f.bias", p)),
                layer: decoder_layer,
            }),
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
    fn test_gpt2_metadata_and_layout() {
        let config: Gpt2Config = serde_json::from_str(GPT2_CONFIG_JSON).unwrap();
        let meta = config.metadata();
        let layout = config.layout();

        // --- Get the nested decoder layout ---
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("GPT-2 should have a decoder layout");
        let layer_layout = &decoder_layout.layer;

        // --- Verify Architectural Correctness ---
        assert!(
            layout.encoder.is_none(),
            "GPT-2 is decoder-only, encoder should be None"
        );
        assert!(
            layer_layout.cross_attn.is_none(),
            "GPT-2 has no cross-attention"
        );

        // --- Verify Naming Conventions ---
        assert_eq!(meta.hidden_size, 768);
        assert_eq!(layout.token_embedding, "wte.weight");
        assert_eq!(
            decoder_layout.position_embedding.as_ref().unwrap(),
            "wpe.weight"
        );
        assert_eq!(layout.lm_head, "wte.weight");

        // Verify a self-attention name
        assert_eq!(
            layer_layout.self_attn.q_weight.replace("{}", "0"),
            "h.0.attn.c_attn.weight"
        );

        // Verify an FFN name
        assert_eq!(
            layer_layout.ffn.down_weight.replace("{}", "11"),
            "h.11.mlp.c_proj.weight"
        );

        // Verify a final norm name
        assert_eq!(
            decoder_layout.final_norm_weight.as_ref().unwrap(),
            "ln_f.weight"
        );
    }

    #[test]
    fn test_distilgpt2_metadata_and_layout() {
        let config: Gpt2Config = serde_json::from_str(DISTILGPT2_CONFIG_JSON).unwrap();
        let meta = config.metadata();
        let layout = config.layout();

        // --- Get the nested decoder layout ---
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("DistilGPT2 should have a decoder layout");
        let layer_layout = &decoder_layout.layer;

        // --- Verify Architectural Correctness ---
        assert!(
            layout.encoder.is_none(),
            "DistilGPT2 is decoder-only, encoder should be None"
        );
        assert!(
            layer_layout.cross_attn.is_none(),
            "DistilGPT2 has no cross-attention"
        );

        // --- Verify Naming Conventions ---
        assert_eq!(meta.num_layers, 6);
        assert_eq!(layout.token_embedding, "transformer.wte.weight");
        assert_eq!(
            decoder_layout.position_embedding.as_ref().unwrap(),
            "transformer.wpe.weight"
        );
        assert_eq!(layout.lm_head, "transformer.wte.weight");

        // Verify an FFN name
        assert_eq!(
            layer_layout.ffn.up_weight.replace("{}", "0"),
            "transformer.h.0.mlp.c_fc.weight"
        );

        // Verify a self-attention name
        assert_eq!(
            layer_layout
                .self_attn
                .o_bias
                .as_ref()
                .unwrap()
                .replace("{}", "5"),
            "transformer.h.5.attn.c_proj.bias"
        );
    }
}
