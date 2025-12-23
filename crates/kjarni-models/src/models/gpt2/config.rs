use kjarni_transformers::activations::Activation;
use kjarni_transformers::traits::{ModelConfig, ModelLayout, ModelMetadata};
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

        ModelLayout {
            // --- Root Level ---
            token_embedding: format!("{}wte.weight", p),
            position_embedding: Some(format!("{}wpe.weight", p)),
            token_type_embedding: None,
            embedding_norm: None,
            embedding_norm_bias: None,

            final_norm: format!("{}ln_f.weight", p),
            // GPT-2 uses biases in LayerNorm
            final_norm_bias: Some(format!("{}ln_f.bias", p)),

            // GPT-2 shares weights with word embeddings
            lm_head: format!("{}wte.weight", p),

            // --- Attention Templates ---
            // Note: GPT-2 uses fused QKV projections named `c_attn`
            attn_q: format!("{}.attn.c_attn.weight", lp),
            attn_q_bias: Some(format!("{}.attn.c_attn.bias", lp)),

            // k and v are part of c_attn in GPT-2, so we leave templates empty
            // to indicate the model should use the fused projection logic
            attn_k: String::new(),
            attn_k_bias: None,
            attn_v: String::new(),
            attn_v_bias: None,

            attn_o: format!("{}.attn.c_proj.weight", lp),
            attn_o_bias: Some(format!("{}.attn.c_proj.bias", lp)),

            attn_norm: format!("{}.ln_1.weight", lp),
            attn_norm_bias: Some(format!("{}.ln_1.bias", lp)),

            // --- FFN Templates ---
            ffn_gate: None, // GPT-2 is not SwiGLU
            ffn_up: format!("{}.mlp.c_fc.weight", lp),
            ffn_up_bias: Some(format!("{}.mlp.c_fc.bias", lp)),
            ffn_down: format!("{}.mlp.c_proj.weight", lp),
            ffn_down_bias: Some(format!("{}.mlp.c_proj.bias", lp)),
            ffn_norm: format!("{}.ln_2.weight", lp),
            ffn_norm_bias: Some(format!("{}.ln_2.bias", lp)),

            // --- Seq2Seq ---
            cross_attn_q: None,
            cross_attn_k: None,
            cross_attn_v: None,
            cross_attn_o: None,
            cross_attn_norm: None,
            cross_attn_q_bias: None,
            cross_attn_k_bias: None,
            cross_attn_v_bias: None,
            cross_attn_o_bias: None,
            cross_attn_norm_bias: None,
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

        assert_eq!(meta.hidden_size, 768);
        assert_eq!(layout.token_embedding, "wte.weight");
        assert_eq!(layout.position_embedding.unwrap(), "wpe.weight");
        assert_eq!(layout.attn_q.replace("{}", "0"), "h.0.attn.c_attn.weight");
        assert_eq!(layout.lm_head, "wte.weight");
    }

    #[test]
    fn test_distilgpt2_metadata_and_layout() {
        let config: Gpt2Config = serde_json::from_str(DISTILGPT2_CONFIG_JSON).unwrap();
        let meta = config.metadata();
        let layout = config.layout();

        assert_eq!(meta.num_layers, 6);
        assert_eq!(layout.token_embedding, "transformer.wte.weight");
        assert_eq!(layout.position_embedding.unwrap(), "transformer.wpe.weight");
        assert_eq!(
            layout.ffn_up.replace("{}", "0"),
            "transformer.h.0.mlp.c_fc.weight"
        );
        assert_eq!(layout.lm_head, "transformer.wte.weight");
    }
}
