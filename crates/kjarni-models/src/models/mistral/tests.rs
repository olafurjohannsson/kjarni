#[cfg(test)]
mod tests {
    use super::super::config::MistralConfig;
    use kjarni_transformers::traits::ModelConfig;

    #[test]
    fn test_mistral_config_deserialization() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_act": "silu",
            "max_position_embeddings": 32768,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "model_type": "mistral"
        }"#;

        let config: MistralConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.model_type(), "mistral");
    }
}