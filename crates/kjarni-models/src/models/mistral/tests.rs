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
            "model_type": "mistral",
            "bos_token_id": [2, 3],
            "eos_token_id": [1, 5]
        }"#;

        let config: MistralConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.model_type(), "mistral");
        assert_eq!(config.vocab_size(), 32000);
        assert_eq!(config.hidden_size(), 4096);
        assert_eq!(config.intermediate_size, 14336);
        assert_eq!(config.num_attention_heads(), 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.hidden_act, "silu");
        assert_eq!(config.model_type, "mistral");
    }
}

/// Mistral against transformers, on the tiny checkpoint in `testdata`.
mod tiny {
    use std::path::PathBuf;

    use kjarni_transformers::Device;

    use super::super::MistralModel;
    use crate::models::tiny_fixture::assert_matches_transformers;

    fn dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models/mistral/testdata")
    }

    #[test]
    fn logits_and_greedy_decoding_match_transformers() {
        let model = MistralModel::from_pretrained(&dir(), Device::Cpu, None, None, None).unwrap();
        assert_matches_transformers(&model, &dir(), 1e-4);
    }
}
