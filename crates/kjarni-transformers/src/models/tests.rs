use ndarray::arr2;

use crate::{
    models::{
        base::{AutoregressiveLoop, ModelInput, ModelLoadConfig, RopeScalingConfig},
        registry::ModelTask,
    },
    tensor::DType,
};

use super::*;
use std::path::PathBuf;

// =========================================================================
//  Metadata Tests
// =========================================================================

#[test]
fn test_model_info_retrieval() {
    // Test a specific model (Llama 3.2 1B)
    let model = ModelType::Llama3_2_1B_Instruct;
    let info = model.info();

    assert_eq!(info.architecture, ModelArchitecture::Llama);
    assert_eq!(info.task, ModelTask::Chat);
    assert!(info.description.contains("Meta edge model"));
    assert_eq!(info.params_millions, 1230);
    assert!(info.paths.weights_url.contains("model.safetensors"));
    assert!(info.paths.gguf_url.is_some());
}

#[test]
fn test_embedding_model_info() {
    let model = ModelType::MiniLML6V2;
    let info = model.info();

    assert_eq!(info.architecture, ModelArchitecture::Bert);
    assert_eq!(info.task, ModelTask::Embedding);
    // Embeddings usually don't have GGUF by default in this registry
    assert!(info.paths.gguf_url.is_none());
}

#[test]
fn test_model_input_from_tokens_slice() {
    let tokens = vec![1, 2, 3];
    let input = ModelInput::from_tokens(&tokens);

    assert!(input.is_tokens());
    assert!(!input.is_gpu());
    assert_eq!(input.batch_size(), 1);
    assert_eq!(input.seq_len(), 3);

    if let ModelInput::TokensCpu(view) = input {
        assert_eq!(view, arr2(&[[1, 2, 3]]));
    } else {
        panic!("Wrong variant");
    }
}

#[test]
fn test_model_input_from_2d_array() {
    let data = arr2(&[[1, 2], [3, 4]]); // Batch 2, Seq 2
    let input = ModelInput::from_array(data.view());

    assert_eq!(input.batch_size(), 2);
    assert_eq!(input.seq_len(), 2);
}

// ========================================================================
//  ModelLoadConfig Tests
// ========================================================================

#[test]
fn test_config_defaults() {
    let config = ModelLoadConfig::default();
    assert!(!config.offload_embeddings);
    assert!(!config.offload_lm_head);
    assert!(config.gpu_layers.is_none());
    assert!(config.target_dtype.is_none());
}

#[test]
fn test_config_full_gpu() {
    let config = ModelLoadConfig::full_gpu();
    // Should be same as default basically, implies all layers
    assert!(config.gpu_layers.is_none());
}

#[test]
fn test_config_offload_embeddings() {
    let config = ModelLoadConfig::set_offload_embeddings();
    assert!(config.offload_embeddings);

    // Test chainable builder
    let config2 = ModelLoadConfig::default().with_offload_embeddings(true);
    assert!(config2.offload_embeddings);
}

#[test]
fn test_config_quantized() {
    let config = ModelLoadConfig::quantized(DType::Q4_K);
    assert_eq!(config.target_dtype, Some(DType::Q4_K));

    let config2 = ModelLoadConfig::default().with_target_dtype(DType::Q8_0);
    assert_eq!(config2.target_dtype, Some(DType::Q8_0));
}

#[test]
fn test_config_partial_gpu() {
    let config = ModelLoadConfig::partial_gpu(0, 10);
    assert_eq!(config.gpu_layer_range, Some((0, 10)));

    let config2 = ModelLoadConfig::default().with_gpu_layer_range(5, 15);
    assert_eq!(config2.gpu_layer_range, Some((5, 15)));
}

#[test]
fn test_config_lm_head() {
    let config = ModelLoadConfig::default().with_quantized_lm_head(DType::Q8_0);
    assert_eq!(config.quantize_lm_head, Some(DType::Q8_0));
}

#[test]
fn test_config_prealloc() {
    let config = ModelLoadConfig::default()
        .with_max_batch_size(32)
        .with_max_sequence_length(2048);

    assert_eq!(config.max_batch_size, Some(32));
    assert_eq!(config.max_sequence_length, Some(2048));
}

// ========================================================================
//  RopeScalingConfig Tests
// ========================================================================

#[test]
fn test_rope_scaling_serialization() {
    let config = RopeScalingConfig {
        factor: 8.0,
        high_freq_factor: 4.0,
        low_freq_factor: 1.0,
        original_max_position_embeddings: 8192,
        rope_type: "llama3".to_string(),
    };

    // Verify it can serialize/deserialize (common source of bugs in config loading)
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: RopeScalingConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(config, deserialized);
}

#[test]
fn test_autoregressive_loop_variants() {
    // Just ensuring the enum exists and derives work
    let l1 = AutoregressiveLoop::Pipelined;
    let l2 = AutoregressiveLoop::Legacy;
    assert_ne!(l1, l2);
    let _copy = l1; // Test Copy trait
}
#[test]
fn test_seq2seq_model_info() {
    let model = ModelType::FlanT5Base;
    let info = model.info();

    assert_eq!(info.architecture, ModelArchitecture::T5);
    assert_eq!(info.task, ModelTask::Seq2Seq);
}

// =========================================================================
//  Helper Method Tests
// =========================================================================

#[test]
fn test_is_instruct_model() {
    assert!(ModelType::Llama3_2_1B_Instruct.is_instruct_model());
    assert!(ModelType::Phi3_5_Mini_Instruct.is_instruct_model());
    assert!(ModelType::FlanT5Base.is_instruct_model()); // Seq2Seq is instruct-capable

    assert!(!ModelType::MiniLML6V2.is_instruct_model()); // Embedding
    assert!(!ModelType::DistilBertSST2.is_instruct_model()); // Classifier
}

#[test]
fn test_architecture_grouping() {
    assert!(ModelType::Llama3_1_8B_Instruct.is_llama_model());
    assert!(ModelType::DeepSeek_R1_Distill_Llama_8B.is_llama_model()); // It's Llama arch
    assert!(!ModelType::Mistral7B_v0_3_Instruct.is_llama_model());

    assert!(ModelType::Gpt2.is_gpt2_model());
    assert!(ModelType::DistilGpt2.is_gpt2_model());

    assert!(ModelType::Qwen2_5_0_5B_Instruct.is_qwen_model());
}

#[test]
fn test_display_group() {
    assert_eq!(
        ModelType::Llama3_2_1B_Instruct.display_group(),
        "LLM (Decoder)"
    );
    assert_eq!(ModelType::NomicEmbedText.display_group(), "Embedding");
    assert_eq!(ModelType::FlanT5Base.display_group(), "Seq2Seq");
    assert_eq!(
        ModelType::MiniLML6V2CrossEncoder.display_group(),
        "Re-Ranker"
    );
}

#[test]
fn test_repo_id_extraction() {
    let model = ModelType::Llama3_2_1B_Instruct;
    // URL: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/...
    assert_eq!(model.repo_id(), "meta-llama/Llama-3.2-1B-Instruct");

    let bert = ModelType::MiniLML6V2;
    // URL: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/...
    assert_eq!(bert.repo_id(), "sentence-transformers/all-MiniLM-L6-v2");
}

// =========================================================================
//  Search & Lookup Tests
// =========================================================================

#[test]
fn test_from_cli_name() {
    assert_eq!(
        ModelType::from_cli_name("llama3.2-1b"),
        Some(ModelType::Llama3_2_1B_Instruct)
    );
    assert_eq!(
        ModelType::from_cli_name("minilm-l6-v2"),
        Some(ModelType::MiniLML6V2)
    );

    // Test case insensitivity if implemented (currently exact match in code, but usually good practice)
    // Based on code: `let normalized = name.to_lowercase();` -> So it IS case insensitive!
    assert_eq!(
        ModelType::from_cli_name("Llama3.2-1B"),
        Some(ModelType::Llama3_2_1B_Instruct)
    );

    assert_eq!(ModelType::from_cli_name("non-existent-model"), None);
}

#[test]
fn test_find_similar() {
    // Typo: "lama" instead of "llama"
    let suggestions = ModelType::find_similar("lama3.2");
    assert!(!suggestions.is_empty());
    assert!(
        suggestions
            .iter()
            .any(|(name, _)| name == "llama3.2-1b" || name == "llama3.2-3b")
    );
}

#[test]
fn test_search_functionality() {
    // Search by name substring
    let results = ModelType::search("nomic");
    assert!(!results.is_empty());
    assert_eq!(results[0].0, ModelType::NomicEmbedText);

    // Search by description keyword (e.g., "reasoning")
    let results = ModelType::search("reasoning");
    assert!(
        results
            .iter()
            .any(|(m, _)| *m == ModelType::Phi3_5_Mini_Instruct)
    );
}

// =========================================================================
//  Formatting Utility Tests
// =========================================================================

#[test]
fn test_format_params() {
    assert_eq!(format_params(500), "500M");
    assert_eq!(format_params(1200), "1.2B");
    assert_eq!(format_params(8030), "8.0B");
    assert_eq!(format_params(70), "70M");
}

#[test]
fn test_format_size() {
    assert_eq!(format_size(500), "500 MB");
    assert_eq!(format_size(1500), "1.5 GB");
    assert_eq!(format_size(16000), "16.0 GB");
}

// =========================================================================
//  Path & Cache Tests
// =========================================================================

#[test]
fn test_cache_directory_structure() {
    let base_dir = PathBuf::from("/tmp/cache");
    let model = ModelType::Llama3_2_1B_Instruct;

    let model_dir = model.cache_dir(&base_dir);

    // Should replace '/' with '_' in repo ID
    // meta-llama/Llama-3.2-1B-Instruct -> meta-llama_Llama-3.2-1B-Instruct
    let expected = base_dir.join("meta-llama_Llama-3.2-1B-Instruct");
    assert_eq!(model_dir, expected);
}

#[test]
fn test_get_default_cache_dir() {
    // Just verify it doesn't panic and returns a valid path
    let path = get_default_cache_dir();
    assert!(path.is_absolute());

    // On Linux/Mac it usually ends with 'kjarni'
    if cfg!(unix) {
        assert!(path.to_string_lossy().contains("kjarni"));
    }
}

// =========================================================================
//  Architecture Tests
// =========================================================================

#[test]
fn test_architecture_categories() {
    assert_eq!(ModelArchitecture::Llama.category(), "decoder");
    assert_eq!(ModelArchitecture::Bert.category(), "encoder");
    assert_eq!(ModelArchitecture::T5.category(), "encoder-decoder");
}

#[test]
fn test_architecture_display_names() {
    assert_eq!(ModelArchitecture::Llama.display_name(), "Llama (Standard)");
    assert_eq!(ModelArchitecture::Mistral.display_name(), "Mistral (SWA)");
}
