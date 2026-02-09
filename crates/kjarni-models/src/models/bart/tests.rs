


mod bart_tests_config {
    

    // BartModel architectural and config tests
// Verifies config.json + generation_config.json are parsed correctly
// and trait implementations expose the right values.
use anyhow::Result;
use kjarni_transformers::{Device, LanguageModel, ModelType, common::{DecodingStrategy, GenerationConfig}, encoder_decoder::{EncoderDecoderGenerator, EncoderDecoderLanguageModel}, traits::ModelConfig};

use crate::models::bart::{config::BartConfig, model::BartModel};

async fn load_bart_large_cnn() -> Result<BartModel> {
    BartModel::from_registry(
        ModelType::BartLargeCnn,
        None,
        Device::Cpu,
        None,
        None,
    )
    .await
}

#[tokio::test]
async fn test_bart_large_cnn_greedy_generation_news() -> Result<()> {
    let model = BartModel::from_registry(
        ModelType::BartLargeCnn, None, Device::Cpu, None, None,
    ).await?;

    let input = concat!(
        "The Federal Reserve announced today that it would hold interest rates ",
        "steady at their current level, citing ongoing concerns about inflation ",
        "and economic uncertainty. The decision was widely expected by analysts ",
        "and marks the third consecutive meeting where rates have remained unchanged. ",
        "Fed Chair Jerome Powell stated that the central bank remains committed to ",
        "bringing inflation down to its two percent target but acknowledged that ",
        "progress has been slower than anticipated. Markets reacted positively to ",
        "the announcement, with the S&P 500 rising half a percent in afternoon trading.",
    );

    let expected = concat!(
        "The Federal Reserve announced today that it would hold interest rates steady at their current level. ",
        "The decision was widely expected by analysts and marks the third consecutive meeting where rates have remained unchanged.",
    );

    let config = GenerationConfig {
        max_length: 80,
        min_length: 20,
        no_repeat_ngram_size: 0,
        repetition_penalty: 1.0,
        max_new_tokens: None,
        add_bos_token: false,
        strategy: DecodingStrategy::Greedy,
        speculation: None,
    };

    let generator = EncoderDecoderGenerator::new(Box::new(model))?;
    let output = generator.generate(input, Some(&config)).await?;

    assert_eq!(
        output.trim(), expected.trim(),
        "Direct BartModel greedy generation mismatch for news input"
    );

    Ok(())
}


#[tokio::test]
async fn test_bart_large_cnn_greedy_generation_ai() -> Result<()> {
    let model = BartModel::from_registry(
        ModelType::BartLargeCnn, None, Device::Cpu, None, None,
    ).await?;

    let input = concat!(
        "Artificial intelligence has rapidly become a central component of modern ",
        "software systems, influencing industries such as healthcare, finance, ",
        "transportation, and education. Advances in machine learning, particularly ",
        "deep learning and transformer-based models, have enabled computers to ",
        "process natural language, images, and large volumes of data with ",
        "unprecedented accuracy. As a result, organizations are increasingly ",
        "adopting AI-driven tools to automate tasks, improve decision-making, and ",
        "gain competitive advantages. However, the growing reliance on AI also ",
        "raises important concerns related to data privacy, model transparency, ",
        "bias, and ethical responsibility. Governments and regulatory bodies are ",
        "beginning to introduce frameworks to ensure that AI systems are developed ",
        "and deployed in a safe and accountable manner. At the same time, ",
        "researchers continue to explore ways to make models more efficient, ",
        "interpretable, and accessible, balancing innovation with societal impact.",
    );

    let expected = concat!(
        "Artificial intelligence has rapidly become a central component of modern software systems. ",
        "Advances in machine learning have enabled computers to process natural language, images, ",
        "and large volumes of data with unprecedented accuracy. Organizations are increasingly ",
        "adopting AI-driven tools to automate tasks, improve decision-making, and gain competitive advantages.",
    );

    let config = GenerationConfig {
        max_length: 142,
        min_length: 30,
        no_repeat_ngram_size: 0,
        repetition_penalty: 1.0,
        max_new_tokens: None,
        add_bos_token: false,
        strategy: DecodingStrategy::Greedy,
        speculation: None,
    };

    let generator = EncoderDecoderGenerator::new(Box::new(model))?;
    let output = generator.generate(input, Some(&config)).await?;

    assert_eq!(
        output.trim(), expected.trim(),
        "Direct BartModel greedy generation mismatch for AI input"
    );

    Ok(())
}

#[tokio::test]
async fn test_bart_large_cnn_decoder_start_token() -> Result<()> {
    let model = BartModel::from_registry(
        ModelType::BartLargeCnn, None, Device::Cpu, None, None,
    ).await?;

    assert_eq!(model.decoder_start_token_id(), 2);
    assert_ne!(model.decoder_start_token_id(), 0);

    Ok(())
}

#[test]
fn test_bart_large_cnn_config_parsing() {
    // Parse the actual config.json content
    let json = r#"{
        "_num_labels": 3,
        "activation_function": "gelu",
        "bos_token_id": 0,
        "d_model": 1024,
        "decoder_attention_heads": 16,
        "decoder_ffn_dim": 4096,
        "decoder_layers": 12,
        "decoder_start_token_id": 2,
        "encoder_attention_heads": 16,
        "encoder_ffn_dim": 4096,
        "encoder_layers": 12,
        "eos_token_id": 2,
        "extra_pos_embeddings": 0,
        "force_bos_token_to_be_generated": true,
        "forced_bos_token_id": 0,
        "forced_eos_token_id": 2,
        "is_encoder_decoder": true,
        "max_position_embeddings": 1024,
        "model_type": "bart",
        "normalize_before": false,
        "pad_token_id": 1,
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

    let config = BartConfig::from_json(json).unwrap();

    // Architecture
    assert_eq!(config.d_model, 1024);
    assert_eq!(config.encoder_layers, 12);
    assert_eq!(config.decoder_layers, 12);
    assert_eq!(config.encoder_attention_heads, 16);
    assert_eq!(config.decoder_attention_heads, 16);
    assert_eq!(config.encoder_ffn_dim, 4096);
    assert_eq!(config.decoder_ffn_dim, 4096);
    assert_eq!(config.vocab_size, 50264);
    assert_eq!(config.max_position_embeddings, 1024);
    assert!(!config.scale_embedding);
    assert!(!config.normalize_before);

    // Critical token IDs
    assert_eq!(config.bos_token_id, 0);
    assert_eq!(config.eos_token_id, 2);
    assert_eq!(config.pad_token_id, 1);
    assert_eq!(config.decoder_start_token_id, 2);
    assert_eq!(config.forced_bos_token_id, Some(0));
    assert_eq!(config.forced_eos_token_id, Some(2));

    // Task-specific params
    let task_params = config.task_specific_params.as_ref().unwrap();
    let summary = task_params.summarization.as_ref().unwrap();
    assert_eq!(summary.max_length, 142);
    assert_eq!(summary.min_length, 56);
    assert_eq!(summary.num_beams, 4);
    assert_eq!(summary.no_repeat_ngram_size, 3);
    assert!((summary.length_penalty - 2.0).abs() < f32::EPSILON);
    assert!(summary.early_stopping);
}

#[tokio::test]
async fn test_bart_large_cnn_architectural_properties() -> Result<()> {
    let model = load_bart_large_cnn().await?;

    // Trait methods should reflect config.json values
    assert_eq!(model.vocab_size(), 50264);
    assert_eq!(model.hidden_size(), 1024);
    assert_eq!(model.num_layers(), 12);
    assert_eq!(model.num_heads(), 16);
    assert_eq!(model.context_size(), 1024);

    // Critical token IDs from config — NOT from generation_config.json
    assert_eq!(model.bos_token_id(), Some(0), "bos_token_id should be 0");
    assert_eq!(model.eos_token_id(), Some(2), "eos_token_id should be 2");
    assert_eq!(model.pad_token_id(), Some(1), "pad_token_id should be 1");

    // Decoder start token — this is the one that matters most for generation
    assert_eq!(
        model.config.decoder_start_token_id, 2,
        "decoder_start_token_id must be 2 (EOS), not 0 (BOS)"
    );

    // Forced tokens
    assert_eq!(
        model.forced_bos_token_id(), Some(0),
        "forced_bos_token_id should be 0"
    );
    assert_eq!(
        model.forced_eos_token_id(), Some(2),
        "forced_eos_token_id should be 2"
    );

    Ok(())
}

#[tokio::test]
async fn test_bart_large_cnn_trait_config_consistency() -> Result<()> {
    let model = load_bart_large_cnn().await?;
    let config = &model.config;
    let meta = config.metadata();

    // Verify trait values come from config, not hardcoded
    assert_eq!(
        model.vocab_size(), meta.vocab_size,
        "vocab_size: trait vs metadata mismatch"
    );
    assert_eq!(
        model.hidden_size(), meta.hidden_size,
        "hidden_size: trait vs metadata mismatch"
    );
    assert_eq!(
        model.num_layers(), meta.num_layers,
        "num_layers: trait vs metadata mismatch"
    );
    assert_eq!(
        model.num_heads(), meta.num_attention_heads,
        "num_heads: trait vs metadata mismatch"
    );
    assert_eq!(
        model.context_size(), meta.max_seq_len,
        "context_size: trait vs metadata mismatch"
    );

    Ok(())
}

#[tokio::test]
async fn test_bart_large_cnn_generation_config_does_not_override_token_ids() -> Result<()> {
    // This test exists because the generation_config.json that ships with
    // facebook/bart-large-cnn has WRONG values:
    //   eos_token_id: 1 (should be 2)
    //   pad_token_id: 0 (should be 1)
    //   decoder_start_token_id: 0 (should be 2)
    //
    // The model MUST use config.json values for these critical token IDs.

    let model = load_bart_large_cnn().await?;

    // These must match config.json, regardless of what generation_config.json says
    assert_ne!(
        model.eos_token_id(), Some(1),
        "eos_token_id is 1 — generation_config.json is overriding config.json!"
    );
    assert_ne!(
        model.pad_token_id(), Some(0),
        "pad_token_id is 0 — generation_config.json is overriding config.json!"
    );
    assert_ne!(
        model.config.decoder_start_token_id, 0,
        "decoder_start_token_id is 0 — generation_config.json is overriding config.json!"
    );

    // Positive assertions for the correct values
    assert_eq!(model.eos_token_id(), Some(2));
    assert_eq!(model.pad_token_id(), Some(1));
    assert_eq!(model.config.decoder_start_token_id, 2);

    Ok(())
}

#[tokio::test]
async fn test_bart_large_cnn_default_generation_config() -> Result<()> {
    let model = load_bart_large_cnn().await?;
    let gen_config = model.get_default_generation_config();

    // Should come from task_specific_params.summarization in config.json
    assert_eq!(gen_config.max_length, 142);
    assert_eq!(gen_config.min_length, 56);
    assert_eq!(gen_config.no_repeat_ngram_size, 3);

    match &gen_config.strategy {
        DecodingStrategy::BeamSearch(params) => {
            assert_eq!(params.num_beams, 4);
            assert!((params.length_penalty - 2.0).abs() < f32::EPSILON);
            assert!(params.early_stopping);
        }
        other => panic!(
            "Expected BeamSearch strategy from task_specific_params, got {:?}",
            other
        ),
    }

    Ok(())
}
}