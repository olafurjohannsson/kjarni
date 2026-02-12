use super::*;
use crate::generation::GenerationOverrides;
use crate::generator::{available_models, is_generator_model};


mod types_tests {
    use super::*;

    #[test]
    fn test_generated_token_display() {
        let token = GeneratedToken {
            id: 123,
            text: "hello".to_string(),
            is_special: false,
        };
        assert_eq!(token.text, "hello");
        assert_eq!(token.id, 123);
        assert!(!token.is_special);
    }

    #[test]
    fn test_generated_token_special() {
        let token = GeneratedToken {
            text: "<|endoftext|>".to_string(),
            id: 0,
            is_special: true,
        };
        assert!(token.is_special);
    }

    #[test]
    fn test_error_display() {
        let err = GeneratorError::UnknownModel("foo".to_string());
        let msg = err.to_string();
        assert!(msg.contains("foo"));

        let err = GeneratorError::ModelNotDownloaded("bar".to_string());
        let msg = err.to_string();
        assert!(msg.contains("bar"));
        assert!(msg.to_lowercase().contains("download"));

        let err = GeneratorError::GpuUnavailable;
        let msg = err.to_string();
        assert!(msg.to_lowercase().contains("gpu") || msg.to_lowercase().contains("unavailable"));

        let err = GeneratorError::InvalidModel("test".to_string(), "reason".to_string());
        let msg = err.to_string();
        assert!(msg.contains("test"));
        assert!(msg.contains("reason"));

        let err = GeneratorError::InvalidConfig("bad config".to_string());
        let msg = err.to_string();
        assert!(msg.contains("bad config"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GeneratorError>();
    }
}


// Unit Tests - Validation


mod validation_tests {
    
    use crate::generator::validation::*;
    use kjarni_transformers::models::ModelType;

    #[test]
    fn test_validate_decoder_models() {
        // Decoder models should be valid for generation
        let valid_models = ["llama3.2-1b-instruct", "qwen2.5-0.5b-instruct"];

        for name in valid_models {
            if let Some(model_type) = ModelType::from_cli_name(name) {
                let result = validate_for_generation(model_type);
                assert!(
                    result.is_ok(),
                    "Model {} should be valid for generation",
                    name
                );
            }
        }
    }

    #[test]
    fn test_validate_encoder_models_rejected() {
        // Encoder models should be rejected
        let encoder_models = ["minilm-l6-v2", "nomic-embed-text"];

        for name in encoder_models {
            if let Some(model_type) = ModelType::from_cli_name(name) {
                let result = validate_for_generation(model_type);
                assert!(
                    result.is_err(),
                    "Encoder {} should fail generation validation",
                    name
                );
            }
        }
    }

    #[test]
    fn test_get_generator_models() {
        let models = get_generator_models();
        assert!(!models.is_empty());
        // Should contain LLM models
        let has_llm = models.iter().any(|m| {
            m.contains("llama") || m.contains("qwen") || m.contains("mistral") || m.contains("phi")
        });
        assert!(has_llm, "Should contain LLM models");
    }

    #[test]
    fn test_suggest_generator_models() {
        let models = suggest_generator_models();
        assert!(!models.is_empty());
    }
}


// Unit Tests - Presets


mod preset_tests {
    
    use crate::generator::presets::*;

    #[test]
    fn test_preset_fast() {
        assert!(!GENERATOR_FAST_V1.model.is_empty());
        assert!(GENERATOR_FAST_V1.memory_mb > 0);
        assert!(GENERATOR_FAST_V1.default_max_tokens > 0);
        assert!(!GENERATOR_FAST_V1.name.is_empty());
        assert!(!GENERATOR_FAST_V1.architecture.is_empty());
    }

    #[test]
    fn test_preset_quality() {
        assert!(!GENERATOR_QUALITY_V1.model.is_empty());
        assert!(GENERATOR_QUALITY_V1.memory_mb >= GENERATOR_FAST_V1.memory_mb);
        assert!(GENERATOR_QUALITY_V1.default_max_tokens >= GENERATOR_FAST_V1.default_max_tokens);
    }

    #[test]
    fn test_preset_balanced() {
        assert!(!GENERATOR_BALANCED_V1.model.is_empty());
        assert!(GENERATOR_BALANCED_V1.memory_mb >= GENERATOR_FAST_V1.memory_mb);
    }

    #[test]
    fn test_preset_creative() {
        assert!(!GENERATOR_CREATIVE_V1.model.is_empty());
        assert!(GENERATOR_CREATIVE_V1.temperature.unwrap() > 0.8);
    }

    #[test]
    fn test_preset_code() {
        assert!(!GENERATOR_CODE_V1.model.is_empty());
        assert!(GENERATOR_CODE_V1.temperature.unwrap() < 0.5);
    }

    #[test]
    fn test_find_preset() {
        assert!(find_preset("GENERATOR_FAST_V1").is_some());
        assert!(find_preset("generator_fast_v1").is_some());
        assert!(find_preset("GENERATOR_QUALITY_V1").is_some());
        assert!(find_preset("nonexistent").is_none());
    }

    #[test]
    fn test_tier_resolution() {
        let fast = GeneratorTier::Fast.resolve();
        let balanced = GeneratorTier::Balanced.resolve();
        let quality = GeneratorTier::Quality.resolve();

        assert!(!fast.model.is_empty());
        assert!(!balanced.model.is_empty());
        assert!(!quality.model.is_empty());
    }

    #[test]
    fn test_tier_default() {
        assert!(matches!(GeneratorTier::default(), GeneratorTier::Balanced));
    }

    #[test]
    fn test_all_presets_valid() {
        for preset in ALL_V1_PRESETS {
            assert!(!preset.name.is_empty(), "Preset name should not be empty");
            assert!(!preset.model.is_empty(), "Preset model should not be empty");
            assert!(
                preset.memory_mb > 0,
                "Preset should have memory requirement"
            );
            assert!(
                preset.default_max_tokens > 0,
                "Preset should have max tokens"
            );
        }
    }

    #[test]
    fn test_all_presets_unique_names() {
        let names: Vec<_> = ALL_V1_PRESETS.iter().map(|p| p.name).collect();
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(
            names.len(),
            unique.len(),
            "All preset names should be unique"
        );
    }

    #[test]
    fn test_list_presets() {
        let presets = list_presets();
        assert!(!presets.is_empty());
        assert!(presets.contains(&"GENERATOR_FAST_V1"));
    }

    #[test]
    fn test_legacy_presets_still_work() {
        assert!(!GeneratorPreset::GPT2.model.is_empty());
        assert!(!GeneratorPreset::FAST.model.is_empty());
        assert!(!GeneratorPreset::QUALITY.model.is_empty());
    }
}


// Unit Tests - Builder


mod builder_tests {
    use super::*;
    use crate::{common::KjarniDevice, generator::presets::GENERATOR_FAST_V1};

    #[test]
    fn test_builder_default_state() {
        let builder = GeneratorBuilder::new("llama3.2-1b-instruct");
        assert_eq!(builder.model, "llama3.2-1b-instruct");
        assert!(!builder.quiet);
    }

    #[test]
    fn test_builder_device_methods() {
        let cpu = GeneratorBuilder::new("llama3.2-1b-instruct").cpu();
        assert!(matches!(cpu.device, KjarniDevice::Cpu));

        let gpu = GeneratorBuilder::new("llama3.2-1b-instruct").gpu();
        assert!(matches!(gpu.device, KjarniDevice::Gpu));
    }

    #[test]
    fn test_builder_generation_config() {
        let builder = GeneratorBuilder::new("llama3.2-1b-instruct")
            .temperature(0.8)
            .max_tokens(256)
            .top_p(0.9);

        assert_eq!(builder.generation_overrides.temperature, Some(0.8));
        assert_eq!(builder.generation_overrides.max_new_tokens, Some(256));
        assert_eq!(builder.generation_overrides.top_p, Some(0.9));
    }

    #[test]
    fn test_builder_greedy() {
        let builder = GeneratorBuilder::new("llama3.2-1b-instruct").greedy();
        assert_eq!(builder.generation_overrides.temperature, Some(0.0));
    }

    #[test]
    fn test_builder_cache_dir() {
        let builder = GeneratorBuilder::new("llama3.2-1b-instruct").cache_dir("/tmp/test");
        assert_eq!(
            builder.cache_dir,
            Some(std::path::PathBuf::from("/tmp/test"))
        );
    }
    #[test]
    fn test_builder_creative() {
        let builder = GeneratorBuilder::new("llama3.2-1b-instruct").creative();
        assert!(builder.generation_overrides.temperature.unwrap_or(0.0) > 0.8);
    }
    #[test]
    fn test_builder_from_preset() {
        let builder = GeneratorBuilder::from_preset(&GENERATOR_FAST_V1);
        assert_eq!(builder.model, GENERATOR_FAST_V1.model);
    }

    #[test]
    fn test_builder_quiet() {
        let builder = GeneratorBuilder::new("llama3.2-1b-instruct").quiet();
        assert!(builder.quiet);
    }

    #[test]
    fn test_builder_offline() {
        use crate::common::DownloadPolicy;

        let builder = GeneratorBuilder::new("llama3.2-1b-instruct").offline();
        assert!(matches!(builder.download_policy, DownloadPolicy::Never));
    }
}

mod module_function_tests {
    use super::*;

    #[test]
    fn test_available_models() {
        let models = available_models();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_suggested_models() {
        let models = suggested_models();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_is_generator_model_valid() {
        assert!(is_generator_model("llama3.2-1b-instruct").is_ok());
        assert!(is_generator_model("qwen2.5-0.5b-instruct").is_ok());
    }

    #[test]
    fn test_is_generator_model_invalid() {
        assert!(is_generator_model("not-a-model").is_err());
        assert!(is_generator_model("minilm-l6-v2").is_err());
    }
}


#[cfg(test)]
mod integration_tests {
    use crate::generator::model::generate;

    use super::*;
    use futures::StreamExt;
    use kjarni_transformers::common::DecodingStrategy;

    fn model_available(model: &str) -> bool {
        let cache_dir = dirs::cache_dir().expect("no cache dir").join("kjarni");
        kjarni_transformers::models::ModelType::from_cli_name(model)
            .map(|m| m.is_downloaded(&cache_dir))
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn test_basic_generation() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let r#gen = Generator::builder("qwen2.5-0.5b-instruct")
            .cpu()
            .quiet()
            .max_tokens(50)
            .build()
            .await
            .expect("Failed to load generator");

        let output = r#gen
            .generate("Hello, how are you?")
            .await
            .expect("Generation failed");

        assert!(!output.is_empty(), "Output should not be empty");
        assert!(
            output.chars().any(|c| c.is_alphabetic()),
            "Output should contain letters: {}",
            output
        );
    }

    #[tokio::test]
    async fn test_generation_produces_text() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let r#gen = Generator::builder("qwen2.5-0.5b-instruct")
            .cpu()
            .quiet()
            .max_tokens(100)
            .build()
            .await
            .unwrap();

        let output = r#gen.generate("The capital of France is").await.unwrap();

        assert!(!output.is_empty());
        // Should produce coherent continuation
        let output_lower = output.to_lowercase();
        let mentions_paris = output_lower.contains("paris");
        let has_text = output.chars().filter(|c| c.is_alphabetic()).count() > 5;

        assert!(
            mentions_paris || has_text,
            "Output should mention Paris or contain meaningful text: {}",
            output
        );
    }

    #[tokio::test]
    async fn test_generation_with_max_tokens() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let r#gen = Generator::builder("qwen2.5-0.5b-instruct")
            .cpu()
            .quiet()
            .max_tokens(10)
            .build()
            .await
            .unwrap();

        let output = r#gen.generate("Tell me a long story about").await.unwrap();

        assert!(!output.is_empty());
    }

    #[tokio::test]
    async fn test_greedy_generation() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let r#gen = Generator::builder("qwen2.5-0.5b-instruct")
            .cpu()
            .quiet()
            .greedy()
            .max_tokens(20)
            .build()
            .await
            .unwrap();

        let output1 = r#gen.generate("1 + 1 = ").await.unwrap();
        let output2 = r#gen.generate("1 + 1 = ").await.unwrap();

        assert!(!output1.is_empty());
        assert!(!output2.is_empty());
        assert_eq!(
            output1, output2,
            "Greedy generation should be deterministic"
        );
    }

    #[tokio::test]
    async fn test_greedy_actually_greedy() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let r#gen = Generator::builder("qwen2.5-0.5b-instruct")
            .cpu()
            .quiet()
            .greedy()
            .max_tokens(20)
            .build()
            .await
            .unwrap();

        // Check what strategy is actually resolved
        let config = r#gen.generation_config();
        println!("Resolved strategy: {:?}", config.inner.strategy);

        assert!(
            matches!(config.inner.strategy, DecodingStrategy::Greedy),
            "Expected Greedy strategy, got {:?}",
            config.inner.strategy
        );
    }

    #[tokio::test]
    async fn test_generation_with_config_override() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let r#gen = Generator::builder("qwen2.5-0.5b-instruct")
            .cpu()
            .quiet()
            .max_tokens(50)
            .build()
            .await
            .unwrap();

        // Override at runtime
        let overrides = GenerationOverrides {
            temperature: Some(0.0), // Greedy
            max_new_tokens: Some(10),
            ..Default::default()
        };

        let output = r#gen
            .generate_with_config("Hello", &overrides)
            .await
            .unwrap();
        assert!(!output.is_empty());
    }

    #[tokio::test]
    async fn test_streaming_generation() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let r#gen = Generator::builder("qwen2.5-0.5b-instruct")
            .cpu()
            .quiet()
            .max_tokens(30)
            .build()
            .await
            .unwrap();

        let mut stream = r#gen.stream("Hello world").await.unwrap();

        let mut tokens = Vec::new();
        while let Some(result) = stream.next().await {
            let token = result.expect("Stream error");
            tokens.push(token);
        }

        assert!(!tokens.is_empty(), "Should produce at least one token");

        let full_text: String = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(!full_text.is_empty(), "Combined text should not be empty");
    }

    #[tokio::test]
    async fn test_stream_matches_generate() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let r#gen = Generator::builder("qwen2.5-0.5b-instruct")
            .cpu()
            .quiet()
            .greedy() // Deterministic
            .max_tokens(20)
            .build()
            .await
            .unwrap();

        // Non-streaming
        let direct = r#gen.generate("Test input").await.unwrap();

        // Streaming
        let mut stream = r#gen.stream("Test input").await.unwrap();
        let mut tokens = Vec::new();
        while let Some(result) = stream.next().await {
            tokens.push(result.unwrap());
        }
        let streamed: String = tokens.iter().map(|t| t.text.as_str()).collect();

        assert_eq!(
            direct, streamed,
            "Streaming and direct should match with greedy"
        );
    }

    #[tokio::test]
    async fn test_stream_text_convenience() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let r#gen = Generator::builder("qwen2.5-0.5b-instruct")
            .cpu()
            .quiet()
            .max_tokens(20)
            .build()
            .await
            .unwrap();

        let mut stream = r#gen.stream_text("Hello").await.unwrap();

        let mut chunks = Vec::new();
        while let Some(result) = stream.next().await {
            chunks.push(result.unwrap());
        }

        assert!(!chunks.is_empty());
        let full: String = chunks.into_iter().collect();
        assert!(!full.is_empty());
    }

    #[tokio::test]
    async fn test_unknown_model_error() {
        let result = Generator::new("not-a-real-model").await;
        assert!(matches!(result, Err(GeneratorError::UnknownModel(_))));
    }

    #[tokio::test]
    async fn test_offline_mode_not_downloaded() {
        let result = Generator::builder("nonexistent-model-xyz")
            .offline()
            .cpu()
            .quiet()
            .build()
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_generator_accessors() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let r#gen = Generator::builder("qwen2.5-0.5b-instruct")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        assert_eq!(r#gen.model_name(), "qwen2.5-0.5b-instruct");
        assert!(matches!(
            r#gen.device(),
            kjarni_transformers::traits::Device::Cpu
        ));
        assert!(r#gen.context_size() > 0);
        assert!(r#gen.vocab_size() > 0);
    }

    #[tokio::test]
    async fn test_sequential_generation() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let r#gen = Generator::builder("qwen2.5-0.5b-instruct")
            .cpu()
            .quiet()
            .max_tokens(20)
            .build()
            .await
            .unwrap();

        let prompts = vec!["Hello", "Goodbye", "Thanks"];
        let mut outputs = Vec::new();

        for prompt in &prompts {
            let output = r#gen.generate(prompt).await.unwrap();
            outputs.push(output);
        }

        assert_eq!(outputs.len(), prompts.len());
        for (i, output) in outputs.iter().enumerate() {
            assert!(!output.is_empty(), "Output {} should not be empty", i);
        }
    }

    #[tokio::test]
    async fn test_concurrent_generation() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let r#gen = std::sync::Arc::new(
            Generator::builder("qwen2.5-0.5b-instruct")
                .cpu()
                .quiet()
                .max_tokens(20)
                .build()
                .await
                .unwrap(),
        );

        let prompts = vec!["One", "Two", "Three"];

        let handles: Vec<_> = prompts
            .into_iter()
            .map(|prompt| {
                let r#generator = r#gen.clone();
                let prompt = prompt.to_string();
                tokio::spawn(async move { generator.generate(&prompt).await })
            })
            .collect();

        for (i, handle) in handles.into_iter().enumerate() {
            let result = handle.await.expect("Task panicked");
            let output = result.expect("Generation failed");
            assert!(
                !output.is_empty(),
                "Concurrent output {} should not be empty",
                i
            );
        }
    }

    #[tokio::test]
    async fn test_module_generate_function() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let result = generate("qwen2.5-0.5b-instruct", "Hello").await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_generator_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Generator>();
    }
}
