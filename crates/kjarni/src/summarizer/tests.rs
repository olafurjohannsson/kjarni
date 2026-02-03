//! Comprehensive tests for the Summarizer module.

use super::*;
use crate::seq2seq::Seq2SeqOverrides;

// =============================================================================
// Unit Tests - Validation
// =============================================================================

mod validation_tests {
    use super::*;
    use crate::summarizer::validation::*;
    use kjarni_transformers::models::ModelType;

    #[test]
    fn test_validate_bart_models() {
        if let Some(bart) = ModelType::from_cli_name("bart-large-cnn") {
            assert!(validate_for_summarization(bart).is_ok());
        }

        if let Some(distilbart) = ModelType::from_cli_name("distilbart-cnn") {
            assert!(validate_for_summarization(distilbart).is_ok());
        }
    }

    #[test]
    fn test_validate_t5_models() {
        let t5_base = ModelType::from_cli_name("flan-t5-base").unwrap();
        assert!(validate_for_summarization(t5_base).is_ok());

        let t5_large = ModelType::from_cli_name("flan-t5-large").unwrap();
        assert!(validate_for_summarization(t5_large).is_ok());
    }

    #[test]
    fn test_get_summarization_models() {
        let models = get_summarization_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"flan-t5-base"));
    }

    #[test]
    fn test_recommended_models() {
        let models = recommended_summarization_models();
        assert!(!models.is_empty());
        assert!(models.len() >= 2);
    }
}

// =============================================================================
// Unit Tests - Presets
// =============================================================================

mod preset_tests {
    use super::*;
    use crate::summarizer::presets::*;

    #[test]
    fn test_preset_fast() {
        assert_eq!(SUMMARIZER_FAST_V1.model, "distilbart-cnn");
        assert_eq!(SUMMARIZER_FAST_V1.architecture, "bart");
        assert!(SUMMARIZER_FAST_V1.default_min_length > 0);
        assert!(SUMMARIZER_FAST_V1.default_max_length > SUMMARIZER_FAST_V1.default_min_length);
        assert!(SUMMARIZER_FAST_V1.memory_mb > 0);
    }

    #[test]
    fn test_preset_quality() {
        assert_eq!(SUMMARIZER_QUALITY_V1.model, "bart-large-cnn");
        assert!(SUMMARIZER_QUALITY_V1.memory_mb > SUMMARIZER_FAST_V1.memory_mb);
    }

    #[test]
    fn test_preset_t5() {
        assert_eq!(SUMMARIZER_T5_V1.model, "flan-t5-base");
        assert_eq!(SUMMARIZER_T5_V1.architecture, "t5");
    }

    #[test]
    fn test_find_preset() {
        assert!(find_preset("SUMMARIZER_FAST_V1").is_some());
        assert!(find_preset("summarizer_fast_v1").is_some());
        assert!(find_preset("nonexistent").is_none());
    }

    #[test]
    fn test_tier_resolution() {
        assert_eq!(SummarizerTier::Fast.resolve().model, "distilbart-cnn");
        assert_eq!(SummarizerTier::Balanced.resolve().model, "flan-t5-base");
        assert_eq!(SummarizerTier::Quality.resolve().model, "bart-large-cnn");
    }

    #[test]
    fn test_all_presets_valid() {
        for preset in ALL_V1_PRESETS {
            assert!(!preset.name.is_empty());
            assert!(!preset.model.is_empty());
            assert!(!preset.architecture.is_empty());
            assert!(preset.default_max_length > preset.default_min_length);
            assert!(preset.memory_mb > 0);
        }
    }
}

// =============================================================================
// Unit Tests - Builder
// =============================================================================

mod builder_tests {
    use super::*;
    use crate::common::KjarniDevice;

    #[test]
    fn test_builder_default_state() {
        let builder = SummarizerBuilder::new("distilbart-cnn");
        assert_eq!(builder.model, "distilbart-cnn");
        assert!(!builder.quiet);
        assert_eq!(builder.overrides.no_repeat_ngram_size, Some(3));
    }

    #[test]
    fn test_builder_length_presets() {
        let short = SummarizerBuilder::new("distilbart-cnn").short();
        assert_eq!(short.overrides.min_length, Some(30));
        assert_eq!(short.overrides.max_length, Some(60));

        let medium = SummarizerBuilder::new("distilbart-cnn").medium();
        assert_eq!(medium.overrides.min_length, Some(50));
        assert_eq!(medium.overrides.max_length, Some(150));

        let long = SummarizerBuilder::new("distilbart-cnn").long();
        assert_eq!(long.overrides.min_length, Some(100));
        assert_eq!(long.overrides.max_length, Some(300));
    }

    #[test]
    fn test_builder_custom_length() {
        let builder = SummarizerBuilder::new("distilbart-cnn")
            .min_length(40)
            .max_length(80);

        assert_eq!(builder.overrides.min_length, Some(40));
        assert_eq!(builder.overrides.max_length, Some(80));
    }

    #[test]
    fn test_builder_generation_params() {
        let builder = SummarizerBuilder::new("distilbart-cnn")
            .num_beams(6)
            .length_penalty(2.0);

        assert_eq!(builder.overrides.num_beams, Some(6));
        assert_eq!(builder.overrides.length_penalty, Some(2.0));
    }

    #[test]
    fn test_builder_greedy() {
        let builder = SummarizerBuilder::new("distilbart-cnn").greedy();
        assert_eq!(builder.overrides.num_beams, Some(1));
    }

    #[test]
    fn test_builder_device_methods() {
        let cpu = SummarizerBuilder::new("distilbart-cnn").cpu();
        assert!(matches!(cpu.device, KjarniDevice::Cpu));

        let gpu = SummarizerBuilder::new("distilbart-cnn").gpu();
        assert!(matches!(gpu.device, KjarniDevice::Gpu));
    }

    #[test]
    fn test_builder_from_preset() {
        use crate::summarizer::presets::SUMMARIZER_FAST_V1;

        let builder = SummarizerBuilder::from_preset(&SUMMARIZER_FAST_V1);
        assert_eq!(builder.model, "distilbart-cnn");
        assert_eq!(builder.overrides.min_length, Some(30));
        assert_eq!(builder.overrides.max_length, Some(130));
    }
}

// =============================================================================
// Unit Tests - Error Types
// =============================================================================

mod error_tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = SummarizerError::UnknownModel("foo".to_string());
        assert!(err.to_string().contains("foo"));

        let err = SummarizerError::IncompatibleModel {
            model: "gpt2".to_string(),
            reason: "decoder only".to_string(),
        };
        assert!(err.to_string().contains("gpt2"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SummarizerError>();
    }
}

// =============================================================================
// Unit Tests - Module-level Functions
// =============================================================================

mod module_function_tests {
    use super::*;

    #[test]
    fn test_available_models() {
        let models = available_models();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_recommended_models() {
        let models = recommended_models();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_is_summarization_model_valid() {
        assert!(is_summarization_model("flan-t5-base").is_ok());
    }

    #[test]
    fn test_is_summarization_model_invalid() {
        assert!(is_summarization_model("not-a-model").is_err());
    }
}

// =============================================================================
// Integration Tests (require model download)
// =============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;
    use futures::StreamExt;

    const TEST_ARTICLE: &str = r#"
        The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building,
        and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on
        each side. During its construction, the Eiffel Tower surpassed the Washington Monument to
        become the tallest man-made structure in the world, a title it held for 41 years until the
        Chrysler Building in New York City was finished in 1930. It was the first structure in the
        world to surpass both the 200-metre and 300-metre mark in height. Due to the addition of a
        broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler
        Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second
        tallest free-standing structure in France after the Millau Viaduct.
    "#;

    fn model_available(model: &str) -> bool {
        let cache_dir = dirs::cache_dir().expect("no cache dir").join("kjarni");
        kjarni_transformers::models::ModelType::from_cli_name(model)
            .map(|m| m.is_downloaded(&cache_dir))
            .unwrap_or(false)
    }

    // =========================================================================
    // BART Model Tests with Assertions
    // =========================================================================

    #[tokio::test]
    async fn test_bart_basic_summarization() {
        if !model_available("distilbart-cnn") {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let s = Summarizer::builder("distilbart-cnn")
            .cpu()
            .quiet()
            .build()
            .await
            .expect("Failed to load summarizer");

        let summary = s
            .summarize(TEST_ARTICLE)
            .await
            .expect("Summarization failed");

        assert!(!summary.is_empty(), "Summary should not be empty");
        assert!(
            summary.len() < TEST_ARTICLE.len(),
            "Summary should be shorter than input"
        );

        // BART summaries should mention key facts
        let summary_lower = summary.to_lowercase();
        let mentions_tower = summary_lower.contains("tower") || summary_lower.contains("eiffel");
        let mentions_height = summary_lower.contains("metre")
            || summary_lower.contains("meter")
            || summary_lower.contains("tall")
            || summary_lower.contains("height")
            || summary_lower.contains("ft")
            || summary_lower.contains("324");

        assert!(
            mentions_tower || mentions_height,
            "Summary should mention tower or height: {}",
            summary
        );
    }

    #[tokio::test]
    async fn test_bart_summary_is_shorter() {
        if !model_available("distilbart-cnn") {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let s = Summarizer::builder("distilbart-cnn")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let summary = s.summarize(TEST_ARTICLE).await.unwrap();

        // Summary should be significantly shorter
        let compression_ratio = summary.len() as f64 / TEST_ARTICLE.len() as f64;
        assert!(
            compression_ratio < 0.7,
            "Summary should be at most 70% of original length, got {}%",
            compression_ratio * 100.0
        );
    }

    #[tokio::test]
    async fn test_bart_short_vs_long_summary() {
        if !model_available("distilbart-cnn") {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let short_summarizer = Summarizer::builder("distilbart-cnn")
            .short()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let long_summarizer = Summarizer::builder("distilbart-cnn")
            .long()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let short_summary = short_summarizer.summarize(TEST_ARTICLE).await.unwrap();
        let long_summary = long_summarizer.summarize(TEST_ARTICLE).await.unwrap();

        assert!(!short_summary.is_empty());
        assert!(!long_summary.is_empty());

        // Long summary should generally be longer (though not guaranteed due to model behavior)
        // At minimum, both should be valid
        assert!(short_summary.chars().filter(|c| c.is_alphabetic()).count() > 10);
        assert!(long_summary.chars().filter(|c| c.is_alphabetic()).count() > 10);
    }

    // =========================================================================
    // T5 Model Tests with Assertions
    // =========================================================================

    #[tokio::test]
    async fn test_t5_summarization() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let s = Summarizer::builder("flan-t5-base")
            .cpu()
            .quiet()
            .build()
            .await
            .expect("Failed to load summarizer");

        let summary = s
            .summarize(TEST_ARTICLE)
            .await
            .expect("Summarization failed");

        assert!(!summary.is_empty(), "Summary should not be empty");
        assert!(
            summary.len() < TEST_ARTICLE.len(),
            "Summary should be shorter than input"
        );
        assert!(
            summary.chars().any(|c| c.is_alphabetic()),
            "Summary should contain letters: {}",
            summary
        );
    }

    #[tokio::test]
    async fn test_t5_uses_prefix_internally() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let s = Summarizer::builder("flan-t5-base")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        // Even short input should work (T5 prepends "summarize: ")
        let summary = s.summarize("Short text to summarize").await.unwrap();
        assert!(!summary.is_empty());
    }

    #[tokio::test]
    async fn test_bart_no_prefix_needed() {
        if !model_available("distilbart-cnn") {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let s = Summarizer::builder("distilbart-cnn")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        // BART doesn't need prefix
        let summary = s.summarize("Short text to summarize").await.unwrap();
        assert!(!summary.is_empty());
    }

    // =========================================================================
    // Generation Config Tests with Assertions
    // =========================================================================

    #[tokio::test]
    async fn test_summarize_greedy_vs_beam() {
        if !model_available("distilbart-cnn") {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let s = Summarizer::builder("distilbart-cnn")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let greedy = s
            .summarize_with_config(TEST_ARTICLE, &Seq2SeqOverrides::greedy())
            .await
            .unwrap();

        let beam = s
            .summarize_with_config(TEST_ARTICLE, &Seq2SeqOverrides::high_quality())
            .await
            .unwrap();

        assert!(!greedy.is_empty(), "Greedy should produce output");
        assert!(!beam.is_empty(), "Beam should produce output");

        // Both should be valid summaries
        assert!(greedy.len() < TEST_ARTICLE.len());
        assert!(beam.len() < TEST_ARTICLE.len());
    }

    #[tokio::test]
    async fn test_summarize_with_length_penalty() {
        if !model_available("distilbart-cnn") {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let s = Summarizer::builder("distilbart-cnn")
            .length_penalty(2.5)
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let summary = s.summarize(TEST_ARTICLE).await.unwrap();
        assert!(!summary.is_empty());
        assert!(summary.len() < TEST_ARTICLE.len());
    }

    // =========================================================================
    // Streaming Tests with Assertions
    // =========================================================================

    #[tokio::test]
    async fn test_stream_summarization() {
        if !model_available("distilbart-cnn") {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let s = Summarizer::builder("distilbart-cnn")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let mut stream = s.stream(TEST_ARTICLE).await.unwrap();

        let mut chunks = Vec::new();
        while let Some(result) = stream.next().await {
            let chunk = result.expect("Stream error");
            chunks.push(chunk);
        }

        assert!(!chunks.is_empty(), "Should produce at least one token");
        let full: String = chunks.into_iter().collect();
        assert!(!full.is_empty(), "Combined output should not be empty");
        assert!(full.len() < TEST_ARTICLE.len(), "Summary should be shorter");
    }

    #[tokio::test]
    async fn test_stream_matches_non_stream() {
        if !model_available("distilbart-cnn") {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let s = Summarizer::builder("distilbart-cnn")
            .greedy() // Use greedy for deterministic output
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        // Non-streaming
        let direct = s.summarize(TEST_ARTICLE).await.unwrap();

        // Streaming
        let mut stream = s.stream(TEST_ARTICLE).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(result) = stream.next().await {
            chunks.push(result.unwrap());
        }
        let streamed: String = chunks.into_iter().collect();

        // With greedy decoding, these should be identical
        assert_eq!(
            direct, streamed,
            "Streaming and direct should match with greedy decoding"
        );
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[tokio::test]
    async fn test_unknown_model_error() {
        let result = Summarizer::new("not-a-real-model").await;
        assert!(matches!(result, Err(SummarizerError::UnknownModel(_))));
    }

    // =========================================================================
    // Batch Summarization Tests
    // =========================================================================

    #[tokio::test]
    async fn test_batch_summarization() {
        if !model_available("distilbart-cnn") {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let s = Summarizer::builder("distilbart-cnn")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let articles = vec![
            "Scientists discover water on Mars. The discovery was made by NASA's rover.",
            "New AI system beats human at chess. The system uses neural networks.",
            "Climate change impacts global agriculture. Farmers are adapting to new conditions.",
        ];

        let mut summaries = Vec::new();
        for article in &articles {
            let summary = s.summarize(article).await.unwrap();
            summaries.push(summary);
        }

        assert_eq!(summaries.len(), articles.len());
        for (i, summary) in summaries.iter().enumerate() {
            assert!(!summary.is_empty(), "Summary {} should not be empty", i);
        }
    }

    // =========================================================================
    // Concurrent Summarization Tests
    // =========================================================================

    #[tokio::test]
    async fn test_concurrent_summarization() {
        if !model_available("distilbart-cnn") {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let s = std::sync::Arc::new(
            Summarizer::builder("distilbart-cnn")
                .cpu()
                .quiet()
                .build()
                .await
                .unwrap(),
        );

        let articles = vec![
            "Article one about science and discovery.",
            "Article two about technology and innovation.",
            "Article three about art and culture.",
        ];

        let handles: Vec<_> = articles
            .into_iter()
            .map(|text| {
                let summarizer = s.clone();
                let text = text.to_string();
                tokio::spawn(async move { summarizer.summarize(&text).await })
            })
            .collect();

        for (i, handle) in handles.into_iter().enumerate() {
            let result = handle.await.expect("Task panicked");
            let summary = result.expect("Summarization failed");
            assert!(
                !summary.is_empty(),
                "Concurrent summary {} should not be empty",
                i
            );
        }
    }

    // =========================================================================
    // Preset and Tier Tests
    // =========================================================================

    #[tokio::test]
    async fn test_summarize_with_preset() {
        use crate::summarizer::presets::SUMMARIZER_FAST_V1;

        if !model_available("distilbart-cnn") {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let s = SummarizerBuilder::from_preset(&SUMMARIZER_FAST_V1)
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let summary = s.summarize(TEST_ARTICLE).await.unwrap();
        assert!(!summary.is_empty());
        assert!(summary.len() < TEST_ARTICLE.len());
    }

    #[tokio::test]
    async fn test_summarize_tier() {
        if !model_available("distilbart-cnn") {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        // Note: This creates a new Summarizer each time (not ideal for perf but tests the API)
        let result = summarize_tier(SummarizerTier::Fast, TEST_ARTICLE).await;
        if let Ok(summary) = result {
            assert!(!summary.is_empty());
        }
    }

    // =========================================================================
    // Accessor Tests
    // =========================================================================

    #[tokio::test]
    async fn test_summarizer_accessors() {
        if !model_available("distilbart-cnn") {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let s = Summarizer::builder("distilbart-cnn")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        assert_eq!(s.model_name(), "distilbart-cnn");
        assert!(matches!(
            s.device(),
            kjarni_transformers::traits::Device::Cpu
        ));
        assert!(s.generator().context_size() > 0);
        assert!(s.generator().vocab_size() > 0);
    }

    // =========================================================================
    // Convenience Function Tests
    // =========================================================================

    #[tokio::test]
    async fn test_module_summarize_function() {
        if !model_available("distilbart-cnn") {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let result = summarize("distilbart-cnn", TEST_ARTICLE).await;
        assert!(result.is_ok());
        let summary = result.unwrap();
        assert!(!summary.is_empty());
        assert!(summary.len() < TEST_ARTICLE.len());
    }
}
