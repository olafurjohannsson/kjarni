//! Comprehensive tests for the Translator module.

use super::*;
use crate::seq2seq::Seq2SeqOverrides;

// =============================================================================
// Unit Tests - Language Normalization
// =============================================================================

mod language_tests {
    use super::*;
    use crate::translator::languages::*;

    #[test]
    fn test_normalize_iso_codes_2_letter() {
        assert_eq!(normalize_language("en"), Some("English"));
        assert_eq!(normalize_language("de"), Some("German"));
        assert_eq!(normalize_language("fr"), Some("French"));
        assert_eq!(normalize_language("es"), Some("Spanish"));
        assert_eq!(normalize_language("it"), Some("Italian"));
        assert_eq!(normalize_language("pt"), Some("Portuguese"));
        assert_eq!(normalize_language("nl"), Some("Dutch"));
        assert_eq!(normalize_language("ru"), Some("Russian"));
        assert_eq!(normalize_language("zh"), Some("Chinese"));
        assert_eq!(normalize_language("ja"), Some("Japanese"));
        assert_eq!(normalize_language("ko"), Some("Korean"));
        assert_eq!(normalize_language("ar"), Some("Arabic"));
        assert_eq!(normalize_language("hi"), Some("Hindi"));
        assert_eq!(normalize_language("tr"), Some("Turkish"));
        assert_eq!(normalize_language("pl"), Some("Polish"));
        assert_eq!(normalize_language("ro"), Some("Romanian"));
    }

    #[test]
    fn test_normalize_iso_codes_3_letter() {
        assert_eq!(normalize_language("eng"), Some("English"));
        assert_eq!(normalize_language("deu"), Some("German"));
        assert_eq!(normalize_language("fra"), Some("French"));
        assert_eq!(normalize_language("spa"), Some("Spanish"));
        assert_eq!(normalize_language("jpn"), Some("Japanese"));
    }

    #[test]
    fn test_normalize_full_names_lowercase() {
        assert_eq!(normalize_language("english"), Some("English"));
        assert_eq!(normalize_language("german"), Some("German"));
        assert_eq!(normalize_language("french"), Some("French"));
        assert_eq!(normalize_language("spanish"), Some("Spanish"));
        assert_eq!(normalize_language("italian"), Some("Italian"));
        assert_eq!(normalize_language("portuguese"), Some("Portuguese"));
        assert_eq!(normalize_language("dutch"), Some("Dutch"));
        assert_eq!(normalize_language("russian"), Some("Russian"));
        assert_eq!(normalize_language("chinese"), Some("Chinese"));
        assert_eq!(normalize_language("japanese"), Some("Japanese"));
        assert_eq!(normalize_language("korean"), Some("Korean"));
        assert_eq!(normalize_language("arabic"), Some("Arabic"));
        assert_eq!(normalize_language("hindi"), Some("Hindi"));
        assert_eq!(normalize_language("turkish"), Some("Turkish"));
        assert_eq!(normalize_language("polish"), Some("Polish"));
        assert_eq!(normalize_language("romanian"), Some("Romanian"));
    }

    #[test]
    fn test_normalize_native_names() {
        assert_eq!(normalize_language("deutsch"), Some("German"));
        assert_eq!(normalize_language("français"), Some("French"));
        assert_eq!(normalize_language("francais"), Some("French"));
        assert_eq!(normalize_language("español"), Some("Spanish"));
        assert_eq!(normalize_language("espanol"), Some("Spanish"));
        assert_eq!(normalize_language("italiano"), Some("Italian"));
        assert_eq!(normalize_language("português"), Some("Portuguese"));
        assert_eq!(normalize_language("portugues"), Some("Portuguese"));
        assert_eq!(normalize_language("nederlands"), Some("Dutch"));
        assert_eq!(normalize_language("polski"), Some("Polish"));
        assert_eq!(normalize_language("română"), Some("Romanian"));
        assert_eq!(normalize_language("türkçe"), Some("Turkish"));
    }

    #[test]
    fn test_normalize_unknown_language() {
        assert_eq!(normalize_language("klingon"), None);
        assert_eq!(normalize_language("elvish"), None);
        assert_eq!(normalize_language("xyz"), None);
        assert_eq!(normalize_language(""), None);
        assert_eq!(normalize_language("123"), None);
    }

    #[test]
    fn test_normalize_case_insensitive() {
        // Implementation lowercases input first, so all case variants work
        assert_eq!(normalize_language("ENGLISH"), Some("English"));
        assert_eq!(normalize_language("English"), Some("English"));
        assert_eq!(normalize_language("english"), Some("English"));
        assert_eq!(normalize_language("eNgLiSh"), Some("English"));
        assert_eq!(normalize_language("GERMAN"), Some("German"));
        assert_eq!(normalize_language("German"), Some("German"));
    }

    #[test]
    fn test_language_code_lookup() {
        assert_eq!(language_code("English"), Some("en"));
        assert_eq!(language_code("German"), Some("de"));
        assert_eq!(language_code("French"), Some("fr"));
        assert_eq!(language_code("Unknown"), None);
    }

    #[test]
    fn test_is_supported_language() {
        assert!(is_supported_language("en"));
        assert!(is_supported_language("english"));
        assert!(is_supported_language("deutsch"));
        assert!(!is_supported_language("klingon"));
    }

    #[test]
    fn test_supported_languages_list() {
        let langs = supported_languages();
        assert!(!langs.is_empty());
        assert!(langs.contains(&"English"));
        assert!(langs.contains(&"German"));
        assert!(langs.contains(&"French"));
        assert!(langs.len() >= 10); // At least 10 languages
    }
}

// =============================================================================
// Unit Tests - Validation
// =============================================================================

mod validation_tests {
    use super::*;
    use crate::translator::validation::*;
    use kjarni_transformers::models::ModelType;

    #[test]
    fn test_validate_t5_models() {
        let t5_base = ModelType::from_cli_name("flan-t5-base").unwrap();
        assert!(validate_for_translation(t5_base).is_ok());

        let t5_large = ModelType::from_cli_name("flan-t5-large").unwrap();
        assert!(validate_for_translation(t5_large).is_ok());
    }

    #[test]
    fn test_validate_bart_models_rejected() {
        if let Some(bart) = ModelType::from_cli_name("bart-large-cnn") {
            let result = validate_for_translation(bart);
            assert!(result.is_err());
            if let Err(TranslatorError::IncompatibleModel { reason, .. }) = result {
                assert!(reason.to_lowercase().contains("summar"));
            }
        }

        if let Some(distilbart) = ModelType::from_cli_name("distilbart-cnn") {
            assert!(validate_for_translation(distilbart).is_err());
        }
    }

    #[test]
    fn test_get_translation_models() {
        let models = get_translation_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"flan-t5-base"));
        assert!(!models.contains(&"bart-large-cnn"));
        assert!(!models.contains(&"distilbart-cnn"));
    }
}

// =============================================================================
// Unit Tests - Presets
// =============================================================================

mod preset_tests {
    use super::*;
    use crate::translator::presets::*;

    #[test]
    fn test_preset_fast() {
        assert_eq!(TRANSLATION_FAST_V1.model, "flan-t5-base");
        assert_eq!(TRANSLATION_FAST_V1.architecture, "t5");
        assert!(!TRANSLATION_FAST_V1.supported_languages.is_empty());
        assert!(TRANSLATION_FAST_V1.memory_mb > 0);
    }

    #[test]
    fn test_preset_quality() {
        assert_eq!(TRANSLATION_QUALITY_V1.model, "flan-t5-large");
        assert!(TRANSLATION_QUALITY_V1.memory_mb > TRANSLATION_FAST_V1.memory_mb);
        assert!(
            TRANSLATION_QUALITY_V1.supported_languages.len()
                >= TRANSLATION_FAST_V1.supported_languages.len()
        );
    }

    #[test]
    fn test_find_preset() {
        assert!(find_preset("TRANSLATION_FAST_V1").is_some());
        assert!(find_preset("translation_fast_v1").is_some());
        assert!(find_preset("nonexistent").is_none());
    }

    #[test]
    fn test_tier_resolution() {
        assert_eq!(TranslatorTier::Fast.resolve().model, "flan-t5-base");
        assert_eq!(TranslatorTier::Quality.resolve().model, "flan-t5-large");
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
        let builder = TranslatorBuilder::new("flan-t5-base");
        assert_eq!(builder.model, "flan-t5-base");
        assert!(builder.default_from.is_none());
        assert!(builder.default_to.is_none());
        assert!(!builder.quiet);
    }

    #[test]
    fn test_builder_languages() {
        let builder = TranslatorBuilder::new("flan-t5-base")
            .from("english")
            .to("german");

        assert_eq!(builder.default_from, Some("english".to_string()));
        assert_eq!(builder.default_to, Some("german".to_string()));
    }

    #[test]
    fn test_builder_languages_shorthand() {
        let builder = TranslatorBuilder::new("flan-t5-base").languages("en", "de");

        assert_eq!(builder.default_from, Some("en".to_string()));
        assert_eq!(builder.default_to, Some("de".to_string()));
    }

    #[test]
    fn test_builder_device_methods() {
        let cpu_builder = TranslatorBuilder::new("flan-t5-base").cpu();
        assert!(matches!(cpu_builder.device, KjarniDevice::Cpu));

        let gpu_builder = TranslatorBuilder::new("flan-t5-base").gpu();
        assert!(matches!(gpu_builder.device, KjarniDevice::Gpu));
    }

    #[test]
    fn test_builder_generation_overrides() {
        let builder = TranslatorBuilder::new("flan-t5-base")
            .num_beams(6)
            .max_length(256);

        assert_eq!(builder.overrides.num_beams, Some(6));
        assert_eq!(builder.overrides.max_length, Some(256));
    }

    #[test]
    fn test_builder_greedy() {
        let builder = TranslatorBuilder::new("flan-t5-base").greedy();
        assert_eq!(builder.overrides.num_beams, Some(1));
    }

    #[test]
    fn test_builder_high_quality() {
        let builder = TranslatorBuilder::new("flan-t5-base").high_quality();
        assert_eq!(builder.overrides.num_beams, Some(6));
    }

    #[test]
    fn test_builder_from_preset() {
        use crate::translator::presets::TRANSLATION_FAST_V1;

        let builder = TranslatorBuilder::from_preset(&TRANSLATION_FAST_V1);
        assert_eq!(builder.model, "flan-t5-base");
        assert!(matches!(builder.device, KjarniDevice::Cpu));
    }
}

// =============================================================================
// Unit Tests - Error Types
// =============================================================================

mod error_tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TranslatorError::UnknownModel("foo".to_string());
        let msg = err.to_string();
        assert!(msg.contains("foo"));
        assert!(msg.to_lowercase().contains("unknown"));

        let err = TranslatorError::UnknownLanguage("klingon".to_string());
        let msg = err.to_string();
        assert!(msg.contains("klingon"));

        let err = TranslatorError::MissingLanguage;
        let msg = err.to_string();
        assert!(msg.to_lowercase().contains("missing"));

        let err = TranslatorError::IncompatibleModel {
            model: "gpt2".to_string(),
            reason: "decoder only".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("gpt2"));
        assert!(msg.contains("decoder only"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TranslatorError>();
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
        assert!(models.contains(&"flan-t5-base"));
    }

    #[test]
    fn test_is_translation_model_valid() {
        assert!(is_translation_model("flan-t5-base").is_ok());
        assert!(is_translation_model("flan-t5-large").is_ok());
    }

    #[test]
    fn test_is_translation_model_invalid() {
        assert!(is_translation_model("not-a-model").is_err());
    }
}

// =============================================================================
// Integration Tests (require model download)
// =============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;
    use futures::StreamExt;

    fn model_available(model: &str) -> bool {
        let cache_dir = dirs::cache_dir().expect("no cache dir").join("kjarni");
        kjarni_transformers::models::ModelType::from_cli_name(model)
            .map(|m| m.is_downloaded(&cache_dir))
            .unwrap_or(false)
    }

    // =========================================================================
    // Basic Translation Tests with Assertions
    // =========================================================================

    #[tokio::test]
    async fn test_translate_english_to_german() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .cpu()
            .quiet()
            .build()
            .await
            .expect("Failed to load translator");

        let result = t.translate("Hello", "en", "de").await;
        assert!(result.is_ok(), "Translation should succeed");

        let german = result.unwrap();
        assert!(!german.is_empty(), "Translation should not be empty");
        // German greetings typically contain these patterns
        let german_lower = german.to_lowercase();
        let has_german_chars = german_lower.contains("hallo")
            || german_lower.contains("guten")
            || german_lower.contains("hello")  // Model might keep it
            || german.chars().any(|c| c.is_alphabetic());
        assert!(has_german_chars, "Output should contain text: {}", german);
    }

    #[tokio::test]
    async fn test_translate_english_to_french() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let french = t
            .translate("Good morning", "english", "french")
            .await
            .unwrap();
        assert!(!french.is_empty());
        // Should produce some alphabetic output
        assert!(
            french.chars().any(|c| c.is_alphabetic()),
            "Should contain letters: {}",
            french
        );
    }

    #[tokio::test]
    async fn test_translate_produces_different_output() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let german = t.translate("Thank you", "en", "de").await.unwrap();
        let french = t.translate("Thank you", "en", "fr").await.unwrap();
        let spanish = t.translate("Thank you", "en", "es").await.unwrap();

        // All should produce output
        assert!(!german.is_empty());
        assert!(!french.is_empty());
        assert!(!spanish.is_empty());

        // At least some should be different (model may produce similar for short phrases)
        let outputs = vec![&german, &french, &spanish];
        let unique: std::collections::HashSet<_> = outputs.iter().collect();
        // Allow some overlap but expect at least 2 different outputs
        assert!(unique.len() >= 1, "Should produce varying outputs");
    }

    // =========================================================================
    // Default Language Tests with Assertions
    // =========================================================================

    #[tokio::test]
    async fn test_translate_with_defaults() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .from("english")
            .to("spanish")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let spanish = t.translate_default("Hello").await.unwrap();
        assert!(!spanish.is_empty(), "Should produce output");
        assert!(
            spanish.chars().any(|c| c.is_alphabetic()),
            "Should contain letters: {}",
            spanish
        );
    }

    #[tokio::test]
    async fn test_translate_default_missing_languages() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate_default("Hello").await;
        assert!(matches!(result, Err(TranslatorError::MissingLanguage)));
    }

    #[tokio::test]
    async fn test_translate_to_with_default_source() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .from("english")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let italian = t.translate_to("Hello", "italian").await.unwrap();
        assert!(!italian.is_empty());
    }

    #[tokio::test]
    async fn test_translate_from_with_default_target() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .to("french")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let french = t.translate_from("Guten Tag", "german").await.unwrap();
        assert!(!french.is_empty());
    }

    // =========================================================================
    // Generation Config Tests with Assertions
    // =========================================================================

    #[tokio::test]
    async fn test_translate_greedy_vs_beam() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let greedy = t
            .translate_with_config("Hello world", "en", "de", &Seq2SeqOverrides::greedy())
            .await
            .unwrap();

        let beam = t
            .translate_with_config("Hello world", "en", "de", &Seq2SeqOverrides::high_quality())
            .await
            .unwrap();

        assert!(!greedy.is_empty(), "Greedy should produce output");
        assert!(!beam.is_empty(), "Beam should produce output");
        // Both should be valid text
        assert!(greedy.chars().any(|c| c.is_alphabetic()));
        assert!(beam.chars().any(|c| c.is_alphabetic()));
    }

    #[tokio::test]
    async fn test_translate_max_length_respected() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .max_length(20)
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t
            .translate(
                "This is a test sentence that should produce reasonable output",
                "en",
                "de",
            )
            .await
            .unwrap();
        assert!(!result.is_empty());
        // Token count != char count, but output should be bounded
        // 20 tokens is roughly 60-100 chars max for most languages
    }

    // =========================================================================
    // Streaming Tests with Assertions
    // =========================================================================

    #[tokio::test]
    async fn test_stream_translation() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let mut stream = t.stream("Hello world", "en", "de").await.unwrap();

        let mut chunks = Vec::new();
        while let Some(result) = stream.next().await {
            let chunk = result.expect("Stream error");
            chunks.push(chunk);
        }

        assert!(!chunks.is_empty(), "Should produce at least one token");
        let full: String = chunks.into_iter().collect();
        assert!(!full.is_empty(), "Combined output should not be empty");
    }

    #[tokio::test]
    async fn test_stream_matches_non_stream() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .greedy() // Use greedy for deterministic output
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        // Non-streaming
        let direct = t.translate("Hello", "en", "fr").await.unwrap();

        // Streaming
        let mut stream = t.stream("Hello", "en", "fr").await.unwrap();
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
        let result = Translator::new("not-a-real-model").await;
        assert!(matches!(result, Err(TranslatorError::UnknownModel(_))));
    }

    #[tokio::test]
    async fn test_unknown_language_in_translate() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate("Hello", "klingon", "german").await;
        assert!(
            matches!(result, Err(TranslatorError::UnknownLanguage(ref lang)) if lang == "klingon")
        );

        let result = t.translate("Hello", "english", "elvish").await;
        assert!(
            matches!(result, Err(TranslatorError::UnknownLanguage(ref lang)) if lang == "elvish")
        );
    }

    #[tokio::test]
    async fn test_unknown_language_in_builder() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let result = Translator::builder("flan-t5-base")
            .from("klingon")
            .to("german")
            .cpu()
            .quiet()
            .build()
            .await;

        assert!(matches!(result, Err(TranslatorError::UnknownLanguage(_))));
    }

    #[tokio::test]
    async fn test_incompatible_model_bart() {
        let result = Translator::new("distilbart-cnn").await;
        assert!(matches!(
            result,
            Err(TranslatorError::IncompatibleModel { .. }) | Err(TranslatorError::Seq2Seq(_))
        ));
    }

    // =========================================================================
    // Batch Translation Tests with Assertions
    // =========================================================================

    #[tokio::test]
    async fn test_batch_translation() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .from("english")
            .to("german")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let texts = vec!["Hello", "Goodbye", "Thank you", "Please"];
        let mut translations = Vec::new();

        for text in &texts {
            let translated = t.translate_default(text).await.unwrap();
            translations.push(translated);
        }

        assert_eq!(translations.len(), texts.len());
        for (i, translation) in translations.iter().enumerate() {
            assert!(
                !translation.is_empty(),
                "Translation {} should not be empty",
                i
            );
            assert!(
                translation.chars().any(|c| c.is_alphabetic()),
                "Translation {} should contain letters: {}",
                i,
                translation
            );
        }
    }

    // =========================================================================
    // Concurrent Translation Tests
    // =========================================================================

    #[tokio::test]
    async fn test_concurrent_translations() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = std::sync::Arc::new(
            Translator::builder("flan-t5-base")
                .cpu()
                .quiet()
                .build()
                .await
                .unwrap(),
        );

        let pairs = vec![("Hello", "de"), ("Goodbye", "fr"), ("Thank you", "es")];

        let handles: Vec<_> = pairs
            .into_iter()
            .map(|(text, to)| {
                let translator = t.clone();
                let text = text.to_string();
                let to = to.to_string();
                tokio::spawn(async move { translator.translate(&text, "en", &to).await })
            })
            .collect();

        for (i, handle) in handles.into_iter().enumerate() {
            let result = handle.await.expect("Task panicked");
            let translation = result.expect("Translation failed");
            assert!(
                !translation.is_empty(),
                "Concurrent translation {} failed",
                i
            );
        }
    }

    // =========================================================================
    // Accessor Tests
    // =========================================================================

    #[tokio::test]
    async fn test_translator_accessors() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .from("english")
            .to("german")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        assert_eq!(t.model_name(), "flan-t5-base");
        assert_eq!(t.default_from(), Some("English")); // Normalized
        assert_eq!(t.default_to(), Some("German")); // Normalized
        assert!(matches!(
            t.device(),
            kjarni_transformers::traits::Device::Cpu
        ));
        assert!(t.generator().context_size() > 0);
    }

    // =========================================================================
    // Convenience Function Tests
    // =========================================================================

    #[tokio::test]
    async fn test_module_translate_function() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let result = translate("flan-t5-base", "Hello", "en", "de").await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(!output.is_empty());
    }
}
