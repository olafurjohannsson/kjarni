use super::*;
use futures::StreamExt;

// Expected Outputs from PyTorch Reference

mod expected {
    pub const FLAN_T5_BASE_EN_TO_DE_HELLO_GREEDY: &str = "Hello,";
    #[expect(dead_code, reason = "test scaffolding for the suite in this module")]
    pub const FLAN_T5_BASE_EN_TO_DE_HELLO_BEAM: &str = "Hello!";
    pub const FLAN_T5_BASE_EN_TO_DE_HOW_OLD_GREEDY: &str = "Wie old sind Sie?";
    #[expect(dead_code, reason = "test scaffolding for the suite in this module")]
    pub const FLAN_T5_BASE_EN_TO_DE_HOW_OLD_BEAM: &str = "How old are you?";
    pub const FLAN_T5_BASE_EN_TO_DE_GOOD_MORNING_GREEDY: &str = "Good morning!";
    #[expect(dead_code, reason = "test scaffolding for the suite in this module")]
    pub const FLAN_T5_BASE_EN_TO_DE_GOOD_MORNING_BEAM: &str = "Good morning!";
    pub const FLAN_T5_BASE_EN_TO_DE_THANK_YOU_GREEDY: &str = "Vielen Dank!";
    #[expect(dead_code, reason = "test scaffolding for the suite in this module")]
    pub const FLAN_T5_BASE_EN_TO_DE_THANK_YOU_BEAM: &str = "Dankeschön!";
    pub const FLAN_T5_BASE_EN_TO_FR_HELLO_GREEDY: &str = "Bonjour,";
    #[expect(dead_code, reason = "test scaffolding for the suite in this module")]
    pub const FLAN_T5_BASE_EN_TO_FR_HELLO_BEAM: &str = "Bonjour!";
    pub const FLAN_T5_BASE_EN_TO_FR_GOOD_MORNING_GREEDY: &str = "Good morning";
    #[expect(dead_code, reason = "test scaffolding for the suite in this module")]
    pub const FLAN_T5_BASE_EN_TO_FR_GOOD_MORNING_BEAM: &str = "Good morning";
    pub const FLAN_T5_BASE_EN_TO_ES_HELLO_GREEDY: &str = "Hi,";
    #[expect(dead_code, reason = "test scaffolding for the suite in this module")]
    pub const FLAN_T5_BASE_EN_TO_ES_HELLO_BEAM: &str = "Hombre!";
    pub const FLAN_T5_BASE_EN_TO_ES_THANK_YOU_GREEDY: &str = "Gracias";
    #[expect(dead_code, reason = "test scaffolding for the suite in this module")]
    pub const FLAN_T5_BASE_EN_TO_ES_THANK_YOU_BEAM: &str = "Gracias";
    pub const FLAN_T5_LARGE_EN_TO_DE_HELLO_GREEDY: &str = "Hello!";
    #[expect(dead_code, reason = "test scaffolding for the suite in this module")]
    pub const FLAN_T5_LARGE_EN_TO_DE_HELLO_BEAM: &str = "Hello!";
    pub const FLAN_T5_LARGE_EN_TO_DE_HOW_OLD_GREEDY: &str = "Wie alte sind Sie?";
    #[expect(dead_code, reason = "test scaffolding for the suite in this module")]
    pub const FLAN_T5_LARGE_EN_TO_DE_HOW_OLD_BEAM: &str = "Wie alte sind Sie?";
    pub const FLAN_T5_LARGE_EN_TO_DE_GOOD_MORNING_GREEDY: &str =
        "Ich freue mich, dass Sie daran reisen.";
    #[expect(dead_code, reason = "test scaffolding for the suite in this module")]
    pub const FLAN_T5_LARGE_EN_TO_DE_GOOD_MORNING_BEAM: &str = "Sehr gute Morgen!";
    pub const FLAN_T5_LARGE_EN_TO_DE_THANK_YOU_GREEDY: &str = "Danke!";
    #[expect(dead_code, reason = "test scaffolding for the suite in this module")]
    pub const FLAN_T5_LARGE_EN_TO_DE_THANK_YOU_BEAM: &str = "Vielen Dank!";
    pub const FLAN_T5_LARGE_EN_TO_FR_HELLO_GREEDY: &str = "Hello!";
    #[expect(dead_code, reason = "test scaffolding for the suite in this module")]
    pub const FLAN_T5_LARGE_EN_TO_FR_HELLO_BEAM: &str = "Hello!";
    pub const FLAN_T5_LARGE_EN_TO_FR_GOOD_MORNING_GREEDY: &str =
        " l'heure actuelle, il y a d'autres possibilités de l'emploi.";
    #[expect(dead_code, reason = "test scaffolding for the suite in this module")]
    pub const FLAN_T5_LARGE_EN_TO_FR_GOOD_MORNING_BEAM: &str = "Bienvenue !";
    pub const FLAN_T5_LARGE_EN_TO_ES_HELLO_GREEDY: &str = "Hello!";
    #[expect(dead_code, reason = "test scaffolding for the suite in this module")]
    pub const FLAN_T5_LARGE_EN_TO_ES_HELLO_BEAM: &str = "Hello!";
    pub const FLAN_T5_LARGE_EN_TO_ES_THANK_YOU_GREEDY: &str = "Gracias";
    #[expect(dead_code, reason = "test scaffolding for the suite in this module")]
    pub const FLAN_T5_LARGE_EN_TO_ES_THANK_YOU_BEAM: &str = "Gracias";
}

mod language_tests {
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
        assert!(langs.len() >= 16);
        assert!(langs.contains(&"English"));
        assert!(langs.contains(&"German"));
        assert!(langs.contains(&"French"));
        assert!(langs.contains(&"Spanish"));
        assert!(langs.contains(&"Chinese"));
        assert!(langs.contains(&"Japanese"));
    }
}

// Unit Tests - Validation

mod validation_tests {
    use crate::translator::TranslatorError;
    use crate::translator::validation::*;
    use kjarni_transformers::models::ModelType;

    #[expect(dead_code, reason = "test scaffolding for the suite in this module")]
    fn test_validate_t5_models_accepted() {
        let t5_base = ModelType::from_cli_name("flan-t5-base").unwrap();
        let result = crate::summarizer::validate_for_summarization(t5_base);
        assert!(
            result.is_ok(),
            "T5 should be accepted for summarization: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_validate_bart_models_rejected() {
        if let Some(bart) = ModelType::from_cli_name("bart-large-cnn") {
            let result = validate_for_translation(bart);
            assert!(result.is_err());
            match result {
                Err(TranslatorError::IncompatibleModel { reason, .. }) => {
                    assert!(reason.to_lowercase().contains("summar"));
                }
                _ => panic!("Expected IncompatibleModel error"),
            }
        }

        if let Some(distilbart) = ModelType::from_cli_name("distilbart-cnn") {
            assert!(validate_for_translation(distilbart).is_err());
        }
    }

    #[test]
    fn test_get_translation_models_returns_only_t5() {
        let models = get_translation_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"flan-t5-base"));
        assert!(models.contains(&"flan-t5-large"));
        assert!(!models.contains(&"bart-large-cnn"));
        assert!(!models.contains(&"distilbart-cnn"));
    }
}

// Unit Tests - Presets

mod preset_tests {
    use crate::translator::presets::*;

    #[test]
    fn test_preset_fast_values() {
        assert_eq!(TRANSLATION_FAST_V1.model, "flan-t5-base");
        assert_eq!(TRANSLATION_FAST_V1.architecture, "t5");
        assert!(TRANSLATION_FAST_V1.supported_languages.len() == 6);
        const { assert!(TRANSLATION_FAST_V1.memory_mb >= 500) };
        const { assert!(TRANSLATION_FAST_V1.memory_mb <= 2000) };
    }

    #[test]
    fn test_preset_quality_values() {
        assert_eq!(TRANSLATION_QUALITY_V1.model, "flan-t5-large");
        assert_eq!(TRANSLATION_QUALITY_V1.architecture, "t5");
        const { assert!(TRANSLATION_QUALITY_V1.memory_mb > TRANSLATION_FAST_V1.memory_mb) };
        assert!(
            TRANSLATION_QUALITY_V1.supported_languages.len()
                >= TRANSLATION_FAST_V1.supported_languages.len()
        );
    }

    #[test]
    fn test_find_preset_case_insensitive() {
        assert!(find_preset("TRANSLATION_FAST_V1").is_some());
        assert!(find_preset("translation_fast_v1").is_some());
        assert!(find_preset("Translation_Fast_V1").is_some());
        assert!(find_preset("nonexistent").is_none());
    }

    #[test]
    fn test_tier_resolution() {
        assert_eq!(TranslatorTier::Fast.resolve().model, "flan-t5-base");
        assert_eq!(TranslatorTier::Quality.resolve().model, "flan-t5-large");
    }
}

// Unit Tests - Builder

mod builder_tests {
    use crate::common::KjarniDevice;
    use crate::translator::TranslatorBuilder;

    #[test]
    fn test_builder_default_state() {
        let builder = TranslatorBuilder::new("flan-t5-base");
        assert_eq!(builder.model, "flan-t5-base");
        assert!(builder.default_from.is_none());
        assert!(builder.default_to.is_none());
        assert!(!builder.quiet);
        assert!(builder.overrides.is_empty());
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
    fn test_builder_overrides_are_explicit() {
        let builder = TranslatorBuilder::new("flan-t5-base")
            .num_beams(6)
            .max_length(256);

        assert_eq!(builder.overrides.num_beams, Some(6));
        assert_eq!(builder.overrides.max_length, Some(256));
        assert!(builder.overrides.min_length.is_none());
        assert!(builder.overrides.length_penalty.is_none());
        assert!(builder.overrides.no_repeat_ngram_size.is_none());
    }

    #[test]
    fn test_builder_greedy_sets_num_beams_1() {
        let builder = TranslatorBuilder::new("flan-t5-base").greedy();
        assert_eq!(builder.overrides.num_beams, Some(1));
    }

    #[test]
    fn test_builder_high_quality_sets_num_beams_6() {
        let builder = TranslatorBuilder::new("flan-t5-base").high_quality();
        assert_eq!(builder.overrides.num_beams, Some(6));
    }
}

mod error_tests {
    use crate::translator::TranslatorError;

    #[test]
    fn test_error_unknown_language_message() {
        let err = TranslatorError::UnknownLanguage("klingon".to_string());
        let msg = err.to_string();
        assert!(msg.contains("klingon"));
    }

    #[test]
    fn test_error_missing_language_message() {
        let err = TranslatorError::MissingLanguage;
        let msg = err.to_string();
        assert!(msg.to_lowercase().contains("missing") || msg.to_lowercase().contains("language"));
    }

    #[test]
    fn test_error_incompatible_model_message() {
        let err = TranslatorError::IncompatibleModel {
            model: "bart-cnn".to_string(),
            reason: "summarization model".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("bart-cnn"));
        assert!(msg.contains("summarization model"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TranslatorError>();
    }
}

// Unit Tests - Module Functions

mod module_function_tests {
    use crate::translator::{available_models, is_translation_model};

    #[test]
    fn test_available_models_contains_t5() {
        let models = available_models();
        assert!(models.contains(&"flan-t5-base"));
        assert!(models.contains(&"flan-t5-large"));
    }

    #[test]
    fn test_available_models_excludes_bart() {
        let models = available_models();
        assert!(!models.contains(&"bart-large-cnn"));
        assert!(!models.contains(&"distilbart-cnn"));
    }

    #[test]
    fn test_is_translation_model_valid() {
        assert!(is_translation_model("flan-t5-base").is_ok());
        assert!(is_translation_model("flan-t5-large").is_ok());
    }

    #[test]
    fn test_is_translation_model_invalid() {
        assert!(is_translation_model("not-a-model").is_err());
        assert!(is_translation_model("bart-large-cnn").is_err());
    }
}

// Integration Tests - Model Output Verification (flan-t5-base)

#[cfg(test)]
mod flan_t5_base_tests {
    use super::expected::*;
    use super::*;

    fn model_available() -> bool {
        let cache_dir = dirs::cache_dir().expect("no cache dir").join("kjarni");
        kjarni_transformers::models::ModelType::from_cli_name("flan-t5-base")
            .map(|m| m.is_downloaded(&cache_dir))
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn test_greedy_en_to_de_hello() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate("Hello", "en", "de").await.unwrap();
        assert_eq!(result, FLAN_T5_BASE_EN_TO_DE_HELLO_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_en_to_de_how_old() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate("How old are you?", "en", "de").await.unwrap();
        assert_eq!(result, FLAN_T5_BASE_EN_TO_DE_HOW_OLD_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_en_to_de_good_morning() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate("Good morning", "en", "de").await.unwrap();
        assert_eq!(result, FLAN_T5_BASE_EN_TO_DE_GOOD_MORNING_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_en_to_de_thank_you() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate("Thank you", "en", "de").await.unwrap();
        assert_eq!(result, FLAN_T5_BASE_EN_TO_DE_THANK_YOU_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_en_to_fr_hello() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate("Hello", "en", "fr").await.unwrap();
        assert_eq!(result, FLAN_T5_BASE_EN_TO_FR_HELLO_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_en_to_fr_good_morning() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate("Good morning", "en", "fr").await.unwrap();
        assert_eq!(result, FLAN_T5_BASE_EN_TO_FR_GOOD_MORNING_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_en_to_es_hello() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate("Hello", "en", "es").await.unwrap();
        assert_eq!(result, FLAN_T5_BASE_EN_TO_ES_HELLO_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_en_to_es_thank_you() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate("Thank you", "en", "es").await.unwrap();
        assert_eq!(result, FLAN_T5_BASE_EN_TO_ES_THANK_YOU_GREEDY);
    }

    #[tokio::test]
    async fn test_streaming_matches_non_streaming() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let direct = t.translate("Thank you", "en", "de").await.unwrap();

        let mut stream = t.stream("Thank you", "en", "de").await.unwrap();
        let mut chunks = Vec::new();
        while let Some(result) = stream.next().await {
            chunks.push(result.unwrap());
        }
        let streamed: String = chunks.into_iter().collect();

        assert_eq!(direct, streamed);
        assert_eq!(direct, FLAN_T5_BASE_EN_TO_DE_THANK_YOU_GREEDY);
    }

    /// Streaming with default settings gives the text `translate` does: the same
    /// settings for the task, and the winning beam's tokens. The expected text is
    /// transformers' output for this prompt on the same weights.
    #[tokio::test]
    async fn streaming_with_beams_matches_non_streaming() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let direct = t
            .translate("The house is small.", "en", "de")
            .await
            .unwrap();

        let mut stream = t.stream("The house is small.", "en", "de").await.unwrap();
        let mut chunks = Vec::new();
        while let Some(result) = stream.next().await {
            chunks.push(result.unwrap());
        }
        assert!(chunks.len() > 1, "expected tokens, got {chunks:?}");
        let streamed: String = chunks.into_iter().collect();

        assert_eq!(streamed, direct);
        assert_eq!(streamed, "Das Haus ist klein.");
    }
}

// Integration Tests - Model Output Verification (flan-t5-large)

#[cfg(test)]
mod flan_t5_large_tests {
    use super::expected::*;
    use super::*;

    fn model_available() -> bool {
        let cache_dir = dirs::cache_dir().expect("no cache dir").join("kjarni");
        kjarni_transformers::models::ModelType::from_cli_name("flan-t5-large")
            .map(|m| m.is_downloaded(&cache_dir))
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn test_greedy_en_to_de_hello() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-large not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-large")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate("Hello", "en", "de").await.unwrap();
        assert_eq!(result, FLAN_T5_LARGE_EN_TO_DE_HELLO_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_en_to_de_how_old() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-large not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-large")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate("How old are you?", "en", "de").await.unwrap();
        assert_eq!(result, FLAN_T5_LARGE_EN_TO_DE_HOW_OLD_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_en_to_de_good_morning() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-large not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-large")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate("Good morning", "en", "de").await.unwrap();
        assert_eq!(result, FLAN_T5_LARGE_EN_TO_DE_GOOD_MORNING_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_en_to_de_thank_you() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-large not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-large")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate("Thank you", "en", "de").await.unwrap();
        assert_eq!(result, FLAN_T5_LARGE_EN_TO_DE_THANK_YOU_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_en_to_fr_hello() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-large not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-large")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate("Hello", "en", "fr").await.unwrap();
        assert_eq!(result, FLAN_T5_LARGE_EN_TO_FR_HELLO_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_en_to_fr_good_morning() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-large not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-large")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate("Good morning", "en", "fr").await.unwrap();
        assert_eq!(result, FLAN_T5_LARGE_EN_TO_FR_GOOD_MORNING_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_en_to_es_hello() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-large not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-large")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate("Hello", "en", "es").await.unwrap();
        assert_eq!(result, FLAN_T5_LARGE_EN_TO_ES_HELLO_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_en_to_es_thank_you() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-large not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-large")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate("Thank you", "en", "es").await.unwrap();
        assert_eq!(result, FLAN_T5_LARGE_EN_TO_ES_THANK_YOU_GREEDY);
    }

    #[tokio::test]
    async fn test_streaming_matches_non_streaming() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-large not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-large")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let direct = t.translate("Thank you", "en", "de").await.unwrap();

        let mut stream = t.stream("Thank you", "en", "de").await.unwrap();
        let mut chunks = Vec::new();
        while let Some(result) = stream.next().await {
            chunks.push(result.unwrap());
        }
        let streamed: String = chunks.into_iter().collect();

        assert_eq!(direct, streamed);
        assert_eq!(direct, FLAN_T5_LARGE_EN_TO_DE_THANK_YOU_GREEDY);
    }
}

// Integration Tests - Error Handling

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    fn model_available(model: &str) -> bool {
        let cache_dir = dirs::cache_dir().expect("no cache dir").join("kjarni");
        kjarni_transformers::models::ModelType::from_cli_name(model)
            .map(|m| m.is_downloaded(&cache_dir))
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn test_unknown_model_error() {
        let result = Translator::new("not-a-real-model-xyz").await;
        match result {
            Err(TranslatorError::UnknownModel(ref m)) => assert_eq!(m, "not-a-real-model-xyz"),
            other => panic!("Expected UnknownModel error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_incompatible_model_bart() {
        let result = Translator::new("distilbart-cnn").await;
        assert!(
            matches!(result, Err(TranslatorError::IncompatibleModel { .. }))
                || matches!(result, Err(TranslatorError::Seq2Seq(_)))
        );
    }

    #[tokio::test]
    async fn test_unknown_source_language() {
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
        match result {
            Err(TranslatorError::UnknownLanguage(ref lang)) => assert_eq!(lang, "klingon"),
            other => panic!("Expected UnknownLanguage error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_unknown_target_language() {
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

        let result = t.translate("Hello", "english", "elvish").await;
        match result {
            Err(TranslatorError::UnknownLanguage(ref lang)) => assert_eq!(lang, "elvish"),
            other => panic!("Expected UnknownLanguage error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_missing_default_languages() {
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

        match result {
            Err(TranslatorError::UnknownLanguage(ref lang)) => assert_eq!(lang, "klingon"),
            other => panic!("Expected UnknownLanguage error, got {:?}", other),
        }
    }
}

// Integration Tests - Default Languages

#[cfg(test)]
mod default_language_tests {
    use super::expected::*;
    use super::*;

    fn model_available() -> bool {
        let cache_dir = dirs::cache_dir().expect("no cache dir").join("kjarni");
        kjarni_transformers::models::ModelType::from_cli_name("flan-t5-base")
            .map(|m| m.is_downloaded(&cache_dir))
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn test_translate_default_uses_configured_languages() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .from("english")
            .to("german")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate_default("Thank you").await.unwrap();
        assert_eq!(result, FLAN_T5_BASE_EN_TO_DE_THANK_YOU_GREEDY);
    }

    #[tokio::test]
    async fn test_translate_to_with_default_source() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .from("english")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate_to("Hello", "french").await.unwrap();
        assert_eq!(result, FLAN_T5_BASE_EN_TO_FR_HELLO_GREEDY);
    }

    #[tokio::test]
    async fn test_translate_from_with_default_target() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Translator::builder("flan-t5-base")
            .to("german")
            .greedy()
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = t.translate_from("Thank you", "english").await.unwrap();
        assert_eq!(result, FLAN_T5_BASE_EN_TO_DE_THANK_YOU_GREEDY);
    }
}

// Integration Tests - Accessors

#[cfg(test)]
mod accessor_tests {
    use super::*;

    fn model_available() -> bool {
        let cache_dir = dirs::cache_dir().expect("no cache dir").join("kjarni");
        kjarni_transformers::models::ModelType::from_cli_name("flan-t5-base")
            .map(|m| m.is_downloaded(&cache_dir))
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn test_translator_accessors() {
        if !model_available() {
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
        assert_eq!(t.default_from(), Some("English"));
        assert_eq!(t.default_to(), Some("German"));
        assert!(matches!(
            t.device(),
            kjarni_transformers::traits::Device::Cpu
        ));
        assert!(t.generator().context_size() >= 512);
        assert!(t.generator().vocab_size() >= 30000);
    }
}

// Integration Tests - Concurrent Usage

#[cfg(test)]
mod concurrency_tests {
    use super::expected::*;
    use super::*;
    use std::sync::Arc;

    fn model_available() -> bool {
        let cache_dir = dirs::cache_dir().expect("no cache dir").join("kjarni");
        kjarni_transformers::models::ModelType::from_cli_name("flan-t5-base")
            .map(|m| m.is_downloaded(&cache_dir))
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn test_concurrent_translations_deterministic() {
        if !model_available() {
            eprintln!("Skipping: flan-t5-base not downloaded");
            return;
        }

        let t = Arc::new(
            Translator::builder("flan-t5-base")
                .greedy()
                .cpu()
                .quiet()
                .build()
                .await
                .unwrap(),
        );

        let handles: Vec<_> = (0..3)
            .map(|_| {
                let translator = t.clone();
                tokio::spawn(async move { translator.translate("Thank you", "en", "de").await })
            })
            .collect();

        for handle in handles {
            let result = handle
                .await
                .expect("Task panicked")
                .expect("Translation failed");
            assert_eq!(result, FLAN_T5_BASE_EN_TO_DE_THANK_YOU_GREEDY);
        }
    }
}

mod opus_mt {
    use super::super::validation::translation_input;
    use super::*;
    use kjarni_transformers::models::ModelType;

    /// From transformers' `MarianMTModel.generate` with the model's own generation config.
    const EN_DE: &[(&str, &str)] = &[
        ("Where is the station?", "Wo ist der Bahnhof?"),
        ("Good morning, how are you?", "Guten Morgen, wie geht's?"),
        (
            "The maintenance team replaced the worn seal on the second pump.",
            "Das Wartungsteam ersetzte die abgenutzte Dichtung an der zweiten Pumpe.",
        ),
        (
            "Please restart the application after installing the update.",
            "Bitte starten Sie die Anwendung nach der Installation des Updates neu.",
        ),
        (
            "My grandmother grew up in a small village near the sea.",
            "Meine Großmutter wuchs in einem kleinen Dorf am Meer auf.",
        ),
        (
            "If the battery runs out, the device saves your work automatically.",
            "Wenn die Batterie ausläuft, spart das Gerät Ihre Arbeit automatisch.",
        ),
        (
            "We cannot guarantee delivery before Friday.",
            "Wir können die Lieferung nicht vor Freitag garantieren.",
        ),
        (
            "Translation models run locally on your own computer.",
            "Übersetzungsmodelle laufen lokal auf Ihrem eigenen Computer.",
        ),
    ];
    const DE_EN: &[(&str, &str)] = &[
        ("Wo ist der Bahnhof?", "Where's the station?"),
        (
            "Guten Morgen, wie geht es dir?",
            "Good morning, how are you?",
        ),
        (
            "Das Wartungsteam hat die verschlissene Dichtung an der zweiten Pumpe ersetzt.",
            "The maintenance team has replaced the worn seal on the second pump.",
        ),
        (
            "Bitte starten Sie die Anwendung nach der Installation des Updates neu.",
            "Please restart the application after installing the update.",
        ),
        (
            "Meine Großmutter ist in einem kleinen Dorf am Meer aufgewachsen.",
            "My grandmother grew up in a small village by the sea.",
        ),
        (
            "Wenn der Akku leer ist, speichert das Gerät Ihre Arbeit automatisch.",
            "When the battery is empty, the device stores your work automatically.",
        ),
        (
            "Wir können die Lieferung vor Freitag nicht garantieren.",
            "We can't guarantee delivery before Friday.",
        ),
        (
            "Übersetzungsmodelle laufen lokal auf Ihrem eigenen Computer.",
            "Translation models run locally on your own computer.",
        ),
    ];

    async fn assert_matches_transformers(
        model: &str,
        from: &str,
        to: &str,
        pairs: &[(&str, &str)],
    ) {
        let translator = Translator::builder(model).quiet().build().await.unwrap();
        for (source, expected) in pairs {
            let translated = translator.translate(source, from, to).await.unwrap();
            assert_eq!(translated.trim(), *expected, "{source}");
        }
    }

    #[tokio::test]
    #[ignore = "needs opus-mt-en-de in the local model cache"]
    async fn en_de_matches_transformers() {
        assert_matches_transformers("opus-mt-en-de", "en", "de", EN_DE).await;
    }

    #[tokio::test]
    #[ignore = "needs opus-mt-de-en in the local model cache"]
    async fn de_en_matches_transformers() {
        assert_matches_transformers("opus-mt-de-en", "de", "en", DE_EN).await;
    }

    async fn assert_gpu_matches_transformers(
        model: &str,
        from: &str,
        to: &str,
        pairs: &[(&str, &str)],
    ) {
        let translator = Translator::builder(model)
            .gpu()
            .quiet()
            .build()
            .await
            .unwrap();
        for (source, expected) in pairs {
            let translated = translator.translate(source, from, to).await.unwrap();
            assert_eq!(translated.trim(), *expected, "{source}");
        }
    }

    /// The sentences together, in one text, come out as they do one at a time.
    #[tokio::test]
    #[ignore = "needs opus-mt-en-de in the local model cache"]
    async fn long_text_loses_no_sentences() {
        let translator = Translator::builder("opus-mt-en-de")
            .quiet()
            .build()
            .await
            .unwrap();
        let join = |column: fn(&(&'static str, &'static str)) -> &'static str, gap: &str| {
            let sentences: Vec<&str> = EN_DE.iter().chain(EN_DE).chain(EN_DE).map(column).collect();
            sentences.join(gap)
        };

        let translated = translator
            .translate(&join(|p| p.0, " "), "en", "de")
            .await
            .unwrap();
        assert_eq!(translated, join(|p| p.1, " "));

        let translated = translator
            .translate(&join(|p| p.0, "\n\n"), "en", "de")
            .await
            .unwrap();
        assert_eq!(translated, join(|p| p.1, "\n\n"));
    }

    /// Streaming produces the same text as `translate`.
    #[tokio::test]
    #[ignore = "needs opus-mt-en-de in the local model cache"]
    async fn long_text_streams_every_sentence() {
        use futures::StreamExt;

        let translator = Translator::builder("opus-mt-en-de")
            .quiet()
            .build()
            .await
            .unwrap();
        let source: Vec<&str> = EN_DE.iter().map(|p| p.0).collect();
        let expected: Vec<&str> = EN_DE.iter().map(|p| p.1).collect();

        let mut stream = translator
            .stream(&source.join("\n"), "en", "de")
            .await
            .unwrap();
        let mut streamed = String::new();
        while let Some(token) = stream.next().await {
            streamed.push_str(&token.unwrap());
        }
        assert_eq!(streamed, expected.join("\n"));
    }

    #[tokio::test]
    #[ignore = "needs a GPU and opus-mt-en-de in the local model cache"]
    async fn en_de_on_the_gpu_matches_transformers() {
        assert_gpu_matches_transformers("opus-mt-en-de", "en", "de", EN_DE).await;
    }

    #[tokio::test]
    #[ignore = "needs a GPU and opus-mt-de-en in the local model cache"]
    async fn de_en_on_the_gpu_matches_transformers() {
        assert_gpu_matches_transformers("opus-mt-de-en", "de", "en", DE_EN).await;
    }

    #[test]
    fn opus_mt_gets_the_bare_text() {
        let input = translation_input(ModelType::OpusMtEnDe, "Hello", "English", "German").unwrap();
        assert_eq!(input, "Hello");
    }

    #[test]
    fn opus_mt_refuses_the_other_direction() {
        let err = translation_input(ModelType::OpusMtEnDe, "Hallo", "German", "English");
        assert!(matches!(
            err,
            Err(TranslatorError::IncompatibleModel { .. })
        ));
    }

    #[test]
    fn flan_t5_keeps_its_prompt() {
        let input = translation_input(ModelType::FlanT5Base, "Hello", "English", "German").unwrap();
        assert_eq!(input, "translate English to German: Hello");
    }
}
