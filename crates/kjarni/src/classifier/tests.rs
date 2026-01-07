//! Tests for the classifier module.
//!
//! Run all tests: `cargo test --package kjarni classifier`
//! Run integration tests (requires model): `cargo test --package kjarni classifier -- --ignored`

use super::*;
use crate::common::{DownloadPolicy, KjarniDevice, LoadConfig};

// =============================================================================
// Type Tests
// =============================================================================

mod types_tests {
    use super::*;

    #[test]
    fn test_classification_result_from_scores() {
        let scores = vec![
            ("positive".to_string(), 0.8),
            ("negative".to_string(), 0.15),
            ("neutral".to_string(), 0.05),
        ];

        let result = ClassificationResult::from_scores(scores).unwrap();

        assert_eq!(result.label, "positive");
        assert!((result.score - 0.8).abs() < 0.001);
        assert_eq!(result.all_scores.len(), 3);
        // Should be sorted by score descending
        assert_eq!(result.all_scores[0].0, "positive");
        assert_eq!(result.all_scores[1].0, "negative");
        assert_eq!(result.all_scores[2].0, "neutral");
    }

    #[test]
    fn test_classification_result_from_empty_scores() {
        let scores: Vec<(String, f32)> = vec![];
        let result = ClassificationResult::from_scores(scores);
        assert!(result.is_none());
    }

    #[test]
    fn test_classification_result_top_k() {
        let scores = vec![
            ("a".to_string(), 0.5),
            ("b".to_string(), 0.3),
            ("c".to_string(), 0.2),
        ];

        let result = ClassificationResult::from_scores(scores).unwrap();
        let top_2 = result.top_k(2);

        assert_eq!(top_2.len(), 2);
        assert_eq!(top_2[0].0, "a");
        assert_eq!(top_2[1].0, "b");
    }

    #[test]
    fn test_classification_result_is_confident() {
        let scores = vec![("positive".to_string(), 0.85)];
        let result = ClassificationResult::from_scores(scores).unwrap();

        assert!(result.is_confident(0.8));
        assert!(result.is_confident(0.85));
        assert!(!result.is_confident(0.9));
    }

    #[test]
    fn test_classification_result_display() {
        let scores = vec![("positive".to_string(), 0.85)];
        let result = ClassificationResult::from_scores(scores).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("positive"));
        assert!(display.contains("85"));
    }

    #[test]
    fn test_classification_overrides_default() {
        let overrides = ClassificationOverrides::default();

        assert!(overrides.top_k.is_none());
        assert!(overrides.threshold.is_none());
        assert!(!overrides.return_logits);
    }

    #[test]
    fn test_classification_overrides_top_1() {
        let overrides = ClassificationOverrides::top_1();

        assert_eq!(overrides.top_k, Some(1));
    }

    #[test]
    fn test_classification_overrides_with_threshold() {
        let overrides = ClassificationOverrides::with_threshold(0.5);

        assert_eq!(overrides.threshold, Some(0.5));
    }
}

// =============================================================================
// Builder Tests
// =============================================================================

mod builder_tests {
    use super::*;

    #[test]
    fn test_builder_default_values() {
        let builder = ClassifierBuilder::new("test-model");

        assert_eq!(builder.model, "test-model");
        assert!(builder.model_path.is_none());
        assert_eq!(builder.device, KjarniDevice::Cpu);
        assert!(builder.context.is_none());
        assert!(builder.cache_dir.is_none());
        assert!(builder.load_config.is_none());
        assert_eq!(builder.download_policy, DownloadPolicy::IfMissing);
        assert!(!builder.quiet);
    }

    #[test]
    fn test_builder_cpu() {
        let builder = ClassifierBuilder::new("test-model").cpu();
        assert_eq!(builder.device, KjarniDevice::Cpu);
    }

    #[test]
    fn test_builder_gpu() {
        let builder = ClassifierBuilder::new("test-model").gpu();
        assert_eq!(builder.device, KjarniDevice::Gpu);
    }

    #[test]
    fn test_builder_auto_device() {
        let builder = ClassifierBuilder::new("test-model").auto_device();
        assert_eq!(builder.device, KjarniDevice::Auto);
    }

    #[test]
    fn test_builder_cache_dir() {
        let builder = ClassifierBuilder::new("test-model")
            .cache_dir("/tmp/test-cache");

        assert_eq!(
            builder.cache_dir,
            Some(std::path::PathBuf::from("/tmp/test-cache"))
        );
    }

    #[test]
    fn test_builder_model_path() {
        let builder = ClassifierBuilder::new("test-model")
            .model_path("/path/to/model");

        assert_eq!(
            builder.model_path,
            Some(std::path::PathBuf::from("/path/to/model"))
        );
    }

    #[test]
    fn test_builder_top_k() {
        let builder = ClassifierBuilder::new("test-model").top_k(5);
        assert_eq!(builder.overrides.top_k, Some(5));
    }

    #[test]
    fn test_builder_threshold() {
        let builder = ClassifierBuilder::new("test-model").threshold(0.5);
        assert_eq!(builder.overrides.threshold, Some(0.5));
    }

    #[test]
    fn test_builder_return_logits() {
        let builder = ClassifierBuilder::new("test-model").return_logits(true);
        assert!(builder.overrides.return_logits);
    }

    #[test]
    fn test_builder_offline() {
        let builder = ClassifierBuilder::new("test-model").offline();
        assert_eq!(builder.download_policy, DownloadPolicy::Never);
    }

    #[test]
    fn test_builder_quiet() {
        let builder = ClassifierBuilder::new("test-model").quiet(true);
        assert!(builder.quiet);
    }

    #[test]
    fn test_builder_chain() {
        let builder = ClassifierBuilder::new("test-model")
            .gpu()
            .top_k(3)
            .threshold(0.1)
            .quiet(true)
            .offline();

        assert_eq!(builder.device, KjarniDevice::Gpu);
        assert_eq!(builder.overrides.top_k, Some(3));
        assert_eq!(builder.overrides.threshold, Some(0.1));
        assert!(builder.quiet);
        assert_eq!(builder.download_policy, DownloadPolicy::Never);
    }

    #[test]
    fn test_builder_from_preset() {
        let builder = ClassifierBuilder::from_preset(&presets::SENTIMENT_V1);

        assert_eq!(builder.model, presets::SENTIMENT_V1.model);
        assert_eq!(builder.device, presets::SENTIMENT_V1.recommended_device);
    }

    #[test]
    fn test_builder_with_load_config() {
        let builder = ClassifierBuilder::new("test-model")
            .with_load_config(|b| b.offload_embeddings(true).max_batch_size(32));

        assert!(builder.load_config.is_some());
        let config = builder.load_config.unwrap();
        assert!(config.inner.offload_embeddings);
        assert_eq!(config.inner.max_batch_size, Some(32));
    }
}

// =============================================================================
// Preset Tests
// =============================================================================

mod preset_tests {
    use super::*;

    #[test]
    fn test_sentiment_v1_preset() {
        let preset = &presets::SENTIMENT_V1;

        assert_eq!(preset.name, "SENTIMENT_V1");
        assert!(!preset.model.is_empty());
        assert_eq!(preset.task, presets::ClassificationTask::Sentiment);
    }

    #[test]
    fn test_find_preset_exists() {
        let preset = presets::find_preset("SENTIMENT_V1");
        assert!(preset.is_some());
        assert_eq!(preset.unwrap().name, "SENTIMENT_V1");
    }

    #[test]
    fn test_find_preset_case_insensitive() {
        let preset = presets::find_preset("sentiment_v1");
        assert!(preset.is_some());
    }

    #[test]
    fn test_find_preset_not_found() {
        let preset = presets::find_preset("NONEXISTENT");
        assert!(preset.is_none());
    }

    #[test]
    fn test_classifier_tier_resolve() {
        let fast = presets::ClassifierTier::Fast.resolve();
        let balanced = presets::ClassifierTier::Balanced.resolve();
        let accurate = presets::ClassifierTier::Accurate.resolve();

        // All should resolve to valid presets
        assert!(!fast.model.is_empty());
        assert!(!balanced.model.is_empty());
        assert!(!accurate.model.is_empty());
    }

    #[test]
    fn test_all_presets_valid() {
        for preset in presets::ALL_V1_PRESETS {
            assert!(!preset.name.is_empty());
            assert!(!preset.model.is_empty());
            assert!(preset.memory_mb > 0);
            assert!(!preset.description.is_empty());
        }
    }
}

// =============================================================================
// Validation Tests
// =============================================================================

mod validation_tests {
    use super::*;
    use kjarni_transformers::models::ModelType;

    #[test]
    fn test_validate_classifier_model() {
        // Cross-encoder should be valid
        if let Some(model_type) = ModelType::from_cli_name("minilm-l6-v2-cross-encoder") {
            let result = validation::validate_for_classification(model_type);
            assert!(result.is_ok(), "Cross-encoder should be valid for classification");
        }
    }

    #[test]
    fn test_validate_non_classifier_model() {
        // Decoder models should fail
        if let Some(model_type) = ModelType::from_cli_name("llama3.2-1b") {
            let result = validation::validate_for_classification(model_type);
            assert!(result.is_err(), "Decoder should not be valid for classification");
        }
    }

    #[test]
    fn test_get_classifier_models() {
        let models = validation::get_classifier_models();
        // Should return some models (or empty if none configured)
        // At minimum, test that it doesn't panic
        assert!(models.len() >= 0);
    }
}

// =============================================================================
// Error Tests
// =============================================================================

mod error_tests {
    use super::*;

    #[test]
    fn test_error_display_unknown_model() {
        let err = ClassifierError::UnknownModel("fake-model".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("fake-model"));
        assert!(msg.contains("Unknown model"));
    }

    #[test]
    fn test_error_display_incompatible_model() {
        let err = ClassifierError::IncompatibleModel {
            model: "test-model".to_string(),
            reason: "not an encoder".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("test-model"));
        assert!(msg.contains("not an encoder"));
    }

    #[test]
    fn test_error_display_not_downloaded() {
        let err = ClassifierError::ModelNotDownloaded("test-model".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test-model"));
        assert!(msg.contains("not downloaded"));
    }

    #[test]
    fn test_error_display_no_labels() {
        let err = ClassifierError::NoLabels;
        let msg = format!("{}", err);
        assert!(msg.contains("label"));
    }
}

// =============================================================================
// Convenience Function Tests
// =============================================================================

mod convenience_tests {
    use super::*;

    #[test]
    fn test_is_classifier_model_unknown() {
        let result = is_classifier_model("nonexistent-model-12345");
        assert!(result.is_err());
    }

    #[test]
    fn test_available_models_returns_list() {
        let models = available_models();
        // Should not panic, may return empty or populated list
        assert!(models.len() >= 0);
    }
}

// =============================================================================
// Integration Tests (require model download)
// =============================================================================

mod integration_tests {
    use super::*;

    /// Test that we can create a classifier (requires model).
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_classifier_new() {
        let classifier = Classifier::new("minilm-l6-v2-cross-encoder").await;
        assert!(classifier.is_ok(), "Failed to create classifier: {:?}", classifier.err());
    }

    /// Test single text classification.
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_classify_single() {
        let classifier = Classifier::new("minilm-l6-v2-cross-encoder")
            .await
            .expect("Failed to load classifier");

        let result = classifier.classify("I love this product! It's amazing!")
            .await
            .expect("Classification failed");

        assert!(!result.label.is_empty());
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert!(!result.all_scores.is_empty());
    }

    /// Test batch classification.
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_classify_batch() {
        let classifier = Classifier::new("minilm-l6-v2-cross-encoder")
            .await
            .expect("Failed to load classifier");

        let texts = [
            "I love this!",
            "This is terrible.",
            "It's okay I guess.",
        ];

        let results = classifier.classify_batch(&texts)
            .await
            .expect("Batch classification failed");

        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(!result.label.is_empty());
            assert!(result.score >= 0.0 && result.score <= 1.0);
        }
    }

    /// Test classification with custom overrides.
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_classify_with_config() {
        let classifier = Classifier::new("minilm-l6-v2-cross-encoder")
            .await
            .expect("Failed to load classifier");

        let overrides = ClassificationOverrides {
            top_k: Some(2),
            threshold: Some(0.01),
            return_logits: false,
            ..Default::default()
        };

        let result = classifier
            .classify_with_config("Great product!", &overrides)
            .await
            .expect("Classification failed");

        // top_k should limit results (but we get all_scores which may have more before filtering)
        assert!(result.all_scores.len() <= 2 || result.all_scores.iter().all(|(_, s)| *s >= 0.01));
    }

    /// Test raw scores without labels.
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_classify_scores() {
        let classifier = Classifier::new("minilm-l6-v2-cross-encoder")
            .await
            .expect("Failed to load classifier");

        let scores = classifier.classify_scores("Test text")
            .await
            .expect("Getting scores failed");

        assert!(!scores.is_empty());
        // Scores should be probabilities (after softmax)
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Scores should sum to ~1.0");
    }

    /// Test GPU classification (if available).
    #[tokio::test]
    #[ignore = "Requires GPU and model download"]
    async fn test_classify_gpu() {
        let classifier = Classifier::builder("minilm-l6-v2-cross-encoder")
            .gpu()
            .build()
            .await
            .expect("Failed to load classifier on GPU");

        let result = classifier.classify("Test GPU classification")
            .await
            .expect("GPU classification failed");

        assert!(!result.label.is_empty());
    }

    /// Test classifier accessors.
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_classifier_accessors() {
        let classifier = Classifier::new("minilm-l6-v2-cross-encoder")
            .await
            .expect("Failed to load classifier");

        assert!(!classifier.model_name().is_empty());
        assert!(classifier.max_seq_length() > 0);
        // Labels may or may not be present depending on model
    }

    /// Test one-liner classify function.
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_classify_convenience_function() {
        let result = classify("minilm-l6-v2-cross-encoder", "I love this!")
            .await
            .expect("Classify function failed");

        assert!(!result.label.is_empty());
    }

    /// Test unknown model error.
    #[tokio::test]
    async fn test_unknown_model_error() {
        let result = Classifier::new("completely-fake-model-that-does-not-exist").await;
        assert!(matches!(result, Err(ClassifierError::UnknownModel(_))));
    }

    /// Test offline mode with missing model.
    #[tokio::test]
    async fn test_offline_missing_model() {
        let result = Classifier::builder("minilm-l6-v2-cross-encoder")
            .offline()
            .cache_dir("/tmp/kjarni-test-empty-cache-12345")
            .build()
            .await;

        // Should fail because model not downloaded and offline mode
        assert!(result.is_err());
    }
}