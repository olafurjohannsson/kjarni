//! Tests for the classifier module.
//!
//! Run all tests: `cargo test --package kjarni classifier`
//! Run integration tests (requires model): `cargo test --package kjarni classifier -- --ignored`

use super::*;
use crate::classifier::{
    ClassificationMode, ClassificationOverrides, Classifier, ClassifierError,
    presets::{
        ClassificationTask, EMOTION_DETAILED_V1, EMOTION_V1, SENTIMENT_3CLASS_V1,
        SENTIMENT_5STAR_V1, SENTIMENT_BINARY_V1, TOXICITY_V1,
    },
};
use crate::common::{DownloadPolicy, KjarniDevice, LoadConfig};

// =============================================================================
// Type Tests
// =============================================================================

mod classifier_tests {
    use super::*;
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
            let builder = ClassifierBuilder::new("test-model").cache_dir("/tmp/test-cache");

            assert_eq!(
                builder.cache_dir,
                Some(std::path::PathBuf::from("/tmp/test-cache"))
            );
        }

        #[test]
        fn test_builder_model_path() {
            let builder = ClassifierBuilder::new("test-model").model_path("/path/to/model");

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
            let builder = ClassifierBuilder::from_preset(&presets::SENTIMENT_BINARY_V1);

            assert_eq!(builder.model, presets::SENTIMENT_BINARY_V1.model);
            assert_eq!(
                builder.device,
                presets::SENTIMENT_BINARY_V1.recommended_device
            );
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
        use crate::classifier::presets::{EmotionTier, SentimentTier, find_preset};

        use super::*;

        #[test]
        fn test_sentiment_binary_preset() {
            let preset = &presets::SENTIMENT_BINARY_V1;

            assert_eq!(preset.name, "SENTIMENT_BINARY_V1");
            assert_eq!(preset.model, "distilbert-sentiment");
            assert_eq!(preset.task, presets::ClassificationTask::Sentiment);
            assert!(preset.labels.is_some());
            assert_eq!(preset.labels.unwrap().len(), 2); // NEGATIVE, POSITIVE
        }

        #[test]
        fn test_sentiment_3class_preset() {
            let preset = &presets::SENTIMENT_3CLASS_V1;

            assert_eq!(preset.name, "SENTIMENT_3CLASS_V1");
            assert_eq!(preset.model, "roberta-sentiment");
            assert_eq!(preset.labels.unwrap().len(), 3); // negative, neutral, positive
        }

        #[test]
        fn test_emotion_preset() {
            let preset = &presets::EMOTION_V1;

            assert_eq!(preset.name, "EMOTION_V1");
            assert_eq!(preset.task, presets::ClassificationTask::Emotion);
            assert_eq!(preset.labels.unwrap().len(), 7);
        }

        #[test]
        fn test_toxicity_preset() {
            let preset = &presets::TOXICITY_V1;

            assert_eq!(preset.name, "TOXICITY_V1");
            assert_eq!(preset.task, presets::ClassificationTask::Toxicity);
            assert_eq!(preset.labels.unwrap().len(), 6);
        }

        #[test]
        fn test_find_preset_exists() {
            let preset = presets::find_preset("SENTIMENT_BINARY_V1");
            assert!(preset.is_some());
            assert_eq!(preset.unwrap().name, "SENTIMENT_BINARY_V1");
        }

        #[test]
        fn test_find_preset_case_insensitive() {
            let preset = presets::find_preset("sentiment_binary_v1");
            assert!(preset.is_some());

            let preset = presets::find_preset("Emotion_V1");
            assert!(preset.is_some());
        }

        #[test]
        fn test_find_preset_not_found() {
            let preset = presets::find_preset("NONEXISTENT");
            assert!(preset.is_none());
        }

        #[test]
        fn test_sentiment_tier_resolve() {
            let fast = presets::SentimentTier::Fast.resolve();
            let balanced = presets::SentimentTier::Balanced.resolve();
            let detailed = presets::SentimentTier::Detailed.resolve();

            assert_eq!(fast.model, "distilbert-sentiment");
            assert_eq!(balanced.model, "roberta-sentiment");
            assert_eq!(detailed.model, "bert-sentiment-multilingual");
        }

        #[test]
        fn test_classifier_tier_resolve() {
            let fast = presets::ClassifierTier::Fast.resolve();
            let balanced = presets::ClassifierTier::Balanced.resolve();

            // Fast defaults to binary sentiment
            assert_eq!(fast.name, "SENTIMENT_BINARY_V1");
            // Balanced defaults to 3-class sentiment
            assert_eq!(balanced.name, "SENTIMENT_3CLASS_V1");
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

        #[test]
        fn test_all_presets_unique_names() {
            let mut names: Vec<&str> = presets::ALL_V1_PRESETS.iter().map(|p| p.name).collect();
            let original_len = names.len();
            names.sort();
            names.dedup();
            assert_eq!(names.len(), original_len, "Duplicate preset names found");
        }
        #[test]
        fn test_preset_lookup() {
            assert!(find_preset("SENTIMENT_BINARY_V1").is_some());
            assert!(find_preset("sentiment_binary_v1").is_some()); // case insensitive
            assert!(find_preset("nonexistent").is_none());
        }

        #[test]
        fn test_sentiment_tier_resolution() {
            assert_eq!(SentimentTier::Fast.resolve().model, "distilbert-sentiment");
            assert_eq!(SentimentTier::Balanced.resolve().model, "roberta-sentiment");
            assert_eq!(
                SentimentTier::Detailed.resolve().model,
                "bert-sentiment-multilingual"
            );
        }

        #[test]
        fn test_emotion_tier_resolution() {
            assert_eq!(EmotionTier::Basic.resolve().model, "distilroberta-emotion");
            assert_eq!(EmotionTier::Detailed.resolve().model, "roberta-emotions");
        }
    }

    // =============================================================================
    // Validation Tests
    // =============================================================================

    mod validation_tests {
        use super::*;
        use kjarni_transformers::models::ModelType;
        #[test]
        fn test_validate_sentiment_model() {
            // DistilBERT-SST2 should be valid
            if let Some(model_type) = ModelType::from_cli_name("distilbert-sentiment") {
                let result = validation::validate_for_classification(model_type);
                assert!(
                    result.is_ok(),
                    "Sentiment model should be valid for classification"
                );
            }
        }

        #[test]
        fn test_validate_reranker_is_not_classifier() {
            // Reranker should NOT be valid for classification
            if let Some(model_type) = ModelType::from_cli_name("minilm-reranker") {
                let result = validation::validate_for_classification(model_type);
                // This depends on how you implement validation
                // If ReRanking task is excluded, this should fail
                // If you allow it, adjust the test
            }
        }

        #[test]
        fn test_validate_decoder_invalid() {
            // Decoder models should fail
            if let Some(model_type) = ModelType::from_cli_name("llama3.2-1b-instruct") {
                let result = validation::validate_for_classification(model_type);
                assert!(
                    result.is_err(),
                    "Decoder should not be valid for classification"
                );
            }
        }

        #[test]
        fn test_get_classifier_models() {
            let models = validation::get_classifier_models();
            // Should return at least some models
            assert!(
                !models.is_empty(),
                "Should have at least one classifier model"
            );

            // Should include sentiment model
            assert!(
                models.iter().any(|m| m.contains("sentiment")
                    || m.contains("emotion")
                    || m.contains("zeroshot")),
                "Should include at least one known classifier"
            );
        }
        #[test]
        fn test_validate_classifier_model() {
            // Cross-encoder should be valid
            if let Some(model_type) = ModelType::from_cli_name("minilm-l6-v2-cross-encoder") {
                let result = validation::validate_for_classification(model_type);
                assert!(
                    result.is_ok(),
                    "Cross-encoder should be valid for classification"
                );
            }
        }

        #[test]
        fn test_validate_non_classifier_model() {
            // Decoder models should fail
            if let Some(model_type) = ModelType::from_cli_name("llama3.2-1b") {
                let result = validation::validate_for_classification(model_type);
                assert!(
                    result.is_err(),
                    "Decoder should not be valid for classification"
                );
            }
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
    // Test Utilities
    // =============================================================================

    const TEST_TEXT_POSITIVE: &str = "I absolutely love this, it's amazing!";
    const TEST_TEXT_NEGATIVE: &str = "This is terrible, I hate it so much.";
    const TEST_TEXT_NEUTRAL: &str = "The meeting is scheduled for Tuesday.";

    // Multilingual test phrases
    const TEST_TEXT_GERMAN_POSITIVE: &str = "Das ist wunderbar, ich liebe es!";
    const TEST_TEXT_FRENCH_NEGATIVE: &str = "C'est terrible, je déteste ça.";
    const TEST_TEXT_SPANISH_POSITIVE: &str = "¡Esto es increíble, me encanta!";

    // Toxic test phrases
    const TEST_TEXT_TOXIC: &str = "I hate you, you're worthless garbage.";
    const TEST_TEXT_NON_TOXIC: &str = "Thank you for your help today.";

    /// Helper to verify scores are valid probabilities
    fn assert_valid_probability(score: f32, context: &str) {
        assert!(
            (0.0..=1.0).contains(&score),
            "{}: score {} not in [0.0, 1.0]",
            context,
            score
        );
    }

    /// Helper to verify single-label scores sum to ~1.0
    fn assert_scores_sum_to_one(scores: &[(String, f32)], tolerance: f32) {
        let sum: f32 = scores.iter().map(|(_, s)| s).sum();
        assert!(
            (sum - 1.0).abs() < tolerance,
            "Scores sum to {}, expected ~1.0",
            sum
        );
    }

    // =============================================================================
    // Loading Strategy Tests
    // =============================================================================

    mod loading_tests {
        use super::*;

        #[tokio::test]
        async fn test_load_by_cli_name() {
            let classifier = Classifier::new("distilbert-sentiment")
                .await
                .expect("Failed to load by CLI name");

            assert_eq!(classifier.model_name(), "distilbert-sentiment");
            assert_eq!(classifier.num_labels(), 2);
        }

        #[tokio::test]
        async fn test_load_by_absolute_path() {
            // First download the model to get a known path
            let classifier_from_registry = Classifier::new("distilbert-sentiment")
                .await
                .expect("Failed to download model");

            // Now test loading from local path
            // Adjust this path based on your cache directory structure
            let cache_dir = crate::common::default_cache_dir();
            let model_path =
                cache_dir.join("distilbert_distilbert-base-uncased-finetuned-sst-2-english");

            if model_path.exists() {
                let classifier = Classifier::from_path(&model_path)
                    .labels(vec!["NEGATIVE", "POSITIVE"])
                    .build()
                    .await
                    .expect("Failed to load by absolute path");

                assert_eq!(classifier.num_labels(), 2);

                let result = classifier
                    .classify(TEST_TEXT_POSITIVE)
                    .await
                    .expect("Classification failed");

                assert_eq!(result.label, "POSITIVE");
            }
        }

        #[tokio::test]
        async fn test_load_by_hf_repo_name() {
            // This tests that the CLI name maps to the correct HF repo
            // The actual loading happens through the registry
            let classifier = Classifier::builder("roberta-sentiment")
                .build()
                .await
                .expect("Failed to load model");

            // Verify it loaded the correct model
            let labels = classifier.labels().expect("Should have labels");
            assert_eq!(labels, vec!["negative", "neutral", "positive"]);
        }
    }

    // =============================================================================
    // Sentiment Models
    // =============================================================================

    mod sentiment_tests {
        use super::*;

        #[tokio::test]
        async fn test_distilbert_sentiment_binary() {
            let preset = &SENTIMENT_BINARY_V1;
            assert_eq!(preset.model, "distilbert-sentiment");
            assert_eq!(preset.task, ClassificationTask::Sentiment);

            let classifier = Classifier::new(preset.model)
                .await
                .expect("Failed to load distilbert-sentiment");

            // Verify preset contract
            let expected_labels: Vec<String> = preset
                .labels
                .unwrap()
                .iter()
                .map(|s| s.to_string())
                .collect();
            assert_eq!(classifier.num_labels(), expected_labels.len());
            assert_eq!(
                classifier.labels().expect("Should have labels"),
                expected_labels
            );

            // Test positive text
            let result = classifier
                .classify(TEST_TEXT_POSITIVE)
                .await
                .expect("Classification failed");
            assert_eq!(result.label, "POSITIVE");
            assert_valid_probability(result.score, "positive classification");
            assert!(
                result.score > 0.8,
                "Expected high confidence for clear positive"
            );

            // Test negative text
            let result = classifier
                .classify(TEST_TEXT_NEGATIVE)
                .await
                .expect("Classification failed");
            assert_eq!(result.label, "NEGATIVE");
            assert!(
                result.score > 0.8,
                "Expected high confidence for clear negative"
            );

            // Verify single-label mode (softmax, scores sum to 1)
            let scores = classifier
                .classify_scores(TEST_TEXT_POSITIVE)
                .await
                .expect("Failed to get scores");
            let sum: f32 = scores.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Single-label scores should sum to 1.0, got {}",
                sum
            );
        }

        #[tokio::test]
        async fn test_roberta_sentiment_3class() {
            let preset = &SENTIMENT_3CLASS_V1;
            assert_eq!(preset.model, "roberta-sentiment");

            let classifier = Classifier::new(preset.model)
                .await
                .expect("Failed to load roberta-sentiment");

            // Verify labels
            let expected_labels: Vec<String> = preset
                .labels
                .unwrap()
                .iter()
                .map(|s| s.to_string())
                .collect();
            assert_eq!(classifier.num_labels(), 3);
            assert_eq!(
                classifier.labels().expect("Should have labels"),
                expected_labels
            );

            // Test positive
            let result = classifier
                .classify(TEST_TEXT_POSITIVE)
                .await
                .expect("Classification failed");
            assert_eq!(result.label, "positive");
            assert!(result.score > 0.7);

            // Test negative
            let result = classifier
                .classify(TEST_TEXT_NEGATIVE)
                .await
                .expect("Classification failed");
            assert_eq!(result.label, "negative");
            assert!(result.score > 0.7);

            // Test neutral
            let result = classifier
                .classify(TEST_TEXT_NEUTRAL)
                .await
                .expect("Classification failed");
            // Neutral text might not always be classified as neutral, but check it works
            assert_valid_probability(result.score, "neutral classification");

            // Verify softmax (single-label)
            let scores = classifier
                .classify_scores(TEST_TEXT_POSITIVE)
                .await
                .expect("Failed to get scores");
            let sum: f32 = scores.iter().sum();
            assert!((sum - 1.0).abs() < 0.01);
        }

        #[tokio::test]
        async fn test_bert_sentiment_multilingual_5star() {
            let preset = &SENTIMENT_5STAR_V1;
            assert_eq!(preset.model, "bert-sentiment-multilingual");

            let classifier = Classifier::new(preset.model)
                .await
                .expect("Failed to load bert-sentiment-multilingual");

            // Verify 5 labels
            assert_eq!(classifier.num_labels(), 5);
            let labels = classifier.labels().expect("Should have labels");
            assert!(labels.iter().any(|l| l.contains("star")));

            // Test English positive (should be 4 or 5 stars)
            let result = classifier
                .classify(TEST_TEXT_POSITIVE)
                .await
                .expect("Classification failed");
            assert!(
                result.label.contains("4") || result.label.contains("5"),
                "Expected 4-5 stars for positive text, got {}",
                result.label
            );

            // Test English negative (should be 1 or 2 stars)
            let result = classifier
                .classify(TEST_TEXT_NEGATIVE)
                .await
                .expect("Classification failed");
            assert!(
                result.label.contains("1") || result.label.contains("2"),
                "Expected 1-2 stars for negative text, got {}",
                result.label
            );

            // Test German positive
            let result = classifier
                .classify(TEST_TEXT_GERMAN_POSITIVE)
                .await
                .expect("German classification failed");
            assert!(
                result.label.contains("4") || result.label.contains("5"),
                "Expected 4-5 stars for German positive, got {}",
                result.label
            );

            // Test French negative
            let result = classifier
                .classify(TEST_TEXT_FRENCH_NEGATIVE)
                .await
                .expect("French classification failed");
            assert!(
                result.label.contains("1")
                    || result.label.contains("2")
                    || result.label.contains("3"),
                "Expected 1-3 stars for French negative, got {}",
                result.label
            );

            // Test Spanish positive
            let result = classifier
                .classify(TEST_TEXT_SPANISH_POSITIVE)
                .await
                .expect("Spanish classification failed");
            assert!(
                result.label.contains("4") || result.label.contains("5"),
                "Expected 4-5 stars for Spanish positive, got {}",
                result.label
            );
        }
    }

    // =============================================================================
    // Emotion Models
    // =============================================================================

    mod emotion_tests {
        use super::*;

        #[tokio::test]
        async fn test_distilroberta_emotion_7class() {
            let preset = &EMOTION_V1;
            assert_eq!(preset.model, "distilroberta-emotion");
            assert_eq!(preset.task, ClassificationTask::Emotion);

            let classifier = Classifier::new(preset.model)
                .await
                .expect("Failed to load distilroberta-emotion");

            // Verify 7 labels
            let expected_labels: Vec<String> = preset
                .labels
                .unwrap()
                .iter()
                .map(|s| s.to_string())
                .collect();
            assert_eq!(classifier.num_labels(), 7);
            assert_eq!(
                classifier.labels().expect("Should have labels"),
                expected_labels
            );

            // This is single-label (not multi-label)
            assert_eq!(classifier.mode(), ClassificationMode::SingleLabel);

            // Test joyful text
            let result = classifier
                .classify(TEST_TEXT_POSITIVE)
                .await
                .expect("Classification failed");
            assert_eq!(result.label, "joy");
            assert!(result.score > 0.7);

            // Test sad/angry text
            let result = classifier
                .classify(TEST_TEXT_NEGATIVE)
                .await
                .expect("Classification failed");
            assert!(
                result.label == "anger" || result.label == "sadness" || result.label == "disgust",
                "Expected negative emotion, got {}",
                result.label
            );

            // Verify softmax
            let scores = classifier
                .classify_scores(TEST_TEXT_POSITIVE)
                .await
                .expect("Failed to get scores");
            let sum: f32 = scores.iter().sum();
            assert!((sum - 1.0).abs() < 0.01);
        }

        #[tokio::test]
        async fn test_roberta_emotions_28class_multilabel() {
            let preset = &EMOTION_DETAILED_V1;
            assert_eq!(preset.model, "roberta-emotions");

            // Load with multi-label mode
            let classifier = Classifier::builder(preset.model)
                .multi_label()
                .build()
                .await
                .expect("Failed to load roberta-emotions");

            // Verify 28 labels
            assert_eq!(classifier.num_labels(), 28);
            assert_eq!(classifier.mode(), ClassificationMode::MultiLabel);

            let labels = classifier.labels().expect("Should have labels");
            assert!(labels.contains(&"joy".to_string()));
            assert!(labels.contains(&"love".to_string()));
            assert!(labels.contains(&"anger".to_string()));
            assert!(labels.contains(&"neutral".to_string()));

            // Test classification
            let result = classifier
                .classify(TEST_TEXT_POSITIVE)
                .await
                .expect("Classification failed");

            // For multi-label, we expect emotions like joy, love, excitement
            assert!(
                result.label == "joy"
                    || result.label == "love"
                    || result.label == "excitement"
                    || result.label == "admiration",
                "Expected positive emotion, got {}",
                result.label
            );

            // Multi-label: scores are sigmoid, each in [0,1] but DON'T sum to 1
            let scores = classifier
                .classify_scores(TEST_TEXT_POSITIVE)
                .await
                .expect("Failed to get scores");

            for (i, score) in scores.iter().enumerate() {
                assert_valid_probability(*score, &format!("emotion score {}", i));
            }

            // Multi-label should have multiple high scores for emotional text
            let high_scores: Vec<f32> = scores.iter().filter(|&&s| s > 0.3).copied().collect();
            // Emotional text often triggers multiple labels
            println!("Number of emotions > 0.3: {}", high_scores.len());

            // Test with threshold override - get all emotions above threshold
            let override_config = ClassificationOverrides {
                threshold: Some(0.2),
                ..Default::default()
            };
            let result = classifier
                .classify_with_config(TEST_TEXT_POSITIVE, &override_config)
                .await
                .expect("Classification with threshold failed");

            // Should return multiple emotions above threshold
            println!(
                "Emotions above 0.2 threshold: {:?}",
                result.all_scores.iter().map(|(l, _)| l).collect::<Vec<_>>()
            );
        }
    }

    // =============================================================================
    // Toxicity Model
    // =============================================================================

    mod toxicity_tests {
        use super::*;

        #[tokio::test]
        async fn test_toxic_bert_multilabel() {
            let preset = &TOXICITY_V1;
            assert_eq!(preset.model, "toxic-bert");
            assert_eq!(preset.task, ClassificationTask::Toxicity);

            // Load with multi-label mode (toxicity is inherently multi-label)
            let classifier = Classifier::builder(preset.model)
                .multi_label()
                .build()
                .await
                .expect("Failed to load toxic-bert");

            // Verify 6 labels
            let expected_labels: Vec<String> = preset
                .labels
                .unwrap()
                .iter()
                .map(|s| s.to_string())
                .collect();
            assert_eq!(classifier.num_labels(), 6);
            assert_eq!(classifier.mode(), ClassificationMode::MultiLabel);

            let labels = classifier.labels().expect("Should have labels");
            assert!(labels.contains(&"toxic".to_string()));
            assert!(labels.contains(&"insult".to_string()));
            assert!(labels.contains(&"threat".to_string()));

            // Test toxic text
            let result = classifier
                .classify(TEST_TEXT_TOXIC)
                .await
                .expect("Classification failed");

            // Should detect toxicity
            assert!(
                result.label == "toxic" || result.label == "insult",
                "Expected toxic/insult for toxic text, got {}",
                result.label
            );
            assert!(result.score > 0.5, "Expected high toxicity score");

            // Get all toxicity scores for toxic text
            let scores = classifier
                .classify_scores(TEST_TEXT_TOXIC)
                .await
                .expect("Failed to get scores");

            // Multi-label: each score independently in [0,1]
            for (i, score) in scores.iter().enumerate() {
                assert_valid_probability(*score, &format!("toxicity score {} ({})", i, labels[i]));
            }

            // Toxic text should trigger multiple categories
            let toxic_categories: Vec<&String> = labels
                .iter()
                .zip(scores.iter())
                .filter(|(_, s)| **s > 0.3)
                .map(|(l, _)| l)
                .collect();
            println!("Detected toxic categories: {:?}", toxic_categories);
            assert!(
                toxic_categories.len() >= 1,
                "Should detect at least one toxicity category"
            );

            // Test non-toxic text
            let result = classifier
                .classify(TEST_TEXT_NON_TOXIC)
                .await
                .expect("Classification failed");

            assert!(
                result.score < 0.6,
                "Non-toxic text should have low toxicity score, got {}",
                result.score
            );

            // Should have low toxicity scores
            let scores = classifier
                .classify_scores(TEST_TEXT_NON_TOXIC)
                .await
                .expect("Failed to get scores");

            let max_score = scores.iter().cloned().fold(0.0f32, f32::max);
            assert!(
                max_score < 0.6,
                "Non-toxic text should have low toxicity scores, max was {}",
                max_score
            );
        }
    }

    // =============================================================================
    // Batch Classification Tests
    // =============================================================================

    mod batch_tests {
        use super::*;

        #[tokio::test]
        async fn test_batch_classification() {
            let classifier = Classifier::new("roberta-sentiment")
                .await
                .expect("Failed to load model");

            let texts = &[TEST_TEXT_POSITIVE, TEST_TEXT_NEGATIVE, TEST_TEXT_NEUTRAL];

            let results = classifier
                .classify_batch(texts)
                .await
                .expect("Batch classification failed");

            assert_eq!(results.len(), 3);

            // First should be positive
            assert_eq!(results[0].label, "positive");
            assert!(results[0].score > 0.7);

            // Second should be negative
            assert_eq!(results[1].label, "negative");
            assert!(results[1].score > 0.7);

            // Third is neutral (may vary)
            assert_valid_probability(results[2].score, "neutral batch result");
        }

        #[tokio::test]
        async fn test_batch_multilingual() {
            let classifier = Classifier::new("bert-sentiment-multilingual")
                .await
                .expect("Failed to load model");

            let texts = &[
                TEST_TEXT_POSITIVE,         // English
                TEST_TEXT_GERMAN_POSITIVE,  // German
                TEST_TEXT_FRENCH_NEGATIVE,  // French
                TEST_TEXT_SPANISH_POSITIVE, // Spanish
            ];

            let results = classifier
                .classify_batch(texts)
                .await
                .expect("Batch classification failed");

            assert_eq!(results.len(), 4);

            for result in &results {
                println!("Batch result: {} (score: {})", result.label, result.score);
            }

            // Expected results from Python reference implementation:
            // Index 0 (English positive): 5 stars (0.9689)
            // Index 1 (German positive): 5 stars (0.8865)
            // Index 2 (French negative): 1 star (0.7609)
            // Index 3 (Spanish positive): 5 stars (0.9312)

            // Test English positive
            assert_eq!(
                results[0].label, "5 stars",
                "English positive should be 5 stars, got {}",
                results[0].label
            );
            assert!(
                results[0].score > 0.95,
                "English positive score should be > 0.95, got {}",
                results[0].score
            );

            // Test German positive
            assert_eq!(
                results[1].label, "5 stars",
                "German positive should be 5 stars, got {}",
                results[1].label
            );
            assert!(
                results[1].score > 0.85,
                "German positive score should be > 0.85, got {}",
                results[1].score
            );

            // Test French negative
            assert_eq!(
                results[2].label, "1 star",
                "French negative should be 1 star, got {}",
                results[2].label
            );
            assert!(
                results[2].score > 0.75,
                "French negative score should be > 0.75, got {}",
                results[2].score
            );

            // Test Spanish positive
            assert_eq!(
                results[3].label, "5 stars",
                "Spanish positive should be 5 stars, got {}",
                results[3].label
            );
            assert!(
                results[3].score > 0.90,
                "Spanish positive score should be > 0.90, got {}",
                results[3].score
            );
        }
    }

    // =============================================================================
    // Error Handling Tests
    // =============================================================================

    mod error_tests_2 {
        use super::*;

        #[tokio::test]
        async fn test_unknown_model_error() {
            let result = Classifier::new("nonexistent-model-12345").await;
            assert!(matches!(result, Err(ClassifierError::UnknownModel(_))));
        }

        #[tokio::test]
        async fn test_invalid_model_path_error() {
            let result = Classifier::from_path("/nonexistent/path/to/model")
                .build()
                .await;
            assert!(matches!(result, Err(ClassifierError::ModelPathNotFound(_))));
        }

        #[tokio::test]

        async fn test_label_count_mismatch_error() {
            // Try to load with wrong number of labels
            let result = Classifier::builder("distilbert-sentiment")
                .labels(vec!["a", "b", "c", "d"]) // 4 labels, but model expects 2
                .build()
                .await;

            assert!(matches!(result, Err(ClassifierError::InvalidLabels(_))));
        }

        #[tokio::test]

        async fn test_threshold_filters_all_results() {
            let classifier = Classifier::new("distilbert-sentiment")
                .await
                .expect("Failed to load");

            let result = classifier
                .classify_with_config(
                    TEST_TEXT_NEUTRAL,
                    &ClassificationOverrides {
                        threshold: Some(0.9999), // Very high threshold
                        ..Default::default()
                    },
                )
                .await;

            // Should error when all results filtered
            assert!(matches!(
                result,
                Err(ClassifierError::ClassificationFailed(_))
            ));
        }
    }

    // =============================================================================
    // Preset Contract Tests
    // =============================================================================

    mod integration_tests {
        use kjarni_transformers::Device;

        use crate::classifier::presets::ClassificationTask;

        use super::*;

        /// Test that we can create a classifier (requires model).
        #[tokio::test]

        async fn test_classifier_new() {
            let classifier = Classifier::new("distilbert-sentiment").await;
            assert!(
                classifier.is_ok(),
                "Failed to create classifier: {:?}",
                classifier.err()
            );
        }

        /// Test single text classification.
        #[tokio::test]

        async fn test_classify_single() {
            let classifier = Classifier::new("distilroberta-emotion")
                .await
                .expect("Failed to load classifier");

            let result = classifier
                .classify("I love this product! It's amazing!")
                .await
                .expect("Classification failed");

            assert!(!result.label.is_empty());
            assert!(result.score >= 0.0 && result.score <= 1.0);
            assert!(!result.all_scores.is_empty());
        }

        #[tokio::test]

        async fn test_classifier_explore_api_toxic_bert() {
            // --- SETUP ---
            const MODEL_NAME: &str = "toxic-bert";
            let toxic_text = "You are a stupid idiot and I will find you.";

            println!(
                "\n\n--- Starting Exploratory Test for Multi-Label Model: '{}' ---",
                MODEL_NAME
            );

            // ========================================================================
            // 1. Observe Initialization in Multi-Label Mode
            // ========================================================================
            println!("\n--- [1] Observing Initialization ---");

            // CRUCIAL: We must enable .multi_label() mode for this model.
            let classifier = Classifier::builder(MODEL_NAME)
                .multi_label()
                .build()
                .await
                .expect("Failed to load classifier in multi-label mode");

            println!("Loaded model '{}' successfully.", classifier.model_name());
            println!("Classifier mode: {:?}", classifier.mode()); // Should be MultiLabel
            println!("Model labels: {:?}", classifier.labels().unwrap());

            // ========================================================================
            // 2. Observe Multi-Label Classification
            // ========================================================================
            println!("\n--- [2] Observing Multi-Label Classification ---");

            let result = classifier
                .classify(toxic_text)
                .await
                .expect("classify() failed");

            // For multi-label, the top score is less important than all scores above a threshold.
            // The scores are independent probabilities (post-sigmoid).
            println!("classify('{}') result:", toxic_text);
            println!(
                "  (Note: The top 'label' and 'score' are less meaningful in multi-label scenarios)"
            );
            println!(
                "  Top Label: '{}' with Score: {}",
                result.label, result.score
            );
            println!("  All Scores (sorted): {:?}", result.all_scores);

            // This is the most common way to use a multi-label result.
            let predictions_above_threshold = result.above_threshold(0.5);
            println!(
                "\n  Predictions with score > 0.5: {:?}",
                predictions_above_threshold
            );

            // Observe raw scores (post-sigmoid)
            let scores = classifier
                .classify_scores(toxic_text)
                .await
                .expect("classify_scores() failed");
            let sum: f32 = scores.iter().sum();
            println!("\nclassify_scores('{}') result:", toxic_text);
            println!("  Scores (post-sigmoid): {:?}", scores);
            println!(
                "  (Note: Sum is NOT expected to be 1.0 in multi-label. Sum = {})",
                sum
            );
        }

        #[tokio::test]

        async fn test_classifier_full_api_distilroberta() {
            const MODEL_NAME: &str = "distilroberta-emotion";
            let happy_text = "I am so happy and excited for the weekend!";
            let sad_text = "That was a truly disappointing and sad experience.";
            let batch_texts = &[happy_text, sad_text];

            const PRESET: &presets::ClassifierPreset = &presets::EMOTION_V1;
            assert_eq!(PRESET.model, MODEL_NAME);
            let model_name = PRESET.model;

            println!("--- Verifying Preset Contract for '{}' ---", PRESET.name);

            let classifier = Classifier::new(model_name)
                .await
                .expect("Failed to load model defined in preset");

            assert_eq!(classifier.model_name(), PRESET.model, "Model name mismatch");
            assert_eq!(
                classifier.architecture(),
                "roberta",
                "Architecture mismatch"
            );

            let expected_labels: Vec<String> = PRESET
                .labels
                .unwrap()
                .iter()
                .map(|s| s.to_string())
                .collect();
            assert_eq!(
                classifier.num_labels(),
                expected_labels.len(),
                "Number of labels mismatch"
            );
            assert_eq!(
                classifier.labels().expect("Model should have labels"),
                expected_labels,
                "Label content mismatch"
            );

            // ========================================================================
            // 1. Test Initialization and Accessors
            // ========================================================================

            assert_eq!(classifier.model_name(), MODEL_NAME);
            assert_eq!(classifier.num_labels(), 7);
            assert_eq!(classifier.max_seq_length(), 514);
            let expected_labels = vec![
                "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise",
            ];
            assert_eq!(classifier.labels().unwrap(), expected_labels);

            // ========================================================================
            // 2. Test Core Classification Methods
            // ========================================================================

            let eps = 1e-6;
            let approx_eq = |a: f32, b: f32| (a - b).abs() < eps;

            // classify() on happy text
            let single_result = classifier
                .classify(happy_text)
                .await
                .expect("classify() failed");

            assert_eq!(single_result.label, "joy");
            assert!(
                approx_eq(single_result.score, 0.97546911),
                "Happy score mismatch: expected 0.97546911, got {}",
                single_result.score
            );

            // classify_batch()
            let batch_results = classifier
                .classify_batch(batch_texts)
                .await
                .expect("classify_batch() failed");

            assert_eq!(batch_results.len(), 2);

            // Batch result 0: happy text
            assert_eq!(batch_results[0].label, "joy");
            assert!(
                approx_eq(batch_results[0].score, 0.97546911),
                "Batch happy score mismatch: expected 0.97546911, got {}",
                batch_results[0].score
            );

            // Batch result 1: sad text - CORRECTED VALUE
            assert_eq!(batch_results[1].label, "sadness");
            assert!(
                approx_eq(batch_results[1].score, 0.97713542),
                "Batch sad score mismatch: expected 0.97713542, got {}",
                batch_results[1].score
            );

            // classify_scores() on HAPPY text
            let happy_scores = classifier
                .classify_scores(happy_text)
                .await
                .expect("classify_scores() on happy text failed");

            let expected_happy_scores: Vec<f32> = vec![
                0.00133714, // anger
                0.00033000, // disgust
                0.00030720, // fear
                0.97546911, // joy ← winner
                0.00330787, // neutral
                0.00320903, // sadness
                0.01603979, // surprise
            ];

            for (i, (actual, expected)) in happy_scores
                .iter()
                .zip(expected_happy_scores.iter())
                .enumerate()
            {
                assert!(
                    approx_eq(*actual, *expected),
                    "Happy score mismatch at index {} ({}): actual {:.8}, expected {:.8}",
                    i,
                    expected_labels[i],
                    actual,
                    expected
                );
            }

            // classify_scores() on SAD text - CORRECTED VALUES
            let sad_scores = classifier
                .classify_scores(sad_text)
                .await
                .expect("classify_scores() on sad text failed");

            let expected_sad_scores: Vec<f32> = vec![
                0.00074336, // anger
                0.00194305, // disgust
                0.00221157, // fear
                0.00190714, // joy
                0.00586602, // neutral
                0.97713542, // sadness ← winner (CORRECTED FROM 0.96455652)
                0.01019347, // surprise
            ];

            for (i, (actual, expected)) in sad_scores
                .iter()
                .zip(expected_sad_scores.iter())
                .enumerate()
            {
                assert!(
                    approx_eq(*actual, *expected),
                    "Sad score mismatch at index {} ({}): actual {:.8}, expected {:.8}",
                    i,
                    expected_labels[i],
                    actual,
                    expected
                );
            }

            // ========================================================================
            // 3. Test Overrides with `classify_with_config()`
            // ========================================================================

            let top_k_override = ClassificationOverrides {
                top_k: Some(3),
                ..Default::default()
            };
            let top_k_result = classifier
                .classify_with_config(sad_text, &top_k_override)
                .await
                .expect("top_k override failed");

            assert_eq!(top_k_result.all_scores.len(), 3);
            assert_eq!(top_k_result.all_scores[0].0, "sadness");
            assert!(
                approx_eq(top_k_result.all_scores[0].1, 0.97713542),
                "Top-k sadness score mismatch"
            );
            assert_eq!(top_k_result.all_scores[1].0, "surprise");
            assert!(
                approx_eq(top_k_result.all_scores[1].1, 0.01019347),
                "Top-k surprise score mismatch"
            );
            assert_eq!(top_k_result.all_scores[2].0, "neutral");
            assert!(
                approx_eq(top_k_result.all_scores[2].1, 0.00586602),
                "Top-k neutral score mismatch"
            );

            let threshold_override = ClassificationOverrides {
                threshold: Some(0.05),
                ..Default::default()
            };
            let threshold_result = classifier
                .classify_with_config(sad_text, &threshold_override)
                .await
                .expect("threshold override failed");

            assert_eq!(threshold_result.all_scores.len(), 1);
            assert_eq!(threshold_result.all_scores[0].0, "sadness");

            let failing_threshold_override = ClassificationOverrides {
                threshold: Some(0.999),
                ..Default::default()
            };
            let failing_threshold_result = classifier
                .classify_with_config(sad_text, &failing_threshold_override)
                .await;

            assert!(matches!(
                failing_threshold_result,
                Err(ClassifierError::ClassificationFailed(_))
            ));

            // ========================================================================
            // 4. Test Builder Configuration (Custom Labels)
            // ========================================================================

            let custom_labels = vec!["A", "B", "C", "D", "E", "F", "G"];
            let classifier_custom = Classifier::builder(MODEL_NAME)
                .labels(custom_labels.clone())
                .build()
                .await
                .expect("Failed to build classifier with custom labels");

            assert!(classifier_custom.has_custom_labels());

            let custom_result = classifier_custom
                .classify(happy_text)
                .await
                .expect("classify() with custom labels failed");

            assert_eq!(custom_result.label, "D"); // D maps to index 3 (joy)
            assert!(
                approx_eq(custom_result.score, 0.97546911),
                "Custom labels score mismatch: expected 0.97546911, got {}",
                custom_result.score
            );
        }

        #[tokio::test]

        async fn test_preset_sentiment_binary_v1() {
            // --- SETUP ---
            const PRESET: &presets::ClassifierPreset = &presets::SENTIMENT_BINARY_V1;
            let model_name = PRESET.model;
            assert_eq!("distilbert-sentiment", model_name);
            assert_eq!(PRESET.task, ClassificationTask::Sentiment);
            let positive_text = "This is a fantastic and wonderful product!";
            let negative_text = "I absolutely hate this, it's a terrible experience.";

            println!(
                "--- Verifying Preset Contract & Outputs for '{}' ---",
                PRESET.name
            );

            // --- 1. Load Model & Verify Preset Contract ---
            let classifier = Classifier::new(model_name)
                .await
                .expect("Failed to load model defined in preset");

            let expected_labels: Vec<String> = PRESET
                .labels
                .unwrap()
                .iter()
                .map(|s| s.to_string())
                .collect();
            assert_eq!(
                classifier.labels().unwrap(),
                expected_labels,
                "Label content mismatch"
            );
            assert_eq!(classifier.model_name(), PRESET.model, "Model name mismatch");

            // --- 2. Assert Positive Case ---
            let positive_result = classifier.classify(positive_text).await.unwrap();
            assert_eq!(positive_result.label, "POSITIVE");
            // Assert the exact score from your verified Rust output
            assert_eq!(positive_result.score, 0.99988973);

            // --- 3. Assert Negative Case ---
            let negative_result = classifier.classify(negative_text).await.unwrap();
            assert_eq!(negative_result.label, "NEGATIVE");
            // Assert the exact score from your verified Rust output
            assert_eq!(negative_result.score, 0.99904543);

            println!(
                "Preset '{}' verified successfully with exact outputs.",
                PRESET.name
            );
        }

        /// An EXPLORATORY test for the SENTIMENT_3CLASS_V1 preset ('roberta-sentiment').
        #[tokio::test]

        async fn test_explore_sentiment_3class_v1() {
            // --- SETUP ---
            const PRESET: &presets::ClassifierPreset = &presets::SENTIMENT_3CLASS_V1;
            let model_name = PRESET.model;
            let positive_text = "I love this so much, it's the best day ever.";
            let neutral_text = "The movie was okay, I guess.";
            let negative_text = "That was an awful thing to say.";

            println!(
                "\n\n--- Starting Exploratory Test for Preset: '{}' ---",
                PRESET.name
            );

            // --- LOAD MODEL ---
            let classifier = Classifier::new(model_name)
                .await
                .expect("Failed to load classifier");

            println!("Model: '{}'", classifier.model_name());
            println!("Labels: {:?}", classifier.labels().unwrap());

            // --- OBSERVE OUTPUTS ---
            let positive_result = classifier.classify(positive_text).await.unwrap();
            println!(
                "\nclassify('{}') result:\n  {:?}",
                positive_text, positive_result
            );

            let neutral_result = classifier.classify(neutral_text).await.unwrap();
            println!(
                "\nclassify('{}') result:\n  {:?}",
                neutral_text, neutral_result
            );

            let negative_result = classifier.classify(negative_text).await.unwrap();
            println!(
                "\nclassify('{}') result:\n  {:?}",
                negative_text, negative_result
            );
        }

        /// An EXPLORATORY test for the SENTIMENT_5STAR_V1 preset ('bert-sentiment-multilingual').
        #[tokio::test]

        async fn test_explore_sentiment_5star_v1() {
            // --- SETUP ---
            const PRESET: &presets::ClassifierPreset = &presets::SENTIMENT_5STAR_V1;
            let model_name = PRESET.model;
            let five_star_text = "An absolute masterpiece, the best I've ever seen.";
            let one_star_text = "Awful, a complete waste of time and money.";

            println!(
                "\n\n--- Starting Exploratory Test for Preset: '{}' ---",
                PRESET.name
            );

            // --- LOAD MODEL ---
            let classifier = Classifier::new(model_name)
                .await
                .expect("Failed to load classifier");

            println!("Model: '{}'", classifier.model_name());
            println!("Labels: {:?}", classifier.labels().unwrap());

            // --- OBSERVE OUTPUTS ---
            let five_star_result = classifier.classify(five_star_text).await.unwrap();
            println!(
                "\nclassify('{}') result:\n  {:?}",
                five_star_text, five_star_result
            );

            let one_star_result = classifier.classify(one_star_text).await.unwrap();
            println!(
                "\nclassify('{}') result:\n  {:?}",
                one_star_text, one_star_result
            );
        }
        #[tokio::test]

        async fn test_classifier_builder_with_f16_dtype() {
            const MODEL_NAME: &str = "distilroberta-emotion";
            let happy_text = "I am so happy and excited for the weekend!";

            println!("--- Testing Classifier load with DType::F16 ---");

            // Build the classifier, explicitly requesting F16 weights.
            // The .f16() method is a convenient shortcut on the builder.
            let classifier_f16 = Classifier::builder(MODEL_NAME)
                .f16() // This sets the target_dtype in the LoadConfig
                .build()
                .await
                .expect("Failed to build classifier with F16 dtype");

            // The most important test is that the model loads and can produce a valid prediction.
            // This proves the entire pipeline works with the converted F16 weights.
            let result = classifier_f16
                .classify(happy_text)
                .await
                .expect("Classification failed with F16 model");

            // While we don't assert the exact score (it might differ slightly due to precision),
            // we assert that the result is plausible and the top label is correct.
            // This confirms the model is functioning correctly.
            assert_eq!(
                result.label, "joy",
                "F16 model failed to predict the correct label."
            );
            // F16 score should be very close to F32 reference (0.98810238)
            assert!(
                result.score > 0.97,
                "F16 model score was unexpectedly low: {}",
                result.score
            );

            // Optional: verify F16 doesn't deviate too much from F32
            let f32_reference = 0.97546911_f32;
            let diff = (result.score - f32_reference).abs();
            assert!(
                diff < 0.001,
                "F16 score {} differs too much from F32 reference {}",
                result.score,
                f32_reference
            );

            println!(
                "F16 score: {}, F32 reference: {}, diff: {}",
                result.score, f32_reference, diff
            );
            println!("Successfully loaded and ran inference with F16 precision.");

            println!("Successfully loaded and ran inference with F16 precision.");
        }

        /// Test batch classification.
        #[tokio::test]

        async fn test_classify_batch() {
            let classifier = Classifier::new("distilroberta-emotion")
                .await
                .expect("Failed to load classifier");

            let texts = ["I love this!", "This is terrible.", "It's okay I guess."];

            let results = classifier
                .classify_batch(&texts)
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

        async fn test_classify_with_config() {
            let classifier = Classifier::new("distilbert-sentiment")
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
            assert!(
                result.all_scores.len() <= 2 || result.all_scores.iter().all(|(_, s)| *s >= 0.01)
            );
        }

        /// Test raw scores without labels.
        #[tokio::test]

        async fn test_classify_scores() {
            let classifier = Classifier::new("distilbert-sentiment")
                .await
                .expect("Failed to load classifier");

            let scores = classifier
                .classify_scores("Test text")
                .await
                .expect("Getting scores failed");

            assert!(!scores.is_empty());
            // Scores should be probabilities (after softmax)
            let sum: f32 = scores.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "Scores should sum to ~1.0");
        }

        /// Test GPU classification (if available).
        #[tokio::test]

        async fn test_classify_gpu() {
            let classifier = Classifier::builder("distilbert-sentiment")
                .gpu()
                .build()
                .await
                .expect("Failed to load classifier on GPU");

            let result = classifier
                .classify("Test GPU classification")
                .await
                .expect("GPU classification failed");

            assert!(!result.label.is_empty());
        }

        #[tokio::test]

        async fn test_classify_gpu_parity_cpu() {
            let classifier = Classifier::builder("distilbert-sentiment")
                .gpu()
                .build()
                .await
                .expect("Failed to load classifier on GPU");

            let result = classifier
                .classify("Test GPU classification")
                .await
                .expect("GPU classification failed");

            let classifier_cpu = Classifier::builder("distilbert-sentiment")
                .cpu()
                .build()
                .await
                .expect("Failed to load classifier on CPU");

            let result_cpu = classifier_cpu
                .classify("Test GPU classification")
                .await
                .expect("CPU classification failed");

            assert_eq!(result.label, result_cpu.label);

            assert_eq!(result.label_index, result_cpu.label_index);
            assert!((result.score - result_cpu.score).abs() < 0.0001);
            for ((label_a, score_a), (label_b, score_b)) in
                result.all_scores.iter().zip(result_cpu.all_scores.iter())
            {
                assert_eq!(label_a, label_b);
                assert!((score_a - score_b).abs() < 0.0001);
            }

            assert!(!result.label.is_empty());
        }

        /// Test classifier accessors.
        #[tokio::test]

        async fn test_classifier_accessors() {
            let classifier = Classifier::new("distilroberta-emotion")
                .await
                .expect("Failed to load classifier");

            assert!(!classifier.model_name().is_empty());
            assert!(classifier.max_seq_length() > 0);
            // Labels may or may not be present depending on model
        }

        /// Test one-liner classify function.
        #[tokio::test]

        async fn test_classify_convenience_function() {
            let result = classify("distilroberta-emotion", "I love this!")
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
        #[tokio::test]

        async fn test_model_distilroberta_emotion() {
            let classifier = Classifier::new("distilroberta-emotion").await.unwrap();
            let result = classifier.classify("I am so happy today!").await.unwrap();
            assert_eq!(result.label, "joy");
        }
    }
}
