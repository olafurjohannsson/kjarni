//! Tests for the classifier module.
use super::*;
use crate::classifier::{
    ClassificationMode, ClassificationOverrides, Classifier, ClassifierError,
};
use crate::common::{DownloadPolicy, KjarniDevice};

// PyTorch Reference Constants

mod expected {
    pub const INPUT_POSITIVE: &str = "I absolutely love this, it's amazing!";
    pub const INPUT_NEGATIVE: &str = "This is terrible, I hate it so much.";
    pub const INPUT_NEUTRAL: &str = "The meeting is scheduled for Tuesday.";
    pub const INPUT_GERMAN_POSITIVE: &str = "Das ist wunderbar, ich liebe es!";
    pub const INPUT_FRENCH_NEGATIVE: &str = "C'est terrible, je déteste ça.";
    pub const INPUT_SPANISH_POSITIVE: &str = "¡Esto es increíble, me encanta!";
    pub const INPUT_TOXIC: &str = "I hate you, you're worthless garbage.";
    pub const INPUT_NON_TOXIC: &str = "Thank you for your help today.";
    pub const INPUT_HAPPY: &str = "I am so happy and excited for the weekend!";
    pub const INPUT_SAD: &str = "That was a truly disappointing and sad experience.";
    pub const DISTILBERT_SENTIMENT_POSITIVE_LABEL: &str = "POSITIVE";
    pub const DISTILBERT_SENTIMENT_POSITIVE_SCORE: f32 = 0.99987566;
    pub const DISTILBERT_SENTIMENT_POSITIVE_LABEL_INDEX: usize = 1;
    pub const DISTILBERT_SENTIMENT_NEGATIVE_LABEL: &str = "NEGATIVE";
    pub const DISTILBERT_SENTIMENT_NEGATIVE_SCORE: f32 = 0.99950767;
    pub const DISTILBERT_SENTIMENT_NEGATIVE_LABEL_INDEX: usize = 0;
    pub const DISTILBERT_SENTIMENT_NEUTRAL_LABEL: &str = "POSITIVE";
    pub const DISTILBERT_SENTIMENT_NEUTRAL_SCORE: f32 = 0.98554265;
    pub const DISTILBERT_SENTIMENT_NEUTRAL_LABEL_INDEX: usize = 1;
    pub const DISTILBERT_SENTIMENT_POSITIVE_SCORES: &[f32] = &[0.00012436, 0.99987566];
    pub const DISTILBERT_SENTIMENT_NEGATIVE_SCORES: &[f32] = &[0.99950767, 0.00049234];
    pub const DISTILBERT_SENTIMENT_NEUTRAL_SCORES: &[f32] = &[0.01445730, 0.98554265];
    pub const ROBERTA_SENTIMENT_POSITIVE_LABEL: &str = "positive";
    pub const ROBERTA_SENTIMENT_POSITIVE_SCORE: f32 = 0.98318094;
    pub const ROBERTA_SENTIMENT_POSITIVE_LABEL_INDEX: usize = 2;
    pub const ROBERTA_SENTIMENT_NEGATIVE_LABEL: &str = "negative";
    pub const ROBERTA_SENTIMENT_NEGATIVE_SCORE: f32 = 0.94799244;
    pub const ROBERTA_SENTIMENT_NEGATIVE_LABEL_INDEX: usize = 0;
    pub const ROBERTA_SENTIMENT_NEUTRAL_LABEL: &str = "neutral";
    pub const ROBERTA_SENTIMENT_NEUTRAL_SCORE: f32 = 0.92984450;
    pub const ROBERTA_SENTIMENT_NEUTRAL_LABEL_INDEX: usize = 1;
    pub const ROBERTA_SENTIMENT_POSITIVE_SCORES: &[f32] = &[0.00723555, 0.00958345, 0.98318094];
    pub const ROBERTA_SENTIMENT_NEGATIVE_SCORES: &[f32] = &[0.94799244, 0.04338233, 0.00862524];
    pub const ROBERTA_SENTIMENT_NEUTRAL_SCORES: &[f32] = &[0.01340395, 0.92984450, 0.05675153];
    pub const BERT_MULTILINGUAL_POSITIVE_LABEL: &str = "5 stars";
    pub const BERT_MULTILINGUAL_POSITIVE_SCORE: f32 = 0.96888399;
    pub const BERT_MULTILINGUAL_POSITIVE_LABEL_INDEX: usize = 4;
    pub const BERT_MULTILINGUAL_NEGATIVE_LABEL: &str = "1 star";
    pub const BERT_MULTILINGUAL_NEGATIVE_SCORE: f32 = 0.93407023;
    pub const BERT_MULTILINGUAL_NEGATIVE_LABEL_INDEX: usize = 0;
    pub const BERT_MULTILINGUAL_GERMAN_POSITIVE_LABEL: &str = "5 stars";
    pub const BERT_MULTILINGUAL_GERMAN_POSITIVE_SCORE: f32 = 0.88652706;
    pub const BERT_MULTILINGUAL_GERMAN_POSITIVE_LABEL_INDEX: usize = 4;
    pub const BERT_MULTILINGUAL_FRENCH_NEGATIVE_LABEL: &str = "1 star";
    pub const BERT_MULTILINGUAL_FRENCH_NEGATIVE_SCORE: f32 = 0.76093268;
    pub const BERT_MULTILINGUAL_FRENCH_NEGATIVE_LABEL_INDEX: usize = 0;
    pub const BERT_MULTILINGUAL_SPANISH_POSITIVE_LABEL: &str = "5 stars";
    pub const BERT_MULTILINGUAL_SPANISH_POSITIVE_SCORE: f32 = 0.93122756;
    pub const BERT_MULTILINGUAL_SPANISH_POSITIVE_LABEL_INDEX: usize = 4;
    pub const DISTILROBERTA_EMOTION_POSITIVE_LABEL: &str = "joy";
    pub const DISTILROBERTA_EMOTION_POSITIVE_SCORE: f32 = 0.86246377;
    pub const DISTILROBERTA_EMOTION_POSITIVE_LABEL_INDEX: usize = 3;
    pub const DISTILROBERTA_EMOTION_NEGATIVE_LABEL: &str = "disgust";
    pub const DISTILROBERTA_EMOTION_NEGATIVE_SCORE: f32 = 0.78679681;
    pub const DISTILROBERTA_EMOTION_NEGATIVE_LABEL_INDEX: usize = 1;
    pub const DISTILROBERTA_EMOTION_HAPPY_LABEL: &str = "joy";
    pub const DISTILROBERTA_EMOTION_HAPPY_SCORE: f32 = 0.98810238;
    pub const DISTILROBERTA_EMOTION_HAPPY_LABEL_INDEX: usize = 3;
    pub const DISTILROBERTA_EMOTION_SAD_LABEL: &str = "sadness";
    pub const DISTILROBERTA_EMOTION_SAD_SCORE: f32 = 0.96455652;
    pub const DISTILROBERTA_EMOTION_SAD_LABEL_INDEX: usize = 5;
    pub const DISTILROBERTA_EMOTION_NEUTRAL_LABEL: &str = "neutral";
    pub const DISTILROBERTA_EMOTION_NEUTRAL_SCORE: f32 = 0.73899311;
    pub const DISTILROBERTA_EMOTION_NEUTRAL_LABEL_INDEX: usize = 4;
    pub const DISTILROBERTA_EMOTION_POSITIVE_SCORES: &[f32] = &[
        0.00477491, 0.00191224, 0.00230405, 0.86246377, 0.01962744, 0.00249050, 0.10642719,
    ];
    pub const DISTILROBERTA_EMOTION_NEGATIVE_SCORES: &[f32] = &[
        0.11187306, 0.78679681, 0.07801756, 0.00140430, 0.00470836, 0.01464303, 0.00255702,
    ];
    pub const DISTILROBERTA_EMOTION_HAPPY_SCORES: &[f32] = &[
        0.00100080, 0.00032470, 0.00030043, 0.98810238, 0.00171235, 0.00189455, 0.00666480,
    ];
    pub const DISTILROBERTA_EMOTION_SAD_SCORES: &[f32] = &[
        0.00093757, 0.00351960, 0.00305090, 0.00237514, 0.01091948, 0.96455652, 0.01464079,
    ];
    pub const DISTILROBERTA_EMOTION_NEUTRAL_SCORES: &[f32] = &[
        0.01819254, 0.00930141, 0.04236472, 0.10131434, 0.73899311, 0.03730628, 0.05252752,
    ];

    pub const ROBERTA_EMOTIONS_POSITIVE_LABEL: &str = "love";
    pub const ROBERTA_EMOTIONS_POSITIVE_SCORE: f32 = 0.87999564;
    pub const ROBERTA_EMOTIONS_POSITIVE_LABEL_INDEX: usize = 18;
    pub const ROBERTA_EMOTIONS_NEGATIVE_LABEL: &str = "fear";
    pub const ROBERTA_EMOTIONS_NEGATIVE_SCORE: f32 = 0.42559999;
    pub const ROBERTA_EMOTIONS_NEGATIVE_LABEL_INDEX: usize = 14;
    pub const ROBERTA_EMOTIONS_HAPPY_LABEL: &str = "excitement";
    pub const ROBERTA_EMOTIONS_HAPPY_SCORE: f32 = 0.74202043;
    pub const ROBERTA_EMOTIONS_HAPPY_LABEL_INDEX: usize = 13;
    pub const ROBERTA_EMOTIONS_SAD_LABEL: &str = "sadness";
    pub const ROBERTA_EMOTIONS_SAD_SCORE: f32 = 0.63250113;
    pub const ROBERTA_EMOTIONS_SAD_LABEL_INDEX: usize = 25;
    pub const ROBERTA_EMOTIONS_NEUTRAL_LABEL: &str = "neutral";
    pub const ROBERTA_EMOTIONS_NEUTRAL_SCORE: f32 = 0.94425792;
    pub const ROBERTA_EMOTIONS_NEUTRAL_LABEL_INDEX: usize = 27;
    pub const TOXIC_BERT_TOXIC_LABEL: &str = "toxic";
    pub const TOXIC_BERT_TOXIC_SCORE: f32 = 0.99004936;
    pub const TOXIC_BERT_TOXIC_LABEL_INDEX: usize = 0;
    pub const TOXIC_BERT_NON_TOXIC_LABEL: &str = "toxic";
    pub const TOXIC_BERT_NON_TOXIC_SCORE: f32 = 0.00050768;
    pub const TOXIC_BERT_NON_TOXIC_LABEL_INDEX: usize = 0;
    pub const TOXIC_BERT_POSITIVE_LABEL: &str = "toxic";
    pub const TOXIC_BERT_POSITIVE_SCORE: f32 = 0.00057647;
    pub const TOXIC_BERT_POSITIVE_LABEL_INDEX: usize = 0;
}

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

    
    // Builder Tests
    

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
            assert_eq!(preset.labels.unwrap().len(), 2);
        }

        #[test]
        fn test_sentiment_3class_preset() {
            let preset = &presets::SENTIMENT_3CLASS_V1;

            assert_eq!(preset.name, "SENTIMENT_3CLASS_V1");
            assert_eq!(preset.model, "roberta-sentiment");
            assert_eq!(preset.labels.unwrap().len(), 3);
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

            assert_eq!(fast.name, "SENTIMENT_BINARY_V1");
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
            assert!(find_preset("sentiment_binary_v1").is_some());
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

    
    // Validation Tests
    

    mod validation_tests {
        use super::*;
        use kjarni_transformers::models::ModelType;

        #[test]
        fn test_validate_sentiment_model() {
            if let Some(model_type) = ModelType::from_cli_name("distilbert-sentiment") {
                let result = validation::validate_for_classification(model_type);
                assert!(
                    result.is_ok(),
                    "Sentiment model should be valid for classification"
                );
            }
        }

        #[test]
        fn test_validate_decoder_invalid() {
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
            assert!(
                !models.is_empty(),
                "Should have at least one classifier model"
            );
            assert!(
                models.iter().any(|m| m.contains("sentiment")
                    || m.contains("emotion")
                    || m.contains("zeroshot")),
                "Should include at least one known classifier"
            );
        }

        #[test]
        fn test_validate_classifier_model() {
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
            if let Some(model_type) = ModelType::from_cli_name("llama3.2-1b") {
                let result = validation::validate_for_classification(model_type);
                assert!(
                    result.is_err(),
                    "Decoder should not be valid for classification"
                );
            }
        }
    }

    
    // Error Tests
    

    mod error_tests {
        use super::*;
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

    mod classifier_golden_values_test {
        use crate::Classifier;

        fn assert_approx_eq(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
            assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
            for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - e).abs() < tolerance,
                    "{label}[{i}]: expected {e:.6}, got {a:.6} (diff: {:.6})",
                    (a - e).abs()
                );
            }
        }

        #[tokio::test]
        async fn golden_classify_love_product() {
            let clf = Classifier::builder("distilbert-sentiment")
                .quiet(true)
                .build()
                .await
                .unwrap();
            let result = clf.classify("I love this product").await.unwrap();
            assert_eq!(result.label, "POSITIVE");
            assert!(
                result.score > 0.999,
                "Expected score > 0.999, got {}",
                result.score
            );
        }

        #[tokio::test]
        async fn golden_classify_terrible() {
            let clf = Classifier::builder("distilbert-sentiment")
                .quiet(true)
                .build()
                .await
                .unwrap();
            let result = clf.classify("This is terrible").await.unwrap();
            assert_eq!(result.label, "NEGATIVE");
            assert!(
                result.score > 0.999,
                "Expected score > 0.999, got {}",
                result.score
            );
        }

        #[tokio::test]
        async fn golden_classify_weather_okay() {
            let clf = Classifier::builder("distilbert-sentiment")
                .quiet(true)
                .build()
                .await
                .unwrap();
            let result = clf.classify("The weather is okay").await.unwrap();
            assert_eq!(result.label, "POSITIVE");
            assert!(
                result.score > 0.999,
                "Expected score > 0.999, got {}",
                result.score
            );
        }

        #[tokio::test]
        async fn golden_classify_love_kjarni() {
            let clf = Classifier::builder("distilbert-sentiment")
                .quiet(true)
                .build()
                .await
                .unwrap();
            let result = clf.classify("I love kjarni").await.unwrap();
            assert_eq!(result.label, "POSITIVE");
            assert!(
                result.score > 0.999,
                "Expected score > 0.999, got {}",
                result.score
            );
        }

        #[tokio::test]
        async fn golden_classify_not_bad() {
            let clf = Classifier::builder("distilbert-sentiment")
                .quiet(true)
                .build()
                .await
                .unwrap();
            let result = clf.classify("not bad").await.unwrap();
            assert_eq!(result.label, "POSITIVE");
            assert!(
                result.score > 0.999,
                "Expected score > 0.999, got {}",
                result.score
            );
        }

        #[tokio::test]
        async fn golden_classify_not_happy() {
            let clf = Classifier::builder("distilbert-sentiment")
                .quiet(true)
                .build()
                .await
                .unwrap();
            let result = clf.classify("I'm not happy").await.unwrap();
            assert_eq!(result.label, "NEGATIVE");
            assert!(
                result.score > 0.999,
                "Expected score > 0.999, got {}",
                result.score
            );
        }
    }

    
    // Convenience Function Tests
    

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
            assert!(models.len() >= 0);
        }
    }

    
    // Test Utilities
    

    const TEST_TEXT_POSITIVE: &str = "I absolutely love this, it's amazing!";
    const TEST_TEXT_NEGATIVE: &str = "This is terrible, I hate it so much.";
    const TEST_TEXT_NEUTRAL: &str = "The meeting is scheduled for Tuesday.";
    const TEST_TEXT_GERMAN_POSITIVE: &str = "Das ist wunderbar, ich liebe es!";
    const TEST_TEXT_FRENCH_NEGATIVE: &str = "C'est terrible, je déteste ça.";
    const TEST_TEXT_SPANISH_POSITIVE: &str = "¡Esto es increíble, me encanta!";
    const TEST_TEXT_TOXIC: &str = "I hate you, you're worthless garbage.";
    const TEST_TEXT_NON_TOXIC: &str = "Thank you for your help today.";

    fn assert_valid_probability(score: f32, context: &str) {
        assert!(
            (0.0..=1.0).contains(&score),
            "{}: score {} not in [0.0, 1.0]",
            context,
            score
        );
    }

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    
    // distilbert-sentiment Integration Tests
    

    mod distilbert_sentiment_tests {
        use super::*;

        #[tokio::test]
        async fn test_positive_text() {
            let classifier = Classifier::new("distilbert-sentiment")
                .await
                .expect("Failed to load distilbert-sentiment");

            let result = classifier
                .classify(expected::INPUT_POSITIVE)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::DISTILBERT_SENTIMENT_POSITIVE_LABEL);
            assert_eq!(
                result.label_index,
                expected::DISTILBERT_SENTIMENT_POSITIVE_LABEL_INDEX
            );
            assert!(
                approx_eq(
                    result.score,
                    expected::DISTILBERT_SENTIMENT_POSITIVE_SCORE,
                    1e-5
                ),
                "Score mismatch: expected {}, got {}",
                expected::DISTILBERT_SENTIMENT_POSITIVE_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_negative_text() {
            let classifier = Classifier::new("distilbert-sentiment")
                .await
                .expect("Failed to load distilbert-sentiment");

            let result = classifier
                .classify(expected::INPUT_NEGATIVE)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::DISTILBERT_SENTIMENT_NEGATIVE_LABEL);
            assert_eq!(
                result.label_index,
                expected::DISTILBERT_SENTIMENT_NEGATIVE_LABEL_INDEX
            );
            assert!(
                approx_eq(
                    result.score,
                    expected::DISTILBERT_SENTIMENT_NEGATIVE_SCORE,
                    1e-5
                ),
                "Score mismatch: expected {}, got {}",
                expected::DISTILBERT_SENTIMENT_NEGATIVE_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_neutral_text() {
            let classifier = Classifier::new("distilbert-sentiment")
                .await
                .expect("Failed to load distilbert-sentiment");

            let result = classifier
                .classify(expected::INPUT_NEUTRAL)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::DISTILBERT_SENTIMENT_NEUTRAL_LABEL);
            assert_eq!(
                result.label_index,
                expected::DISTILBERT_SENTIMENT_NEUTRAL_LABEL_INDEX
            );
            assert!(
                approx_eq(
                    result.score,
                    expected::DISTILBERT_SENTIMENT_NEUTRAL_SCORE,
                    1e-5
                ),
                "Score mismatch: expected {}, got {}",
                expected::DISTILBERT_SENTIMENT_NEUTRAL_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_scores_exact_match() {
            let classifier = Classifier::new("distilbert-sentiment")
                .await
                .expect("Failed to load distilbert-sentiment");

            let scores = classifier
                .classify_scores(expected::INPUT_POSITIVE)
                .await
                .expect("Failed to get scores");

            assert_eq!(scores.len(), 2);
            for (i, (actual, &expected_score)) in scores
                .iter()
                .zip(expected::DISTILBERT_SENTIMENT_POSITIVE_SCORES.iter())
                .enumerate()
            {
                assert!(
                    approx_eq(*actual, expected_score, 1e-5),
                    "Score mismatch at index {}: expected {}, got {}",
                    i,
                    expected_score,
                    actual
                );
            }
        }

        #[tokio::test]
        async fn test_scores_sum_to_one() {
            let classifier = Classifier::new("distilbert-sentiment")
                .await
                .expect("Failed to load distilbert-sentiment");

            let scores = classifier
                .classify_scores(expected::INPUT_POSITIVE)
                .await
                .expect("Failed to get scores");

            let sum: f32 = scores.iter().sum();
            assert!(
                approx_eq(sum, 1.0, 0.001),
                "Scores should sum to 1.0, got {}",
                sum
            );
        }
    }

    
    // roberta-sentiment Integration Tests
    

    mod roberta_sentiment_tests {
        use super::*;

        #[tokio::test]
        async fn test_positive_text() {
            let classifier = Classifier::new("roberta-sentiment")
                .await
                .expect("Failed to load roberta-sentiment");

            let result = classifier
                .classify(expected::INPUT_POSITIVE)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::ROBERTA_SENTIMENT_POSITIVE_LABEL);
            assert_eq!(
                result.label_index,
                expected::ROBERTA_SENTIMENT_POSITIVE_LABEL_INDEX
            );
            assert!(
                approx_eq(
                    result.score,
                    expected::ROBERTA_SENTIMENT_POSITIVE_SCORE,
                    1e-5
                ),
                "Score mismatch: expected {}, got {}",
                expected::ROBERTA_SENTIMENT_POSITIVE_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_negative_text() {
            let classifier = Classifier::new("roberta-sentiment")
                .await
                .expect("Failed to load roberta-sentiment");

            let result = classifier
                .classify(expected::INPUT_NEGATIVE)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::ROBERTA_SENTIMENT_NEGATIVE_LABEL);
            assert_eq!(
                result.label_index,
                expected::ROBERTA_SENTIMENT_NEGATIVE_LABEL_INDEX
            );
            assert!(
                approx_eq(
                    result.score,
                    expected::ROBERTA_SENTIMENT_NEGATIVE_SCORE,
                    1e-5
                ),
                "Score mismatch: expected {}, got {}",
                expected::ROBERTA_SENTIMENT_NEGATIVE_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_neutral_text() {
            let classifier = Classifier::new("roberta-sentiment")
                .await
                .expect("Failed to load roberta-sentiment");

            let result = classifier
                .classify(expected::INPUT_NEUTRAL)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::ROBERTA_SENTIMENT_NEUTRAL_LABEL);
            assert_eq!(
                result.label_index,
                expected::ROBERTA_SENTIMENT_NEUTRAL_LABEL_INDEX
            );
            assert!(
                approx_eq(
                    result.score,
                    expected::ROBERTA_SENTIMENT_NEUTRAL_SCORE,
                    1e-5
                ),
                "Score mismatch: expected {}, got {}",
                expected::ROBERTA_SENTIMENT_NEUTRAL_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_scores_exact_match() {
            let classifier = Classifier::new("roberta-sentiment")
                .await
                .expect("Failed to load roberta-sentiment");

            let scores = classifier
                .classify_scores(expected::INPUT_POSITIVE)
                .await
                .expect("Failed to get scores");

            assert_eq!(scores.len(), 3);
            for (i, (actual, &expected_score)) in scores
                .iter()
                .zip(expected::ROBERTA_SENTIMENT_POSITIVE_SCORES.iter())
                .enumerate()
            {
                assert!(
                    approx_eq(*actual, expected_score, 1e-5),
                    "Score mismatch at index {}: expected {}, got {}",
                    i,
                    expected_score,
                    actual
                );
            }
        }
    }

    
    // bert-sentiment-multilingual Integration Tests
    

    mod bert_multilingual_tests {
        use super::*;

        #[tokio::test]
        async fn test_english_positive() {
            let classifier = Classifier::new("bert-sentiment-multilingual")
                .await
                .expect("Failed to load bert-sentiment-multilingual");

            let result = classifier
                .classify(expected::INPUT_POSITIVE)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::BERT_MULTILINGUAL_POSITIVE_LABEL);
            assert_eq!(
                result.label_index,
                expected::BERT_MULTILINGUAL_POSITIVE_LABEL_INDEX
            );
            assert!(
                approx_eq(
                    result.score,
                    expected::BERT_MULTILINGUAL_POSITIVE_SCORE,
                    1e-5
                ),
                "Score mismatch: expected {}, got {}",
                expected::BERT_MULTILINGUAL_POSITIVE_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_english_negative() {
            let classifier = Classifier::new("bert-sentiment-multilingual")
                .await
                .expect("Failed to load bert-sentiment-multilingual");

            let result = classifier
                .classify(expected::INPUT_NEGATIVE)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::BERT_MULTILINGUAL_NEGATIVE_LABEL);
            assert_eq!(
                result.label_index,
                expected::BERT_MULTILINGUAL_NEGATIVE_LABEL_INDEX
            );
            assert!(
                approx_eq(
                    result.score,
                    expected::BERT_MULTILINGUAL_NEGATIVE_SCORE,
                    1e-5
                ),
                "Score mismatch: expected {}, got {}",
                expected::BERT_MULTILINGUAL_NEGATIVE_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_german_positive() {
            let classifier = Classifier::new("bert-sentiment-multilingual")
                .await
                .expect("Failed to load bert-sentiment-multilingual");

            let result = classifier
                .classify(expected::INPUT_GERMAN_POSITIVE)
                .await
                .expect("Classification failed");

            assert_eq!(
                result.label,
                expected::BERT_MULTILINGUAL_GERMAN_POSITIVE_LABEL
            );
            assert_eq!(
                result.label_index,
                expected::BERT_MULTILINGUAL_GERMAN_POSITIVE_LABEL_INDEX
            );
            assert!(
                approx_eq(
                    result.score,
                    expected::BERT_MULTILINGUAL_GERMAN_POSITIVE_SCORE,
                    1e-5
                ),
                "Score mismatch: expected {}, got {}",
                expected::BERT_MULTILINGUAL_GERMAN_POSITIVE_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_french_negative() {
            let classifier = Classifier::new("bert-sentiment-multilingual")
                .await
                .expect("Failed to load bert-sentiment-multilingual");

            let result = classifier
                .classify(expected::INPUT_FRENCH_NEGATIVE)
                .await
                .expect("Classification failed");

            assert_eq!(
                result.label,
                expected::BERT_MULTILINGUAL_FRENCH_NEGATIVE_LABEL
            );
            assert_eq!(
                result.label_index,
                expected::BERT_MULTILINGUAL_FRENCH_NEGATIVE_LABEL_INDEX
            );
            assert!(
                approx_eq(
                    result.score,
                    expected::BERT_MULTILINGUAL_FRENCH_NEGATIVE_SCORE,
                    1e-5
                ),
                "Score mismatch: expected {}, got {}",
                expected::BERT_MULTILINGUAL_FRENCH_NEGATIVE_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_spanish_positive() {
            let classifier = Classifier::new("bert-sentiment-multilingual")
                .await
                .expect("Failed to load bert-sentiment-multilingual");

            let result = classifier
                .classify(expected::INPUT_SPANISH_POSITIVE)
                .await
                .expect("Classification failed");

            assert_eq!(
                result.label,
                expected::BERT_MULTILINGUAL_SPANISH_POSITIVE_LABEL
            );
            assert_eq!(
                result.label_index,
                expected::BERT_MULTILINGUAL_SPANISH_POSITIVE_LABEL_INDEX
            );
            assert!(
                approx_eq(
                    result.score,
                    expected::BERT_MULTILINGUAL_SPANISH_POSITIVE_SCORE,
                    1e-5
                ),
                "Score mismatch: expected {}, got {}",
                expected::BERT_MULTILINGUAL_SPANISH_POSITIVE_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_batch_multilingual() {
            let classifier = Classifier::new("bert-sentiment-multilingual")
                .await
                .expect("Failed to load model");

            let texts = &[
                expected::INPUT_POSITIVE,
                expected::INPUT_GERMAN_POSITIVE,
                expected::INPUT_FRENCH_NEGATIVE,
                expected::INPUT_SPANISH_POSITIVE,
            ];

            let results = classifier
                .classify_batch(texts)
                .await
                .expect("Batch classification failed");

            assert_eq!(results.len(), 4);

            // English positive
            assert_eq!(results[0].label, expected::BERT_MULTILINGUAL_POSITIVE_LABEL);
            assert!(approx_eq(
                results[0].score,
                expected::BERT_MULTILINGUAL_POSITIVE_SCORE,
                1e-5
            ));

            // German positive
            assert_eq!(
                results[1].label,
                expected::BERT_MULTILINGUAL_GERMAN_POSITIVE_LABEL
            );
            assert!(approx_eq(
                results[1].score,
                expected::BERT_MULTILINGUAL_GERMAN_POSITIVE_SCORE,
                1e-5
            ));

            // French negative
            assert_eq!(
                results[2].label,
                expected::BERT_MULTILINGUAL_FRENCH_NEGATIVE_LABEL
            );
            assert!(approx_eq(
                results[2].score,
                expected::BERT_MULTILINGUAL_FRENCH_NEGATIVE_SCORE,
                1e-5
            ));

            // Spanish positive
            assert_eq!(
                results[3].label,
                expected::BERT_MULTILINGUAL_SPANISH_POSITIVE_LABEL
            );
            assert!(approx_eq(
                results[3].score,
                expected::BERT_MULTILINGUAL_SPANISH_POSITIVE_SCORE,
                1e-5
            ));
        }
    }

    
    // distilroberta-emotion Integration Tests
    

    mod distilroberta_emotion_tests {
        use super::*;

        #[tokio::test]
        async fn test_positive_text() {
            let classifier = Classifier::new("distilroberta-emotion")
                .await
                .expect("Failed to load distilroberta-emotion");

            let result = classifier
                .classify(expected::INPUT_POSITIVE)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::DISTILROBERTA_EMOTION_POSITIVE_LABEL);
            assert_eq!(
                result.label_index,
                expected::DISTILROBERTA_EMOTION_POSITIVE_LABEL_INDEX
            );
            assert!(
                approx_eq(
                    result.score,
                    expected::DISTILROBERTA_EMOTION_POSITIVE_SCORE,
                    1e-5
                ),
                "Score mismatch: expected {}, got {}",
                expected::DISTILROBERTA_EMOTION_POSITIVE_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_negative_text() {
            let classifier = Classifier::new("distilroberta-emotion")
                .await
                .expect("Failed to load distilroberta-emotion");

            let result = classifier
                .classify(expected::INPUT_NEGATIVE)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::DISTILROBERTA_EMOTION_NEGATIVE_LABEL);
            assert_eq!(
                result.label_index,
                expected::DISTILROBERTA_EMOTION_NEGATIVE_LABEL_INDEX
            );
            assert!(
                approx_eq(
                    result.score,
                    expected::DISTILROBERTA_EMOTION_NEGATIVE_SCORE,
                    1e-5
                ),
                "Score mismatch: expected {}, got {}",
                expected::DISTILROBERTA_EMOTION_NEGATIVE_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_happy_text() {
            let classifier = Classifier::new("distilroberta-emotion")
                .await
                .expect("Failed to load distilroberta-emotion");

            let result = classifier
                .classify(expected::INPUT_HAPPY)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::DISTILROBERTA_EMOTION_HAPPY_LABEL);
            assert_eq!(
                result.label_index,
                expected::DISTILROBERTA_EMOTION_HAPPY_LABEL_INDEX
            );
            assert!(
                approx_eq(
                    result.score,
                    expected::DISTILROBERTA_EMOTION_HAPPY_SCORE,
                    1e-5
                ),
                "Score mismatch: expected {}, got {}",
                expected::DISTILROBERTA_EMOTION_HAPPY_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_sad_text() {
            let classifier = Classifier::new("distilroberta-emotion")
                .await
                .expect("Failed to load distilroberta-emotion");

            let result = classifier
                .classify(expected::INPUT_SAD)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::DISTILROBERTA_EMOTION_SAD_LABEL);
            assert_eq!(
                result.label_index,
                expected::DISTILROBERTA_EMOTION_SAD_LABEL_INDEX
            );
            assert!(
                approx_eq(
                    result.score,
                    expected::DISTILROBERTA_EMOTION_SAD_SCORE,
                    1e-5
                ),
                "Score mismatch: expected {}, got {}",
                expected::DISTILROBERTA_EMOTION_SAD_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_neutral_text() {
            let classifier = Classifier::new("distilroberta-emotion")
                .await
                .expect("Failed to load distilroberta-emotion");

            let result = classifier
                .classify(expected::INPUT_NEUTRAL)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::DISTILROBERTA_EMOTION_NEUTRAL_LABEL);
            assert_eq!(
                result.label_index,
                expected::DISTILROBERTA_EMOTION_NEUTRAL_LABEL_INDEX
            );
            assert!(
                approx_eq(
                    result.score,
                    expected::DISTILROBERTA_EMOTION_NEUTRAL_SCORE,
                    1e-5
                ),
                "Score mismatch: expected {}, got {}",
                expected::DISTILROBERTA_EMOTION_NEUTRAL_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_happy_scores_exact_match() {
            let classifier = Classifier::new("distilroberta-emotion")
                .await
                .expect("Failed to load distilroberta-emotion");

            let scores = classifier
                .classify_scores(expected::INPUT_HAPPY)
                .await
                .expect("Failed to get scores");

            assert_eq!(scores.len(), 7);
            for (i, (actual, &expected_score)) in scores
                .iter()
                .zip(expected::DISTILROBERTA_EMOTION_HAPPY_SCORES.iter())
                .enumerate()
            {
                assert!(
                    approx_eq(*actual, expected_score, 1e-5),
                    "Score mismatch at index {}: expected {}, got {}",
                    i,
                    expected_score,
                    actual
                );
            }
        }

        #[tokio::test]
        async fn test_sad_scores_exact_match() {
            let classifier = Classifier::new("distilroberta-emotion")
                .await
                .expect("Failed to load distilroberta-emotion");

            let scores = classifier
                .classify_scores(expected::INPUT_SAD)
                .await
                .expect("Failed to get scores");

            assert_eq!(scores.len(), 7);
            for (i, (actual, &expected_score)) in scores
                .iter()
                .zip(expected::DISTILROBERTA_EMOTION_SAD_SCORES.iter())
                .enumerate()
            {
                assert!(
                    approx_eq(*actual, expected_score, 1e-5),
                    "Score mismatch at index {}: expected {}, got {}",
                    i,
                    expected_score,
                    actual
                );
            }
        }
    }

    
    // roberta-emotions (Multi-Label) Integration Tests
    

    mod roberta_emotions_tests {
        use super::*;

        #[tokio::test]
        async fn test_positive_text_multilabel() {
            let classifier = Classifier::builder("roberta-emotions")
                .multi_label()
                .build()
                .await
                .expect("Failed to load roberta-emotions");

            assert_eq!(classifier.mode(), ClassificationMode::MultiLabel);
            assert_eq!(classifier.num_labels(), 28);

            let result = classifier
                .classify(expected::INPUT_POSITIVE)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::ROBERTA_EMOTIONS_POSITIVE_LABEL);
            assert_eq!(
                result.label_index,
                expected::ROBERTA_EMOTIONS_POSITIVE_LABEL_INDEX
            );
            assert!(
                approx_eq(
                    result.score,
                    expected::ROBERTA_EMOTIONS_POSITIVE_SCORE,
                    1e-5
                ),
                "Score mismatch: expected {}, got {}",
                expected::ROBERTA_EMOTIONS_POSITIVE_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_happy_text_multilabel() {
            let classifier = Classifier::builder("roberta-emotions")
                .multi_label()
                .build()
                .await
                .expect("Failed to load roberta-emotions");

            let result = classifier
                .classify(expected::INPUT_HAPPY)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::ROBERTA_EMOTIONS_HAPPY_LABEL);
            assert_eq!(
                result.label_index,
                expected::ROBERTA_EMOTIONS_HAPPY_LABEL_INDEX
            );
            assert!(
                approx_eq(result.score, expected::ROBERTA_EMOTIONS_HAPPY_SCORE, 1e-5),
                "Score mismatch: expected {}, got {}",
                expected::ROBERTA_EMOTIONS_HAPPY_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_sad_text_multilabel() {
            let classifier = Classifier::builder("roberta-emotions")
                .multi_label()
                .build()
                .await
                .expect("Failed to load roberta-emotions");

            let result = classifier
                .classify(expected::INPUT_SAD)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::ROBERTA_EMOTIONS_SAD_LABEL);
            assert_eq!(
                result.label_index,
                expected::ROBERTA_EMOTIONS_SAD_LABEL_INDEX
            );
            assert!(
                approx_eq(result.score, expected::ROBERTA_EMOTIONS_SAD_SCORE, 1e-5),
                "Score mismatch: expected {}, got {}",
                expected::ROBERTA_EMOTIONS_SAD_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_neutral_text_multilabel() {
            let classifier = Classifier::builder("roberta-emotions")
                .multi_label()
                .build()
                .await
                .expect("Failed to load roberta-emotions");

            let result = classifier
                .classify(expected::INPUT_NEUTRAL)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::ROBERTA_EMOTIONS_NEUTRAL_LABEL);
            assert_eq!(
                result.label_index,
                expected::ROBERTA_EMOTIONS_NEUTRAL_LABEL_INDEX
            );
            assert!(
                approx_eq(result.score, expected::ROBERTA_EMOTIONS_NEUTRAL_SCORE, 1e-5),
                "Score mismatch: expected {}, got {}",
                expected::ROBERTA_EMOTIONS_NEUTRAL_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_multilabel_scores_do_not_sum_to_one() {
            let classifier = Classifier::builder("roberta-emotions")
                .multi_label()
                .build()
                .await
                .expect("Failed to load roberta-emotions");

            let scores = classifier
                .classify_scores(expected::INPUT_POSITIVE)
                .await
                .expect("Failed to get scores");

            // Multi-label scores are sigmoid, NOT softmax - they don't sum to 1
            let sum: f32 = scores.iter().sum();
            assert!(
                sum != 1.0,
                "Multi-label scores should NOT sum to 1.0, got {}",
                sum
            );

            // Each score should be valid probability
            for (i, &score) in scores.iter().enumerate() {
                assert_valid_probability(score, &format!("score at index {}", i));
            }
        }
    }

    
    // toxic-bert (Multi-Label) Integration Tests
    

    mod toxic_bert_tests {
        use super::*;

        #[tokio::test]
        async fn test_toxic_text() {
            let classifier = Classifier::builder("toxic-bert")
                .multi_label()
                .build()
                .await
                .expect("Failed to load toxic-bert");

            assert_eq!(classifier.mode(), ClassificationMode::MultiLabel);
            assert_eq!(classifier.num_labels(), 6);

            let result = classifier
                .classify(expected::INPUT_TOXIC)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, expected::TOXIC_BERT_TOXIC_LABEL);
            assert_eq!(result.label_index, expected::TOXIC_BERT_TOXIC_LABEL_INDEX);
            assert!(
                approx_eq(result.score, expected::TOXIC_BERT_TOXIC_SCORE, 1e-5),
                "Score mismatch: expected {}, got {}",
                expected::TOXIC_BERT_TOXIC_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_non_toxic_text() {
            let classifier = Classifier::builder("toxic-bert")
                .multi_label()
                .build()
                .await
                .expect("Failed to load toxic-bert");

            let result = classifier
                .classify(expected::INPUT_NON_TOXIC)
                .await
                .expect("Classification failed");

            // Non-toxic text should have very low toxicity score
            assert!(
                approx_eq(result.score, expected::TOXIC_BERT_NON_TOXIC_SCORE, 1e-5),
                "Score mismatch: expected {}, got {}",
                expected::TOXIC_BERT_NON_TOXIC_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_positive_text_not_toxic() {
            let classifier = Classifier::builder("toxic-bert")
                .multi_label()
                .build()
                .await
                .expect("Failed to load toxic-bert");

            let result = classifier
                .classify(expected::INPUT_POSITIVE)
                .await
                .expect("Classification failed");

            // Positive text should have very low toxicity score
            assert!(
                approx_eq(result.score, expected::TOXIC_BERT_POSITIVE_SCORE, 1e-5),
                "Score mismatch: expected {}, got {}",
                expected::TOXIC_BERT_POSITIVE_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_toxic_above_threshold() {
            let classifier = Classifier::builder("toxic-bert")
                .multi_label()
                .build()
                .await
                .expect("Failed to load toxic-bert");

            let result = classifier
                .classify(expected::INPUT_TOXIC)
                .await
                .expect("Classification failed");

            // Check above_threshold for toxic text
            let above_50 = result.above_threshold(0.5);
            assert!(
                above_50.len() >= 1,
                "Should have at least one toxicity category above 0.5"
            );
            assert!(
                above_50.iter().any(|(label, _)| label == "toxic"),
                "Should detect 'toxic' category"
            );
        }
    }

    
    // Batch Classification Tests
    

    mod batch_tests {
        use kjarni_transformers::LanguageModel;

        use super::*;
        #[tokio::test]
        async fn test_roberta_sentiment_tokenization_parity() {
            use crate::SequenceClassifier;
            use kjarni_transformers::models::ModelType;
            use kjarni_transformers::traits::Device;

            // PyTorch reference values
            const EXPECTED_INPUT_IDS: &[u32] = &[0, 100, 3668, 657, 42, 6, 24, 18, 2770, 328, 2];
            const EXPECTED_ATTENTION_MASK: &[u32] = &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
            const EXPECTED_LENGTH: usize = 11;

            let model_type = ModelType::from_cli_name("roberta-sentiment").unwrap();
            let cache_dir = crate::common::default_cache_dir();

            let classifier = SequenceClassifier::from_registry(
                model_type,
                Some(cache_dir),
                Device::Cpu,
                None,
                None,
            )
            .await
            .expect("Failed to load");

            let text = "I absolutely love this, it's amazing!";
            let tokenizer = classifier.tokenizer();
            let encoding = tokenizer.encode(text, true).expect("Failed to tokenize");

            assert_eq!(encoding.get_ids(), EXPECTED_INPUT_IDS, "Input IDs mismatch");
            assert_eq!(
                encoding.get_attention_mask(),
                EXPECTED_ATTENTION_MASK,
                "Attention mask mismatch"
            );
            assert_eq!(encoding.len(), EXPECTED_LENGTH, "Sequence length mismatch");
        }

        #[tokio::test]
        async fn test_roberta_sentiment_scores_parity() {
            // PyTorch reference values
            const EXPECTED_PROBS: [f32; 3] = [0.00723555, 0.00958345, 0.98318094];
            const EXPECTED_LABEL: &str = "positive";
            const EXPECTED_SCORE: f32 = 0.98318094;
            const TOLERANCE: f32 = 1e-4;

            let classifier = Classifier::new("roberta-sentiment")
                .await
                .expect("Failed to load model");

            let text = "I absolutely love this, it's amazing!";

            // Test raw scores
            let scores = classifier
                .classify_scores(text)
                .await
                .expect("Failed to get scores");

            assert_eq!(scores.len(), 3, "Expected 3 scores");
            for (i, (&actual, &expected)) in scores.iter().zip(EXPECTED_PROBS.iter()).enumerate() {
                assert!(
                    approx_eq(actual, expected, TOLERANCE),
                    "Score mismatch at index {}: expected {}, got {}",
                    i,
                    expected,
                    actual
                );
            }

            // Verify scores sum to 1.0 (softmax property)
            let sum: f32 = scores.iter().sum();
            assert!(
                approx_eq(sum, 1.0, 1e-5),
                "Scores should sum to 1.0, got {}",
                sum
            );

            // Test classification result
            let result = classifier
                .classify(text)
                .await
                .expect("Classification failed");

            assert_eq!(result.label, EXPECTED_LABEL, "Label mismatch");
            assert!(
                approx_eq(result.score, EXPECTED_SCORE, TOLERANCE),
                "Score mismatch: expected {}, got {}",
                EXPECTED_SCORE,
                result.score
            );
        }

        #[tokio::test]
        async fn test_roberta_sentiment_logits_parity() {
            use crate::SequenceClassifier;
            use kjarni_transformers::models::ModelType;
            use kjarni_transformers::traits::Device;

            // PyTorch reference values
            const EXPECTED_LOGITS: [f32; 3] = [-1.653445, -1.372414, 3.2583416];
            const TOLERANCE: f32 = 1e-4;

            let model_type = ModelType::from_cli_name("roberta-sentiment").unwrap();
            let cache_dir = crate::common::default_cache_dir();

            let classifier = SequenceClassifier::from_registry(
                model_type,
                Some(cache_dir),
                Device::Cpu,
                None,
                None,
            )
            .await
            .expect("Failed to load");

            let text = "I absolutely love this, it's amazing!";
            let logits = classifier.predict_logits(&[text]).await.expect("Failed");

            assert_eq!(logits.len(), 1, "Expected 1 batch result");
            assert_eq!(logits[0].len(), 3, "Expected 3 logits");

            for (i, (&actual, &expected)) in
                logits[0].iter().zip(EXPECTED_LOGITS.iter()).enumerate()
            {
                assert!(
                    approx_eq(actual, expected, TOLERANCE),
                    "Logit mismatch at index {}: expected {}, got {}",
                    i,
                    expected,
                    actual
                );
            }
        }

        #[tokio::test]
        async fn test_batch_roberta_sentiment() {
            let classifier = Classifier::new("roberta-sentiment")
                .await
                .expect("Failed to load model");

            let texts = &[
                expected::INPUT_POSITIVE,
                expected::INPUT_NEGATIVE,
                expected::INPUT_NEUTRAL,
            ];

            let results = classifier
                .classify_batch(texts)
                .await
                .expect("Batch classification failed");

            assert_eq!(results.len(), 3);

            // Positive
            println!("label: {} score: {}", results[0].label, results[0].score);
            println!(
                "expected label: {} expected score: {}",
                expected::ROBERTA_SENTIMENT_POSITIVE_LABEL,
                expected::ROBERTA_SENTIMENT_POSITIVE_SCORE,
            );
            assert_eq!(results[0].label, expected::ROBERTA_SENTIMENT_POSITIVE_LABEL);
            assert!(approx_eq(
                results[0].score,
                expected::ROBERTA_SENTIMENT_POSITIVE_SCORE,
                1e-5
            ));

            // Negative
            println!("label: {} score: {}", results[1].label, results[1].score);
            println!(
                "expected label: {} expected score: {}",
                expected::ROBERTA_SENTIMENT_NEGATIVE_LABEL,
                expected::ROBERTA_SENTIMENT_NEGATIVE_SCORE,
            );
            assert_eq!(results[1].label, expected::ROBERTA_SENTIMENT_NEGATIVE_LABEL);
            assert!(approx_eq(
                results[1].score,
                expected::ROBERTA_SENTIMENT_NEGATIVE_SCORE,
                1e-5
            ));

            // Neutral
            println!("label: {} score: {}", results[2].label, results[2].score);
            println!(
                "expected label: {} expected score: {}",
                expected::ROBERTA_SENTIMENT_NEUTRAL_LABEL,
                expected::ROBERTA_SENTIMENT_NEUTRAL_SCORE,
            );
            assert_eq!(results[2].label, expected::ROBERTA_SENTIMENT_NEUTRAL_LABEL);
            assert!(approx_eq(
                results[2].score,
                expected::ROBERTA_SENTIMENT_NEUTRAL_SCORE,
                1e-5
            ));
        }
    }

    
    // Error Handling Tests
    

    mod error_handling_tests {
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
            let result = Classifier::builder("distilbert-sentiment")
                .labels(vec!["a", "b", "c", "d"])
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
                    expected::INPUT_NEUTRAL,
                    &ClassificationOverrides {
                        threshold: Some(0.9999),
                        ..Default::default()
                    },
                )
                .await;

            assert!(matches!(
                result,
                Err(ClassifierError::ClassificationFailed(_))
            ));
        }

        #[tokio::test]
        async fn test_offline_missing_model() {
            let result = Classifier::builder("minilm-l6-v2-cross-encoder")
                .offline()
                .cache_dir("/tmp/kjarni-test-empty-cache-12345")
                .build()
                .await;

            assert!(result.is_err());
        }
    }

    
    // GPU Parity Tests
    

    mod gpu_tests {
        use super::*;

        #[tokio::test]
        async fn test_gpu_classification() {
            let classifier = Classifier::builder("distilbert-sentiment")
                .gpu()
                .build()
                .await
                .expect("Failed to load classifier on GPU");

            let result = classifier
                .classify(expected::INPUT_POSITIVE)
                .await
                .expect("GPU classification failed");

            assert_eq!(result.label, expected::DISTILBERT_SENTIMENT_POSITIVE_LABEL);
        }

        #[tokio::test]
        async fn test_gpu_cpu_parity() {
            let gpu_classifier = Classifier::builder("distilbert-sentiment")
                .gpu()
                .build()
                .await
                .expect("Failed to load classifier on GPU");

            let cpu_classifier = Classifier::builder("distilbert-sentiment")
                .cpu()
                .build()
                .await
                .expect("Failed to load classifier on CPU");

            let gpu_result = gpu_classifier
                .classify(expected::INPUT_POSITIVE)
                .await
                .expect("GPU classification failed");

            let cpu_result = cpu_classifier
                .classify(expected::INPUT_POSITIVE)
                .await
                .expect("CPU classification failed");

            assert_eq!(gpu_result.label, cpu_result.label);
            assert_eq!(gpu_result.label_index, cpu_result.label_index);
            assert!(
                approx_eq(gpu_result.score, cpu_result.score, 1e-4),
                "GPU/CPU score mismatch: GPU={}, CPU={}",
                gpu_result.score,
                cpu_result.score
            );
        }
    }

    
    // F16 Precision Tests
    

    mod f16_tests {
        use super::*;

        #[tokio::test]
        async fn test_f16_classification() {
            let classifier = Classifier::builder("distilroberta-emotion")
                .f16()
                .build()
                .await
                .expect("Failed to build classifier with F16 dtype");

            let result = classifier
                .classify(expected::INPUT_HAPPY)
                .await
                .expect("Classification failed with F16 model");

            assert_eq!(result.label, expected::DISTILROBERTA_EMOTION_HAPPY_LABEL);
            assert!(
                result.score > 0.97,
                "F16 model score was unexpectedly low: {}",
                result.score
            );

            // F16 should be close to F32 reference
            let diff = (result.score - expected::DISTILROBERTA_EMOTION_HAPPY_SCORE).abs();
            assert!(
                diff < 0.01,
                "F16 score {} differs too much from F32 reference {}",
                result.score,
                expected::DISTILROBERTA_EMOTION_HAPPY_SCORE
            );
        }
    }

    
    // Accessor Tests
    

    mod accessor_tests {
        use super::*;

        #[tokio::test]
        async fn test_distilbert_accessors() {
            let classifier = Classifier::new("distilbert-sentiment")
                .await
                .expect("Failed to load classifier");

            assert_eq!(classifier.model_name(), "distilbert-sentiment");
            assert_eq!(classifier.num_labels(), 2);
            assert!(classifier.max_seq_length() > 0);
            assert_eq!(classifier.mode(), ClassificationMode::SingleLabel);

            let labels = classifier.labels().expect("Should have labels");
            assert_eq!(labels, vec!["NEGATIVE", "POSITIVE"]);
        }

        #[tokio::test]
        async fn test_roberta_sentiment_accessors() {
            let classifier = Classifier::new("roberta-sentiment")
                .await
                .expect("Failed to load classifier");

            assert_eq!(classifier.model_name(), "roberta-sentiment");
            assert_eq!(classifier.num_labels(), 3);

            let labels = classifier.labels().expect("Should have labels");
            assert_eq!(labels, vec!["negative", "neutral", "positive"]);
        }

        #[tokio::test]
        async fn test_distilroberta_emotion_accessors() {
            let classifier = Classifier::new("distilroberta-emotion")
                .await
                .expect("Failed to load classifier");

            assert_eq!(classifier.model_name(), "distilroberta-emotion");
            assert_eq!(classifier.num_labels(), 7);
            assert_eq!(classifier.architecture(), "roberta");
            assert_eq!(classifier.max_seq_length(), 514);

            let labels = classifier.labels().expect("Should have labels");
            assert_eq!(
                labels,
                vec![
                    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
                ]
            );
        }
    }

    
    // Convenience Function Tests
    

    mod convenience_function_tests {
        use super::*;

        #[tokio::test]
        async fn test_classify_function() {
            let result = classify("distilroberta-emotion", expected::INPUT_HAPPY)
                .await
                .expect("Classify function failed");

            assert_eq!(result.label, expected::DISTILROBERTA_EMOTION_HAPPY_LABEL);
        }
    }

    
    // Override Tests
    

    mod override_tests {
        use super::*;

        #[tokio::test]
        async fn test_top_k_override() {
            let classifier = Classifier::new("distilroberta-emotion")
                .await
                .expect("Failed to load classifier");

            let override_config = ClassificationOverrides {
                top_k: Some(3),
                ..Default::default()
            };

            let result = classifier
                .classify_with_config(expected::INPUT_SAD, &override_config)
                .await
                .expect("Classification with top_k failed");

            assert_eq!(result.all_scores.len(), 3);
            assert_eq!(result.all_scores[0].0, "sadness");
        }

        #[tokio::test]
        async fn test_threshold_override() {
            let classifier = Classifier::new("distilroberta-emotion")
                .await
                .expect("Failed to load classifier");

            let override_config = ClassificationOverrides {
                threshold: Some(0.05),
                ..Default::default()
            };

            let result = classifier
                .classify_with_config(expected::INPUT_SAD, &override_config)
                .await
                .expect("Classification with threshold failed");

            assert_eq!(result.all_scores.len(), 1);
            assert_eq!(result.all_scores[0].0, "sadness");
        }
    }

    mod custom_labels_tests {
        use super::*;

        #[tokio::test]
        async fn test_custom_labels() {
            let custom_labels = vec!["A", "B", "C", "D", "E", "F", "G"];
            let classifier = Classifier::builder("distilroberta-emotion")
                .labels(custom_labels.clone())
                .build()
                .await
                .expect("Failed to build classifier with custom labels");

            assert!(classifier.has_custom_labels());

            let result = classifier
                .classify(expected::INPUT_HAPPY)
                .await
                .expect("Classification with custom labels failed");

            // D maps to index 3 (joy)
            assert_eq!(result.label, "D");
            assert!(approx_eq(
                result.score,
                expected::DISTILROBERTA_EMOTION_HAPPY_SCORE,
                1e-5
            ));
        }
    }

    
    // Loading Tests
    

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
                    .classify(expected::INPUT_POSITIVE)
                    .await
                    .expect("Classification failed");

                assert_eq!(result.label, expected::DISTILBERT_SENTIMENT_POSITIVE_LABEL);
            }
        }
    }
}
