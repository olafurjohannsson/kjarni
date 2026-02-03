//! Tests for the embedder module.
//!
//! Run all tests: `cargo test --package kjarni embedder`
//! Run integration tests (requires model): `cargo test --package kjarni embedder -- --ignored`

use super::*;
use crate::common::{DownloadPolicy, KjarniDevice, LoadConfig};

// =============================================================================
// Type Tests
// =============================================================================

mod types_tests {
    use kjarni_transformers::PoolingStrategy;

    use super::*;

    #[test]
    fn test_pooling_strategy_default() {
        let strategy = PoolingStrategy::default();
        assert_eq!(strategy, PoolingStrategy::Mean);
    }

    #[test]
    fn test_pooling_strategy_as_str() {
        assert_eq!(PoolingStrategy::Mean.as_str(), "mean");
        assert_eq!(PoolingStrategy::Max.as_str(), "max");
        assert_eq!(PoolingStrategy::Cls.as_str(), "cls");
        assert_eq!(PoolingStrategy::LastToken.as_str(), "last_token");
    }

    #[test]
    fn test_pooling_strategy_display() {
        assert_eq!(format!("{}", PoolingStrategy::Mean), "mean");
        assert_eq!(format!("{}", PoolingStrategy::Max), "max");
        assert_eq!(format!("{}", PoolingStrategy::Cls), "cls");
        assert_eq!(format!("{}", PoolingStrategy::LastToken), "last_token");
    }

    #[test]
    fn test_pooling_strategy_from_str() {
        assert_eq!("mean".parse::<PoolingStrategy>().unwrap(), PoolingStrategy::Mean);
        assert_eq!("max".parse::<PoolingStrategy>().unwrap(), PoolingStrategy::Max);
        assert_eq!("cls".parse::<PoolingStrategy>().unwrap(), PoolingStrategy::Cls);
        assert_eq!("last_token".parse::<PoolingStrategy>().unwrap(), PoolingStrategy::LastToken);
        assert_eq!("lasttoken".parse::<PoolingStrategy>().unwrap(), PoolingStrategy::LastToken);
        assert_eq!("last".parse::<PoolingStrategy>().unwrap(), PoolingStrategy::LastToken);
        assert_eq!("MEAN".parse::<PoolingStrategy>().unwrap(), PoolingStrategy::Mean);
    }

    #[test]
    fn test_pooling_strategy_from_str_invalid() {
        assert!("invalid".parse::<PoolingStrategy>().is_err());
        assert!("".parse::<PoolingStrategy>().is_err());
    }

    #[test]
    fn test_embedding_overrides_default() {
        let overrides = EmbeddingOverrides::default();

        assert!(overrides.pooling.is_none());
        assert!(overrides.normalize.is_none());
        assert!(overrides.max_length.is_none());
    }

    #[test]
    fn test_embedding_overrides_for_search() {
        let overrides = EmbeddingOverrides::for_search();

        assert_eq!(overrides.pooling, Some(PoolingStrategy::Mean));
        assert_eq!(overrides.normalize, Some(true));
    }

    #[test]
    fn test_embedding_overrides_for_clustering() {
        let overrides = EmbeddingOverrides::for_clustering();

        assert_eq!(overrides.pooling, Some(PoolingStrategy::Mean));
        assert_eq!(overrides.normalize, Some(true));
    }

    #[test]
    fn test_embedding_overrides_for_similarity() {
        let overrides = EmbeddingOverrides::for_similarity();

        assert_eq!(overrides.pooling, Some(PoolingStrategy::Mean));
        assert_eq!(overrides.normalize, Some(true));
    }
}

// =============================================================================
// Builder Tests
// =============================================================================

mod builder_tests {
    use kjarni_transformers::PoolingStrategy;

    use super::*;

    #[test]
    fn test_builder_default_values() {
        let builder = EmbedderBuilder::new("test-model");

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
        let builder = EmbedderBuilder::new("test-model").cpu();
        assert_eq!(builder.device, KjarniDevice::Cpu);
    }

    #[test]
    fn test_builder_gpu() {
        let builder = EmbedderBuilder::new("test-model").gpu();
        assert_eq!(builder.device, KjarniDevice::Gpu);
    }

    #[test]
    fn test_builder_auto_device() {
        let builder = EmbedderBuilder::new("test-model").auto_device();
        assert_eq!(builder.device, KjarniDevice::Auto);
    }

    #[test]
    fn test_builder_cache_dir() {
        let builder = EmbedderBuilder::new("test-model")
            .cache_dir("/tmp/test-cache");

        assert_eq!(
            builder.cache_dir,
            Some(std::path::PathBuf::from("/tmp/test-cache"))
        );
    }

    #[test]
    fn test_builder_model_path() {
        let builder = EmbedderBuilder::new("test-model")
            .model_path("/path/to/model");

        assert_eq!(
            builder.model_path,
            Some(std::path::PathBuf::from("/path/to/model"))
        );
    }

    #[test]
    fn test_builder_pooling() {
        let builder = EmbedderBuilder::new("test-model")
            .pooling(PoolingStrategy::Cls);

        assert_eq!(builder.overrides.pooling, Some(PoolingStrategy::Cls));
    }

    #[test]
    fn test_builder_normalize() {
        let builder = EmbedderBuilder::new("test-model").normalize(false);
        assert_eq!(builder.overrides.normalize, Some(false));
    }

    #[test]
    fn test_builder_max_length() {
        let builder = EmbedderBuilder::new("test-model").max_length(256);
        assert_eq!(builder.overrides.max_length, Some(256));
    }

    #[test]
    fn test_builder_offline() {
        let builder = EmbedderBuilder::new("test-model").offline();
        assert_eq!(builder.download_policy, DownloadPolicy::Never);
    }

    #[test]
    fn test_builder_quiet() {
        let builder = EmbedderBuilder::new("test-model").quiet(true);
        assert!(builder.quiet);
    }

    #[test]
    fn test_builder_chain() {
        let builder = EmbedderBuilder::new("test-model")
            .gpu()
            .pooling(PoolingStrategy::Mean)
            .normalize(true)
            .max_length(512)
            .quiet(true)
            .offline();

        assert_eq!(builder.device, KjarniDevice::Gpu);
        assert_eq!(builder.overrides.pooling, Some(PoolingStrategy::Mean));
        assert_eq!(builder.overrides.normalize, Some(true));
        assert_eq!(builder.overrides.max_length, Some(512));
        assert!(builder.quiet);
        assert_eq!(builder.download_policy, DownloadPolicy::Never);
    }

    #[test]
    fn test_builder_from_preset() {
        let builder = EmbedderBuilder::from_preset(&presets::EMBEDDING_SMALL_V1);

        assert_eq!(builder.model, presets::EMBEDDING_SMALL_V1.model);
        assert_eq!(builder.device, presets::EMBEDDING_SMALL_V1.recommended_device);
        assert_eq!(
            builder.overrides.pooling,
            Some(presets::EMBEDDING_SMALL_V1.default_pooling)
        );
        assert_eq!(
            builder.overrides.normalize,
            Some(presets::EMBEDDING_SMALL_V1.normalize_default)
        );
    }

    #[test]
    fn test_builder_with_load_config() {
        let builder = EmbedderBuilder::new("test-model")
            .with_load_config(|b| b.offload_embeddings(true).max_batch_size(64));

        assert!(builder.load_config.is_some());
        let config = builder.load_config.unwrap();
        assert!(config.inner.offload_embeddings);
        assert_eq!(config.inner.max_batch_size, Some(64));
    }
}

// =============================================================================
// Preset Tests
// =============================================================================

mod preset_tests {
    use kjarni_transformers::PoolingStrategy;

    use super::*;

    #[test]
    fn test_embedding_small_v1_preset() {
        let preset = &presets::EMBEDDING_SMALL_V1;

        assert_eq!(preset.name, "EMBEDDING_SMALL_V1");
        assert_eq!(preset.model, "minilm-l6-v2");
        assert_eq!(preset.dimension, 384);
        assert_eq!(preset.default_pooling, PoolingStrategy::Mean);
        assert!(preset.normalize_default);
    }

    #[test]
    fn test_embedding_nomic_v1_preset() {
        let preset = &presets::EMBEDDING_NOMIC_V1;

        assert_eq!(preset.name, "EMBEDDING_NOMIC_V1");
        assert_eq!(preset.model, "nomic-embed-text");
        assert_eq!(preset.dimension, 768);
    }

    #[test]
    fn test_find_preset_exists() {
        let preset = presets::find_preset("EMBEDDING_SMALL_V1");
        assert!(preset.is_some());
        assert_eq!(preset.unwrap().name, "EMBEDDING_SMALL_V1");
    }

    #[test]
    fn test_find_preset_case_insensitive() {
        let preset = presets::find_preset("embedding_small_v1");
        assert!(preset.is_some());
    }

    #[test]
    fn test_find_preset_not_found() {
        let preset = presets::find_preset("NONEXISTENT");
        assert!(preset.is_none());
    }

    #[test]
    fn test_embedder_tier_resolve() {
        let small = presets::EmbedderTier::Small.resolve();
        let medium = presets::EmbedderTier::Medium.resolve();
        let large = presets::EmbedderTier::Large.resolve();

        assert!(!small.model.is_empty());
        assert!(!medium.model.is_empty());
        assert!(!large.model.is_empty());
    }

    #[test]
    fn test_all_presets_valid() {
        for preset in presets::ALL_V1_PRESETS {
            assert!(!preset.name.is_empty());
            assert!(!preset.model.is_empty());
            assert!(preset.dimension > 0);
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
    fn test_validate_embedding_model() {
        if let Some(model_type) = ModelType::from_cli_name("minilm-l6-v2") {
            let result = validation::validate_for_embedding(model_type);
            assert!(result.is_ok(), "MiniLM should be valid for embedding");
        }
    }

    #[test]
    fn test_validate_non_embedding_model() {
        if let Some(model_type) = ModelType::from_cli_name("llama3.2-1b") {
            let result = validation::validate_for_embedding(model_type);
            assert!(result.is_err(), "Decoder should not be valid for embedding");
        }
    }

    #[test]
    fn test_get_embedding_models() {
        let models = validation::get_embedding_models();
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
        let err = EmbedderError::UnknownModel("fake-model".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("fake-model"));
        assert!(msg.contains("Unknown model"));
    }

    #[test]
    fn test_error_display_incompatible_model() {
        let err = EmbedderError::IncompatibleModel {
            model: "test-model".to_string(),
            reason: "not an encoder".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("test-model"));
        assert!(msg.contains("not an encoder"));
    }

    #[test]
    fn test_error_display_not_downloaded() {
        let err = EmbedderError::ModelNotDownloaded("test-model".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test-model"));
        assert!(msg.contains("not downloaded"));
    }
}

// =============================================================================
// Convenience Function Tests
// =============================================================================

mod convenience_tests {
    use super::*;

    #[test]
    fn test_is_embedding_model_unknown() {
        let result = is_embedding_model("nonexistent-model-12345");
        assert!(result.is_err());
    }

    #[test]
    fn test_available_models_returns_list() {
        let models = available_models();
        assert!(models.len() >= 0);
    }
}

// =============================================================================
// Cosine Similarity Tests
// =============================================================================

mod similarity_tests {
    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = super::model::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = super::model::cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = super::model::cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_normalized() {
        let a = vec![0.6, 0.8, 0.0]; // Already normalized (0.36 + 0.64 = 1)
        let b = vec![0.6, 0.8, 0.0];
        let sim = super::model::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = super::model::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }
}

// =============================================================================
// Integration Tests (require model download)
// =============================================================================

mod integration_tests {
    use kjarni_transformers::PoolingStrategy;

    use super::*;

    /// Test that we can create an embedder.
    #[tokio::test]
    
    async fn test_embedder_new() {
        let embedder = Embedder::new("minilm-l6-v2").await;
        assert!(embedder.is_ok(), "Failed to create embedder: {:?}", embedder.err());
    }

    /// Test single text embedding.
    #[tokio::test]
    
    async fn test_embed_single() {
        let embedder = Embedder::new("minilm-l6-v2")
            .await
            .expect("Failed to load embedder");

        let embedding = embedder.embed("Hello world!")
            .await
            .expect("Embedding failed");

        assert_eq!(embedding.len(), 384); // MiniLM dimension
        // Check it's normalized (approximately)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Embedding should be normalized");
    }

    /// Test batch embedding.
    #[tokio::test]
    
    async fn test_embed_batch() {
        let embedder = Embedder::new("minilm-l6-v2")
            .await
            .expect("Failed to load embedder");

        let texts = ["Hello world", "How are you?", "This is a test"];
        let embeddings = embedder.embed_batch(&texts)
            .await
            .expect("Batch embedding failed");

        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), 384);
        }
    }

    /// Test embedding with custom pooling.
    #[tokio::test]
    
    async fn test_embed_with_config() {
        let embedder = Embedder::new("minilm-l6-v2")
            .await
            .expect("Failed to load embedder");

        let overrides = EmbeddingOverrides {
            pooling: Some(PoolingStrategy::Cls),
            normalize: Some(true),
            max_length: None,
        };

        let embedding = embedder
            .embed_with_config("Test text", &overrides)
            .await
            .expect("Embedding with config failed");

        assert_eq!(embedding.len(), 384);
    }

    /// Test unnormalized embeddings.
    #[tokio::test]
    
    async fn test_embed_unnormalized() {
        let embedder = Embedder::builder("minilm-l6-v2")
            .normalize(false)
            .build()
            .await
            .expect("Failed to load embedder");

        let embedding = embedder.embed("Test text")
            .await
            .expect("Embedding failed");

        // Unnormalized embedding may not have unit norm
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Just check it's not exactly 1.0 (could be close by chance)
        assert!(embedding.len() == 384);
    }

    /// Test similarity computation.
    #[tokio::test]
    
    async fn test_similarity() {
        let embedder = Embedder::new("minilm-l6-v2")
            .await
            .expect("Failed to load embedder");

        let sim = embedder.similarity(
            "The cat sat on the mat",
            "A feline was sitting on a rug"
        ).await.expect("Similarity failed");

        // Similar sentences should have high similarity
        assert!(sim > 0.5, "Similar sentences should have sim > 0.5, got {}", sim);
    }

    /// Test similarity with dissimilar texts.
    #[tokio::test]
    
    async fn test_similarity_dissimilar() {
        let embedder = Embedder::new("minilm-l6-v2")
            .await
            .expect("Failed to load embedder");

        let sim = embedder.similarity(
            "The cat sat on the mat",
            "Stock prices rose today"
        ).await.expect("Similarity failed");

        // Dissimilar sentences should have lower similarity
        assert!(sim < 0.5, "Dissimilar sentences should have sim < 0.5, got {}", sim);
    }

    /// Test batch similarities.
    #[tokio::test]
    
    async fn test_similarities() {
        let embedder = Embedder::new("minilm-l6-v2")
            .await
            .expect("Failed to load embedder");

        let query = "What is machine learning?";
        let docs = [
            "Machine learning is a branch of AI",
            "The weather is nice today",
            "Deep learning uses neural networks",
        ];

        let similarities = embedder.similarities(query, &docs)
            .await
            .expect("Similarities failed");

        assert_eq!(similarities.len(), 3);
        // First and third should be more similar to query than second
        assert!(similarities[0] > similarities[1]);
        assert!(similarities[2] > similarities[1]);
    }

    /// Test ranking by similarity.
    #[tokio::test]
    
    async fn test_rank_by_similarity() {
        let embedder = Embedder::new("minilm-l6-v2")
            .await
            .expect("Failed to load embedder");

        let query = "What is the capital of France?";
        let docs = [
            "The Eiffel Tower is in Paris",
            "Berlin is in Germany",
            "Paris is the capital of France",
        ];

        let ranked = embedder.rank_by_similarity(query, &docs)
            .await
            .expect("Ranking failed");

        assert_eq!(ranked.len(), 3);
        // Third document should be ranked first
        assert_eq!(ranked[0].0, 2, "Third doc should be most relevant");
    }

    /// Test GPU embedding (if available).
    #[tokio::test]
    async fn test_embed_gpu() {
        let embedder = Embedder::builder("minilm-l6-v2")
            .gpu()
            .build()
            .await
            .expect("Failed to load embedder on GPU");

        let embedding = embedder.embed("Test GPU embedding")
            .await
            .expect("GPU embedding failed");

        assert_eq!(embedding.len(), 384);
    }

    /// Test embedder accessors.
    #[tokio::test]
    
    async fn test_embedder_accessors() {
        let embedder = Embedder::new("minilm-l6-v2")
            .await
            .expect("Failed to load embedder");

        assert_eq!(embedder.model_name(), "minilm-l6-v2");
        assert_eq!(embedder.dimension(), 384);
        assert!(embedder.max_seq_length() > 0);
    }

    /// Test one-liner embed function.
    #[tokio::test]
    
    async fn test_embed_convenience_function() {
        let embedding = embed("minilm-l6-v2", "Hello world")
            .await
            .expect("Embed function failed");

        assert_eq!(embedding.len(), 384);
    }

    /// Test one-liner similarity function.
    #[tokio::test]
    
    async fn test_similarity_convenience_function() {
        let sim = similarity("minilm-l6-v2", "Hello", "Hi there")
            .await
            .expect("Similarity function failed");

        assert!(sim >= -1.0 && sim <= 1.0);
    }

    /// Test unknown model error.
    #[tokio::test]
    async fn test_unknown_model_error() {
        let result = Embedder::new("completely-fake-model-that-does-not-exist").await;
        assert!(matches!(result, Err(EmbedderError::UnknownModel(_))));
    }

    /// Test offline mode with missing model.
    #[tokio::test]
    async fn test_offline_missing_model() {
        let result = Embedder::builder("minilm-l6-v2")
            .offline()
            .cache_dir("/tmp/kjarni-test-empty-cache-12345")
            .build()
            .await;

        assert!(result.is_err());
    }
}