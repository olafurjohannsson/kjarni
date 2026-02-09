//! Comprehensive tests for the Summarizer module.

use super::*;
use futures::StreamExt;

// Expected Outputs from PyTorch Reference

mod expected {
    // =========================================================================
    // Test Inputs
    // =========================================================================
    pub const INPUT_EIFFEL_TOWER: &str = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. It was the first structure to reach a height of 300 metres. Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.";
    pub const INPUT_AMAZON_RAINFOREST: &str = "The Amazon rainforest produces about 20 percent of the world's oxygen. It is the largest tropical rainforest in the world, covering over 5.5 million square kilometers. The forest is home to approximately 10 percent of all species on Earth.";
    pub const INPUT_PYTHON_LANGUAGE: &str = "Python is a high-level programming language known for its simple syntax and readability. It was created by Guido van Rossum and first released in 1991. Python is widely used in web development, data science, and artificial intelligence.";

    // =========================================================================
    // bart-large-cnn
    // =========================================================================
    pub const BART_LARGE_CNN_EIFFEL_TOWER_GREEDY: &str = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.";
    pub const BART_LARGE_CNN_EIFFEL_TOWER_BEAM: &str = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. Excluding transmitters, it is the second tallest free-standing structure in France after the Millau Viaduct.";
    pub const BART_LARGE_CNN_AMAZON_RAINFOREST_GREEDY: &str = "The Amazon rainforest produces about 20 percent of the world's oxygen. It is the largest tropical rainforest in the world, covering over 5.5 million square kilometers. The forest is home to approximately 10 percent of all species on Earth.";
    pub const BART_LARGE_CNN_AMAZON_RAINFOREST_BEAM: &str = "The Amazon rainforest produces about 20 percent of the world's oxygen. It is the largest tropical rainforest in the world, covering over 5.5 million square kilometers.";
    pub const BART_LARGE_CNN_PYTHON_LANGUAGE_GREEDY: &str = "Python is a high-level programming language. It was created by Guido van Rossum and first released in 1991.";
    pub const BART_LARGE_CNN_PYTHON_LANGUAGE_BEAM: &str = "Python is a high-level programming language. It was created by Guido van Rossum and first released in 1991. Python is widely used in web development, data science, and artificial intelligence.";

    // =========================================================================
    // distilbart-cnn
    // =========================================================================
    pub const DISTILBART_CNN_EIFFEL_TOWER_GREEDY: &str = " The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building . Excluding transmitters, it is the second tallest free-standing structure in France after the Millau Viaduct .";
    pub const DISTILBART_CNN_EIFFEL_TOWER_BEAM: &str = " The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building . Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct .";
    pub const DISTILBART_CNN_AMAZON_RAINFOREST_GREEDY: &str = " Amazon rainforest produces 20 percent of the world's oxygen . It is the largest tropical rainforest in the world, covering over 5.5 million square kilometers . The forest is home to approximately 10 percent of all species on Earth .";
    pub const DISTILBART_CNN_AMAZON_RAINFOREST_BEAM: &str = " The Amazon rainforest produces 20 percent of the world's oxygen . It is the largest tropical rainforest in the world, covering over 5.5 million square kilometers . The forest is home to approximately 10 percent of all species on Earth .";
    pub const DISTILBART_CNN_PYTHON_LANGUAGE_GREEDY: &str = " Python is widely used in web development, data science, and artificial intelligence . It was created by Guido van Rossum and first released in 1991 .";
    pub const DISTILBART_CNN_PYTHON_LANGUAGE_BEAM: &str = " Python is widely used in web development, data science, and artificial intelligence . It was created by Guido van Rossum and first released in 1991 .";
}

// Unit Tests - Validation

mod validation_tests {
    use crate::summarizer::SummarizerError;
    use crate::summarizer::validation::*;
    use kjarni_transformers::models::ModelType;

    #[test]
    fn test_validate_bart_models_accepted() {
        if let Some(bart) = ModelType::from_cli_name("bart-large-cnn") {
            assert!(validate_for_summarization(bart).is_ok());
        }

        if let Some(distilbart) = ModelType::from_cli_name("distilbart-cnn") {
            assert!(validate_for_summarization(distilbart).is_ok());
        }
    }

    #[test]
    fn test_validate_t5_models_accepted() {
        let t5_models = ["flan-t5-base", "flan-t5-small"];

        for name in t5_models {
            if let Some(model_type) = ModelType::from_cli_name(name) {
                let result = validate_for_summarization(model_type);
                assert!(
                    result.is_ok(),
                    "T5 model {} should be valid for summarization",
                    name
                );
            }
        }
    }

    #[test]
    fn test_get_summarization_models() {
        let models = get_summarization_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"bart-large-cnn"));
        assert!(models.contains(&"distilbart-cnn"));
        assert!(models.contains(&"flan-t5-base"));
        assert!(models.contains(&"flan-t5-large"));
    }
}

// Unit Tests - Presets

mod preset_tests {
    use crate::summarizer::presets::*;

    #[test]
    fn test_preset_fast_values() {
        assert_eq!(SUMMARIZER_FAST_V1.model, "distilbart-cnn");
        assert_eq!(SUMMARIZER_FAST_V1.architecture, "bart");
        assert!(SUMMARIZER_FAST_V1.memory_mb >= 500);
        assert!(SUMMARIZER_FAST_V1.memory_mb <= 2000);
    }

    #[test]
    fn test_preset_quality_values() {
        assert_eq!(SUMMARIZER_QUALITY_V1.model, "bart-large-cnn");
        assert_eq!(SUMMARIZER_QUALITY_V1.architecture, "bart");
        assert!(SUMMARIZER_QUALITY_V1.memory_mb > SUMMARIZER_FAST_V1.memory_mb);
    }

    #[test]
    fn test_find_preset_case_insensitive() {
        assert!(find_preset("SUMMARIZER_FAST_V1").is_some());
        assert!(find_preset("summarizer_fast_v1").is_some());
        assert!(find_preset("Summarizer_Fast_V1").is_some());
        assert!(find_preset("nonexistent").is_none());
    }

    #[test]
    fn test_tier_resolution() {
        assert_eq!(SummarizerTier::Fast.resolve().model, "distilbart-cnn");
        assert_eq!(SummarizerTier::Quality.resolve().model, "bart-large-cnn");
    }
}

// Unit Tests - Builder

mod builder_tests {
    use crate::common::KjarniDevice;
    use crate::summarizer::SummarizerBuilder;

    #[test]
    fn test_builder_default_state() {
        let builder = SummarizerBuilder::new("bart-large-cnn");
        assert_eq!(builder.model, "bart-large-cnn");
        assert!(!builder.quiet);
        assert!(builder.overrides.is_empty());
    }

    #[test]
    fn test_builder_device_methods() {
        let cpu_builder = SummarizerBuilder::new("bart-large-cnn").cpu();
        assert!(matches!(cpu_builder.device, KjarniDevice::Cpu));

        let gpu_builder = SummarizerBuilder::new("bart-large-cnn").gpu();
        assert!(matches!(gpu_builder.device, KjarniDevice::Gpu));
    }

    #[test]
    fn test_builder_overrides_are_explicit() {
        let builder = SummarizerBuilder::new("bart-large-cnn")
            .num_beams(6)
            .max_length(256);

        assert_eq!(builder.overrides.num_beams, Some(6));
        assert_eq!(builder.overrides.max_length, Some(256));
        assert!(builder.overrides.length_penalty.is_none());
        assert!(builder.overrides.no_repeat_ngram_size.is_none());
    }

    #[test]
    fn test_builder_greedy_sets_num_beams_1() {
        let builder = SummarizerBuilder::new("bart-large-cnn").greedy();
        assert_eq!(builder.overrides.num_beams, Some(1));
    }
}

mod error_tests {
    use crate::summarizer::SummarizerError;

    #[test]
    fn test_error_incompatible_model_message() {
        let err = SummarizerError::IncompatibleModel {
            model: "flan-t5-base".to_string(),
            reason: "translation model".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("flan-t5-base"));
        assert!(msg.contains("translation model"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SummarizerError>();
    }
}
mod module_function_tests {
    use crate::summarizer::{available_models, is_summarization_model};

    #[test]
    fn test_available_models_contains_bart() {
        let models = available_models();
        assert!(models.contains(&"bart-large-cnn"));
        assert!(models.contains(&"distilbart-cnn"));
    }

    #[test]
    fn test_available_models_includes_t5() {
        let models = available_models();
        assert!(models.contains(&"flan-t5-base"));
        assert!(models.contains(&"flan-t5-large"));
    }

    #[test]
    fn test_is_summarization_model_valid() {
        assert!(is_summarization_model("bart-large-cnn").is_ok());
        assert!(is_summarization_model("distilbart-cnn").is_ok());
    }

    #[test]
    fn test_is_summarization_model_invalid() {
        assert!(is_summarization_model("not-a-model").is_err());
    }

    #[test]
    fn test_is_summarization_model_valid_t5() {
        assert!(is_summarization_model("flan-t5-base").is_ok());
        assert!(is_summarization_model("flan-t5-large").is_ok());
    }
}

#[cfg(test)]
mod bart_large_cnn_tests {
    use super::expected::*;
    use super::*;

    fn model_available() -> bool {
        let cache_dir = dirs::cache_dir().expect("no cache dir").join("kjarni");
        kjarni_transformers::models::ModelType::from_cli_name("bart-large-cnn")
            .map(|m| m.is_downloaded(&cache_dir))
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn test_greedy_eiffel_tower() {
        if !model_available() {
            eprintln!("Skipping: bart-large-cnn not downloaded");
            return;
        }

        let s = Summarizer::builder("bart-large-cnn")
            .greedy()
            .min_length(10)
            .max_length(60)
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = s.summarize(INPUT_EIFFEL_TOWER).await.unwrap();
        assert_eq!(result, BART_LARGE_CNN_EIFFEL_TOWER_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_amazon_rainforest() {
        if !model_available() {
            eprintln!("Skipping: bart-large-cnn not downloaded");
            return;
        }

        let s = Summarizer::builder("bart-large-cnn")
            .greedy()
            .min_length(10)
            .max_length(60)
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = s.summarize(INPUT_AMAZON_RAINFOREST).await.unwrap();
        assert_eq!(result, BART_LARGE_CNN_AMAZON_RAINFOREST_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_python_language() {
        if !model_available() {
            eprintln!("Skipping: bart-large-cnn not downloaded");
            return;
        }

        let s = Summarizer::builder("bart-large-cnn")
            .greedy()
            .min_length(10)
            .max_length(60)
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = s.summarize(INPUT_PYTHON_LANGUAGE).await.unwrap();
        assert_eq!(result, BART_LARGE_CNN_PYTHON_LANGUAGE_GREEDY);
    }

    #[tokio::test]
    async fn test_streaming_matches_non_streaming() {
        if !model_available() {
            eprintln!("Skipping: bart-large-cnn not downloaded");
            return;
        }

        let s = Summarizer::builder("bart-large-cnn")
            .greedy()
            .min_length(10)
            .max_length(60)
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let direct = s.summarize(INPUT_PYTHON_LANGUAGE).await.unwrap();

        let mut stream = s.stream(INPUT_PYTHON_LANGUAGE).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(result) = stream.next().await {
            chunks.push(result.unwrap());
        }
        let streamed: String = chunks.into_iter().collect();

        assert_eq!(direct, streamed);
        assert_eq!(direct, BART_LARGE_CNN_PYTHON_LANGUAGE_GREEDY);
    }
}

mod bart_large_golden_values_test {
    use super::*;

    #[tokio::test]
    async fn golden_summarize_ai_paragraph() {
        let summarizer = Summarizer::builder("bart-large-cnn")
            .quiet()
            .greedy()
            .min_length(30)
            .max_length(142)
            .build()
            .await
            .unwrap();

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

        let summary = summarizer.summarize(input).await.unwrap();
        assert_eq!(
            summary.trim(),
            expected.trim(),
            "AI paragraph summary mismatch"
        );
    }

    #[tokio::test]
    async fn golden_summarize_news_style() {
        let summarizer = Summarizer::builder("bart-large-cnn")
            .quiet()
            .greedy()
            .min_length(20)
            .max_length(80)
            .build()
            .await
            .unwrap();

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

        let summary = summarizer.summarize(input).await.unwrap();
        assert_eq!(summary.trim(), expected.trim(), "News summary mismatch");
    }
}

// Integration Tests - distilbart-cnn Output Verification

#[cfg(test)]
mod distilbart_cnn_tests {
    use super::expected::*;
    use super::*;

    fn model_available() -> bool {
        let cache_dir = dirs::cache_dir().expect("no cache dir").join("kjarni");
        kjarni_transformers::models::ModelType::from_cli_name("distilbart-cnn")
            .map(|m| m.is_downloaded(&cache_dir))
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn test_greedy_eiffel_tower() {
        if !model_available() {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let s = Summarizer::builder("distilbart-cnn")
            .greedy()
            .min_length(10)
            .max_length(60)
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = s.summarize(INPUT_EIFFEL_TOWER).await.unwrap();
        assert_eq!(result, DISTILBART_CNN_EIFFEL_TOWER_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_amazon_rainforest() {
        if !model_available() {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let s = Summarizer::builder("distilbart-cnn")
            .greedy()
            .min_length(10)
            .max_length(60)
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = s.summarize(INPUT_AMAZON_RAINFOREST).await.unwrap();
        assert_eq!(result, DISTILBART_CNN_AMAZON_RAINFOREST_GREEDY);
    }

    #[tokio::test]
    async fn test_greedy_python_language() {
        if !model_available() {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let s = Summarizer::builder("distilbart-cnn")
            .greedy()
            .min_length(10)
            .max_length(60)
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let result = s.summarize(INPUT_PYTHON_LANGUAGE).await.unwrap();
        assert_eq!(result, DISTILBART_CNN_PYTHON_LANGUAGE_GREEDY);
    }

    #[tokio::test]
    async fn test_streaming_matches_non_streaming() {
        if !model_available() {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let s = Summarizer::builder("distilbart-cnn")
            .greedy()
            .min_length(10)
            .max_length(60)
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        let direct = s.summarize(INPUT_PYTHON_LANGUAGE).await.unwrap();

        let mut stream = s.stream(INPUT_PYTHON_LANGUAGE).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(result) = stream.next().await {
            chunks.push(result.unwrap());
        }
        let streamed: String = chunks.into_iter().collect();

        assert_eq!(direct, streamed);
        assert_eq!(direct, DISTILBART_CNN_PYTHON_LANGUAGE_GREEDY);
    }
}

// Integration Tests - Error Handling

#[cfg(test)]
mod error_handling_tests {
    use crate::seq2seq::Seq2SeqOverrides;

    use super::*;

    #[tokio::test]
    async fn test_unknown_model_error() {
        let result = Summarizer::new("not-a-real-model-xyz").await;
        match result {
            Err(SummarizerError::UnknownModel(ref m)) => assert_eq!(m, "not-a-real-model-xyz"),
            other => panic!("Expected UnknownModel error, got {:?}", other),
        }
    }
    #[tokio::test]
    async fn test_debug_t5_token_generation() {
        use crate::seq2seq::Seq2SeqGenerator;
        use futures::StreamExt;

        let generator = Seq2SeqGenerator::new("flan-t5-base")
            .await
            .expect("Should load");

        let prompt = "summarize: The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889 as the entrance arch for the 1889 World's Fair.";

        let overrides = Seq2SeqOverrides {
            max_length: Some(64),
            num_beams: Some(1),
            do_sample: Some(false),
            ..Default::default()
        };

        // Stream to see each token
        let stream = generator.stream_with_config(prompt, overrides).await;

        match stream {
            Ok(mut s) => {
                let mut tokens = Vec::new();
                while let Some(result) = s.next().await {
                    match result {
                        Ok(token) => {
                            println!("Token: {:?}", token);
                            tokens.push(token);
                        }
                        Err(e) => println!("Error: {:?}", e),
                    }
                }
                println!("Total tokens: {}", tokens.len());
            }
            Err(e) => println!("Stream error: {:?}", e),
        }
    }
    #[tokio::test]
    async fn test_debug_t5_generation_config() {
        let summarizer = Summarizer::new("flan-t5-base").await.expect("Should load");

        let text = "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. \
                It was constructed from 1887 to 1889 as the entrance arch for the 1889 World's Fair.";

        let overrides = Seq2SeqOverrides {
            max_length: Some(64),
            num_beams: Some(1),
            do_sample: Some(false), // Add this explicitly
            ..Default::default()
        };

        println!("Overrides: {:?}", overrides);
        println!("is_empty: {}", overrides.is_empty());

        let summary = summarizer
            .summarize_with_config(text, &overrides)
            .await
            .expect("Should summarize");

        println!("Summary: {}", summary);
    }
    #[tokio::test]
    async fn test_t5_model_can_summarize() {
        // PyTorch reference: greedy decoding with max_new_tokens=64
        const EXPECTED_SUMMARY: &str = "Eiffel Tower";

        let summarizer = Summarizer::new("flan-t5-base")
            .await
            .expect("T5 should load for summarization");

        let text = "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. \
                It was constructed from 1887 to 1889 as the entrance arch for the 1889 World's Fair.";

        // Use explicit greedy config matching PyTorch
        let overrides = Seq2SeqOverrides {
            max_length: Some(64),
            num_beams: Some(1),
            ..Default::default()
        };

        let summary = summarizer
            .summarize_with_config(text, &overrides)
            .await
            .expect("T5 should be able to summarize");

        assert_eq!(
            summary, EXPECTED_SUMMARY,
            "Summary mismatch: expected '{}', got '{}'",
            EXPECTED_SUMMARY, summary
        );
    }
}

// Integration Tests - Accessors

#[cfg(test)]
mod accessor_tests {
    use super::*;

    fn model_available() -> bool {
        let cache_dir = dirs::cache_dir().expect("no cache dir").join("kjarni");
        kjarni_transformers::models::ModelType::from_cli_name("distilbart-cnn")
            .map(|m| m.is_downloaded(&cache_dir))
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn test_summarizer_accessors() {
        if !model_available() {
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
        assert_eq!(s.model_type(), ModelType::DistilBartCnn);
        assert!(matches!(
            s.device(),
            kjarni_transformers::traits::Device::Cpu
        ));
        assert!(s.generator().context_size() >= 512);
        assert!(s.generator().vocab_size() >= 30000);
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
        kjarni_transformers::models::ModelType::from_cli_name("distilbart-cnn")
            .map(|m| m.is_downloaded(&cache_dir))
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn test_concurrent_summarizations_deterministic() {
        if !model_available() {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let s = Arc::new(
            Summarizer::builder("distilbart-cnn")
                .greedy()
                .min_length(10)
                .max_length(60)
                .cpu()
                .quiet()
                .build()
                .await
                .unwrap(),
        );

        let handles: Vec<_> = (0..3)
            .map(|_| {
                let summarizer = s.clone();
                let input = INPUT_PYTHON_LANGUAGE.to_string();
                tokio::spawn(async move { summarizer.summarize(&input).await })
            })
            .collect();

        for handle in handles {
            let result = handle
                .await
                .expect("Task panicked")
                .expect("Summarization failed");
            assert_eq!(result, DISTILBART_CNN_PYTHON_LANGUAGE_GREEDY);
        }
    }
}

// Integration Tests - Module-level Convenience Function

#[cfg(test)]
mod convenience_function_tests {
    use super::expected::*;
    use super::*;

    fn model_available() -> bool {
        let cache_dir = dirs::cache_dir().expect("no cache dir").join("kjarni");
        kjarni_transformers::models::ModelType::from_cli_name("distilbart-cnn")
            .map(|m| m.is_downloaded(&cache_dir))
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn test_summarize_function() {
        if !model_available() {
            eprintln!("Skipping: distilbart-cnn not downloaded");
            return;
        }

        let result = summarize("distilbart-cnn", INPUT_PYTHON_LANGUAGE)
            .await
            .unwrap();
        // Default may use beam search, so just verify non-empty and has content
        assert!(!result.is_empty());
        assert!(result.contains("Python"));
    }
}
