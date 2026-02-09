#[cfg(test)]
mod seq2seq_tests {
    use crate::seq2seq::{Seq2SeqError, generate};
    use crate::seq2seq::{
        Seq2SeqGenerator, Seq2SeqOverrides, available_models, is_seq2seq_model,
    };

    

    #[tokio::test]
    async fn test_t5_full_workflow() {
        let r#gen = Seq2SeqGenerator::new("flan-t5-base").await.unwrap();

        // Translation
        let de = r#gen
            .generate("translate English to German: Hello")
            .await
            .unwrap();
        println!("de: {}", de);
        assert!(!de.is_empty());

        // Summarization
        let summary = r#gen
            .generate("summarize: Long text here...")
            .await
            .unwrap();

        println!("summary: {}", summary);

        assert!(!summary.is_empty());
    }

    #[test]
    fn test_overrides_default() {
        let o = Seq2SeqOverrides::default();
        assert!(o.is_empty());
    }

    #[test]
    fn test_overrides_greedy() {
        let o = Seq2SeqOverrides::greedy();
        assert_eq!(o.num_beams, Some(1));
    }

    #[test]
    fn test_overrides_merge() {
        let base = Seq2SeqOverrides {
            num_beams: Some(4),
            max_length: Some(100),
            ..Default::default()
        };
        let override_ = Seq2SeqOverrides {
            num_beams: Some(6),
            min_length: Some(10),
            ..Default::default()
        };

        let merged = base.merge(&override_);

        assert_eq!(merged.num_beams, Some(6)); // overridden
        assert_eq!(merged.max_length, Some(100)); // from base
        assert_eq!(merged.min_length, Some(10)); // from override
    }

    #[test]
    fn test_available_models_not_empty() {
        let models = available_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"flan-t5-base"));
        assert!(models.contains(&"distilbart-cnn"));
    }

    #[test]
    fn test_is_seq2seq_model_valid() {
        assert!(is_seq2seq_model("flan-t5-base").is_ok());
        assert!(is_seq2seq_model("flan-t5-large").is_ok());
        assert!(is_seq2seq_model("distilbart-cnn").is_ok());
        assert!(is_seq2seq_model("bart-large-cnn").is_ok());
    }

    #[test]
    fn test_is_seq2seq_model_invalid() {
        // Decoder-only models
        assert!(is_seq2seq_model("llama3.2-1b-instruct").is_err());
        assert!(is_seq2seq_model("gpt2").is_err());

        // Encoder-only models
        assert!(is_seq2seq_model("minilm-l6-v2").is_err());

        // Unknown model
        assert!(is_seq2seq_model("not-a-real-model").is_err());
    }


    /// Helper to check if a model is downloaded
    fn model_available(model: &str) -> bool {
        let cache_dir = dirs::cache_dir().expect("no cache dir").join("kjarni");

        if let Some(model_type) = kjarni_transformers::models::ModelType::from_cli_name(model) {
            model_type.is_downloaded(&cache_dir)
        } else {
            false
        }
    }

    // =========================================================================
    // T5 Tests
    // =========================================================================

    #[tokio::test]
    async fn test_t5_translation() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping test: flan-t5-base not downloaded");
            return;
        }

        let generator = Seq2SeqGenerator::builder("flan-t5-base")
            .cpu()
            .quiet()
            .build()
            .await
            .expect("Failed to load model");

        let output = generator
            .generate("translate English to German: Hello, how are you?")
            .await
            .expect("Generation failed");

        assert!(!output.is_empty());
        // T5 should produce German text
        println!("T5 translation output: {}", output);
    }

    #[tokio::test]
    async fn test_t5_summarization() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping test: flan-t5-base not downloaded");
            return;
        }

        let generator = Seq2SeqGenerator::builder("flan-t5-base")
            .cpu()
            .quiet()
            .build()
            .await
            .expect("Failed to load model");

        let input = "summarize: The quick brown fox jumps over the lazy dog. \
                     This is a classic pangram that contains every letter of the alphabet. \
                     It has been used for decades to test typewriters and keyboards.";

        let output = generator.generate(input).await.expect("Generation failed");

        assert!(!output.is_empty());
        println!("T5 summary output: {}", output);
    }

    // #[tokio::test]
    // async fn test_t5_greedy_vs_beam() {
    //     if !model_available("flan-t5-base") {
    //         eprintln!("Skipping test: flan-t5-base not downloaded");
    //         return;
    //     }

    //     let generator = Seq2SeqGenerator::builder("flan-t5-base")
    //         .cpu()
    //         .quiet()
    //         .build()
    //         .await
    //         .expect("Failed to load model");

    //     let input = "translate English to French: Good morning";

    //     // Greedy (fast)
    //     let greedy_output = generator
    //         .generate_with_config(input, &Seq2SeqOverrides::greedy())
    //         .await
    //         .expect("Greedy generation failed");

    //     // Beam search (quality)
    //     let beam_output = generator
    //         .generate_with_config(input, &Seq2SeqOverrides::high_quality())
    //         .await
    //         .expect("Beam generation failed");

    //     assert!(!greedy_output.is_empty());
    //     assert!(!beam_output.is_empty());

    //     println!("Greedy: {}", greedy_output);
    //     println!("Beam:   {}", beam_output);
    // }

    // =========================================================================
    // BART Tests
    // =========================================================================

    #[tokio::test]
    async fn test_bart_summarization() {
        if !model_available("distilbart-cnn") {
            eprintln!("Skipping test: distilbart-cnn not downloaded");
            return;
        }

        let generator = Seq2SeqGenerator::builder("distilbart-cnn")
            .cpu()
            .quiet()
            .build()
            .await
            .expect("Failed to load model");

        let article = "The tower is 324 metres (1,063 ft) tall, about the same height as \
                       an 81-storey building, and the tallest structure in Paris. Its base \
                       is square, measuring 125 metres (410 ft) on each side. During its \
                       construction, the Eiffel Tower surpassed the Washington Monument to \
                       become the tallest man-made structure in the world.";

        let summary = generator
            .generate(article)
            .await
            .expect("Generation failed");

        assert!(!summary.is_empty());
        assert!(summary.len() < article.len()); // Summary should be shorter
        println!("BART summary: {}", summary);
    }

    #[tokio::test]
    async fn test_bart_with_length_constraints() {
        if !model_available("distilbart-cnn") {
            eprintln!("Skipping test: distilbart-cnn not downloaded");
            return;
        }

        let generator = Seq2SeqGenerator::builder("distilbart-cnn")
            .min_length(20)
            .max_length(50)
            .cpu()
            .quiet()
            .build()
            .await
            .expect("Failed to load model");

        let article = "Scientists have discovered a new species of deep-sea fish \
                       in the Mariana Trench. The fish, named Pseudoliparis swirei, \
                       was found at a depth of 8,000 meters.";

        let summary = generator
            .generate(article)
            .await
            .expect("Generation failed");

        assert!(!summary.is_empty());
        println!("BART constrained summary: {}", summary);
    }

    // =========================================================================
    // Streaming Tests
    // =========================================================================

    #[tokio::test]
    async fn test_streaming_generation() {
        use futures::StreamExt;

        if !model_available("flan-t5-base") {
            eprintln!("Skipping test: flan-t5-base not downloaded");
            return;
        }

        let generator = Seq2SeqGenerator::builder("flan-t5-base")
            .cpu()
            .quiet()
            .build()
            .await
            .expect("Failed to load model");

        let mut stream = generator
            .stream("translate English to Spanish: Hello world")
            .await
            .expect("Failed to create stream");

        let mut tokens = Vec::new();
        while let Some(result) = stream.next().await {
            let token = result.expect("Token error");
            tokens.push(token);
        }

        assert!(!tokens.is_empty());

        let full_text: String = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(!full_text.is_empty());
        println!("Streamed output: {}", full_text);
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[tokio::test]
    async fn test_unknown_model_error() {
        let result = Seq2SeqGenerator::new("not-a-real-model").await;
        assert!(matches!(result, Err(Seq2SeqError::UnknownModel(_))));
    }

    #[tokio::test]
    async fn test_incompatible_model_error() {
        // Try to use a decoder-only model
        let result = Seq2SeqGenerator::new("gpt2").await;
        assert!(matches!(
            result,
            Err(Seq2SeqError::IncompatibleModel { .. })
        ));
    }

    #[tokio::test]
    async fn test_offline_mode_not_downloaded() {
        // Use a model name that's valid but unlikely to be downloaded
        let result = Seq2SeqGenerator::builder("flan-t5-large")
            .offline()
            .build()
            .await;

        // Either succeeds (if downloaded) or fails with ModelNotDownloaded
        if let Err(e) = result {
            assert!(matches!(e, Seq2SeqError::ModelNotDownloaded(_)));
        }
    }

    // =========================================================================
    // Convenience Function Tests
    // =========================================================================

    #[tokio::test]
    async fn test_generate_convenience_function() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping test: flan-t5-base not downloaded");
            return;
        }

        let output = generate("flan-t5-base", "translate English to German: Thank you")
            .await
            .expect("Generation failed");

        assert!(!output.is_empty());
        println!("Convenience function output: {}", output);
    }

    // =========================================================================
    // Concurrent Usage Tests
    // =========================================================================

    #[tokio::test]
    async fn test_concurrent_generation() {
        if !model_available("flan-t5-base") {
            eprintln!("Skipping test: flan-t5-base not downloaded");
            return;
        }

        let generator = std::sync::Arc::new(
            Seq2SeqGenerator::builder("flan-t5-base")
                .cpu()
                .quiet()
                .build()
                .await
                .expect("Failed to load model"),
        );

        let inputs = vec![
            "translate English to German: Hello",
            "translate English to French: Goodbye",
            "translate English to Spanish: Thank you",
        ];

        let handles: Vec<_> = inputs
            .into_iter()
            .map(|input| {
                let r#gen = generator.clone();
                let input = input.to_string();
                tokio::spawn(async move { r#gen.generate(&input).await })
            })
            .collect();

        for handle in handles {
            let result = handle.await.expect("Task panicked");
            let output = result.expect("Generation failed");
            assert!(!output.is_empty());
            println!("Concurrent output: {}", output);
        }
    }

  
}
