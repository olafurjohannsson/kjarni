
use crate::models::bart::model::BartModel;
use anyhow::Result;
use kjarni_transformers::encoder_decoder::{GpuBackend, run_beam_search};
use kjarni_transformers::models::ModelType;
use kjarni_transformers::common::{DecodingStrategy, BeamSearchParams, GenerationConfig};
use kjarni_transformers::traits::Device;
use kjarni_transformers::{LanguageModel, WgpuContext};
use kjarni_transformers::encoder_decoder::CpuBackend;

#[tokio::test]
async fn test_beam_search_full_4_beams() -> Result<()> {
    

    let model = BartModel::from_registry(ModelType::DistilBartCnn, None, Device::Cpu, None).await?;
    let backend = CpuBackend;

    let context = WgpuContext::new().await?;
    let gpu_model = BartModel::from_registry(ModelType::DistilBartCnn, None, Device::Wgpu, Some(context.clone())).await?;
    let gpu_backend = GpuBackend::new(context)?;

    let text = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency. It enforces memory safety—meaning that all references point to valid memory—without using a garbage collector. To simultaneously enforce memory safety and prevent data races, its \"borrow checker\" tracks the object lifetime of all references in a program during compilation. Rust was influenced by languages like C++, Haskell, and Erlang.";

    let config = GenerationConfig {
        max_length: 142,
        min_length: 56,
        no_repeat_ngram_size: 3,
        repetition_penalty: 1.0,
        max_new_tokens: None,
        add_bos_token: false,
        strategy: DecodingStrategy::BeamSearch(BeamSearchParams {
            num_beams: 4,
            length_penalty: 2.0,
            early_stopping: true,
        }),
    };

    let result = run_beam_search(&model, &backend, text, &config).await?;
    let result_trimmed = result.trim();

    let gpu_result = run_beam_search(&gpu_model, &gpu_backend, text, &config).await?;
    let gpu_result_trimmed = gpu_result.trim();

    assert_eq!(
        result_trimmed, gpu_result_trimmed,
        "CPU and GPU beam search outputs do not match.\nCPU: '{}'\nGPU: '{}'",
        result_trimmed, gpu_result_trimmed
    );

    // The exact output from Hugging Face Transformers (Beam Search)
    let python_golden = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency . It enforces memory safety without using a garbage collector . To simultaneously enforce memory safety and prevent data races, its \"borrow checker\" tracks the object lifetime of all references in a program .";

    // Normalization helper: Python's decoder often leaves spaces before punctuation (e.g. " .")
    // while Rust's might be tighter. We normalize to single spaces to compare content.
    let normalize = |s: &str| -> String {
        s.replace(" .", ".")
            .replace(" ,", ",")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };

    let rust_normalized = normalize(result_trimmed);
    let python_normalized = normalize(python_golden);

    println!("Rust (normalized):   {}", rust_normalized);
    println!("Python (normalized): {}", python_normalized);

    // Strict Assertion
    assert_eq!(
        rust_normalized, python_normalized,
        "Beam search output did not match golden Python output.\nRust: '{}'\nPyth: '{}'",
        rust_normalized, python_normalized
    );

    // Verify key phrases are present (Redundant if assert_eq passes, but good for debugging)
    let required_phrases = [
        "Rust is a multi-paradigm",
        "general-purpose programming language",
        "emphasizes performance",
        "enforces memory safety",
        "garbage collector",
        "borrow checker",
        "tracks the object lifetime",
        "in a program",
    ];

    for phrase in &required_phrases {
        // We search in the raw string to ensure specific spacing wasn't totally mangled,
        // though normalized check is the primary one.
        let phrase_norm = normalize(phrase);
        assert!(
            rust_normalized.contains(&phrase_norm),
            "Missing phrase: '{}'",
            phrase
        );
    }

    // Verify No repeated 3-grams
    let words: Vec<&str> = result_trimmed.split_whitespace().collect();
    for i in 0..words.len().saturating_sub(2) {
        let trigram = format!("{} {} {}", words[i], words[i + 1], words[i + 2]);
        let count = (0..words.len().saturating_sub(2))
            .filter(|&j| format!("{} {} {}", words[j], words[j + 1], words[j + 2]) == trigram)
            .count();
        assert!(count <= 1, "Repeated 3-gram found: '{}'", trigram);
    }

    Ok(())
}

#[tokio::test]
async fn test_beam_search_debug_trace() -> Result<()> {
    let model = BartModel::from_registry(ModelType::DistilBartCnn, None, Device::Cpu, None).await?;
    let backend = CpuBackend;

    let text = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency. It enforces memory safety—meaning that all references point to valid memory—without using a garbage collector. To simultaneously enforce memory safety and prevent data races, its \"borrow checker\" tracks the object lifetime of all references in a program during compilation. Rust was influenced by languages like C++, Haskell, and Erlang.";

    let config = GenerationConfig {
        max_length: 142,
        min_length: 56,
        no_repeat_ngram_size: 3,
        repetition_penalty: 1.0,
        max_new_tokens: None,
        add_bos_token: false,
        strategy: DecodingStrategy::BeamSearch(BeamSearchParams {
            num_beams: 4,
            length_penalty: 2.0,
            early_stopping: true,
        }),
    };
    let result = run_beam_search(&model, &backend, text, &config).await?;
    let tokenizer = model.tokenizer();
    let encoding = tokenizer
        .encode(result.trim(), false)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let token_count = encoding.get_ids().len();

    println!("Output token count: {}", token_count);
    println!("min_length config: {}", config.min_length);
    // torch golden
    let python_output = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency . It enforces memory safety without using a garbage collector . To simultaneously enforce memory safety and prevent data races, its \"borrow checker\" tracks the object lifetime of all references in a program .";

    let python_encoding = tokenizer
        .encode(python_output, false)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let python_token_count = python_encoding.get_ids().len();
    println!("Python output token count: {}", python_token_count);
    assert!(
        token_count >= 40,
        "Token count {} is suspiciously low",
        token_count
    );
    assert_eq!(
        token_count, python_token_count,
        "Token count mismatch: Rust={} vs Python={}. Text might be different.",
        token_count, python_token_count
    );

    Ok(())
}

// Test with full input, greedy (num_beams=1)
#[tokio::test]
async fn test_beam_search_greedy_full_input() -> Result<()> {
    let model = BartModel::from_registry(ModelType::DistilBartCnn, None, Device::Cpu, None).await?;
    let backend = CpuBackend;

    let text = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency. It enforces memory safety—meaning that all references point to valid memory—without using a garbage collector. To simultaneously enforce memory safety and prevent data races, its \"borrow checker\" tracks the object lifetime of all references in a program during compilation. Rust was influenced by languages like C++, Haskell, and Erlang.";

    let config = GenerationConfig {
        max_length: 50,
        min_length: 0,
        no_repeat_ngram_size: 3,
        strategy: DecodingStrategy::Greedy,
        ..Default::default()
    };

    let result = run_beam_search(&model, &backend, text, &config).await?;
    let result_trimmed = result.trim(); // Handle leading/trailing whitespace
    println!("Generated: '{}'", result_trimmed);

    // Golden prefix from Python
    let expected_prefix = "Rust is a multi-paradigm, general-purpose programming language";
    assert!(
        result_trimmed.starts_with(expected_prefix),
        "Output should start with '{}', got '{}'",
        expected_prefix,
        &result_trimmed[..result_trimmed.len().min(80)]
    );

    // Verify key phrases are present
    assert!(
        result_trimmed.contains("emphasizes performance")
            || result_trimmed.contains("It emphasizes performance"),
        "Should contain 'emphasizes performance'"
    );
    assert!(
        result_trimmed.contains("type safety"),
        "Should contain 'type safety'"
    );
    assert!(
        result_trimmed.contains("concurrency"),
        "Should contain 'concurrency'"
    );

    // Check no repeated 3-grams (word-level)
    let words: Vec<&str> = result_trimmed.split_whitespace().collect();
    for i in 0..words.len().saturating_sub(2) {
        let trigram = format!("{} {} {}", words[i], words[i + 1], words[i + 2]);
        let count = (0..words.len().saturating_sub(2))
            .filter(|&j| format!("{} {} {}", words[j], words[j + 1], words[j + 2]) == trigram)
            .count();
        assert!(count <= 1, "Repeated 3-gram found: '{}'", trigram);
    }

    println!("✅ Output matches expected prefix and contains key phrases");
    println!("✅ No repeated 3-grams detected");

    Ok(())
}
