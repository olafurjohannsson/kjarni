use anyhow::Result;
use kjarni_transformers::common::{BeamSearchParams, DecodingStrategy, GenerationConfig};
use kjarni_transformers::encoder_decoder::{CpuBackend, run_beam_search};
use kjarni_transformers::gpu::encoder_decoder::backend::GpuEncoderDecoderBackend;
use kjarni_transformers::models::ModelType;
use kjarni_transformers::traits::Device;
use kjarni_transformers::{LanguageModel, WgpuContext};

use crate::models::bart::model::BartModel;

#[tokio::test]
async fn test_beam_search_4_beams() -> Result<()> {
    let model = BartModel::from_registry(ModelType::DistilBartCnn, None, Device::Cpu, None, None).await?;
    let backend = CpuBackend;

    let context = WgpuContext::new().await?;
    let gpu_model = BartModel::from_registry(ModelType::DistilBartCnn, None, Device::Wgpu, Some(context.clone()), None).await?;
    let gpu_backend = GpuEncoderDecoderBackend::new(context)?;

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
        ..Default::default()
    };

    let result = run_beam_search(&model, &backend, text, &config).await?;
    let result_trimmed = result.trim();

    let gpu_result = run_beam_search(&gpu_model, &gpu_backend, text, &config).await?;
    let gpu_result_trimmed = gpu_result.trim();

    assert_eq!(
        result_trimmed, gpu_result_trimmed,
        "cpu and gpu beam search outputs do not match.\ncpu: '{}'\ngpu: '{}'",
        result_trimmed, gpu_result_trimmed
    );

    let python_golden = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency . It enforces memory safety without using a garbage collector . To simultaneously enforce memory safety and prevent data races, its \"borrow checker\" tracks the object lifetime of all references in a program .";

    let normalize = |s: &str| -> String {
        s.replace(" .", ".")
            .replace(" ,", ",")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    };

    let rust_normalized = normalize(result_trimmed);
    let python_normalized = normalize(python_golden);

    println!("rust (normalized):   {}", rust_normalized);
    println!("python (normalized): {}", python_normalized);

    assert_eq!(
        rust_normalized, python_normalized,
        "beam search output did not match golden python output.\nrust: '{}'\npython: '{}'",
        rust_normalized, python_normalized
    );

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
        let phrase_norm = normalize(phrase);
        assert!(
            rust_normalized.contains(&phrase_norm),
            "missing phrase: '{}'",
            phrase
        );
    }

    let words: Vec<&str> = result_trimmed.split_whitespace().collect();
    for i in 0..words.len().saturating_sub(2) {
        let trigram = format!("{} {} {}", words[i], words[i + 1], words[i + 2]);
        let count = (0..words.len().saturating_sub(2))
            .filter(|&j| format!("{} {} {}", words[j], words[j + 1], words[j + 2]) == trigram)
            .count();
        assert!(count <= 1, "repeated 3-gram found: '{}'", trigram);
    }

    Ok(())
}

#[tokio::test]
async fn test_beam_search_debug_trace() -> Result<()> {
    let model = BartModel::from_registry(ModelType::DistilBartCnn, None, Device::Cpu, None, None).await?;
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
        ..Default::default()
    };

    let result = run_beam_search(&model, &backend, text, &config).await?;
    let tokenizer = model.tokenizer();
    let encoding = tokenizer
        .encode(result.trim(), false)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let token_count = encoding.get_ids().len();

    println!("output token count: {}", token_count);
    println!("min_length config: {}", config.min_length);

    let python_output = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency . It enforces memory safety without using a garbage collector . To simultaneously enforce memory safety and prevent data races, its \"borrow checker\" tracks the object lifetime of all references in a program .";

    let python_encoding = tokenizer
        .encode(python_output, false)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let python_token_count = python_encoding.get_ids().len();

    println!("python output token count: {}", python_token_count);

    assert!(
        token_count >= 40,
        "token count {} is suspiciously low",
        token_count
    );
    assert_eq!(
        token_count, python_token_count,
        "token count mismatch: rust={} vs python={}",
        token_count, python_token_count
    );

    Ok(())
}

#[tokio::test]
async fn test_beam_search_greedy() -> Result<()> {
    let model = BartModel::from_registry(ModelType::DistilBartCnn, None, Device::Cpu, None, None).await?;
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
    let result_trimmed = result.trim();

    println!("generated: '{}'", result_trimmed);

    let expected_prefix = "Rust is a multi-paradigm, general-purpose programming language";
    assert!(
        result_trimmed.starts_with(expected_prefix),
        "output should start with '{}', got '{}'",
        expected_prefix,
        &result_trimmed[..result_trimmed.len().min(80)]
    );

    assert!(
        result_trimmed.contains("emphasizes performance")
            || result_trimmed.contains("It emphasizes performance"),
        "should contain 'emphasizes performance'"
    );
    assert!(
        result_trimmed.contains("type safety"),
        "should contain 'type safety'"
    );
    assert!(
        result_trimmed.contains("concurrency"),
        "should contain 'concurrency'"
    );

    let words: Vec<&str> = result_trimmed.split_whitespace().collect();
    for i in 0..words.len().saturating_sub(2) {
        let trigram = format!("{} {} {}", words[i], words[i + 1], words[i + 2]);
        let count = (0..words.len().saturating_sub(2))
            .filter(|&j| format!("{} {} {}", words[j], words[j + 1], words[j + 2]) == trigram)
            .count();
        assert!(count <= 1, "repeated 3-gram found: '{}'", trigram);
    }

    Ok(())
}