//! Text classification command using the high-level Classifier API.

use anyhow::{anyhow, Result};
use kjarni::DType;
use kjarni::classifier::{Classifier, ClassificationMode, ClassificationOverrides};
use kjarni::common::LoadConfig;


pub async fn run(
    input: &[String],
    model: &str,
    model_path: Option<&str>,
    labels: Option<&str>,
    top_k: usize,
    threshold: Option<f32>,
    max_length: Option<usize>,
    batch_size: Option<usize>,
    multi_label: bool,
    format: &str,
    gpu: bool,
    dtype: Option<&str>,
    quiet: bool,
) -> Result<()> {
    // 1. Resolve input text
    let text = if input.is_empty() {
        crate::commands::util::resolve_input(None)?
    } else {
        input.join(" ")
    };

    if text.trim().is_empty() {
        return Err(anyhow!("No input text provided."));
    }

    // Check for batch mode
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let is_batch = lines.len() > 1;

    // 2. Build classifier
    let mut builder = if let Some(path) = model_path {
        // Load from local path
        Classifier::from_path(path)
    } else {
        // Load from registry
        Classifier::builder(model)
    };

    // Apply options
    builder = builder
        .top_k(top_k)
        .quiet(quiet);

    if let Some(t) = threshold {
        builder = builder.threshold(t);
    }

    if let Some(max_len) = max_length {
        builder = builder.max_length(max_len);
    }

    if let Some(bs) = batch_size {
        builder = builder.batch_size(bs);
    }

    if multi_label {
        builder = builder.multi_label();
    }

    if gpu {
        builder = builder.gpu();
    }

    // Parse dtype
    if let Some(dt) = dtype {
        let dtype_enum = match dt.to_lowercase().as_str() {
            "f32" | "float32" => DType::F32,
            "f16" | "float16" => DType::F16,
            "bf16" | "bfloat16" => DType::BF16,
            _ => return Err(anyhow!("Invalid dtype: {}. Use: f32, f16, bf16", dt)),
        };
        builder = builder.dtype(dtype_enum);
    }

    // Parse labels
    if let Some(labels_str) = labels {
        builder = builder.labels_str(labels_str);
    }

    let classifier = builder
        .build()
        .await
        .map_err(|e| anyhow!("Failed to load classifier: {}", e))?;

    // Show info
    if !quiet {
        eprintln!("Model: {}", classifier.model_name());
        eprintln!("Device: {:?}", classifier.device());
        eprintln!("Labels: {:?}", classifier.labels().unwrap_or_default());
        if multi_label {
            eprintln!("Mode: multi-label (sigmoid)");
        }
    }

    // 3. Run classification
    if is_batch {
        if !quiet {
            eprintln!("Classifying {} texts...", lines.len());
        }
        let results = classifier
            .classify_batch(&lines)
            .await
            .map_err(|e| anyhow!("Classification failed: {}", e))?;
        output_batch_results(&lines, &results, format, quiet)?;
    } else {
        let text_str = lines.first().copied().unwrap_or(&text);
        let result = classifier
            .classify(text_str)
            .await
            .map_err(|e| anyhow!("Classification failed: {}", e))?;
        output_single_result(text_str, &result, format, quiet)?;
    }

    Ok(())
}

fn output_single_result(
    text: &str,
    result: &kjarni::classifier::ClassificationResult,
    format: &str,
    quiet: bool,
) -> Result<()> {
    match format {
        "json" => {
            let output = serde_json::json!({
                "text": text,
                "label": result.label,
                "score": result.score,
                "label_index": result.label_index,
                "predictions": result.all_scores.iter().map(|(label, score)| {
                    serde_json::json!({ "label": label, "score": score })
                }).collect::<Vec<_>>()
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "jsonl" => {
            let output = serde_json::json!({
                "text": text,
                "label": result.label,
                "score": result.score
            });
            println!("{}", serde_json::to_string(&output)?);
        }
        "text" => {
            if quiet {
                println!("{}", result.label);
            } else {
                println!();
                for (label, score) in &result.all_scores {
                    let bar_len = (score * 40.0) as usize;
                    let bar = "█".repeat(bar_len);
                    println!("  {:>16}  {:>6.2}%  {}", label, score * 100.0, bar);
                }
                println!();
            }
        }
        _ => return Err(anyhow!("Unknown format: '{}'. Use: json, jsonl, text", format)),
    }
    Ok(())
}

fn output_batch_results(
    texts: &[&str],
    results: &[kjarni::classifier::ClassificationResult],
    format: &str,
    quiet: bool,
) -> Result<()> {
    match format {
        "json" => {
            let output: Vec<_> = texts
                .iter()
                .zip(results.iter())
                .map(|(text, result)| {
                    serde_json::json!({
                        "text": text,
                        "label": result.label,
                        "score": result.score,
                        "predictions": result.all_scores.iter().map(|(l, s)| {
                            serde_json::json!({ "label": l, "score": s })
                        }).collect::<Vec<_>>()
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "jsonl" => {
            for (text, result) in texts.iter().zip(results.iter()) {
                let output = serde_json::json!({
                    "text": text,
                    "label": result.label,
                    "score": result.score
                });
                println!("{}", serde_json::to_string(&output)?);
            }
        }
        "text" => {
            if quiet {
                for result in results {
                    println!("{}", result.label);
                }
            } else {
                for (text, result) in texts.iter().zip(results.iter()) {
                    let truncated = if text.len() > 50 {
                        format!("{}...", &text[..47])
                    } else {
                        text.to_string()
                    };
                    println!(
                        "{:>16} ({:.1}%)  {}",
                        result.label,
                        result.score * 100.0,
                        truncated
                    );
                }
            }
        }
        _ => return Err(anyhow!("Unknown format: '{}'. Use: json, jsonl, text", format)),
    }
    Ok(())
}