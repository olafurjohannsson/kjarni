//! Classification command with colored terminal output.

use anyhow::{anyhow, Result};
use colored::*;
use kjarni::DType;
use kjarni::classifier::{Classifier, ClassificationResult};

use crate::commands::display;

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
    // Resolve input text
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

    // Build classifier
    let mut builder = if let Some(path) = model_path {
        Classifier::from_path(path)
    } else {
        Classifier::builder(model)
    };

    builder = builder.top_k(top_k).quiet(quiet);

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
    if let Some(dt) = dtype {
        let dtype_enum = parse_dtype(dt)?;
        builder = builder.dtype(dtype_enum);
    }
    if let Some(labels_str) = labels {
        builder = builder.labels_str(labels_str);
    }

    let classifier = builder
        .build()
        .await
        .map_err(|e| anyhow!("Failed to load classifier: {}", e))?;

    if !quiet {
        eprintln!(
            "{}",
            format!(
                "Model: {}  Device: {:?}",
                classifier.model_name(),
                classifier.device()
            )
            .dimmed()
        );
    }

    // Run classification
    if is_batch {
        if !quiet {
            eprintln!("{}", format!("Classifying {} texts...", lines.len()).dimmed());
        }
        let results = classifier
            .classify_batch(&lines)
            .await
            .map_err(|e| anyhow!("Classification failed: {}", e))?;
        let output = format_batch_results(&lines, &results, format, quiet)?;
        print!("{}", output);
    } else {
        let text_str = lines.first().copied().unwrap_or(&text);
        let result = classifier
            .classify(text_str)
            .await
            .map_err(|e| anyhow!("Classification failed: {}", e))?;
        let output = format_single_result(text_str, &result, format, quiet)?;
        print!("{}", output);
    }

    Ok(())
}

fn parse_dtype(dt: &str) -> Result<DType> {
    match dt.to_lowercase().as_str() {
        "f32" | "float32" => Ok(DType::F32),
        "f16" | "float16" => Ok(DType::F16),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        _ => Err(anyhow!("Invalid dtype: {}. Use: f32, f16, bf16", dt)),
    }
}

fn format_single_result(
    text: &str,
    result: &ClassificationResult,
    format: &str,
    quiet: bool,
) -> Result<String> {
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
            Ok(format!("{}\n", serde_json::to_string_pretty(&output)?))
        }
        "jsonl" => {
            let output = serde_json::json!({
                "text": text,
                "label": result.label,
                "score": result.score
            });
            Ok(format!("{}\n", serde_json::to_string(&output)?))
        }
        "text" => {
            if quiet {
                Ok(format!("{}\n", result.label))
            } else {
                Ok(format_pretty_single(text, result))
            }
        }
        _ => Err(anyhow!(
            "Unknown format: '{}'. Use: json, jsonl, text",
            format
        )),
    }
}

fn format_pretty_single(text: &str, result: &ClassificationResult) -> String {
    let mut output = String::new();

    // Input text
    let truncated = truncate_clean(text, 60);
    output.push_str(&format!(
        "\n  {} \"{}\"\n\n",
        "Input".dimmed(),
        truncated.white()
    ));

    // Find the top label
    let top_label = &result.label;

    // All predictions with bars
    for (label, score) in &result.all_scores {
        let is_top = label == top_label;
        let prefix = if is_top {
            "✓".green().bold()
        } else {
            " ".normal()
        };

        let bar = display::score_bar(*score, 20);
        let pct = display::score_pct(*score);

        let label_str = if is_top {
            format!("{:>14}", label).white().bold()
        } else {
            format!("{:>14}", label).dimmed()
        };

        output.push_str(&format!(
            "  {} {}  {}  {}\n",
            prefix, label_str, bar, pct
        ));
    }

    output.push('\n');
    output
}

fn format_batch_results(
    texts: &[&str],
    results: &[ClassificationResult],
    format: &str,
    quiet: bool,
) -> Result<String> {
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
            Ok(format!("{}\n", serde_json::to_string_pretty(&output)?))
        }
        "jsonl" => {
            let mut result_str = String::new();
            for (text, result) in texts.iter().zip(results.iter()) {
                let output = serde_json::json!({
                    "text": text,
                    "label": result.label,
                    "score": result.score
                });
                result_str.push_str(&serde_json::to_string(&output)?);
                result_str.push('\n');
            }
            Ok(result_str)
        }
        "text" => {
            if quiet {
                let mut output = String::new();
                for result in results {
                    output.push_str(&format!("{}\n", result.label));
                }
                Ok(output)
            } else {
                let mut output = String::new();
                output.push('\n');
                for (text, result) in texts.iter().zip(results.iter()) {
                    let truncated = truncate_clean(text, 40);
                    let bar = display::score_bar(result.score, 15);
                    let pct = display::score_pct(result.score);
                    let label_colored = display::colorize_by_score(
                        &format!("{:>14}", result.label),
                        result.score,
                    );

                    output.push_str(&format!(
                        "  {}  {}  {}  \"{}\"\n",
                        label_colored,
                        bar,
                        pct,
                        truncated.dimmed()
                    ));
                }
                output.push('\n');
                Ok(output)
            }
        }
        _ => Err(anyhow!(
            "Unknown format: '{}'. Use: json, jsonl, text",
            format
        )),
    }
}

fn truncate_clean(text: &str, max_len: usize) -> String {
    let clean = text.replace('\n', " ").replace('\r', "");
    if clean.len() <= max_len {
        clean
    } else {
        format!("{}...", &clean[..max_len - 1])
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    fn mock_result(label: &str, score: f32, label_index: usize) -> ClassificationResult {
        ClassificationResult {
            label: label.to_string(),
            score,
            label_index,
            all_scores: vec![
                ("positive".to_string(), 0.8),
                ("negative".to_string(), 0.2),
            ],
        }
    }

    fn mock_result_with_scores(
        label: &str,
        score: f32,
        label_index: usize,
        all_scores: Vec<(&str, f32)>,
    ) -> ClassificationResult {
        ClassificationResult {
            label: label.to_string(),
            score,
            label_index,
            all_scores: all_scores
                .into_iter()
                .map(|(l, s)| (l.to_string(), s))
                .collect(),
        }
    }

    #[test]
    fn test_parse_dtype_f32() {
        assert!(matches!(parse_dtype("f32").unwrap(), DType::F32));
        assert!(matches!(parse_dtype("float32").unwrap(), DType::F32));
        assert!(matches!(parse_dtype("F32").unwrap(), DType::F32));
    }

    #[test]
    fn test_parse_dtype_f16() {
        assert!(matches!(parse_dtype("f16").unwrap(), DType::F16));
        assert!(matches!(parse_dtype("float16").unwrap(), DType::F16));
    }

    #[test]
    fn test_parse_dtype_bf16() {
        assert!(matches!(parse_dtype("bf16").unwrap(), DType::BF16));
        assert!(matches!(parse_dtype("bfloat16").unwrap(), DType::BF16));
    }

    #[test]
    fn test_parse_dtype_invalid() {
        let result = parse_dtype("int8");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid dtype"));
    }

    #[test]
    fn test_format_single_result_json() {
        let result = mock_result("positive", 0.95, 0);
        let output = format_single_result("great movie", &result, "json", false).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed["text"], "great movie");
        assert_eq!(parsed["label"], "positive");
        assert_eq!(parsed["score"], 0.95);
        assert_eq!(parsed["label_index"], 0);
        assert!(parsed["predictions"].is_array());
        assert_eq!(parsed["predictions"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_format_single_result_jsonl() {
        let result = mock_result("negative", 0.7, 1);
        let output = format_single_result("bad movie", &result, "jsonl", false).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed["text"], "bad movie");
        assert_eq!(parsed["label"], "negative");
        assert_eq!(parsed["score"], 0.7);
        // jsonl format doesn't include predictions
        assert!(parsed.get("predictions").is_none());
    }

    #[test]
    fn test_format_single_result_text_quiet() {
        let result = mock_result("positive", 0.9, 0);
        let output = format_single_result("test", &result, "text", true).unwrap();

        assert_eq!(output.trim(), "positive");
    }

    #[test]
    fn test_format_single_result_text_verbose() {
        let result = mock_result_with_scores("positive", 0.8, 0, vec![
            ("positive", 0.8),
            ("negative", 0.2),
        ]);
        let output = format_single_result("test", &result, "text", false).unwrap();

        assert!(output.contains("positive"));
        assert!(output.contains("negative"));
        assert!(output.contains("80.0%"));
        assert!(output.contains("20.0%"));
        assert!(output.contains("█")); // progress bar
    }

    #[test]
    fn test_format_single_result_unknown_format() {
        let result = mock_result("positive", 0.9, 0);
        let output = format_single_result("test", &result, "xml", false);

        assert!(output.is_err());
        assert!(output.unwrap_err().to_string().contains("Unknown format"));
    }
    #[test]
    fn test_format_batch_results_json() {
        let texts = vec!["good", "bad"];
        let results = vec![
            mock_result("positive", 0.9, 0),
            mock_result("negative", 0.8, 1),
        ];

        let output = format_batch_results(&texts, &results, "json", false).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0]["text"], "good");
        assert_eq!(parsed[0]["label"], "positive");
        assert_eq!(parsed[1]["text"], "bad");
        assert_eq!(parsed[1]["label"], "negative");
    }

    #[test]
    fn test_format_batch_results_jsonl() {
        let texts = vec!["a", "b"];
        let results = vec![
            mock_result("pos", 0.9, 0),
            mock_result("neg", 0.1, 1),
        ];

        let output = format_batch_results(&texts, &results, "jsonl", false).unwrap();
        let lines: Vec<&str> = output.trim().lines().collect();

        assert_eq!(lines.len(), 2);

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();

        assert_eq!(first["text"], "a");
        assert_eq!(first["label"], "pos");
        assert_eq!(second["text"], "b");
        assert_eq!(second["label"], "neg");
    }

    #[test]
    fn test_format_batch_results_text_quiet() {
        let texts = vec!["one", "two"];
        let results = vec![
            mock_result("positive", 0.9, 0),
            mock_result("negative", 0.8, 1),
        ];

        let output = format_batch_results(&texts, &results, "text", true).unwrap();
        let lines: Vec<&str> = output.trim().lines().collect();

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "positive");
        assert_eq!(lines[1], "negative");
    }

    #[test]
    fn test_format_batch_results_text_verbose() {
        let texts = vec!["short text", "another one"];
        let results = vec![
            mock_result("positive", 0.95, 0),
            mock_result("negative", 0.75, 1),
        ];

        let output = format_batch_results(&texts, &results, "text", false).unwrap();

        assert!(output.contains("positive"));
        assert!(output.contains("95.0%"));
        assert!(output.contains("negative"));
        assert!(output.contains("75.0%"));
        assert!(output.contains("short text"));
    }

    #[test]
    fn test_format_batch_results_text_truncates_long_text() {
        let long_text = "a".repeat(100);
        let texts = vec![long_text.as_str()];
        let results = vec![mock_result("label", 0.5, 0)];

        let output = format_batch_results(&texts, &results, "text", false).unwrap();

        assert!(output.contains("..."));
        assert!(!output.contains(&long_text));
    }

    #[test]
    fn test_format_batch_results_unknown_format() {
        let output = format_batch_results(&["a"], &[mock_result("x", 0.5, 0)], "csv", false);

        assert!(output.is_err());
        assert!(output.unwrap_err().to_string().contains("Unknown format"));
    }

    #[test]
    fn test_format_batch_results_empty() {
        let texts: Vec<&str> = vec![];
        let results: Vec<ClassificationResult> = vec![];

        let output = format_batch_results(&texts, &results, "json", false).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();

        assert!(parsed.is_empty());
    }

    #[test]
    fn test_format_single_result_special_characters() {
        let result = mock_result("label", 0.5, 0);
        let output = format_single_result("text with \"quotes\" and\nnewlines", &result, "json", false).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert!(parsed["text"].as_str().unwrap().contains("quotes"));
    }

    #[test]
    fn test_format_single_result_unicode() {
        let result = mock_result("jákvætt", 0.9, 0);
        let output = format_single_result("þetta er íslenska", &result, "json", false).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed["text"], "þetta er íslenska");
        assert_eq!(parsed["label"], "jákvætt");
    }

    #[test]
    fn test_format_single_result_zero_score() {
        let result = mock_result_with_scores("none", 0.0, 0, vec![("none", 0.0)]);
        let output = format_single_result("text", &result, "text", false).unwrap();

        assert!(output.contains("0.0%"));
    }

    #[test]
    fn test_format_single_result_full_score() {
        let result = mock_result_with_scores("certain", 1.0, 0, vec![("certain", 1.0)]);
        let output = format_single_result("text", &result, "text", false).unwrap();

        assert!(output.contains("100.0%"));
    }
}
