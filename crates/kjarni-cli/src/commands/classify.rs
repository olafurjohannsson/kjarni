//! Text classification command

use crate::commands::util::resolve_input;
use anyhow::{anyhow, Result};
use kjarni::{registry, Device, ModelTask, ModelType, SequenceClassifier};

pub async fn run(
    input: &[String],
    model: &str,
    top_k: usize,
    format: &str,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    // 1. Resolve input text
    let text = if input.is_empty() {
        resolve_input(None)?
    } else {
        resolve_input(Some(&input.join(" ")))?
    };

    if text.trim().is_empty() {
        return Err(anyhow!("No input text provided."));
    }

    // Check for batch mode (multiple lines)
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let is_batch = lines.len() > 1;

    // 2. Resolve model
    let device = if gpu { Device::Wgpu } else { Device::Cpu };

    let model_type = ModelType::from_cli_name(model).ok_or_else(|| {
        let mut msg = format!("Unknown model: '{}'.", model);
        let suggestions = ModelType::find_similar(model);
        if !suggestions.is_empty() {
            msg.push_str("\n\nDid you mean?");
            for (name, _) in suggestions.iter().take(3) {
                msg.push_str(&format!("\n  - {}", name));
            }
        }
        anyhow!(msg)
    })?;

    // Validate it's a classifier
    let info = model_type.info();
    let is_classifier = matches!(
        info.task,
        ModelTask::SentimentAnalysis | ModelTask::ZeroShotClassification
    );

    if !is_classifier {
        return Err(anyhow!(
            "Model '{}' is not a classifier (task: {:?}).\n\
             Use a classifier model like:\n  \
             - sentiment-distilbert (sentiment analysis)\n  \
             - zeroshot-bart (zero-shot classification)",
            model,
            info.task
        ));
    }

    // 3. Download if needed
    if !registry::is_model_downloaded(model)? {
        if !quiet {
            eprintln!("Downloading model '{}'...", model);
        }
        registry::download_model(model, false).await?;
    }

    // 4. Load classifier
    if !quiet {
        eprintln!("Loading classifier '{}'...", model);
    }

    let classifier =
        SequenceClassifier::from_registry(model_type, None, device, None, None).await?;

    // 5. Run classification
    if is_batch {
        if !quiet {
            eprintln!("Classifying {} texts...", lines.len());
        }
        let all_predictions = classifier.classify_batch(&lines, top_k).await?;
        output_batch_results(&lines, &all_predictions, format, quiet)?;
    } else {
        let text_str = lines.first().map(|s| *s).unwrap_or(&text);
        let predictions = classifier.classify(text_str, top_k).await?;
        output_single_result(text_str, &predictions, format, quiet)?;
    }

    Ok(())
}

fn output_single_result(
    text: &str,
    predictions: &[(String, f32)],
    format: &str,
    quiet: bool,
) -> Result<()> {
    match format {
        "json" => {
            let output = serde_json::json!({
                "text": text,
                "predictions": predictions.iter().map(|(label, score)| {
                    serde_json::json!({
                        "label": label,
                        "score": score
                    })
                }).collect::<Vec<_>>()
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "jsonl" => {
            let output = serde_json::json!({
                "text": text,
                "label": predictions.first().map(|(l, _)| l),
                "score": predictions.first().map(|(_, s)| s),
                "all": predictions
            });
            println!("{}", serde_json::to_string(&output)?);
        }
        "text" => {
            if quiet {
                // Just output top label
                if let Some((label, _)) = predictions.first() {
                    println!("{}", label);
                }
            } else {
                println!();
                for (label, score) in predictions {
                    let bar_len = (score * 40.0) as usize;
                    let bar = "█".repeat(bar_len);
                    println!("  {:>12}  {:>6.2}%  {}", label, score * 100.0, bar);
                }
                println!();
            }
        }
        _ => {
            return Err(anyhow!(
                "Unknown format: '{}'. Use: json, jsonl, text",
                format
            ));
        }
    }
    Ok(())
}

fn output_batch_results(
    texts: &[&str],
    all_predictions: &[Vec<(String, f32)>],
    format: &str,
    quiet: bool,
) -> Result<()> {
    match format {
        "json" => {
            let output: Vec<_> = texts
                .iter()
                .zip(all_predictions.iter())
                .map(|(text, preds)| {
                    serde_json::json!({
                        "text": text,
                        "predictions": preds.iter().map(|(label, score)| {
                            serde_json::json!({ "label": label, "score": score })
                        }).collect::<Vec<_>>()
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "jsonl" => {
            for (text, preds) in texts.iter().zip(all_predictions.iter()) {
                let output = serde_json::json!({
                    "text": text,
                    "label": preds.first().map(|(l, _)| l),
                    "score": preds.first().map(|(_, s)| s)
                });
                println!("{}", serde_json::to_string(&output)?);
            }
        }
        "text" => {
            if quiet {
                for preds in all_predictions {
                    if let Some((label, _)) = preds.first() {
                        println!("{}", label);
                    }
                }
            } else {
                for (text, preds) in texts.iter().zip(all_predictions.iter()) {
                    let truncated = if text.len() > 50 {
                        format!("{}...", &text[..47])
                    } else {
                        text.to_string()
                    };

                    if let Some((label, score)) = preds.first() {
                        println!("{:>12} ({:.1}%)  {}", label, score * 100.0, truncated);
                    }
                }
            }
        }
        _ => {
            return Err(anyhow!(
                "Unknown format: '{}'. Use: json, jsonl, text",
                format
            ));
        }
    }
    Ok(())
}
