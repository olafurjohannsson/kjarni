//! Text classification command

use anyhow::{anyhow, Result};
use std::io::{self, BufRead};

use kjarni::{
    registry,
    SequenceClassifier,
    ModelArchitecture,
    ModelType,
    Device,
};
use crate::commands::util::resolve_input;

pub async fn run(
    input: &[String],
    model: &str,
    top_k: usize,
    format: &str,
    labels: Option<&str>,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    // 1. Resolve input text
    let text = resolve_input(Some(input[0].as_str()))?;

    if text.trim().is_empty() {
        return Err(anyhow!("No input text provided."));
    }

    // 2. Load classifier
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

    // Validate architecture
    if model_type.architecture() != ModelArchitecture::Bert {
        return Err(anyhow!(
            "Model '{}' is not a sequence classifier.\n\
             Use a classifier model like distilbert-sentiment or bert-base-emotion.",
            model
        ));
    }

    if !registry::is_model_downloaded(model)? {
        if !quiet {
            eprintln!("Downloading model '{}'...", model);
        }
        registry::download_model(model, false).await?;
    }

    if !quiet {
        eprintln!("Loading classifier '{}'...", model);
    }

    let classifier = SequenceClassifier::from_registry(model_type, None, device, None, None).await?;

    // 3. Handle custom labels for zero-shot
    let predictions = if let Some(label_str) = labels {
        let custom_labels: Vec<&str> = label_str.split(',').map(|s| s.trim()).collect();
        if !quiet {
            eprintln!("Using custom labels: {:?}", custom_labels);
        }
        // classifier.classify_with_labels(&text, &custom_labels, top_k).await?
    } else {
        // classifier.classify(&text, top_k).await?
        unimplemented!()
    };

    // 4. Output results
    // match format {
    //     "json" => {
    //         let output: Vec<_> = predictions.iter().map(|(label, score)| {
    //             serde_json::json!({
    //                 "label": label,
    //                 "score": score
    //             })
    //         }).collect();
    //         println!("{}", serde_json::to_string_pretty(&output)?);
    //     }
    //     "jsonl" => {
    //         for (label, score) in &predictions {
    //             println!("{}", serde_json::json!({
    //                 "label": label,
    //                 "score": score
    //             }));
    //         }
    //     }
    //     "text" => {
    //         if quiet {
    //             // Just output top label
    //             if let Some((label, _)) = predictions.first() {
    //                 println!("{}", label);
    //             }
    //         } else {
    //             for (label, score) in &predictions {
    //                 println!("{:.4}\t{}", score, label);
    //             }
    //         }
    //     }
    //     _ => {
    //         return Err(anyhow!("Unknown format: '{}'. Use: json, jsonl, text", format));
    //     }
    // }

    Ok(())
}