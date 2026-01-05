//! Rerank documents by relevance to a query using cross-encoder models

use anyhow::{anyhow, Result};
use std::io::{self, BufRead};

use kjarni::{
    registry,
    CrossEncoder,
    ModelArchitecture,
    ModelType,
    Device,
};

pub async fn run(
    query: &str,
    documents: &[String],
    model: &str,
    model_path: Option<&str>,
    top_k: Option<usize>,
    format: &str,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    // 1. Resolve documents - from args or stdin
    let docs: Vec<String> = if documents.is_empty() {
        // Read from stdin (one document per line)
        if !quiet {
            eprintln!("Reading documents from stdin (one per line)...");
        }
        io::stdin()
            .lock()
            .lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .collect()
    } else {
        documents.to_vec()
    };

    if docs.is_empty() {
        return Err(anyhow!("No documents provided. Pass as arguments or pipe via stdin."));
    }

    // 2. Resolve model
    let device = if gpu { Device::Wgpu } else { Device::Cpu };

    if model_path.is_some() {
        return Err(anyhow!("--model-path not yet implemented."));
    }

    let model_type = ModelType::from_cli_name(model).ok_or_else(|| {
        let mut msg = format!("Unknown model: '{}'.", model);
        let suggestions = ModelType::find_similar(model);
        if !suggestions.is_empty() {
            msg.push_str("\n\nDid you mean?");
            for (name, _) in suggestions {
                msg.push_str(&format!("\n  - {}", name));
            }
        }
        anyhow!(msg)
    })?;

    // Validate it's a cross-encoder
    if model_type.architecture() != ModelArchitecture::Bert {
        return Err(anyhow!(
            "Model '{}' is not a cross-encoder. Reranking requires a cross-encoder model like minilm-l6-v2-cross-encoder.",
            model
        ));
    }

    // Check if downloaded
    if !registry::is_model_downloaded(model)? {
        if !quiet {
            eprintln!("Model '{}' not found locally. Downloading...", model);
        }
        registry::download_model(model, false).await?;
        if !quiet {
            eprintln!();
        }
    }

    // 3. Load cross-encoder
    if !quiet {
        eprintln!("Loading model '{}'...", model);
    }

    let encoder = CrossEncoder::from_registry(model_type, None, device, None, None).await?;

    // 4. Rerank
    if !quiet {
        eprintln!("Reranking {} documents against query: {:?}", docs.len(), query);
        eprintln!();
    }

    let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
    let results = encoder.rerank(query, &doc_refs).await?;

    // Apply top_k
    let results: Vec<_> = match top_k {
        Some(k) => results.into_iter().take(k).collect(),
        None => results,
    };

    // 5. Output
    match format {
        "json" => {
            let output: Vec<_> = results.iter().map(|(idx, score)| {
                serde_json::json!({
                    "index": idx,
                    "score": score,
                    "document": docs[*idx]
                })
            }).collect();
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "jsonl" => {
            for (idx, score) in &results {
                let obj = serde_json::json!({
                    "index": idx,
                    "score": score,
                    "document": docs[*idx]
                });
                println!("{}", serde_json::to_string(&obj)?);
            }
        }
        "text" => {
            for (idx, score) in &results {
                // Tab-separated: score, index, document (truncated)
                println!("{:.4}\t{}\t{}", score, idx, truncate(&docs[*idx], 80));
            }
        }
        "docs" => {
            // Just output the documents in ranked order (for piping)
            for (idx, _) in &results {
                println!("{}", docs[*idx]);
            }
        }
        _ => {
            return Err(anyhow!("Unknown format: '{}'. Use: json, jsonl, text, docs", format));
        }
    }

    Ok(())
}

fn truncate(s: &str, max_len: usize) -> String {
    let s = s.replace('\n', " ").replace('\t', " ");
    if s.len() <= max_len {
        s
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}