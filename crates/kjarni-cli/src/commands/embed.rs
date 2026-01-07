//! Text embedding command using the high-level Embedder API.

use anyhow::{anyhow, Result};
use kjarni::{PoolingStrategy, embedder::Embedder};

pub async fn run(
    input: Option<&str>,
    model: &str,
    format: &str,
    normalize: bool,
    pooling: &str,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    // 1. Resolve input
    let text = crate::commands::util::resolve_input(input)?;

    // Check for batch mode
    let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
    let is_batch = lines.len() > 1;

    // 2. Parse pooling strategy
    let pooling_strategy: PoolingStrategy = pooling.parse()
        .map_err(|e: String| anyhow!("Invalid pooling strategy: {}", e))?;

    // 3. Build embedder
    let mut builder = Embedder::builder(model)
        .normalize(normalize)
        .pooling(pooling_strategy)
        .quiet(quiet);

    if gpu {
        builder = builder.gpu();
    }

    let embedder = builder
        .build()
        .await
        .map_err(|e| anyhow!("Failed to load embedder: {}", e))?;

    if !quiet {
        eprintln!("Model: {} (dim={})", embedder.model_name(), embedder.dimension());
    }

    // 4. Generate embeddings
    if is_batch {
        if !quiet {
            eprintln!("Embedding {} texts...", lines.len());
        }
        let embeddings = embedder.embed_batch(&lines).await
            .map_err(|e| anyhow!("Embedding failed: {}", e))?;
        output_batch(&lines, &embeddings, format)?;
    } else {
        let text_str = lines.first().copied().unwrap_or(&text);
        let embedding = embedder.embed(text_str).await
            .map_err(|e| anyhow!("Embedding failed: {}", e))?;
        output_single(text_str, &embedding, model, format)?;
    }

    Ok(())
}

fn output_single(text: &str, embedding: &[f32], model: &str, format: &str) -> Result<()> {
    match format {
        "json" => {
            let output = serde_json::json!({
                "text": text,
                "embedding": embedding,
                "model": model,
                "dim": embedding.len()
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "jsonl" => {
            println!("{}", serde_json::to_string(&embedding)?);
        }
        "raw" => {
            println!("{}", embedding.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(" "));
        }
        _ => return Err(anyhow!("Unknown format: '{}'. Use: json, jsonl, raw", format)),
    }
    Ok(())
}

fn output_batch(texts: &[&str], embeddings: &[Vec<f32>], format: &str) -> Result<()> {
    match format {
        "json" => {
            let output: Vec<_> = texts.iter().zip(embeddings.iter()).map(|(text, emb)| {
                serde_json::json!({
                    "text": text,
                    "embedding": emb
                })
            }).collect();
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "jsonl" => {
            for (text, emb) in texts.iter().zip(embeddings.iter()) {
                let obj = serde_json::json!({
                    "text": text,
                    "embedding": emb
                });
                println!("{}", serde_json::to_string(&obj)?);
            }
        }
        "raw" => {
            for emb in embeddings {
                println!("{}", emb.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(" "));
            }
        }
        _ => return Err(anyhow!("Unknown format: '{}'. Use: json, jsonl, raw", format)),
    }
    Ok(())
}