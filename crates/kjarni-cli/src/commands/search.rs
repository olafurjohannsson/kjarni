//! Search command using production index

use anyhow::{anyhow, Result};

use kjarni::{
    registry,
    SentenceEncoder,
    ModelArchitecture,
    ModelType,
    Device,
};
use kjarni::{IndexReader, SearchMode};

pub async fn run(
    index_path: &str,
    query: &str,
    top_k: usize,
    mode: &str,
    model: &str,
    format: &str,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    // 1. Open index
    let reader = IndexReader::open(index_path)?;

    if reader.is_empty() {
        return Err(anyhow!("Index is empty."));
    }

    if !quiet {
        eprintln!("Loaded index: {} documents in {} segments",
            reader.len(), reader.segment_count());
    }

    // 2. Parse search mode
    let search_mode: SearchMode = mode.parse()
        .map_err(|e: String| anyhow!(e))?;

    // 3. Search based on mode
    let results = match search_mode {
        SearchMode::Keyword => {
            if !quiet {
                eprintln!("Searching with BM25...");
            }
            reader.search_keywords(query, top_k)
        }
        SearchMode::Semantic => {
            let query_embedding = get_query_embedding(query, model, &reader, gpu, quiet).await?;
            if !quiet {
                eprintln!("Searching semantically...");
            }
            reader.search_semantic(&query_embedding, top_k)
        }
        SearchMode::Hybrid => {
            let query_embedding = get_query_embedding(query, model, &reader, gpu, quiet).await?;
            if !quiet {
                eprintln!("Searching with hybrid (BM25 + semantic)...");
            }
            reader.search_hybrid(query, &query_embedding, top_k)
        }
    };

    if results.is_empty() {
        if !quiet {
            eprintln!("No results found.");
        }
        return Ok(());
    }

    // 4. Output results
    output_results(&results, format)?;

    Ok(())
}

async fn get_query_embedding(
    query: &str,
    model: &str,
    reader: &IndexReader,
    gpu: bool,
    quiet: bool,
) -> Result<Vec<f32>> {
    let device = if gpu { Device::Wgpu } else { Device::Cpu };

    let model_type = ModelType::from_cli_name(model)
        .ok_or_else(|| anyhow!("Unknown model: '{}'", model))?;

    if model_type.architecture() != ModelArchitecture::Encoder {
        return Err(anyhow!("Model '{}' is not an encoder.", model));
    }

    if !registry::is_model_downloaded(model)? {
        if !quiet {
            eprintln!("Downloading model '{}'...", model);
        }
        registry::download_model(model).await?;
    }

    if !quiet {
        eprintln!("Loading encoder '{}'...", model);
    }

    let encoder = SentenceEncoder::from_registry(model_type, None, device, None).await?;

    // Verify dimension matches
    let embedding = encoder.encode(query).await?;
    if embedding.len() != reader.dimension() {
        return Err(anyhow!(
            "Dimension mismatch: encoder produces {}, index expects {}.\n\
             Use the same model that created the index.",
            embedding.len(),
            reader.dimension()
        ));
    }

    Ok(embedding)
}

fn output_results(results: &[kjarni::SearchResult], format: &str) -> Result<()> {
    match format {
        "json" => {
            let output: Vec<_> = results.iter().map(|r| {
                serde_json::json!({
                    "score": r.score,
                    "document_id": r.document_id,
                    "text": r.text,
                    "metadata": r.metadata
                })
            }).collect();
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "jsonl" => {
            for r in results {
                let obj = serde_json::json!({
                    "score": r.score,
                    "document_id": r.document_id,
                    "text": r.text,
                    "metadata": r.metadata
                });
                println!("{}", serde_json::to_string(&obj)?);
            }
        }
        "text" => {
            for r in results {
                let source = r.metadata.get("source")
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                let text = r.text.replace('\n', " ");
                let truncated = if text.len() > 60 {
                    format!("{}...", &text[..57])
                } else {
                    text
                };
                println!("{:.4}\t{}\t{}", r.score, source, truncated);
            }
        }
        "docs" => {
            // Just output documents (for piping)
            for r in results {
                println!("{}", r.text);
            }
        }
        _ => {
            return Err(anyhow!(
                "Unknown format: '{}'. Use: json, jsonl, text, docs",
                format
            ));
        }
    }

    Ok(())
}