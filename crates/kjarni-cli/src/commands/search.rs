//! Search command using production index

use anyhow::{anyhow, Result};
use kjarni::{IndexReader, SearchMode};
use kjarni::embedder::Embedder;

pub async fn run(
    index_path: &str,
    query: &str,
    top_k: usize,
    mode: &str,
    model: &str,
    rerank_model: Option<&str>,
    format: &str,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    // Open index
    let reader = IndexReader::open(index_path)?;

    if reader.is_empty() {
        return Err(anyhow!("Index is empty."));
    }

    if !quiet {
        eprintln!("Loaded index: {} documents in {} segments",
            reader.len(), reader.segment_count());
    }

    // Parse search mode
    let search_mode: SearchMode = mode.parse()
        .map_err(|e: String| anyhow!(e))?;

    // If rerank model is provided, we will fetch more results initially
    let fetch_k = if rerank_model.is_some() { top_k * 5 } else { top_k };

    // Search based on mode
    let mut results = match search_mode {
        SearchMode::Keyword => {
            if !quiet {
                eprintln!("Searching with BM25...");
            }
            reader.search_keywords(query, fetch_k)
        }
        SearchMode::Semantic => {
            let query_embedding = get_query_embedding(query, model, &reader, gpu, quiet).await?;
            if !quiet {
                eprintln!("Searching semantically...");
            }
            reader.search_semantic(&query_embedding, fetch_k)
        }
        SearchMode::Hybrid => {
            let query_embedding = get_query_embedding(query, model, &reader, gpu, quiet).await?;
            if !quiet {
                eprintln!("Searching with hybrid (BM25 + semantic)...");
            }
            reader.search_hybrid(query, &query_embedding, fetch_k)
        }
    };


    if results.is_empty() {
        if !quiet {
            eprintln!("No results found.");
        }
        return Ok(());
    }

    // Optional re rank
    if let Some(reranker_name) = rerank_model {
        if !quiet {
            eprintln!("Reranking top {} results with '{}'...", results.len(), reranker_name);
        }

        // Load Reranker
        let mut builder = kjarni::reranker::Reranker::builder(reranker_name).quiet(quiet);
        if gpu { builder = builder.gpu(); }
        
        let reranker = builder.build().await.map_err(|e| anyhow!(e))?;

        // Prepare inputs
        let texts: Vec<&str> = results.iter().map(|r| r.text.as_str()).collect();
        
        // Run Reranker (returns sorted results)
        let reranked_results = reranker.rerank(query, &texts).await.map_err(|e| anyhow!(e))?;
        
        // Reconstruct the final list based on reranker output
        let mut new_results = Vec::with_capacity(reranked_results.len());
        
        for rr in reranked_results {
            // rr.index is the index into our `texts` (and thus `results`) vector
            let mut original_result = results[rr.index].clone();
            
            // Update the score to the high-precision cross-encoder score
            original_result.score = rr.score;
            
            new_results.push(original_result);
        }
        
        // Truncate to the requested top_k (reranker might have returned fewer if threshold used, but here we just clamp)
        if new_results.len() > top_k {
            new_results.truncate(top_k);
        }
        
        results = new_results;
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
    // Use high-level Embedder builder
    let mut builder = Embedder::builder(model).quiet(quiet);
    
    if gpu {
        builder = builder.gpu();
    } else {
        builder = builder.cpu();
    }

    let embedder = builder.build().await.map_err(|e| anyhow!(e))?;

    // Verify dimension matches
    if embedder.dimension() != reader.dimension() {
        return Err(anyhow!(
            "Dimension mismatch: index expects {}, model '{}' produces {}.\n\
             Use the same model that created the index.",
            reader.dimension(),
            embedder.model_name(),
            embedder.dimension()
        ));
    }

    // Generate embedding
    let embedding = embedder.embed(query).await.map_err(|e| anyhow!(e))?;
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