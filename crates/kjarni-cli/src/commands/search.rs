//! Search command using production index

use anyhow::{anyhow, Result};
use kjarni::{IndexReader, SearchMode, SearchResult};
use kjarni::embedder::Embedder;
use std::collections::HashMap;

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

    // Optional rerank
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
            let mut original_result = results[rr.index].clone();
            original_result.score = rr.score;
            new_results.push(original_result);
        }
        
        if new_results.len() > top_k {
            new_results.truncate(top_k);
        }
        
        results = new_results;
    }

    // Output results
    let output = format_results(&results, format)?;
    print!("{}", output);

    Ok(())
}

async fn get_query_embedding(
    query: &str,
    model: &str,
    reader: &IndexReader,
    gpu: bool,
    quiet: bool,
) -> Result<Vec<f32>> {
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

    let embedding = embedder.embed(query).await.map_err(|e| anyhow!(e))?;
    Ok(embedding)
}

fn format_results(results: &[SearchResult], format: &str) -> Result<String> {
    match format {
        "json" => format_results_json(results),
        "jsonl" => format_results_jsonl(results),
        "text" => Ok(format_results_text(results)),
        "docs" => Ok(format_results_docs(results)),
        _ => Err(anyhow!(
            "Unknown format: '{}'. Use: json, jsonl, text, docs",
            format
        )),
    }
}

fn format_results_json(results: &[SearchResult]) -> Result<String> {
    let output: Vec<_> = results.iter().map(|r| {
        serde_json::json!({
            "score": r.score,
            "document_id": r.document_id,
            "text": r.text,
            "metadata": r.metadata
        })
    }).collect();
    Ok(format!("{}\n", serde_json::to_string_pretty(&output)?))
}

fn format_results_jsonl(results: &[SearchResult]) -> Result<String> {
    let mut output = String::new();
    for r in results {
        let obj = serde_json::json!({
            "score": r.score,
            "document_id": r.document_id,
            "text": r.text,
            "metadata": r.metadata
        });
        output.push_str(&serde_json::to_string(&obj)?);
        output.push('\n');
    }
    Ok(output)
}

fn format_results_text(results: &[SearchResult]) -> String {
    let mut output = String::new();
    for r in results {
        let source = r.metadata.get("source")
            .map(|s| s.as_str())
            .unwrap_or("?");
        let truncated = truncate_text(&r.text, 60);
        output.push_str(&format!("{:.4}\t{}\t{}\n", r.score, source, truncated));
    }
    output
}

fn format_results_docs(results: &[SearchResult]) -> String {
    let mut output = String::new();
    for r in results {
        output.push_str(&r.text);
        output.push('\n');
    }
    output
}

fn truncate_text(text: &str, max_len: usize) -> String {
    let text = text.replace('\n', " ");
    if text.len() > max_len {
        format!("{}...", &text[..max_len - 3])
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn mock_result(score: f32, doc_id: usize, text: &str) -> SearchResult {
        SearchResult {
            score,
            document_id: doc_id,
            text: text.to_string(),
            metadata: HashMap::new(),
        }
    }

    fn mock_result_with_metadata(
        score: f32,
        doc_id: usize,
        text: &str,
        metadata: HashMap<String, String>,
    ) -> SearchResult {
        SearchResult {
            score,
            document_id: doc_id,
            text: text.to_string(),
            metadata,
        }
    }
    #[test]
    fn test_truncate_text_short() {
        assert_eq!(truncate_text("hello world", 60), "hello world");
        assert_eq!(truncate_text("short", 10), "short");
    }

    #[test]
    fn test_truncate_text_exact_length() {
        let text = "a".repeat(60);
        assert_eq!(truncate_text(&text, 60), text);
    }

    #[test]
    fn test_truncate_text_long() {
        let text = "a".repeat(100);
        let result = truncate_text(&text, 60);
        assert_eq!(result.len(), 60);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_truncate_text_replaces_newlines() {
        assert_eq!(truncate_text("hello\nworld", 60), "hello world");
        assert_eq!(truncate_text("line1\nline2\nline3", 60), "line1 line2 line3");
    }

    #[test]
    fn test_truncate_text_newlines_then_truncate() {
        let text = "line one\nline two\nline three\nline four\nline five";
        let result = truncate_text(text, 30);
        assert_eq!(result.len(), 30);
        assert!(result.ends_with("..."));
        assert!(!result.contains('\n'));
    }

    #[test]
    fn test_truncate_text_empty() {
        assert_eq!(truncate_text("", 60), "");
    }

    #[test]
    fn test_truncate_text_only_newlines() {
        assert_eq!(truncate_text("\n\n\n", 60), "   ");
    }
    #[test]
    fn test_format_results_json_single() {
        let results = vec![mock_result(0.95, 1, "test document")];
        let output = format_results_json(&results).unwrap();
        
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["score"], 0.95);
        assert_eq!(parsed[0]["document_id"], 1);
        assert_eq!(parsed[0]["text"], "test document");
    }

    #[test]
    fn test_format_results_json_multiple() {
        let results = vec![
            mock_result(0.95, 1, "first doc"),
            mock_result(0.85, 2, "second doc"),
            mock_result(0.75, 3, "third doc"),
        ];
        let output = format_results_json(&results).unwrap();
        
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0]["score"], 0.95);
        assert_eq!(parsed[1]["score"], 0.85);
        assert_eq!(parsed[2]["score"], 0.75);
    }

    #[test]
    fn test_format_results_json_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "file.txt".to_string());
        metadata.insert("page".to_string(), "5".to_string());
        
        let results = vec![mock_result_with_metadata(0.9, 1, "doc", metadata)];
        let output = format_results_json(&results).unwrap();
        
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed[0]["metadata"]["source"], "file.txt");
        assert_eq!(parsed[0]["metadata"]["page"], "5");
    }

    #[test]
    fn test_format_results_json_empty() {
        let results: Vec<SearchResult> = vec![];
        let output = format_results_json(&results).unwrap();
        
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
        assert!(parsed.is_empty());
    }
    #[test]
    fn test_format_results_jsonl_single() {
        let results = vec![mock_result(0.95, 1, "test")];
        let output = format_results_jsonl(&results).unwrap();
        
        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        assert_eq!(parsed["score"], 0.95);
        assert_eq!(parsed["document_id"], 1);
    }

    #[test]
    fn test_format_results_jsonl_multiple() {
        let results = vec![
            mock_result(0.9, 1, "first"),
            mock_result(0.8, 2, "second"),
        ];
        let output = format_results_jsonl(&results).unwrap();
        
        let lines: Vec<&str> = output.trim().lines().collect();
        assert_eq!(lines.len(), 2);
        
        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        
        assert_eq!(first["document_id"], 1);
        assert_eq!(second["document_id"], 2);
    }

    #[test]
    fn test_format_results_jsonl_each_line_valid_json() {
        let results = vec![
            mock_result(0.9, 1, "one"),
            mock_result(0.8, 2, "two"),
            mock_result(0.7, 3, "three"),
        ];
        let output = format_results_jsonl(&results).unwrap();
        
        for line in output.trim().lines() {
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
            assert!(parsed.is_ok(), "Line should be valid JSON: {}", line);
        }
    }

    #[test]
    fn test_format_results_jsonl_empty() {
        let results: Vec<SearchResult> = vec![];
        let output = format_results_jsonl(&results).unwrap();
        assert!(output.is_empty());
    }
    #[test]
    fn test_format_results_text_basic() {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "doc.txt".to_string());
        
        let results = vec![mock_result_with_metadata(0.9512, 1, "test content", metadata)];
        let output = format_results_text(&results);
        
        assert!(output.contains("0.9512"));
        assert!(output.contains("doc.txt"));
        assert!(output.contains("test content"));
    }

    #[test]
    fn test_format_results_text_no_source() {
        let results = vec![mock_result(0.85, 1, "content")];
        let output = format_results_text(&results);
        assert!(output.contains("?"));
    }

    #[test]
    fn test_format_results_text_truncates_long_content() {
        let long_text = "a".repeat(100);
        let results = vec![mock_result(0.9, 1, &long_text)];
        let output = format_results_text(&results);
        assert!(output.contains("..."));
        assert!(!output.contains(&long_text));
    }

    #[test]
    fn test_format_results_text_multiple() {
        let mut m1 = HashMap::new();
        m1.insert("source".to_string(), "file1.txt".to_string());
        let mut m2 = HashMap::new();
        m2.insert("source".to_string(), "file2.txt".to_string());
        
        let results = vec![
            mock_result_with_metadata(0.95, 1, "first", m1),
            mock_result_with_metadata(0.85, 2, "second", m2),
        ];
        let output = format_results_text(&results);
        
        let lines: Vec<&str> = output.trim().lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("file1.txt"));
        assert!(lines[1].contains("file2.txt"));
    }

    #[test]
    fn test_format_results_text_score_precision() {
        let results = vec![mock_result(0.123456789, 1, "test")];
        let output = format_results_text(&results);
        assert!(output.contains("0.1235") || output.contains("0.1234"));
    }

    #[test]
    fn test_format_results_text_empty() {
        let results: Vec<SearchResult> = vec![];
        let output = format_results_text(&results);
        assert!(output.is_empty());
    }
    #[test]
    fn test_format_results_docs_single() {
        let results = vec![mock_result(0.9, 1, "document content here")];
        let output = format_results_docs(&results);
        
        assert_eq!(output.trim(), "document content here");
    }

    #[test]
    fn test_format_results_docs_multiple() {
        let results = vec![
            mock_result(0.9, 1, "first document"),
            mock_result(0.8, 2, "second document"),
        ];
        let output = format_results_docs(&results);
        
        let lines: Vec<&str> = output.trim().lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "first document");
        assert_eq!(lines[1], "second document");
    }

    #[test]
    fn test_format_results_docs_preserves_content() {
        let long_text = "a".repeat(100);
        let results = vec![mock_result(0.9, 1, &long_text)];
        let output = format_results_docs(&results);
        
        assert!(output.contains(&long_text));
    }

    #[test]
    fn test_format_results_docs_ignores_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "file.txt".to_string());
        
        let results = vec![mock_result_with_metadata(0.9, 1, "content only", metadata)];
        let output = format_results_docs(&results);
        assert_eq!(output.trim(), "content only");
        assert!(!output.contains("file.txt"));
    }

    #[test]
    fn test_format_results_docs_empty() {
        let results: Vec<SearchResult> = vec![];
        let output = format_results_docs(&results);
        assert!(output.is_empty());
    }
    #[test]
    fn test_format_results_json_format() {
        let results = vec![mock_result(0.9, 1, "test")];
        let output = format_results(&results, "json").unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn test_format_results_jsonl_format() {
        let results = vec![mock_result(0.9, 1, "test")];
        let output = format_results(&results, "jsonl").unwrap();
        
        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        assert_eq!(parsed["score"], 0.9);
    }

    #[test]
    fn test_format_results_text_format() {
        let results = vec![mock_result(0.9, 1, "test")];
        let output = format_results(&results, "text").unwrap();
        
        assert!(output.contains("0.9"));
        assert!(output.contains("test"));
    }

    #[test]
    fn test_format_results_docs_format() {
        let results = vec![mock_result(0.9, 1, "test content")];
        let output = format_results(&results, "docs").unwrap();
        
        assert_eq!(output.trim(), "test content");
    }

    #[test]
    fn test_format_results_unknown_format() {
        let results = vec![mock_result(0.9, 1, "test")];
        let result = format_results(&results, "xml");
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown format"));
    }

    #[test]
    fn test_format_results_unknown_format_message() {
        let results = vec![mock_result(0.9, 1, "test")];
        let result = format_results(&results, "csv");
        
        let err = result.unwrap_err().to_string();
        assert!(err.contains("csv"));
        assert!(err.contains("json"));
        assert!(err.contains("jsonl"));
        assert!(err.contains("text"));
        assert!(err.contains("docs"));
    }
    #[test]
    fn test_format_results_special_characters_in_text() {
        let results = vec![mock_result(0.9, 1, "text with \"quotes\" and\ttabs")];
        
        let json_output = format_results_json(&results).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_output).unwrap();
        assert!(parsed[0]["text"].as_str().unwrap().contains("quotes"));
    }

    #[test]
    fn test_format_results_unicode() {
        let results = vec![mock_result(0.9, 1, "日本語テキスト")];
        
        let json_output = format_results_json(&results).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_output).unwrap();
        assert_eq!(parsed[0]["text"], "日本語テキスト");
        
        let docs_output = format_results_docs(&results);
        assert!(docs_output.contains("日本語テキスト"));
    }

    #[test]
    fn test_format_results_zero_score() {
        let results = vec![mock_result(0.0, 1, "zero score doc")];
        
        let text_output = format_results_text(&results);
        assert!(text_output.contains("0.0000"));
        
        let json_output = format_results_json(&results).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_output).unwrap();
        assert_eq!(parsed[0]["score"], 0.0);
    }

    #[test]
    fn test_format_results_high_score() {
        let results = vec![mock_result(1.0, 1, "perfect score")];
        
        let text_output = format_results_text(&results);
        assert!(text_output.contains("1.0000"));
    }

    #[test]
    fn test_format_results_negative_score() {
        let results = vec![mock_result(-0.5, 1, "negative score doc")];
        
        let json_output = format_results_json(&results).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_output).unwrap();
        assert_eq!(parsed[0]["score"], -0.5);
        
        let text_output = format_results_text(&results);
        assert!(text_output.contains("-0.5000"));
    }

    #[test]
    fn test_format_results_large_document_id() {
        let results = vec![mock_result(0.9, 999999, "large id")];
        
        let json_output = format_results_json(&results).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_output).unwrap();
        assert_eq!(parsed[0]["document_id"], 999999);
    }

    #[test]
    fn test_format_results_multiline_text_in_docs() {
        let results = vec![mock_result(0.9, 1, "line1\nline2\nline3")];
        
        let output = format_results_docs(&results);
        assert!(output.contains("line1\nline2\nline3"));
    }

    #[test]
    fn test_format_results_multiline_text_in_text_format() {
        let results = vec![mock_result(0.9, 1, "line1\nline2\nline3")];
        let output = format_results_text(&results);
        assert!(output.contains("line1 line2 line3"));
        assert!(!output.contains("line1\nline2"));
    }
}