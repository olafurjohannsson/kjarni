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

/// A rerank result with index, score, and document text
#[derive(Debug, Clone)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
    pub document: String,
}

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
    // Resolve documents - from args or stdin
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

    // Resolve model
    let device = if gpu { Device::Wgpu } else { Device::Cpu };

    if model_path.is_some() {
        return Err(anyhow!("--model-path not yet implemented."));
    }

    let model_type = ModelType::from_cli_name(model).ok_or_else(|| {
        anyhow!(model_not_found_error(model))
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
        registry::download_model(model, false, quiet).await?;
        if !quiet {
            eprintln!();
        }
    }

    // Load cross-encoder
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

    // Convert to RerankResult
    let rerank_results: Vec<RerankResult> = results
        .iter()
        .map(|(idx, score)| RerankResult {
            index: *idx,
            score: *score,
            document: docs[*idx].clone(),
        })
        .collect();

    // 5. Output
    let output = format_results(&rerank_results, format)?;
    print!("{}", output);

    Ok(())
}

fn model_not_found_error(model: &str) -> String {
    let mut msg = format!("Unknown model: '{}'.", model);
    let suggestions = ModelType::find_similar(model);
    if !suggestions.is_empty() {
        msg.push_str("\n\nDid you mean?");
        for (name, _) in suggestions {
            msg.push_str(&format!("\n  - {}", name));
        }
    }
    msg
}

fn format_results(results: &[RerankResult], format: &str) -> Result<String> {
    match format {
        "json" => format_json(results),
        "jsonl" => format_jsonl(results),
        "text" => Ok(format_text(results)),
        "docs" => Ok(format_docs(results)),
        _ => Err(anyhow!("Unknown format: '{}'. Use: json, jsonl, text, docs", format)),
    }
}

fn format_json(results: &[RerankResult]) -> Result<String> {
    let output: Vec<_> = results.iter().map(|r| {
        serde_json::json!({
            "index": r.index,
            "score": r.score,
            "document": r.document
        })
    }).collect();
    Ok(format!("{}\n", serde_json::to_string_pretty(&output)?))
}

fn format_jsonl(results: &[RerankResult]) -> Result<String> {
    let mut output = String::new();
    for r in results {
        let obj = serde_json::json!({
            "index": r.index,
            "score": r.score,
            "document": r.document
        });
        output.push_str(&serde_json::to_string(&obj)?);
        output.push('\n');
    }
    Ok(output)
}

fn format_text(results: &[RerankResult]) -> String {
    let mut output = String::new();
    for r in results {
        output.push_str(&format!("{:.4}\t{}\t{}\n", r.score, r.index, truncate(&r.document, 80)));
    }
    output
}

fn format_docs(results: &[RerankResult]) -> String {
    let mut output = String::new();
    for r in results {
        output.push_str(&r.document);
        output.push('\n');
    }
    output
}

fn truncate(s: &str, max_len: usize) -> String {
    let s = s.replace('\n', " ").replace('\t', " ");
    if s.len() <= max_len {
        s
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn mock_result(index: usize, score: f32, document: &str) -> RerankResult {
        RerankResult {
            index,
            score,
            document: document.to_string(),
        }
    }

    #[test]
    fn test_truncate_short_string() {
        assert_eq!(truncate("hello world", 80), "hello world");
        assert_eq!(truncate("short", 20), "short");
    }

    #[test]
    fn test_truncate_exact_length() {
        let text = "a".repeat(80);
        assert_eq!(truncate(&text, 80), text);
    }

    #[test]
    fn test_truncate_long_string() {
        let text = "a".repeat(100);
        let result = truncate(&text, 80);
        assert_eq!(result.len(), 80);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_truncate_replaces_newlines() {
        assert_eq!(truncate("hello\nworld", 80), "hello world");
        assert_eq!(truncate("line1\nline2\nline3", 80), "line1 line2 line3");
    }

    #[test]
    fn test_truncate_replaces_tabs() {
        assert_eq!(truncate("hello\tworld", 80), "hello world");
        assert_eq!(truncate("col1\tcol2\tcol3", 80), "col1 col2 col3");
    }

    #[test]
    fn test_truncate_mixed_whitespace() {
        assert_eq!(truncate("hello\n\tworld", 80), "hello  world");
    }

    #[test]
    fn test_truncate_newlines_then_truncate() {
        let text = "line one\nline two\nline three\nline four\nline five\nline six";
        let result = truncate(text, 30);
        assert_eq!(result.len(), 30);
        assert!(result.ends_with("..."));
        assert!(!result.contains('\n'));
    }

    #[test]
    fn test_truncate_empty() {
        assert_eq!(truncate("", 80), "");
    }

    #[test]
    fn test_truncate_only_whitespace() {
        assert_eq!(truncate("\n\t\n", 80), "   ");
    }

    #[test]
    fn test_truncate_minimum_length() {
        assert_eq!(truncate("hello", 4), "h...");
    }

    #[test]
    fn test_model_not_found_error_basic() {
        let error = model_not_found_error("unknown-model");
        assert!(error.contains("Unknown model: 'unknown-model'"));
    }

    #[test]
    fn test_model_not_found_error_with_suggestions() {
        let error = model_not_found_error("minilm");
        assert!(error.contains("Unknown model"));
    }

    #[test]
    fn test_model_not_found_error_close_match() {
        let error = model_not_found_error("minilm-l6");
        assert!(error.contains("Unknown model"));
    }
    #[test]
    fn test_format_json_single() {
        let results = vec![mock_result(0, 0.95, "test document")];
        let output = format_json(&results).unwrap();

        let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["index"], 0);
        assert_eq!(parsed[0]["score"], 0.95);
        assert_eq!(parsed[0]["document"], "test document");
    }

    #[test]
    fn test_format_json_multiple() {
        let results = vec![
            mock_result(2, 0.95, "best match"),
            mock_result(0, 0.80, "second match"),
            mock_result(1, 0.60, "third match"),
        ];
        let output = format_json(&results).unwrap();

        let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0]["index"], 2);
        assert_eq!(parsed[1]["index"], 0);
        assert_eq!(parsed[2]["index"], 1);
    }

    #[test]
    fn test_format_json_preserves_order() {
        let results = vec![
            mock_result(5, 0.9, "a"),
            mock_result(1, 0.8, "b"),
            mock_result(3, 0.7, "c"),
        ];
        let output = format_json(&results).unwrap();

        let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed[0]["index"], 5);
        assert_eq!(parsed[1]["index"], 1);
        assert_eq!(parsed[2]["index"], 3);
    }

    #[test]
    fn test_format_json_empty() {
        let results: Vec<RerankResult> = vec![];
        let output = format_json(&results).unwrap();

        let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
        assert!(parsed.is_empty());
    }
    #[test]
    fn test_format_jsonl_single() {
        let results = vec![mock_result(0, 0.95, "doc")];
        let output = format_jsonl(&results).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        assert_eq!(parsed["index"], 0);
        assert_eq!(parsed["score"], 0.95);
    }

    #[test]
    fn test_format_jsonl_multiple() {
        let results = vec![
            mock_result(0, 0.9, "first"),
            mock_result(1, 0.8, "second"),
        ];
        let output = format_jsonl(&results).unwrap();

        let lines: Vec<&str> = output.trim().lines().collect();
        assert_eq!(lines.len(), 2);

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();

        assert_eq!(first["index"], 0);
        assert_eq!(second["index"], 1);
    }

    #[test]
    fn test_format_jsonl_each_line_valid() {
        let results = vec![
            mock_result(0, 0.9, "a"),
            mock_result(1, 0.8, "b"),
            mock_result(2, 0.7, "c"),
        ];
        let output = format_jsonl(&results).unwrap();

        for line in output.trim().lines() {
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
            assert!(parsed.is_ok(), "Line should be valid JSON: {}", line);
        }
    }

    #[test]
    fn test_format_jsonl_empty() {
        let results: Vec<RerankResult> = vec![];
        let output = format_jsonl(&results).unwrap();
        assert!(output.is_empty());
    }
    #[test]
    fn test_format_text_basic() {
        let results = vec![mock_result(0, 0.9512, "test document")];
        let output = format_text(&results);

        assert!(output.contains("0.9512"));
        assert!(output.contains("\t0\t"));
        assert!(output.contains("test document"));
    }

    #[test]
    fn test_format_text_multiple() {
        let results = vec![
            mock_result(2, 0.95, "best"),
            mock_result(0, 0.80, "second"),
        ];
        let output = format_text(&results);

        let lines: Vec<&str> = output.trim().lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("0.9500"));
        assert!(lines[0].contains("\t2\t"));
        assert!(lines[1].contains("0.8000"));
        assert!(lines[1].contains("\t0\t"));
    }

    #[test]
    fn test_format_text_truncates_long_document() {
        let long_doc = "a".repeat(100);
        let results = vec![mock_result(0, 0.9, &long_doc)];
        let output = format_text(&results);

        assert!(output.contains("..."));
        assert!(!output.contains(&long_doc));
    }

    #[test]
    fn test_format_text_score_precision() {
        let results = vec![mock_result(0, 0.123456789, "doc")];
        let output = format_text(&results);

        // Should have 4 decimal places
        assert!(output.contains("0.1235") || output.contains("0.1234"));
    }

    #[test]
    fn test_format_text_empty() {
        let results: Vec<RerankResult> = vec![];
        let output = format_text(&results);
        assert!(output.is_empty());
    }

    #[test]
    fn test_format_text_negative_score() {
        let results = vec![mock_result(0, -0.5, "doc")];
        let output = format_text(&results);
        assert!(output.contains("-0.5000"));
    }
    #[test]
    fn test_format_docs_single() {
        let results = vec![mock_result(0, 0.9, "document content")];
        let output = format_docs(&results);
        assert_eq!(output.trim(), "document content");
    }

    #[test]
    fn test_format_docs_multiple() {
        let results = vec![
            mock_result(2, 0.95, "best match"),
            mock_result(0, 0.80, "second match"),
        ];
        let output = format_docs(&results);

        let lines: Vec<&str> = output.trim().lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "best match");
        assert_eq!(lines[1], "second match");
    }

    #[test]
    fn test_format_docs_preserves_full_content() {
        let long_doc = "a".repeat(200);
        let results = vec![mock_result(0, 0.9, &long_doc)];
        let output = format_docs(&results);
        assert!(output.contains(&long_doc));
    }

    #[test]
    fn test_format_docs_ignores_score() {
        let results = vec![mock_result(0, 0.95, "just the doc")];
        let output = format_docs(&results);

        assert_eq!(output.trim(), "just the doc");
        assert!(!output.contains("0.95"));
    }

    #[test]
    fn test_format_docs_ignores_index() {
        let results = vec![mock_result(42, 0.9, "content only")];
        let output = format_docs(&results);

        assert_eq!(output.trim(), "content only");
        assert!(!output.contains("42"));
    }

    #[test]
    fn test_format_docs_empty() {
        let results: Vec<RerankResult> = vec![];
        let output = format_docs(&results);
        assert!(output.is_empty());
    }
    #[test]
    fn test_format_results_json() {
        let results = vec![mock_result(0, 0.9, "test")];
        let output = format_results(&results, "json").unwrap();

        let parsed: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn test_format_results_jsonl() {
        let results = vec![mock_result(0, 0.9, "test")];
        let output = format_results(&results, "jsonl").unwrap();

        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        assert_eq!(parsed["index"], 0);
    }

    #[test]
    fn test_format_results_text() {
        let results = vec![mock_result(0, 0.9, "test")];
        let output = format_results(&results, "text").unwrap();

        assert!(output.contains("0.9"));
        assert!(output.contains("test"));
    }

    #[test]
    fn test_format_results_docs() {
        let results = vec![mock_result(0, 0.9, "test content")];
        let output = format_results(&results, "docs").unwrap();

        assert_eq!(output.trim(), "test content");
    }

    #[test]
    fn test_format_results_unknown_format() {
        let results = vec![mock_result(0, 0.9, "test")];
        let result = format_results(&results, "xml");

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown format"));
    }

    #[test]
    fn test_format_results_unknown_format_lists_valid() {
        let results = vec![mock_result(0, 0.9, "test")];
        let result = format_results(&results, "csv");

        let err = result.unwrap_err().to_string();
        assert!(err.contains("json"));
        assert!(err.contains("jsonl"));
        assert!(err.contains("text"));
        assert!(err.contains("docs"));
    }
    #[test]
    fn test_special_characters_in_document() {
        let results = vec![mock_result(0, 0.9, "doc with \"quotes\" and\ttabs")];

        let json_output = format_json(&results).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_output).unwrap();
        assert!(parsed[0]["document"].as_str().unwrap().contains("quotes"));
    }

    #[test]
    fn test_unicode_document() {
        let results = vec![mock_result(0, 0.9, "日本語ドキュメント")];

        let json_output = format_json(&results).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_output).unwrap();
        assert_eq!(parsed[0]["document"], "日本語ドキュメント");

        let docs_output = format_docs(&results);
        assert!(docs_output.contains("日本語ドキュメント"));
    }

    #[test]
    fn test_zero_score() {
        let results = vec![mock_result(0, 0.0, "zero score")];

        let text_output = format_text(&results);
        assert!(text_output.contains("0.0000"));

        let json_output = format_json(&results).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_output).unwrap();
        assert_eq!(parsed[0]["score"], 0.0);
    }

    #[test]
    fn test_large_index() {
        let results = vec![mock_result(999999, 0.9, "large index")];

        let json_output = format_json(&results).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_output).unwrap();
        assert_eq!(parsed[0]["index"], 999999);

        let text_output = format_text(&results);
        assert!(text_output.contains("999999"));
    }

    #[test]
    fn test_multiline_document_in_text_format() {
        let results = vec![mock_result(0, 0.9, "line1\nline2\nline3")];
        let output = format_text(&results);
        assert!(output.contains("line1 line2 line3"));
        assert!(!output.contains("line1\nline2"));
    }

    #[test]
    fn test_multiline_document_in_docs_format() {
        let results = vec![mock_result(0, 0.9, "line1\nline2")];

        let output = format_docs(&results);
        assert!(output.contains("line1\nline2"));
    }

    #[test]
    fn test_empty_document() {
        let results = vec![mock_result(0, 0.9, "")];

        let json_output = format_json(&results).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_output).unwrap();
        assert_eq!(parsed[0]["document"], "");

        let docs_output = format_docs(&results);
        assert_eq!(docs_output, "\n");
    }

    #[test]
    fn test_very_high_score() {
        let results = vec![mock_result(0, 15.5, "high score")];

        let text_output = format_text(&results);
        assert!(text_output.contains("15.5000"));
    }

    #[test]
    fn test_many_results() {
        let results: Vec<RerankResult> = (0..100)
            .map(|i| mock_result(i, 1.0 - (i as f32 * 0.01), &format!("doc {}", i)))
            .collect();

        let json_output = format_json(&results).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_output).unwrap();
        assert_eq!(parsed.len(), 100);

        let text_output = format_text(&results);
        let lines: Vec<&str> = text_output.trim().lines().collect();
        assert_eq!(lines.len(), 100);
    }
}