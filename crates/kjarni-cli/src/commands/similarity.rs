//! Compute semantic similarity between two texts

use anyhow::{anyhow, Result};
use std::fs;
use std::path::Path;

use kjarni::{
    registry,
    SentenceEncoder,
    ModelArchitecture,
    ModelType,
    Device,
};

pub async fn run(
    text1: &str,
    text2: &str,
    model: &str,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    // 1. Resolve texts (could be file paths)
    let content1 = resolve_text(text1)?;
    let content2 = resolve_text(text2)?;

    // 2. Load encoder
    let device = if gpu { Device::Wgpu } else { Device::Cpu };
    let model_type = ModelType::from_cli_name(model)
        .ok_or_else(|| anyhow!("Unknown model: '{}'", model))?;

    if model_type.architecture() != ModelArchitecture::Bert {
        return Err(anyhow!("Model '{}' is not an encoder.", model));
    }

    if !registry::is_model_downloaded(model)? {
        if !quiet {
            eprintln!("Downloading model '{}'...", model);
        }
        registry::download_model(model, false, quiet).await?;
    }

    if !quiet {
        eprintln!("Loading encoder '{}'...", model);
    }

    let encoder = SentenceEncoder::from_registry(model_type, None, device, None, None).await?;

    // 3. Encode both texts
    if !quiet {
        eprintln!("Computing similarity...");
        eprintln!();
    }

    let embeddings = encoder.encode_batch(&[&content1, &content2]).await?;
    
    // 4. Compute cosine similarity
    let similarity = cosine_similarity(&embeddings[0], &embeddings[1]);

    // 5. Output
    let output = format_output(similarity, &content1, &content2, quiet);
    print!("{}", output);

    Ok(())
}

fn format_output(similarity: f32, text1: &str, text2: &str, quiet: bool) -> String {
    if quiet {
        format!("{:.6}\n", similarity)
    } else {
        let mut output = String::new();
        output.push_str(&format!("Similarity: {:.4}\n", similarity));
        output.push('\n');
        output.push_str(&format!("Interpretation: {}\n", interpret_similarity(similarity)));
        output.push('\n');
        output.push_str(&format!("Text 1: {}\n", truncate(text1, 50)));
        output.push_str(&format!("Text 2: {}\n", truncate(text2, 50)));
        output
    }
}

fn resolve_text(input: &str) -> Result<String> {
    let path = Path::new(input);
    if path.exists() && path.is_file() {
        fs::read_to_string(path)
            .map_err(|e| anyhow!("Failed to read '{}': {}", input, e))
    } else {
        Ok(input.to_string())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-9);
    dot / denom
}

fn interpret_similarity(score: f32) -> &'static str {
    if score > 0.9 {
        "Very similar / near duplicate"
    } else if score > 0.7 {
        "Highly similar"
    } else if score > 0.5 {
        "Moderately similar"
    } else if score > 0.3 {
        "Somewhat related"
    } else {
        "Not similar"
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    let s = s.replace('\n', " ");
    if s.len() <= max_len {
        s
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    // =========================================================================
    // cosine_similarity tests
    // =========================================================================

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6, "Identical vectors should have similarity 1.0");
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6, "Opposite vectors should have similarity -1.0");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "Orthogonal vectors should have similarity 0.0");
    }

    #[test]
    fn test_cosine_similarity_scaled() {
        // Cosine similarity should be scale-invariant
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0]; // 2x scaled
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6, "Scaled vectors should have similarity 1.0");
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0, "Different length vectors should return 0.0");
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let sim = cosine_similarity(&a, &b);
        // Empty vectors - edge case, should handle gracefully
        assert!(sim.is_finite());
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        // Should handle zero vector without NaN (due to max(1e-9))
        assert!(sim.is_finite());
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_both_zero() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.is_finite());
    }

    #[test]
    fn test_cosine_similarity_realistic_embeddings() {
        // Simulating 384-dim embeddings
        let a: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).cos()).collect();
        let sim = cosine_similarity(&a, &b);
        
        assert!(sim >= -1.0 && sim <= 1.0, "Similarity should be in [-1, 1]");
    }

    #[test]
    fn test_cosine_similarity_negative_values() {
        let a = vec![-1.0, -2.0, -3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_mixed_signs() {
        let a = vec![1.0, -1.0, 1.0];
        let b = vec![-1.0, 1.0, -1.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    // =========================================================================
    // interpret_similarity tests
    // =========================================================================

    #[test]
    fn test_interpret_similarity_very_similar() {
        assert_eq!(interpret_similarity(0.95), "Very similar / near duplicate");
        assert_eq!(interpret_similarity(0.91), "Very similar / near duplicate");
        assert_eq!(interpret_similarity(1.0), "Very similar / near duplicate");
    }

    #[test]
    fn test_interpret_similarity_highly_similar() {
        assert_eq!(interpret_similarity(0.9), "Highly similar");
        assert_eq!(interpret_similarity(0.8), "Highly similar");
        assert_eq!(interpret_similarity(0.71), "Highly similar");
    }

    #[test]
    fn test_interpret_similarity_moderately_similar() {
        assert_eq!(interpret_similarity(0.7), "Moderately similar");
        assert_eq!(interpret_similarity(0.6), "Moderately similar");
        assert_eq!(interpret_similarity(0.51), "Moderately similar");
    }

    #[test]
    fn test_interpret_similarity_somewhat_related() {
        assert_eq!(interpret_similarity(0.5), "Somewhat related");
        assert_eq!(interpret_similarity(0.4), "Somewhat related");
        assert_eq!(interpret_similarity(0.31), "Somewhat related");
    }

    #[test]
    fn test_interpret_similarity_not_similar() {
        assert_eq!(interpret_similarity(0.3), "Not similar");
        assert_eq!(interpret_similarity(0.1), "Not similar");
        assert_eq!(interpret_similarity(0.0), "Not similar");
        assert_eq!(interpret_similarity(-0.5), "Not similar");
    }

    #[test]
    fn test_interpret_similarity_boundary_values() {
        // Test exact boundary values
        assert_eq!(interpret_similarity(0.9), "Highly similar");      // <= 0.9
        assert_eq!(interpret_similarity(0.90001), "Very similar / near duplicate"); // > 0.9
        assert_eq!(interpret_similarity(0.7), "Moderately similar");  // <= 0.7
        assert_eq!(interpret_similarity(0.70001), "Highly similar");  // > 0.7
        assert_eq!(interpret_similarity(0.5), "Somewhat related");    // <= 0.5
        assert_eq!(interpret_similarity(0.50001), "Moderately similar"); // > 0.5
        assert_eq!(interpret_similarity(0.3), "Not similar");         // <= 0.3
        assert_eq!(interpret_similarity(0.30001), "Somewhat related"); // > 0.3
    }

    // =========================================================================
    // truncate tests
    // =========================================================================

    #[test]
    fn test_truncate_short_string() {
        assert_eq!(truncate("hello", 50), "hello");
        assert_eq!(truncate("short text", 20), "short text");
    }

    #[test]
    fn test_truncate_exact_length() {
        let text = "a".repeat(50);
        assert_eq!(truncate(&text, 50), text);
    }

    #[test]
    fn test_truncate_long_string() {
        let text = "a".repeat(100);
        let result = truncate(&text, 50);
        assert_eq!(result.len(), 50);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_truncate_replaces_newlines() {
        assert_eq!(truncate("hello\nworld", 50), "hello world");
        assert_eq!(truncate("line1\nline2\nline3", 50), "line1 line2 line3");
    }

    #[test]
    fn test_truncate_newlines_then_truncate() {
        let text = "line one\nline two\nline three\nline four\nline five";
        let result = truncate(text, 30);
        assert_eq!(result.len(), 30);
        assert!(result.ends_with("..."));
        assert!(!result.contains('\n'));
    }

    #[test]
    fn test_truncate_empty() {
        assert_eq!(truncate("", 50), "");
    }

    #[test]
    fn test_truncate_only_newlines() {
        assert_eq!(truncate("\n\n\n", 50), "   ");
    }

    #[test]
    fn test_truncate_minimum_length() {
        assert_eq!(truncate("hello", 4), "h...");
    }

    // =========================================================================
    // resolve_text tests
    // =========================================================================

    #[test]
    fn test_resolve_text_literal() {
        let result = resolve_text("hello world").unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_resolve_text_literal_with_path_like_string() {
        // A string that looks like a path but doesn't exist
        let result = resolve_text("/nonexistent/path/to/file.txt").unwrap();
        assert_eq!(result, "/nonexistent/path/to/file.txt");
    }

    #[test]
    fn test_resolve_text_from_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "content from file").unwrap();
        
        let result = resolve_text(temp_file.path().to_str().unwrap()).unwrap();
        assert_eq!(result.trim(), "content from file");
    }

    #[test]
    fn test_resolve_text_from_file_multiline() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "line one").unwrap();
        writeln!(temp_file, "line two").unwrap();
        writeln!(temp_file, "line three").unwrap();
        
        let result = resolve_text(temp_file.path().to_str().unwrap()).unwrap();
        assert!(result.contains("line one"));
        assert!(result.contains("line two"));
        assert!(result.contains("line three"));
    }

    #[test]
    fn test_resolve_text_from_file_unicode() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "日本語テキスト").unwrap();
        
        let result = resolve_text(temp_file.path().to_str().unwrap()).unwrap();
        assert!(result.contains("日本語テキスト"));
    }

    #[test]
    fn test_resolve_text_empty_string() {
        let result = resolve_text("").unwrap();
        assert_eq!(result, "");
    }

    // =========================================================================
    // format_output tests
    // =========================================================================

    #[test]
    fn test_format_output_quiet() {
        let output = format_output(0.85, "text one", "text two", true);
        assert_eq!(output.trim(), "0.850000");
    }

    #[test]
    fn test_format_output_quiet_precision() {
        let output = format_output(0.123456789, "a", "b", true);
        assert!(output.contains("0.123457")); // 6 decimal places, rounded
    }

    #[test]
    fn test_format_output_verbose() {
        let output = format_output(0.85, "text one", "text two", false);
        
        assert!(output.contains("Similarity: 0.8500"));
        assert!(output.contains("Interpretation: Highly similar"));
        assert!(output.contains("Text 1: text one"));
        assert!(output.contains("Text 2: text two"));
    }

    #[test]
    fn test_format_output_verbose_truncates_long_text() {
        let long_text = "a".repeat(100);
        let output = format_output(0.5, &long_text, "short", false);
        
        assert!(output.contains("..."));
        assert!(!output.contains(&long_text));
    }

    #[test]
    fn test_format_output_very_similar() {
        let output = format_output(0.95, "a", "b", false);
        assert!(output.contains("Very similar / near duplicate"));
    }

    #[test]
    fn test_format_output_not_similar() {
        let output = format_output(0.1, "a", "b", false);
        assert!(output.contains("Not similar"));
    }

    #[test]
    fn test_format_output_negative_similarity() {
        let output = format_output(-0.5, "a", "b", false);
        assert!(output.contains("-0.5000"));
        assert!(output.contains("Not similar"));
    }

    #[test]
    fn test_format_output_perfect_similarity() {
        let output = format_output(1.0, "a", "b", false);
        assert!(output.contains("1.0000"));
        assert!(output.contains("Very similar / near duplicate"));
    }

    // =========================================================================
    // Integration-like tests
    // =========================================================================

    #[test]
    fn test_full_workflow_literal_texts() {
        // Simulate the non-encoder parts of the workflow
        let text1 = "The quick brown fox jumps over the lazy dog";
        let text2 = "A fast auburn fox leaps above a sleepy canine";
        
        let content1 = resolve_text(text1).unwrap();
        let content2 = resolve_text(text2).unwrap();
        
        assert_eq!(content1, text1);
        assert_eq!(content2, text2);
        
        // Simulate similarity (would come from encoder)
        let similarity = 0.75;
        
        let output = format_output(similarity, &content1, &content2, false);
        assert!(output.contains("Highly similar"));
    }

    #[test]
    fn test_full_workflow_file_and_literal() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "content from a file").unwrap();
        
        let text1 = temp_file.path().to_str().unwrap();
        let text2 = "literal text input";
        
        let content1 = resolve_text(text1).unwrap();
        let content2 = resolve_text(text2).unwrap();
        
        assert!(content1.contains("content from a file"));
        assert_eq!(content2, "literal text input");
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    fn test_cosine_similarity_very_small_values() {
        let a = vec![1e-10, 1e-10, 1e-10];
        let b = vec![1e-10, 1e-10, 1e-10];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.is_finite());
        assert!((sim - 1.0).abs() < 1e-3); // Should be close to 1.0
    }

    #[test]
    fn test_cosine_similarity_very_large_values() {
        let a = vec![1e10, 1e10, 1e10];
        let b = vec![1e10, 1e10, 1e10];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.is_finite());
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_format_output_unicode_text() {
        let output = format_output(0.8, "日本語テキスト", "中文文本", false);
        assert!(output.contains("日本語テキスト"));
        assert!(output.contains("中文文本"));
    }

    #[test]
    fn test_format_output_newlines_in_text() {
        let output = format_output(0.8, "line1\nline2", "other\ntext", false);
        // truncate should replace newlines
        assert!(output.contains("line1 line2"));
        assert!(output.contains("other text"));
    }

    #[test]
    fn test_resolve_text_whitespace_only() {
        let result = resolve_text("   ").unwrap();
        assert_eq!(result, "   ");
    }

    #[test]
    fn test_interpret_similarity_edge_negative() {
        // Very negative similarity
        assert_eq!(interpret_similarity(-1.0), "Not similar");
    }

    #[test]
    fn test_interpret_similarity_edge_over_one() {
        // Due to floating point, might get > 1.0
        assert_eq!(interpret_similarity(1.001), "Very similar / near duplicate");
    }
}