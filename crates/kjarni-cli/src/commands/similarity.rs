//! Compute semantic similarity between two texts

use anyhow::{Result, anyhow};
use std::fs;
use std::path::Path;
use colored::*;
use crate::commands::display;

use kjarni::{Device, ModelArchitecture, ModelType, SentenceEncoder, registry};

pub async fn run(text1: &str, text2: &str, model: &str, gpu: bool, quiet: bool) -> Result<()> {
    let content1 = resolve_text(text1)?;
    let content2 = resolve_text(text2)?;

    let device = if gpu { Device::Wgpu } else { Device::Cpu };

    let model_type =
        ModelType::from_cli_name(model).ok_or_else(|| anyhow!("Unknown model: '{}'", model))?;

    if model_type.architecture() != ModelArchitecture::Bert {
        return Err(anyhow!("Model '{}' is not an encoder.", model));
    }

    if !registry::is_model_downloaded(model)? {
        if !quiet {
            eprintln!("{}", format!("Downloading model '{}'...", model).dimmed());
        }
        registry::download_model(model, false, quiet).await?;
    }

    if !quiet {
        eprintln!("{}", format!("Loading encoder '{}'...", model).dimmed());
        eprintln!("{}", "Computing similarity...".dimmed());
        eprintln!();
    }

    let encoder = SentenceEncoder::from_registry(model_type, None, device, None, None).await?;
    let embeddings = encoder.encode_batch(&[&content1, &content2]).await?;
    let similarity = cosine_similarity(&embeddings[0], &embeddings[1]);

    if quiet {
        print!("{:.6}\n", similarity);
    } else {
        print!("{}", format_pretty(similarity, &content1, &content2));
    }

    Ok(())
}

fn format_pretty(score: f32, text1: &str, text2: &str) -> String {
    let mut output = String::new();

    // Main result: bar + percentage + label
    let bar = display::score_bar(score, 20);
    let pct = display::score_pct(score);
    let label = display::similarity_label(score);

    output.push_str(&format!(
        "  {}  {}  {}\n\n",
        bar, pct, label
    ));

    // Texts being compared
    let t1 = truncate_clean(text1, 60);
    let t2 = truncate_clean(text2, 60);

    output.push_str(&format!(
        "  {} \"{}\"\n",
        "↔".dimmed(),
        t1.white()
    ));
    output.push_str(&format!(
        "  {} \"{}\"\n",
        "↔".dimmed(),
        t2.white()
    ));

    output
}

fn truncate_clean(s: &str, max_len: usize) -> String {
    let clean = s.replace('\n', " ").replace('\r', "");
    if clean.len() <= max_len {
        clean
    } else {
        format!("{}…", &clean[..max_len - 1])
    }
}

fn resolve_text(input: &str) -> Result<String> {
    let path = Path::new(input);
    if path.exists() && path.is_file() {
        fs::read_to_string(path).map_err(|e| anyhow!("Failed to read '{}': {}", input, e))
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


#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Identical vectors should have similarity 1.0"
        );
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - (-1.0)).abs() < 1e-6,
            "Opposite vectors should have similarity -1.0"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "Orthogonal vectors should have similarity 0.0"
        );
    }

    #[test]
    fn test_cosine_similarity_scaled() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0]; 
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Scaled vectors should have similarity 1.0"
        );
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
        assert!(sim.is_finite());
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
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

    #[test]
    fn test_resolve_text_literal() {
        let result = resolve_text("hello world").unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_resolve_text_literal_with_path_like_string() {
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
    #[test]
    fn test_cosine_similarity_small_values() {
        let a = vec![1e-4, 1e-4, 1e-4];
        let b = vec![1e-4, 1e-4, 1e-4];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.is_finite());
        assert!(
            (sim - 1.0).abs() < 1e-3,
            "Identical small vectors should have similarity ~1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_very_small_values_graceful() {
        let a = vec![1e-10, 1e-10, 1e-10];
        let b = vec![1e-10, 1e-10, 1e-10];
        let sim = cosine_similarity(&a, &b);

        assert!(sim.is_finite());
        assert!(!sim.is_nan());
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
    fn test_resolve_text_whitespace_only() {
        let result = resolve_text("   ").unwrap();
        assert_eq!(result, "   ");
    }

}
