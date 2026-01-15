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
    if quiet {
        println!("{:.6}", similarity);
    } else {
        println!("Similarity: {:.4}", similarity);
        println!();
        
        // Interpretation
        let interpretation = if similarity > 0.9 {
            "Very similar / near duplicate"
        } else if similarity > 0.7 {
            "Highly similar"
        } else if similarity > 0.5 {
            "Moderately similar"
        } else if similarity > 0.3 {
            "Somewhat related"
        } else {
            "Not similar"
        };
        
        println!("Interpretation: {}", interpretation);
        
        // Show truncated inputs
        println!();
        println!("Text 1: {}", truncate(&content1, 50));
        println!("Text 2: {}", truncate(&content2, 50));
    }

    Ok(())
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

fn truncate(s: &str, max_len: usize) -> String {
    let s = s.replace('\n', " ");
    if s.len() <= max_len {
        s
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}