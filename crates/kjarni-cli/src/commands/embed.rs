//! Text embedding command using the high-level Embedder API.

use anyhow::{anyhow, Result};
use kjarni::{PoolingStrategy, embedder::Embedder};

pub async fn run(
    input: Option<&str>,
    model: &str,
    model_path: Option<&str>,
    format: &str,
    normalize: bool,
    pooling: &str,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    // Resolve input
    let text = crate::commands::util::resolve_input(input)?;

    // Check for batch mode
    let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
    let is_batch = lines.len() > 1;

    // Parse pooling strategy
    let pooling_strategy: PoolingStrategy = pooling.parse()
        .map_err(|e: String| anyhow!("Invalid pooling strategy: {}", e))?;

    // Build embedder
    let mut builder = if let Some(path) = model_path {
        Embedder::from_path(path)
    } else {
        Embedder::builder(model)
    };

    builder = builder
        .normalize(normalize)
        .pooling(pooling_strategy);

    if gpu {
        builder = builder.gpu();
    }

    let embedder = builder
        .build()
        .await
        .map_err(|e| anyhow!("Failed to load embedder: {}", e))?;

    // Generate embeddings
    if is_batch {
        let embeddings = embedder.embed_batch(&lines).await
            .map_err(|e| anyhow!("Embedding failed: {}", e))?;
        let output = format_batch(&lines, &embeddings, format)?;
        
        print!("{}", output);
    } else {
        let text_str = lines.first().copied().unwrap_or(&text);
        let embedding = embedder.embed(text_str).await
            .map_err(|e| anyhow!("Embedding failed: {}", e))?;
        let output = format_single(text_str, &embedding, model, format)?;
        print!("{}", output);
    }

    Ok(())
}

fn format_single(text: &str, embedding: &[f32], model: &str, format: &str) -> Result<String> {
    match format {
        "json" => {
            let output = serde_json::json!({
                "text": text,
                "embedding": embedding,
                "model": model,
                "dim": embedding.len()
            });
            Ok(format!("{}\n", serde_json::to_string_pretty(&output)?))
        }
        "jsonl" => {
            Ok(format!("{}\n", serde_json::to_string(&embedding)?))
        }
        "raw" => {
            Ok(format!("{}\n", embedding.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(" ")))
        }
        _ => Err(anyhow!("Unknown format: '{}'. Use: json, jsonl, raw", format)),
    }
}

fn format_batch(texts: &[&str], embeddings: &[Vec<f32>], format: &str) -> Result<String> {
    match format {
        "json" => {
            let output: Vec<_> = texts.iter().zip(embeddings.iter()).map(|(text, emb)| {
                serde_json::json!({
                    "text": text,
                    "embedding": emb
                })
            }).collect();
            Ok(format!("{}\n", serde_json::to_string_pretty(&output)?))
        }
        "jsonl" => {
            let mut result = String::new();
            for (text, emb) in texts.iter().zip(embeddings.iter()) {
                let obj = serde_json::json!({
                    "text": text,
                    "embedding": emb
                });
                result.push_str(&serde_json::to_string(&obj)?);
                result.push('\n');
            }
            Ok(result)
        }
        "raw" => {
            let mut result = String::new();
            for emb in embeddings {
                result.push_str(&emb.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(" "));
                result.push('\n');
            }
            Ok(result)
        }
        _ => Err(anyhow!("Unknown format: '{}'. Use: json, jsonl, raw", format)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_single_json() {
        let text = "hello world";
        let embedding = vec![0.1, 0.2, 0.3];
        let model = "test-model";

        let result = format_single(text, &embedding, model, "json").unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

        assert_eq!(parsed["text"], "hello world");
        assert_eq!(parsed["model"], "test-model");
        assert_eq!(parsed["dim"], 3);
        assert_eq!(parsed["embedding"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_format_single_jsonl() {
        let text = "hello";
        let embedding = vec![1.0, 2.0, 3.0];

        let result = format_single(text, &embedding, "model", "jsonl").unwrap();
        let parsed: Vec<f32> = serde_json::from_str(result.trim()).unwrap();

        assert_eq!(parsed, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_format_single_raw() {
        let text = "test";
        let embedding = vec![0.5, 1.5, 2.5];

        let result = format_single(text, &embedding, "model", "raw").unwrap();

        assert_eq!(result.trim(), "0.5 1.5 2.5");
    }

    #[test]
    fn test_format_single_unknown_format() {
        let result = format_single("text", &[0.1], "model", "xml");

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown format"));
    }

    #[test]
    fn test_format_single_empty_embedding() {
        let result = format_single("text", &[], "model", "json").unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

        assert_eq!(parsed["dim"], 0);
        assert!(parsed["embedding"].as_array().unwrap().is_empty());
    }
    #[test]
    fn test_format_batch_json() {
        let texts = vec!["hello", "world"];
        let embeddings = vec![vec![0.1, 0.2], vec![0.3, 0.4]];

        let result = format_batch(&texts, &embeddings, "json").unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0]["text"], "hello");
        assert_eq!(parsed[1]["text"], "world");
        assert_eq!(parsed[0]["embedding"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_format_batch_jsonl() {
        let texts = vec!["a", "b"];
        let embeddings = vec![vec![1.0], vec![2.0]];

        let result = format_batch(&texts, &embeddings, "jsonl").unwrap();
        let lines: Vec<&str> = result.trim().lines().collect();

        assert_eq!(lines.len(), 2);

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();

        assert_eq!(first["text"], "a");
        assert_eq!(second["text"], "b");
    }

    #[test]
    fn test_format_batch_raw() {
        let texts = vec!["x", "y"];
        let embeddings = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let result = format_batch(&texts, &embeddings, "raw").unwrap();
        let lines: Vec<&str> = result.trim().lines().collect();

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "1 2");
        assert_eq!(lines[1], "3 4");
    }

    #[test]
    fn test_format_batch_unknown_format() {
        let result = format_batch(&["a"], &[vec![1.0]], "csv");

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown format"));
    }

    #[test]
    fn test_format_batch_empty() {
        let texts: Vec<&str> = vec![];
        let embeddings: Vec<Vec<f32>> = vec![];

        let result = format_batch(&texts, &embeddings, "json").unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();

        assert!(parsed.is_empty());
    }

    #[test]
    fn test_format_batch_raw_preserves_precision() {
        let texts = vec!["test"];
        let embeddings = vec![vec![0.123456789]];

        let result = format_batch(&texts, &embeddings, "raw").unwrap();

        // Should contain the float representation
        assert!(result.contains("0.123456789") || result.contains("0.12345679"));
    }
}