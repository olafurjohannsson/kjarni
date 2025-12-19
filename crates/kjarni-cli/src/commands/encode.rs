use anyhow::{anyhow, Result};
use kjarni::{
    Device,
    ModelArchitecture,
    ModelType,
    registry,
    SentenceEncoder
};


use super::util::{resolve_input, stdin_has_data};

pub async fn run(
    input: Option<&str>,
    model: &str,
    model_path: Option<&str>,
    format: &str,
    normalize: bool,
    pooling: &str,
    gpu: bool,
) -> Result<()> {
    // 1. Resolve input
    let text = resolve_input(input)?;
    
    // Determine if batch mode (multiple lines)
    let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
    let is_batch = lines.len() > 1;

    // 2. Resolve model
    let device = if gpu { Device::Wgpu } else { Device::Cpu };

    if model_path.is_some() {
        return Err(anyhow!("--model-path not yet implemented. Use a registry model."));
    }

    let model_type = ModelType::from_cli_name(model)
        .ok_or_else(|| anyhow!("Unknown model: '{}'. Run 'kjarni model list --arch encoder' to see available models.", model))?;

    if model_type.architecture() != ModelArchitecture::Encoder {
        return Err(anyhow!(
            "Model '{}' is not an encoder. Use an encoder model for embedding.",
            model
        ));
    }

    // 3. Load encoder
    eprintln!("Loading model '{}'...", model);
    let encoder = SentenceEncoder::from_registry(model_type, None, device, None).await?;

    // 4. Encode
    if is_batch {
        // Batch mode
        eprintln!("Encoding {} texts...", lines.len());
        let embeddings = encoder.encode_batch_with(&lines, Some(pooling), normalize).await?;

        match format {
            "json" => {
                let output: Vec<_> = lines.iter().zip(embeddings.iter()).map(|(text, emb)| {
                    serde_json::json!({
                        "text": text,
                        "embedding": emb
                    })
                }).collect();
                println!("{}", serde_json::to_string_pretty(&output)?);
            }
            "jsonl" => {
                for (text, emb) in lines.iter().zip(embeddings.iter()) {
                    let obj = serde_json::json!({
                        "text": text,
                        "embedding": emb
                    });
                    println!("{}", serde_json::to_string(&obj)?);
                }
            }
            "raw" => {
                for emb in &embeddings {
                    println!("{}", emb.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(" "));
                }
            }
            _ => return Err(anyhow!("Unknown format: '{}'. Use: json, jsonl, raw", format)),
        }
    } else {
        // Single text mode
        let text_str = lines.first().map(|s| *s).unwrap_or(&text);
        let embedding = encoder.encode_with(text_str, Some(pooling), normalize).await?;

        match format {
            "json" => {
                let output = serde_json::json!({
                    "text": text_str,
                    "embedding": embedding,
                    "model": model,
                    "dim": embedding.len()
                });
                println!("{}", serde_json::to_string_pretty(&output)?);
            }
            "jsonl" | "raw" => {
                // For single text, just output the embedding
                if format == "raw" {
                    println!("{}", embedding.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(" "));
                } else {
                    println!("{}", serde_json::to_string(&embedding)?);
                }
            }
            _ => return Err(anyhow!("Unknown format: '{}'. Use: json, jsonl, raw", format)),
        }
    }

    Ok(())
}