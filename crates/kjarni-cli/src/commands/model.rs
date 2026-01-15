use anyhow::Result;
use kjarni::registry;
use kjarni::ModelType;
use kjarni_cli::ModelCommands;
use std::path::Path;

pub async fn run(action: ModelCommands) -> Result<()> {
    match action {
        ModelCommands::List { arch, task, downloaded } => list(arch, task, downloaded),
        ModelCommands::Download { name, gguf, quiet } => download(&name, gguf, quiet).await,
        ModelCommands::Remove { name } => remove(&name),
        ModelCommands::Info { name } => info(&name),
        ModelCommands::Search { query } => search(&query),
    }
}

fn remove(name: &str) -> Result<()> {
    let model_path = registry::model_path(name)?;

    if !model_path.exists() {
        println!("Model '{}' is not downloaded.", name);
        return Ok(());
    }

    // Show what will be deleted
    let size = dir_size(&model_path)?;
    println!();
    println!("This will delete:");
    println!("  Model: {}", name);
    println!("  Path:  {}", model_path.display());
    println!("  Size:  {}", format_bytes(size));
    println!();

    // Confirm
    print!("Are you sure? [y/N] ");
    std::io::Write::flush(&mut std::io::stdout())?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    if input.trim().to_lowercase() == "y" {
        std::fs::remove_dir_all(&model_path)?;
        println!("✓ Removed {}", name);
    } else {
        println!("Cancelled.");
    }

    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    }
}
fn dir_size(path: &Path) -> Result<u64> {
    let mut size = 0;
    for entry in walkdir::WalkDir::new(path) {
        let entry = entry?;
        if entry.file_type().is_file() {
            size += entry.metadata()?.len();
        }
    }
    Ok(size)
}
fn search(query: &str) -> Result<()> {
    let results = ModelType::search(query);

    if results.is_empty() {
        println!("No models found matching '{}'", query);
        return Ok(());
    }

    println!();
    println!("Search results for '{}':", query);
    println!("{}", "-".repeat(60));

    for (model_type, score) in results.iter().take(10) {
        let info = model_type.info();
        let downloaded = model_type.is_downloaded(&registry::cache_dir());
        let status = if downloaded { "✓" } else { " " };

        println!(
            "  {} {:<28} {:>8}  {}",
            status,
            model_type.cli_name(),
            format_params(info.params_millions),
            truncate(info.description, 30)
        );
    }

    println!();
    Ok(())
}

/// Format parameters in a human-readable way (e.g., "1.5B", "345M")
pub fn format_params(millions: usize) -> String {
    if millions >= 1000 {
        format!("{:.1}B", millions as f64 / 1000.0)
    } else {
        format!("{}M", millions)
    }
}


fn list(filter_arch: Option<String>, filter_task: Option<String>, only_downloaded: bool) -> Result<()> {
    let models = registry::list_models();
    let arch_filter = filter_arch.as_deref().map(|s| s.to_lowercase());
    let task_filter = filter_task.as_deref().map(|s| s.to_lowercase());

    let groups = [
        "LLM (Decoder)",
        "Seq2Seq",
        "Embedding",
        "Re-Ranker",
        "Classifier",
    ];

    // Calculate totals
    let downloaded_count = models.iter().filter(|m| m.downloaded).count();
    let total_count = models.len();

    println!();
    println!("Cache: {}", registry::cache_dir().display());
    println!("Models: {}/{} downloaded", downloaded_count, total_count);
    
    // Show active filters
    if arch_filter.is_some() || task_filter.is_some() || only_downloaded {
        let mut filters = Vec::new();
        if let Some(ref a) = arch_filter {
            filters.push(format!("arch={}", a));
        }
        if let Some(ref t) = task_filter {
            filters.push(format!("task={}", t));
        }
        if only_downloaded {
            filters.push("downloaded".to_string());
        }
        println!("Filter: {}", filters.join(", "));
    }
    println!();

    let mut any_shown = false;

    for group in groups {
        let group_models: Vec<_> = models
            .iter()
            .filter(|m| {
                let m_group = m.model_type.display_group();
                if m_group != group {
                    return false;
                }

                // Filter by downloaded
                if only_downloaded && !m.downloaded {
                    return false;
                }

                // Filter by architecture
                if let Some(ref f) = arch_filter {
                    let arch = format!("{:?}", m.architecture).to_lowercase();
                    if !arch.contains(f) {
                        return false;
                    }
                }

                // Filter by task
                if let Some(ref f) = task_filter {
                    let task_match = match f.as_str() {
                        "chat" | "llm" | "decoder" => m_group == "LLM (Decoder)",
                        "embedding" | "embed" | "encoder" => m_group == "Embedding",
                        "classification" | "classify" | "classifier" => m_group == "Classifier",
                        "rerank" | "reranker" | "re-ranker" => m_group == "Re-Ranker",
                        "seq2seq" | "summarization" | "translation" | "summarize" => m_group == "Seq2Seq",
                        _ => {
                            // Also check task enum
                            let task_str = format!("{:?}", m.model_type.info().task).to_lowercase();
                            task_str.contains(f)
                        }
                    };
                    if !task_match {
                        return false;
                    }
                }

                true
            })
            .collect();

        if group_models.is_empty() {
            continue;
        }

        any_shown = true;

        println!("{}", group.to_uppercase());
        println!("{}", "-".repeat(90));

        for m in group_models {
            // Show download status with format info
            let status = if m.downloaded {
                if has_gguf_downloaded(&m.model_type) {
                    "✓ gguf"
                } else {
                    "✓ st  "  // safetensors
                }
            } else {
                "      "
            };

            let gguf_tag = if m.has_gguf { "[GGUF]" } else { "      " };

            println!(
                "  {} {:<28} {:>8} {} {}",
                status,
                m.cli_name,
                m.params,
                gguf_tag,
                truncate(&m.description, 30)
            );
        }
        println!();
    }

    if !any_shown {
        println!("No models found matching the filters.");
        println!();
        println!("Available task filters: chat, embedding, classification, rerank, seq2seq");
        println!("Available arch filters: llama, bert, qwen, t5, bart, mistral, gpt");
        println!();
    }

    println!("Legend: ✓ gguf = GGUF downloaded, ✓ st = SafeTensors downloaded");
    println!("        [GGUF] = GGUF format available for download");
    println!();
    println!("Commands:");
    println!("  kjarni model download <name>        Download SafeTensors");
    println!("  kjarni model download <name> --gguf Download GGUF (smaller)");
    println!("  kjarni model info <name>            Show details");
    println!("  kjarni model remove <name>          Delete from disk");
    println!();
    println!("Filters:");
    println!("  kjarni model list --task chat       Show only chat/LLM models");
    println!("  kjarni model list --task embedding  Show only embedding models");
    println!("  kjarni model list --arch bert       Show only BERT-based models");
    println!("  kjarni model list --downloaded      Show only downloaded models");
    println!();

    Ok(())
}

fn has_gguf_downloaded(model_type: &ModelType) -> bool {
    let cache_dir = registry::cache_dir();
    let model_dir = model_type.cache_dir(&cache_dir);
    model_dir.join("model.gguf").exists()
}

async fn download(name: &str, prefer_gguf: bool, quiet: bool) -> Result<()> {
    registry::download_model(name, prefer_gguf, quiet).await
}

fn info(name: &str) -> Result<()> {
    let model = registry::get_model_info(name)?;
    let model_path = registry::model_path(name)?;

    println!();
    println!("┌─────────────────────────────────────────┐");
    println!("│  {}  ", pad_center(&model.cli_name, 37));
    println!("└─────────────────────────────────────────┘");
    println!();
    println!("  Architecture:  {}", model.architecture.display_name());
    println!("  Parameters:    {}", model.params);
    println!("  Size (est):    {}", model.size);
    println!();

    // Download status section
    println!("  Download Status:");

    let st_path = model_path.join("model.safetensors");
    let st_index_path = model_path.join("model.safetensors.index.json");
    let gguf_path = model_path.join("model.gguf");

    if st_path.exists() || st_index_path.exists() {
        let size = if st_path.exists() {
            st_path.metadata().map(|m| m.len()).unwrap_or(0)
        } else {
            dir_size(&model_path).unwrap_or(0)
        };
        println!("    ✓ SafeTensors  {}", format_bytes(size));
    } else {
        println!("    ○ SafeTensors  (not downloaded)");
    }

    if gguf_path.exists() {
        let size = gguf_path.metadata().map(|m| m.len()).unwrap_or(0);
        println!("    ✓ GGUF         {}", format_bytes(size));
    } else if model.has_gguf {
        println!("    ○ GGUF         (available, use --gguf to download)");
    } else {
        println!("    ✗ GGUF         (not available for this model)");
    }

    println!();
    println!("  Path: {}", model_path.display());
    println!();
    println!("  Description:");
    println!("    {}", model.description);
    println!();

    Ok(())
}

fn pad_center(s: &str, width: usize) -> String {
    if s.len() >= width {
        s.to_string()
    } else {
        let padding = width - s.len();
        let left = padding / 2;
        let right = padding - left;
        format!("{}{}{}", " ".repeat(left), s, " ".repeat(right))
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
