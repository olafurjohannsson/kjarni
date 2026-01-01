use anyhow::{anyhow, Result};
use kjarni::registry;
use kjarni_cli::ModelCommands;
use kjarni::{ModelArchitecture, ModelType};

pub async fn run(action: ModelCommands) -> Result<()> {
    match action {
        ModelCommands::List { arch } => list(arch),
        ModelCommands::Download { name, gguf } => download(&name, gguf).await,
        ModelCommands::Info { name } => info(&name),
        ModelCommands::Search { query } => search(&query),
    }
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
fn list(filter_arg: Option<String>) -> Result<()> {
    let models = registry::list_models();
    let filter = filter_arg.as_deref().map(|s| s.to_lowercase());

    // Define the display order
    let groups = [
        "LLM (Decoder)",
        "Seq2Seq",
        "Embedding",
        "Re-Ranker",     // <-- Now clearly separated
        "Classifier",    // <-- Now clearly separated
    ];

    println!();
    println!("Cache: {}", registry::cache_dir().display());
    println!();

    for group in groups {
        // Filter models by their semantic group
        let group_models: Vec<_> = models.iter().filter(|m| {
            let m_group = m.model_type.display_group(); // Use the new helper
            
            // 1. Must match current group loop
            if m_group != group { return false; }

            // 2. Apply User Filter (flexible matching)
            if let Some(f) = &filter {
                // Allow filtering by group name (e.g. "classifier")
                if m_group.to_lowercase().contains(f) { return true; }
                
                // Allow filtering by specific architecture (e.g. "bert")
                let arch = format!("{:?}", m.architecture).to_lowercase();
                if arch.contains(f) { return true; }
                
                return false;
            }
            true
        }).collect();

        if group_models.is_empty() { continue; }

        println!("{}", group.to_uppercase());
        println!("{}", "-".repeat(85));

        for m in group_models {
            let status = if m.downloaded { "✓" } else { " " };
            let gguf_tag = if m.has_gguf { "[GGUF]" } else { "" };
            
            println!(
                "  {} {:<28} {:>8} {:<6} {}",
                status,
                m.cli_name,
                m.params,
                gguf_tag,
                truncate(&m.description, 35)
            );
        }
        println!();
    }

    println!("✓ = downloaded");
    println!();

    Ok(())
}

async fn download(name: &str, prefer_gguf: bool ) -> Result<()> {
    registry::download_model(name, prefer_gguf).await
}

fn info(name: &str) -> Result<()> {
    let model = registry::get_model_info(name)?;

    println!();
    println!("Model: {}", model.cli_name);
    println!("{}", "-".repeat(40));
    println!("Architecture:  {}", model.architecture.display_name());
    println!("Parameters:    {}", model.params);
    println!("Size:          {}", model.size);
    println!("Downloaded:    {}", if model.downloaded { "Yes" } else { "No" });
    println!("Path:          {}", registry::model_path(&model.cli_name)?.display());
    println!();
    println!("Description:");
    println!("  {}", model.description);
    println!();

    Ok(())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}