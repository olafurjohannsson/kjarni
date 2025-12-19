use anyhow::{anyhow, Result};
use kjarni::registry;
use kjarni_cli::ModelCommands;
use kjarni::{ModelArchitecture, ModelType};

pub async fn run(action: ModelCommands) -> Result<()> {
    match action {
        ModelCommands::List { arch } => list(arch),
        ModelCommands::Download { name } => download(&name).await,
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


fn list(arch_filter: Option<String>) -> Result<()> {
    let models = registry::list_models();

    let arch_filter: Option<ModelArchitecture> = match arch_filter.as_deref() {
        Some("encoder") => Some(ModelArchitecture::Encoder),
        Some("decoder") => Some(ModelArchitecture::Decoder),
        Some("encoder-decoder") => Some(ModelArchitecture::EncoderDecoder),
        Some("cross-encoder") => Some(ModelArchitecture::CrossEncoder),
        Some(other) => {
            return Err(anyhow!(
                "Unknown architecture: '{}'. Use: encoder, decoder, encoder-decoder, cross-encoder",
                other
            ))
        }
        None => None,
    };

    let architectures = [
        ModelArchitecture::Decoder,
        ModelArchitecture::Encoder,
        ModelArchitecture::EncoderDecoder,
        ModelArchitecture::CrossEncoder,
    ];

    println!();
    println!("Cache: {}", registry::cache_dir().display());
    println!();

    for arch in architectures {
        if let Some(filter) = arch_filter {
            if arch != filter {
                continue;
            }
        }

        let arch_models: Vec<_> = models.iter().filter(|m| m.architecture == arch).collect();

        if arch_models.is_empty() {
            continue;
        }

        println!("{}", arch.display_name().to_uppercase());
        println!("{}", "-".repeat(78));

        for m in arch_models {
            let status = if m.downloaded { "✓" } else { " " };
            println!(
                "  {} {:<28} {:>8}  {}",
                status,
                m.cli_name,
                m.params,
                truncate(&m.description, 35)
            );
        }
        println!();
    }

    println!("✓ = downloaded");
    println!();

    Ok(())
}

async fn download(name: &str) -> Result<()> {
    registry::download_model(name).await
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