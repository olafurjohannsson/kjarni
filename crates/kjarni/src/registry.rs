//! Model registry and management
//!
//! Provides high-level functions for listing, downloading, and managing models.

use anyhow::{anyhow, Result};
use kjarni_transformers::models::{
    download_model_files,
    format_params, // Make sure WeightsFormat is exported from your models module
    format_size,
    get_default_cache_dir,
    registry::WeightsFormat,
    ModelArchitecture,
    ModelType,
};
use std::path::PathBuf;

/// Information about a model for display purposes
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub model_type: ModelType,
    pub cli_name: String,
    pub architecture: ModelArchitecture,
    pub description: String,
    pub size: String,
    pub params: String,
    pub downloaded: bool,
    pub has_gguf: bool, // New field!
}

/// List all available models with their download status
pub fn list_models() -> Vec<ModelEntry> {
    let cache_dir = get_default_cache_dir();

    ModelType::all()
        .filter(|model_type| !model_type.cli_name().is_empty())  // Skip internal models
        .map(|model_type| {
            let info = model_type.info();
            ModelEntry {
                model_type,
                cli_name: model_type.cli_name().to_string(),
                architecture: info.architecture,
                description: info.description.to_string(),
                size: format_size(info.size_mb),
                params: format_params(info.params_millions),
                downloaded: model_type.is_downloaded(&cache_dir),
                has_gguf: info.paths.gguf_url.is_some(),
            }
        })
        .collect()
}

/// List models filtered by architecture
pub fn list_models_by_architecture(arch: ModelArchitecture) -> Vec<ModelEntry> {
    list_models()
        .into_iter()
        .filter(|m| m.architecture == arch)
        .collect()
}

/// Get detailed info about a specific model
pub fn get_model_info(name: &str) -> Result<ModelEntry> {
    let model_type = ModelType::from_cli_name(name).ok_or_else(|| {
        anyhow!(
            "Unknown model: '{}'. Run 'kjarni model list' to see available models.",
            name
        )
    })?;

    let cache_dir = get_default_cache_dir();
    let info = model_type.info();

    Ok(ModelEntry {
        model_type,
        cli_name: model_type.cli_name().to_string(),
        architecture: info.architecture,
        description: info.description.to_string(),
        size: format_size(info.size_mb),
        params: format_params(info.params_millions),
        downloaded: model_type.is_downloaded(&cache_dir),
        has_gguf: info.paths.gguf_url.is_some(),
    })
}

/// Download a model by CLI name
/// Updated to accept format preference
pub async fn download_model(name: &str, prefer_gguf: bool) -> Result<()> {
    let model_type = ModelType::from_cli_name(name).ok_or_else(|| {
        anyhow!(
            "Unknown model: '{}'. Run 'kjarni model list' to see available models.",
            name
        )
    })?;

    let cache_dir = get_default_cache_dir();
    let model_dir = model_type.cache_dir(&cache_dir);
    let info = model_type.info();

    // Determine format
    let format = if prefer_gguf && info.paths.gguf_url.is_some() {
        WeightsFormat::GGUF
    } else {
        if prefer_gguf {
            println!("! GGUF requested but not available for this model. Downloading SafeTensors.");
        }
        WeightsFormat::SafeTensors
    };

    println!("Downloading {}...", model_type.cli_name());
    println!("  Repository: {}", model_type.repo_id());
    println!(
        "  Format:     {}",
        if matches!(format, WeightsFormat::GGUF) {
            "GGUF (Optimized)"
        } else {
            "SafeTensors (Standard)"
        }
    );
    println!("  Size:       ~{}", format_size(info.size_mb)); // Note: GGUF size might differ slightly
    println!("  Destination: {}", model_dir.display());
    println!();

    download_model_files(&model_dir, &info.paths, format).await?;

    println!();
    println!("✓ Download complete!");

    Ok(())
}

/// Check if a model is downloaded
pub fn is_model_downloaded(name: &str) -> Result<bool> {
    let model_type =
        ModelType::from_cli_name(name).ok_or_else(|| anyhow!("Unknown model: '{}'", name))?;

    let cache_dir = get_default_cache_dir();
    Ok(model_type.is_downloaded(&cache_dir))
}

pub fn cache_dir() -> PathBuf {
    get_default_cache_dir()
}

pub fn model_path(name: &str) -> Result<PathBuf> {
    let model_type =
        ModelType::from_cli_name(name).ok_or_else(|| anyhow!("Unknown model: '{}'", name))?;

    let cache_dir = get_default_cache_dir();
    Ok(model_type.cache_dir(&cache_dir))
}
