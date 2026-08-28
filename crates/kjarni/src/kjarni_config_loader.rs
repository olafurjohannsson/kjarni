
use std::path::{Path, PathBuf};
use anyhow::Result;
use crate::kjarni_config::KjarniConfig;

/// Load configuration with standard priority:
/// CLI flags > ./kjarni.toml > ~/.config/kjarni/config.toml > defaults
pub fn load_config() -> Result<KjarniConfig> {
    // Try local first
    if let Some(config) = try_load_from_path("./kjarni.toml")? {
        return Ok(config);
    }
    
    // Try user config
    if let Some(config_dir) = dirs::config_dir() {
        let user_config = config_dir.join("kjarni").join("config.toml");
        if let Some(config) = try_load_from_path(&user_config)? {
            return Ok(config);
        }
    }
    
    // Return defaults
    Ok(KjarniConfig::default())
}

/// Load from specific path.
pub fn load_config_from_path(path: &Path) -> Result<KjarniConfig> {
    let contents = std::fs::read_to_string(path)?;
    let config: KjarniConfig = toml::from_str(&contents)?;
    Ok(config)
}

fn try_load_from_path(path: impl AsRef<Path>) -> Result<Option<KjarniConfig>> {
    let path = path.as_ref();
    if path.exists() {
        Ok(Some(load_config_from_path(path)?))
    } else {
        Ok(None)
    }
}

/// Get the cache directory, respecting config.
///
/// Precedence is explicit config, then `KJARNI_CACHE_DIR`, then the platform
/// default. An entry someone wrote in their config file is a deliberate choice
/// and outranks an environment variable that may have been inherited.
pub fn get_cache_dir(config: &KjarniConfig) -> PathBuf {
    config
        .cache
        .dir
        .clone()
        .unwrap_or_else(kjarni_transformers::models::get_default_cache_dir)
}