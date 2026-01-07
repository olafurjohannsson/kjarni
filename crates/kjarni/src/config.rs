use crate::generation::GenerationOverrides;
use anyhow::Result;
use config::{Config, Environment, File};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KjarniConfig {
    #[serde(default)]
    pub system: SystemConfig,
    #[serde(default)]
    pub models: ModelsConfig,
    #[serde(default)]
    pub generation: GenerationOverrides,
    #[serde(default)]
    pub indexing: IndexingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemConfig {
    pub cache_dir: Option<PathBuf>,
    pub device: Option<String>,
    pub log_level: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    pub chat: String,
    pub embedding: String,
    pub reranker: String,
    pub summarization: String,
    pub audio: String,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            chat: "llama-3.2-1b".to_string(),
            embedding: "minilm-l6-v2".to_string(),
            reranker: "minilm-l6-v2-cross-encoder".to_string(),
            summarization: "distilbart-cnn".to_string(),
            audio: "whisper-tiny".to_string(),
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingConfig {
    pub strategy: String, // "recursive", "semantic"
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub semantic_threshold: f32,
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            strategy: "recursive".to_string(),
            chunk_size: 512,
            chunk_overlap: 50,
            semantic_threshold: 0.75,
        }
    }
}

impl KjarniConfig {
    pub fn load(custom_path: Option<PathBuf>) -> Result<Self> {
        let mut s = Config::builder();

        // 1. Load from system/user config dirs
        if let Some(dirs) = directories::ProjectDirs::from("com", "kjarni", "kjarni") {
            let path = dirs.config_dir().join("config.toml");
            if path.exists() {
                s = s.add_source(File::from(path));
            }
        }

        // 2. Load from local directory
        if std::path::Path::new("kjarni.toml").exists() {
            s = s.add_source(File::with_name("kjarni.toml"));
        }

        // 3. Load from explicit CLI argument
        if let Some(path) = custom_path {
            s = s.add_source(File::from(path));
        }

        // 4. Environment variables (KJARNI_GENERATION_TEMPERATURE=0.8)
        s = s.add_source(Environment::with_prefix("KJARNI").separator("_"));

        let config = s.build()?;
        Ok(config.try_deserialize()?)
    }
}