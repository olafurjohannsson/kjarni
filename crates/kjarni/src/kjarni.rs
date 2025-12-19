use std::sync::Arc;
use std::path::PathBuf;
use anyhow::Result;
use kjarni_transformers::prelude::*;
use kjarni_transformers::models::ModelType;
use kjarni_transformers::encoder_decoder::traits::EncoderDecoderLanguageModel;

// pub mod config;
// pub mod utils;
// pub mod model_manager;

// Re-export other modules
pub use kjarni_search as search;
pub use kjarni_rag as rag;
use crate::config_utils::merge_generation_config;
use crate::model_manager::ModelManager;
use crate::config::{KjarniConfig, GenerationDefaults};

pub struct Kjarni {
    pub config: KjarniConfig,
    pub models: Arc<ModelManager>,
    // Context is shared across models
    pub device: Device,
    pub context: Option<Arc<WgpuContext>>,
}

impl Kjarni {
    /// Initialize the library. Loads config from disk automatically.
    pub async fn new(config_path: Option<PathBuf>) -> Result<Self> {
        let config = KjarniConfig::load(config_path)?;
        
        // Determine device from config
        let device = match config.system.device.as_deref() {
            Some("cpu") => Device::Cpu,
            Some("gpu") | Some("wgpu") => Device::Wgpu,
            _ => Device::Cpu, // Default to CPU for safety, or implement auto-detection
        };

        let context = if device.is_gpu() {
            Some(WgpuContext::new().await?)
        } else {
            None
        };

        Ok(Self {
            config,
            models: Arc::new(ModelManager::new()),
            device,
            context,
        })
    }

    /// High-level summarize function
    pub async fn summarize(
        &self, 
        text: &str, 
        model_name_override: Option<&str>, 
        overrides: Option<GenerationDefaults>
    ) -> Result<String> {
        // 1. Determine Model Name (Override > Config > Default)
        let model_name = model_name_override.unwrap_or(&self.config.models.summarization);
        let model_type = ModelType::from_cli_name(model_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown model: {}", model_name))?;

        // 2. Load Model via Manager
        self.models.get_or_load_seq2seq_generator(
            model_type,
            self.config.system.cache_dir.as_deref().map(|p| p.to_str().unwrap()),
            self.device,
            self.context.clone()
        ).await?;

        let guard = self.models.seq2seq_generator.lock().await;
        let generator = guard.as_ref().unwrap();

        // 3. Merge Configs
        let model_defaults = generator.model.get_default_generation_config();
        let runtime_args = overrides.unwrap_or_default();
        
        let final_config = merge_generation_config(
            model_defaults,
            &self.config.generation, // Layer 2 (TOML)
            &runtime_args            // Layer 3 (Args)
        );

        // 4. Generate
        generator.generate(text, Some(&final_config)).await
    }
}