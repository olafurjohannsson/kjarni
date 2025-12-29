// kjarni/src/model.rs

use kjarni_models::prelude::*;
use kjarni_transformers::decoder::prelude::*;
use std::path::Path;

pub struct Model {
    generator: DecoderGenerator,
    model_type: Option<ModelType>,
}

impl Model {
    // ========================================================================
    // Simple Constructors
    // ========================================================================
    
    /// Load from registry with defaults.
    pub async fn from_registry(model_type: ModelType) -> Result<Self> {
        ModelBuilder::new()
            .from_registry(model_type)
            .build()
            .await
    }
    
    /// Load from local path (auto-detects safetensors vs GGUF).
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        ModelBuilder::new()
            .from_path(path)
            .build_sync()
    }
    
    /// Get a builder for advanced configuration.
    pub fn builder() -> ModelBuilder {
        ModelBuilder::new()
    }
    
    // ========================================================================
    // Generation API
    // ========================================================================
    
    /// Generate text with default settings.
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        self.generator.generate(prompt, &GenerationConfig::default()).await
    }
    
    /// Generate text with custom config.
    pub async fn generate_with(&self, prompt: &str, config: GenerationConfig) -> Result<String> {
        self.generator.generate(prompt, &config).await
    }
    
    /// Stream tokens as they're generated.
    pub async fn stream(&self, prompt: &str) -> Result<impl Stream<Item = Result<Token>>> {
        self.stream_with(prompt, GenerationConfig::default()).await
    }
    
    /// Stream tokens with custom config.
    pub async fn stream_with(&self, prompt: &str, config: GenerationConfig) 
        -> Result<impl Stream<Item = Result<Token>>> 
    {
        self.generator.generate_stream(prompt, &config).await
    }
    
    // ========================================================================
    // Metadata
    // ========================================================================
    
    /// Get the model type if loaded from registry.
    pub fn model_type(&self) -> Option<ModelType> {
        self.model_type
    }
    
    /// Get the device this model is running on.
    pub fn device(&self) -> Device {
        self.generator.model.device()
    }
    
    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.generator.model.vocab_size()
    }
}

/// A generated token with metadata.
pub struct Token {
    pub text: String,
    pub id: u32,
    pub is_generated: bool,
}