// kjarni/src/builder.rs

use std::{path::{Path, PathBuf}, sync::Arc};

use kjarni_models::models::llama::LlamaModel;
use kjarni_transformers::{Device, ModelArchitecture, ModelType, WgpuContext, decoder::traits::DecoderLanguageModel, models::base::ModelLoadConfig, tensor::DType};

use crate::model::Model;

pub struct ModelBuilder {
    source: Option<ModelSource>,
    device: Option<Device>,
    cache_dir: Option<PathBuf>,
    offload_embeddings: bool,
    offload_lm_head: bool,
    target_dtype: Option<DType>,
}

enum ModelSource {
    Registry(ModelType),
    Path(PathBuf),
}

impl ModelBuilder {
    pub fn new() -> Self {
        Self {
            source: None,
            device: None,
            cache_dir: None,
            offload_embeddings: false,
            offload_lm_head: false,
            target_dtype: None,
        }
    }
    
    // --- Source ---
    
    pub fn from_registry(mut self, model_type: ModelType) -> Self {
        self.source = Some(ModelSource::Registry(model_type));
        self
    }
    
    pub fn from_path(mut self, path: impl AsRef<Path>) -> Self {
        self.source = Some(ModelSource::Path(path.as_ref().to_path_buf()));
        self
    }
    
    // --- Device ---
    
    pub fn cpu(mut self) -> Self {
        self.device = Some(Device::Cpu);
        self
    }
    
    pub fn gpu(mut self) -> Self {
        self.device = Some(Device::Wgpu);
        self
    }
    
    // --- Memory ---
    
    pub fn offload_embeddings(mut self) -> Self {
        self.offload_embeddings = true;
        self
    }
    
    pub fn offload_lm_head(mut self) -> Self {
        self.offload_lm_head = true;
        self
    }
    
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.target_dtype = Some(dtype);
        self
    }
    
    // --- Storage ---
    
    pub fn cache_dir(mut self, path: impl AsRef<Path>) -> Self {
        self.cache_dir = Some(path.as_ref().to_path_buf());
        self
    }
    
    // --- Build ---
    
    pub async fn build(self) -> anyhow::Result<Model> {
        let source = self.source.ok_or_else(|| anyhow::anyhow!("No model source specified"))?;
        
        // Auto-detect device if not specified
        let device = self.device.unwrap_or_else(|| {
            if WgpuContext::is_available() { Device::Wgpu } else { Device::Cpu }
        });
        
        let load_config = ModelLoadConfig {
            offload_embeddings: self.offload_embeddings,
            offload_lm_head: self.offload_lm_head,
            target_dtype: self.target_dtype,
            ..Default::default()
        };
        
        let context = if device.is_gpu() {
            Some(Arc::new(WgpuContext::new().await?))
        } else {
            None
        };
        
        let (inner, model_type): (Box<dyn DecoderLanguageModel>, Option<ModelType>) = match source {
            ModelSource::Registry(model_type) => {
                let model = load_from_registry(model_type, self.cache_dir, device, context, load_config).await?;
                (model, Some(model_type))
            }
            ModelSource::Path(path) => {
                let model = load_from_path(&path, device, context, load_config)?;
                (model, None)
            }
        };
        
        let generator = DecoderGenerator::new(inner)?;
        
        Ok(Model { generator, model_type })
    }
}

// Helper to load from registry, auto-detecting architecture
async fn load_from_registry(
    model_type: ModelType,
    cache_dir: Option<PathBuf>,
    device: Device,
    context: Option<Arc<WgpuContext>>,
    load_config: ModelLoadConfig,
) -> anyhow::Result<Box<dyn DecoderLanguageModel>> {
    match model_type.info().architecture {
        ModelArchitecture::Decoder => {
            // Dispatch based on model family
            match model_type {
                ModelType::Llama3_2_1B | ModelType::Llama3_2_3B | ModelType::Llama3_8B_Instruct => {
                    let model = LlamaModel::from_registry(model_type, cache_dir, device, context, Some(load_config)).await?;
                    Ok(Box::new(model))
                }
                // Future:
                // ModelType::Phi3_Mini => { ... }
                // ModelType::Mistral_7B => { ... }
                _ => Err(anyhow::anyhow!("Unsupported model type: {:?}", model_type))
            }
        }
        _ => Err(anyhow::anyhow!("Only decoder models supported currently"))
    }
}