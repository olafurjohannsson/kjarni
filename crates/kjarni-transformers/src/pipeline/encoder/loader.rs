use std::{path::{Path, PathBuf}, sync::Arc};

use tokenizers::Tokenizer;

use crate::{Device, ModelType, WgpuContext, cpu::encoder::{
    classifier::CpuSequenceClassificationHead,
    config::PoolingStrategy,
    traits::{CpuEncoder, GpuEncoder},
}, models::{base::ModelLoadConfig, download_model_files, registry::WeightsFormat}, pipeline::encoder::{EncoderPipeline, EncoderPipelineBuilder}, traits::{ModelConfig, ModelLayout, ModelMetadata}, weights::ModelWeights};
use anyhow::{anyhow, Result};

// =============================================================================
// Encoder Model Factory
// =============================================================================

/// Factory trait for encoder-based models.
///
/// Implement this for SentenceEncoder, SequenceClassifier, and CrossEncoder
/// to get automatic loading from registry and pretrained paths.
/// Factory trait for encoder-based models.
pub trait EncoderModelFactory: Sized {
    /// Load the model config from weights (auto-detects config type).
    fn load_config(weights: &ModelWeights) -> Result<Arc<dyn ModelConfig>>;
    
    /// Build the encoder backend(s).
    fn build_backends(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        load_config: ModelLoadConfig,
        context: Option<&Arc<WgpuContext>>,
        device: Device,
    ) -> Result<(Option<Box<dyn CpuEncoder>>, Option<Box<dyn GpuEncoder>>)>;
    
    /// Build the classification head (None for SentenceEncoder).
    fn build_head(
        weights: &ModelWeights,
        load_config: &ModelLoadConfig,
    ) -> Result<Option<CpuSequenceClassificationHead>> {
        Ok(None)
    }
    
    /// Get the pooling strategy for this model type.
    fn pooling_strategy() -> PoolingStrategy {
        PoolingStrategy::Mean
    }
    
    /// Construct the model from a loaded pipeline.
    fn new_from_pipeline(
        pipeline: EncoderPipeline,
        tokenizer: Tokenizer,
        config: Arc<dyn ModelConfig>,
        model_type: Option<ModelType>,
    ) -> Self;
}

// =============================================================================
// Generic Encoder Loader
// =============================================================================

pub struct EncoderLoader;

impl EncoderLoader {
    pub async fn load_from_registry<M: EncoderModelFactory>(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<M> {
        let info = model_type.info();
        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::cache_dir()
                .expect("No cache directory")
                .join("kjarni")
        });
        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));
        
        // Download model files
        download_model_files(&model_dir, &info.paths, WeightsFormat::SafeTensors).await?;
        
        // Create GPU context if needed
        let context = if device.is_gpu() && context.is_none() {
            Some(WgpuContext::new().await?)
        } else {
            context
        };
        
        Self::load_from_pretrained::<M>(&model_dir, device, context, load_config, Some(model_type))
    }
    
    pub fn load_from_pretrained<M: EncoderModelFactory>(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
        model_type: Option<ModelType>,
    ) -> Result<M> {
        log::info!("Loading encoder from {:?}", model_path);
        
        let weights = ModelWeights::new(model_path)?;
        let load_config = load_config.unwrap_or_default();
        
        // 1. Load model-specific config
        let config = M::load_config(&weights)?;
        let meta = config.metadata();
        let layout = config.layout();
        
        // 2. Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(anyhow!("Tokenizer not found at {:?}", tokenizer_path));
        }
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
        
        // Configure truncation and padding
        let _ = tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: meta.max_seq_len,
            ..Default::default()
        }));
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        
        // 3. Build backends
        let (cpu_encoder, gpu_encoder) = M::build_backends(
            &weights,
            &meta,
            &layout,
            load_config.clone(),
            context.as_ref(),
            device,
        )?;
        
        // 4. Build head (if any)
        let cpu_head = M::build_head(&weights, &load_config)?;
        
        // 5. Build pipeline
        let pipeline = EncoderPipelineBuilder::new(&weights, config.clone())
            .with_load_config(load_config)
            .with_backends(cpu_encoder, gpu_encoder)
            .with_head(cpu_head)
            .with_pooling_strategy(M::pooling_strategy())
            .with_context(context)
            .build()?;
        
        // 6. Construct the model
        Ok(M::new_from_pipeline(pipeline, tokenizer, config, model_type))
    }
}