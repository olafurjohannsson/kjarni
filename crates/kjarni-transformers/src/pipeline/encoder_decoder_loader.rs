use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::Device;
use crate::{
    common::HFGenerationDefaults,
    cpu::encoder::{CpuEncoder, GpuEncoder},
    encoder_decoder::traits::{CpuCrossDecoder, GpuCrossDecoder},
    models::{base::ModelLoadConfig, download_model_files, registry::WeightsFormat, ModelType},
    pipeline::{EncoderDecoderPipeline, EncoderDecoderPipelineBuilder},
    traits::{ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
    WgpuContext,
};

/// Factory trait for Seq2Seq models (BART, T5, Whisper).
pub trait EncoderDecoderModelFactory: Sized {
    type Config: ModelConfig + 'static;

    fn load_config(weights: &ModelWeights) -> Result<Arc<Self::Config>>;

    /// Build the specific backend implementations (e.g. BartCpuEncoder, T5GpuDecoder)
    fn build_backends(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        config: &Arc<Self::Config>,
        load_config: &ModelLoadConfig,
        context: Option<&Arc<WgpuContext>>,
        device: Device,
    ) -> Result<(
        Option<Box<dyn CpuEncoder>>,
        Option<Box<dyn GpuEncoder>>,
        Option<Box<dyn CpuCrossDecoder>>,
        Option<Box<dyn GpuCrossDecoder>>,
    )>;

    /// Wrap the generic pipeline into the specific Model struct
    fn new_from_pipeline(
        pipeline: EncoderDecoderPipeline,
        tokenizer: Tokenizer,
        config: Arc<Self::Config>,
        generation_defaults: Option<HFGenerationDefaults>,
    ) -> Self;
}

pub struct Seq2SeqLoader;

impl Seq2SeqLoader {
    pub async fn load_from_registry<M: EncoderDecoderModelFactory>(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: crate::prelude::Device,
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

        download_model_files(&model_dir, &info.paths, WeightsFormat::SafeTensors, true).await?;

        let context = if device.is_gpu() && context.is_none() {
            Some(WgpuContext::new().await?)
        } else {
            context
        };

        Self::load_from_pretrained::<M>(&model_dir, device, context, load_config)
    }

    pub fn load_from_pretrained<M: EncoderDecoderModelFactory>(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<M> {
        let weights = ModelWeights::new(model_path)?;
        let load_config = load_config.unwrap_or_default();

        // 1. Config & Tokenizer
        let config = M::load_config(&weights)?;
        let meta = config.metadata();
        let layout = config.layout();

        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!(e))?;

        // 2. Build Backends (User implementation logic)
        let (cpu_enc, gpu_enc, cpu_dec, gpu_dec) = M::build_backends(
            &weights,
            &meta,
            &layout,
            &config,
            &load_config,
            context.as_ref(),
            device,
        )?;

        // 3. Build Pipeline
        let pipeline = EncoderDecoderPipelineBuilder::new(&weights, config.clone())
            .with_load_config(load_config)
            .with_context(context)
            .with_encoder_backends(cpu_enc, gpu_enc)
            .with_decoder_backends(cpu_dec, gpu_dec)
            .build()?;

        // 4. Generation Defaults
        let gen_defaults =
            if let Ok(json) = std::fs::read_to_string(model_path.join("generation_config.json")) {
                HFGenerationDefaults::from_json(&json).ok()
            } else {
                None
            };

        Ok(M::new_from_pipeline(
            pipeline,
            tokenizer,
            config,
            gen_defaults,
        ))
    }
}
