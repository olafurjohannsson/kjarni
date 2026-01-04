use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::chat::chatml::ChatMLTemplate;
use crate::chat::llama3::Llama3ChatTemplate;
use crate::chat::mistral::MistralChatTemplate;
use crate::common::HFGenerationDefaults;
use crate::decoder::traits::{CpuDecoder, GpuDecoder};
use crate::models::base::ModelLoadConfig;
use crate::models::registry::WeightsFormat;
use crate::models::{download_model_files, ModelArchitecture, ModelType};
use crate::pipeline::{DecoderPipeline, DecoderPipelineBuilder};
use crate::rope::loader::LoadedRoPE;
use crate::tensor::DType;
use crate::traits::{Device, ModelConfig, ModelLayout, ModelMetadata};
use crate::weights::ModelWeights;
use crate::{ChatTemplate, WgpuContext};

/// The Factory trait that every model (Llama, Phi, etc.) implements to
/// plug into the generic loading pipeline.
pub trait DecoderModelFactory: Sized {
    type Config: ModelConfig + 'static;

    fn build_backends(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        rope: &LoadedRoPE, // Handover RoPE
        load_config: ModelLoadConfig,
        context: Option<&Arc<WgpuContext>>,
    ) -> Result<(Option<Box<dyn CpuDecoder>>, Option<Box<dyn GpuDecoder>>)>;

    fn new_from_pipeline(
        pipeline: DecoderPipeline,
        tokenizer: Tokenizer,
        config: Arc<Self::Config>,
        model_type: Option<ModelType>,
        generation_defaults: Option<HFGenerationDefaults>,
        chat_template: Option<Box<dyn ChatTemplate>>,
    ) -> Self;
    fn load_config(weights: &ModelWeights) -> Result<Arc<Self::Config>>;
}

pub struct GenericLoader;

impl GenericLoader {
    pub async fn load_from_registry<M: DecoderModelFactory>(
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

        let config = load_config.clone().unwrap_or_default();

        // Logic: If user asked for Q4/Q6/Q8, OR explicitly set a GGUF flag (if you have one),
        // and the model supports it, download GGUF.
        let is_quantized_request = matches!(
            config.target_dtype,
            Some(DType::Q4_K) | Some(DType::Q6_K) | Some(DType::Q8_0)
        ) || config.use_gguf;
        let format = if is_quantized_request && info.paths.gguf_url.is_some() {
            log::info!("Configuration requests quantization. Preferring GGUF format.");
            WeightsFormat::GGUF
        } else {
            WeightsFormat::SafeTensors
        };

        download_model_files(&model_dir, &info.paths, format).await?;

        let context = if device.is_gpu() && context.is_none() {
            Some(WgpuContext::new().await?)
        } else {
            context
        };

        Self::load_from_pretrained::<M>(&model_dir, device, context, load_config, Some(model_type))
    }

    pub fn load_from_pretrained<M: DecoderModelFactory>(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
        model_type: Option<ModelType>,
    ) -> Result<M> {
        let weights = ModelWeights::new(model_path)?;
        let load_config: ModelLoadConfig = load_config.unwrap_or_default();

        // 1. Load Architecture-Specific Config (Handles GGUF/Safetensors automatically)
        let config = M::load_config(&weights)?;
        let meta = config.metadata();
        let layout = config.layout();

        // 2. Load Tokenizer
        let tokenizer_path = if model_path.is_file() {
            model_path.parent().unwrap().join("tokenizer.json")
        } else {
            model_path.join("tokenizer.json")
        };
        if !tokenizer_path.exists() {
            return Err(anyhow!("Tokenizer not found at {:?}", tokenizer_path));
        }
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!(e))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: meta.max_seq_len,
                ..Default::default()
            }))
            .unwrap();
        tokenizer.with_padding(None);

        let rope = LoadedRoPE::new(context.as_ref(), &meta, device.is_gpu())?;

        // try load generation defaults from generation_config.json
        let generation_defaults = Self::try_load_generation_defaults(model_path);

        // 2. Build Backends (The model-specific loops for Llama/GPT/etc)
        // Notice we pass 'rope' and 'load_config.target_dtype' here!
        let (cpu_decoder, gpu_decoder) = M::build_backends(
            &weights,
            &meta,
            &layout,
            &rope,
            load_config,
            context.as_ref(),
        )?;

        // 3. Build Pipeline via Builder (Coordinates tied weights internally)
        let pipeline: DecoderPipeline = DecoderPipelineBuilder::new(&weights, config.clone())
            .with_load_config(load_config)
            .with_backends(cpu_decoder, gpu_decoder)
            .with_context_opt(context)
            .build()?;
        // Auto-detect template based on ModelType
        let chat_template: Option<Box<dyn ChatTemplate>> = model_type.and_then(|mt| {
            if !mt.is_instruct_model() {
                return None;
            }

            match mt.architecture() {
                ModelArchitecture::Llama => {
                    Some(Box::new(Llama3ChatTemplate::for_generation()) as Box<dyn ChatTemplate>)
                }
                ModelArchitecture::Qwen2 => {
                    Some(Box::new(ChatMLTemplate::new()) as Box<dyn ChatTemplate>)
                }
                ModelArchitecture::Mistral => {
                    Some(Box::new(MistralChatTemplate::new()) as Box<dyn ChatTemplate>)
                }

                // Fallback
                _ => None,
            }
        });
        // 5. Wrap in the specific model struct
        Ok(M::new_from_pipeline(
            pipeline,
            tokenizer,
            config,
            model_type,
            generation_defaults,
            chat_template,
        ))
    }

    fn try_load_generation_defaults(model_path: &Path) -> Option<HFGenerationDefaults> {
        let gen_config_path = if model_path.is_file() {
            model_path.parent()?.join("generation_config.json")
        } else {
            model_path.join("generation_config.json")
        };

        let json = std::fs::read_to_string(&gen_config_path).ok()?;
        HFGenerationDefaults::from_json(&json).ok()
    }
}
