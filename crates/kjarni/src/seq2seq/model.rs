//! Core Seq2SeqGenerator implementation.

use std::sync::Arc;
use std::time::Instant;

use anyhow::anyhow;
use futures::{Stream, StreamExt, pin_mut};
use kjarni_models::models::bart::model::BartModel;
use kjarni_models::models::t5::T5Model;
use log::{debug, info};
use tokio::sync::mpsc;

use kjarni_transformers::WgpuContext;
use kjarni_transformers::common::{GenerationConfig, HFGenerationConfig};
use kjarni_transformers::encoder_decoder::{EncoderDecoderGenerator, EncoderDecoderLanguageModel};
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::models::{ModelArchitecture, ModelType};
use kjarni_transformers::pipeline::Seq2SeqLoader;
use kjarni_transformers::traits::Device;

use crate::common::DownloadPolicy;

use super::builder::Seq2SeqGeneratorBuilder;
use super::resolution::apply_overrides;
use super::types::{Seq2SeqError, Seq2SeqOverrides, Seq2SeqResult, Seq2SeqToken};
use super::validation::{folder_architecture, validate_for_seq2seq};
use kjarni_transformers::models::get_default_cache_dir;

/// Where the model comes from.
enum ModelSource {
    Registry(ModelType),
    Folder {
        path: std::path::PathBuf,
        architecture: ModelArchitecture,
    },
}

impl ModelSource {
    fn architecture(&self) -> ModelArchitecture {
        match self {
            Self::Registry(model_type) => model_type.info().architecture,
            Self::Folder { architecture, .. } => *architecture,
        }
    }
}

/// Generic text-to-text generator for encoder-decoder models
pub struct Seq2SeqGenerator {
    /// The underlying encoder-decoder generator (wrapped in Arc for streaming).
    encoder_decoder: Arc<EncoderDecoderGenerator>,

    /// The registry model, or `None` for a model loaded from a folder.
    model_type: Option<ModelType>,

    /// The registry name, or the folder's name.
    model_name: String,

    /// The model's family.
    architecture: ModelArchitecture,

    /// User-provided overrides (only what the user explicitly set).
    user_overrides: Seq2SeqOverrides,

    /// Device the model is running on.
    device: Device,

    /// GPU context if using GPU.
    context: Option<Arc<WgpuContext>>,
}

impl Seq2SeqGenerator {
    /// Create a Seq2SeqGenerator with default settings.
    pub async fn new(model: &str) -> Seq2SeqResult<Self> {
        Seq2SeqGeneratorBuilder::new(model).build().await
    }

    /// Create a builder for custom configuration.
    pub fn builder(model: &str) -> Seq2SeqGeneratorBuilder {
        Seq2SeqGeneratorBuilder::new(model)
    }

    /// construct from builder.
    pub(crate) async fn from_builder(builder: Seq2SeqGeneratorBuilder) -> Seq2SeqResult<Self> {
        let load_start = Instant::now();

        let source = match &builder.model_path {
            Some(path) => {
                let architecture = folder_architecture(path)?;
                debug!("Folder {path:?} holds a {architecture:?} model");
                ModelSource::Folder {
                    path: path.clone(),
                    architecture,
                }
            }
            None => {
                let model_type = ModelType::from_cli_name(&builder.model)
                    .ok_or_else(|| Seq2SeqError::UnknownModel(builder.model.clone()))?;
                debug!(
                    "Resolved model '{}' -> {:?} ({:?})",
                    builder.model,
                    model_type,
                    model_type.info().architecture
                );
                let validation = validate_for_seq2seq(model_type)?;
                if !builder.quiet && !builder.allow_warnings {
                    for warning in &validation.warnings {
                        eprintln!("warning: {}", warning);
                    }
                }
                ModelSource::Registry(model_type)
            }
        };
        let model_name = match &source {
            ModelSource::Folder { path, .. } => crate::common::folder_name(path),
            ModelSource::Registry(model_type) => model_type.cli_name().to_string(),
        };

        // Determine cache directory
        let cache_dir = builder
            .cache_dir
            .clone()
            .unwrap_or_else(get_default_cache_dir);

        if let ModelSource::Registry(model_type) = source
            && !model_type.is_downloaded(&cache_dir)
        {
            match builder.download_policy {
                DownloadPolicy::Never => {
                    return Err(Seq2SeqError::ModelNotDownloaded(builder.model.clone()));
                }
                DownloadPolicy::IfMissing | DownloadPolicy::Eager => {
                    info!("Downloading model '{}'...", builder.model);
                    if !builder.quiet {
                        eprintln!("downloading model '{}'...", builder.model);
                    }
                    kjarni_transformers::models::download_model_files(
                        &model_type.cache_dir(&cache_dir),
                        &model_type.info().paths,
                        kjarni_transformers::models::registry::WeightsFormat::SafeTensors,
                        builder.quiet,
                    )
                    .await
                    .map_err(|e| Seq2SeqError::DownloadFailed {
                        model: builder.model.clone(),
                        source: e,
                    })?;
                }
            }
        }

        // Determine device
        let device = builder.device.to_device();

        // Create or use provided GPU context
        let context = if device == Device::Wgpu {
            if let Some(ctx) = builder.context {
                Some(ctx)
            } else {
                debug!("Initializing GPU context...");
                Some(
                    WgpuContext::new()
                        .await
                        .map_err(|_| Seq2SeqError::GpuUnavailable)?,
                )
            }
        } else {
            None
        };

        // Get load config
        let load_config = builder
            .load_config
            .map(|c| c.into_inner())
            .unwrap_or_default();

        let architecture = source.architecture();
        info!("Loading {architecture} model '{model_name}' on {device:?}...");

        let model: Box<dyn EncoderDecoderLanguageModel> =
            Self::load_model(&source, &cache_dir, device, context.clone(), load_config)
                .await
                .map_err(|e| Seq2SeqError::LoadFailed {
                    model: model_name.clone(),
                    source: e,
                })?;

        // Create the encoder-decoder generator (takes ownership of model)
        let encoder_decoder = Arc::new(EncoderDecoderGenerator::new(model).map_err(|e| {
            Seq2SeqError::LoadFailed {
                model: model_name.clone(),
                source: e,
            }
        })?);

        let load_elapsed = load_start.elapsed();
        info!(
            "Model '{}' loaded in {:.2}s (vocab={}, context={})",
            model_name,
            load_elapsed.as_secs_f32(),
            encoder_decoder.model.vocab_size(),
            encoder_decoder.model.context_size()
        );

        Ok(Self {
            encoder_decoder,
            model_type: match source {
                ModelSource::Registry(model_type) => Some(model_type),
                ModelSource::Folder { .. } => None,
            },
            model_name,
            architecture,
            user_overrides: builder.overrides,
            device,
            context,
        })
    }

    /// Load the model from the registry cache or its folder.
    async fn load_model(
        source: &ModelSource,
        cache_dir: &std::path::Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: ModelLoadConfig,
    ) -> anyhow::Result<Box<dyn EncoderDecoderLanguageModel>> {
        match (source, source.architecture()) {
            (ModelSource::Folder { path, .. }, ModelArchitecture::T5) => {
                let model: T5Model = Seq2SeqLoader::load_from_pretrained(
                    path,
                    device,
                    context,
                    Some(load_config),
                    HFGenerationConfig::load_or_default(path),
                )?;
                Ok(Box::new(model))
            }
            (
                ModelSource::Folder { path, .. },
                ModelArchitecture::Bart | ModelArchitecture::Marian,
            ) => {
                let model: BartModel = Seq2SeqLoader::load_from_pretrained(
                    path,
                    device,
                    context,
                    Some(load_config),
                    HFGenerationConfig::load_or_default(path),
                )?;
                Ok(Box::new(model))
            }
            (ModelSource::Registry(model_type), ModelArchitecture::T5) => {
                let model = T5Model::from_registry(
                    *model_type,
                    Some(cache_dir.to_path_buf()),
                    device,
                    context,
                    Some(load_config),
                )
                .await?;
                Ok(Box::new(model))
            }
            (
                ModelSource::Registry(model_type),
                ModelArchitecture::Bart | ModelArchitecture::Marian,
            ) => {
                let model = BartModel::from_registry(
                    *model_type,
                    Some(cache_dir.to_path_buf()),
                    device,
                    context,
                    Some(load_config),
                )
                .await?;
                Ok(Box::new(model))
            }
            (_, architecture) => Err(anyhow!(
                "Architecture {architecture:?} is not supported for seq2seq generation. Use T5 or BART."
            )),
        }
    }

    /// Generate output from input text.
    pub async fn generate(&self, input: &str) -> Seq2SeqResult<String> {
        self.generate_with_config(input, &Seq2SeqOverrides::default())
            .await
    }

    /// Generate with runtime overrides for this call only
    pub async fn generate_with_config(
        &self,
        input: &str,
        runtime_overrides: &Seq2SeqOverrides,
    ) -> Seq2SeqResult<String> {
        let merged = self.user_overrides.merge(runtime_overrides);

        // If no overrides, pass None - let model use its defaults
        let config = if merged.is_empty() {
            None
        } else {
            let mut cfg = self
                .encoder_decoder
                .model
                .get_generation_config_for_input(input);
            apply_overrides(&mut cfg, &merged);
            Some(cfg)
        };

        debug!(
            "Generate: input_len={} chars, config={}",
            input.len(),
            if config.is_some() {
                "custom"
            } else {
                "model defaults"
            }
        );

        let output = self
            .encoder_decoder
            .generate(input, config.as_ref())
            .await
            .map_err(Seq2SeqError::GenerationFailed)?;

        Ok(output)
    }

    /// Stream generated tokens.
    pub async fn stream(
        &self,
        input: &str,
    ) -> Seq2SeqResult<std::pin::Pin<Box<dyn Stream<Item = Seq2SeqResult<Seq2SeqToken>> + Send>>>
    {
        self.stream_with_config(input, Seq2SeqOverrides::default())
            .await
    }

    /// Stream with runtime overrides.
    pub async fn stream_with_config(
        &self,
        input: &str,
        runtime_overrides: Seq2SeqOverrides,
    ) -> Seq2SeqResult<std::pin::Pin<Box<dyn Stream<Item = Seq2SeqResult<Seq2SeqToken>> + Send>>>
    {
        let merged = self.user_overrides.merge(&runtime_overrides);

        // If no overrides, pass None - let model use its defaults
        let config = if merged.is_empty() {
            None
        } else {
            let mut cfg = self
                .encoder_decoder
                .model
                .get_generation_config_for_input(input);
            apply_overrides(&mut cfg, &merged);
            Some(cfg)
        };

        debug!(
            "Stream: input_len={} chars, config={}",
            input.len(),
            if config.is_some() {
                "custom"
            } else {
                "model defaults"
            }
        );

        // Clone/own everything we need for the spawned task
        let encoder_decoder = self.encoder_decoder.clone();
        let input = input.to_string();

        let (tx, rx) = mpsc::channel::<Seq2SeqResult<Seq2SeqToken>>(32);

        tokio::spawn(async move {
            let stream = encoder_decoder.generate_stream(&input, config.as_ref());

            pin_mut!(stream);

            while let Some(result) = stream.next().await {
                let msg = match result {
                    Ok(token) => Ok(Seq2SeqToken {
                        text: token.text,
                        id: token.id,
                        is_special: false,
                    }),
                    Err(e) => Err(Seq2SeqError::GenerationFailed(e)),
                };

                if tx.send(msg).await.is_err() {
                    debug!("Stream receiver dropped, stopping generation");
                    break;
                }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }

    /// Stream text
    pub async fn stream_text(
        &self,
        input: &str,
    ) -> Seq2SeqResult<std::pin::Pin<Box<dyn Stream<Item = Seq2SeqResult<String>> + Send>>> {
        let inner_stream = self.stream(input).await?;
        let mapped = inner_stream.map(|result| result.map(|token| token.text));
        Ok(Box::pin(mapped))
    }

    /// Get the model's default generation config.
    pub fn get_model_defaults(&self) -> GenerationConfig {
        self.encoder_decoder.model.get_default_generation_config()
    }

    /// The registry model, or `None` for a model loaded from a folder.
    pub fn model_type(&self) -> Option<ModelType> {
        self.model_type
    }

    /// The registry name, or the name of the folder the model was loaded from.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// The model's family: T5, BART or Marian.
    pub fn architecture(&self) -> ModelArchitecture {
        self.architecture
    }

    /// Get the device the model is running on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get the context window size.
    pub fn context_size(&self) -> usize {
        self.encoder_decoder.model.context_size()
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.encoder_decoder.model.vocab_size()
    }

    /// Get the GPU context if using GPU.
    pub fn gpu_context(&self) -> Option<&Arc<WgpuContext>> {
        self.context.as_ref()
    }

    /// Get a reference to the underlying encoder-decoder generator.
    pub fn encoder_decoder(&self) -> &EncoderDecoderGenerator {
        &self.encoder_decoder
    }
}
