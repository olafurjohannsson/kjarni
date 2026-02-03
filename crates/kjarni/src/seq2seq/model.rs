//! Core Seq2SeqGenerator implementation.

use std::sync::Arc;

use anyhow::anyhow;
use futures::{Stream, StreamExt, pin_mut};
use kjarni_models::models::bart::model::BartModel;
use tokio::sync::mpsc;

use kjarni_transformers::WgpuContext;
use kjarni_transformers::common::GenerationConfig;
use kjarni_transformers::encoder_decoder::{EncoderDecoderGenerator, EncoderDecoderLanguageModel};
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::models::{ModelArchitecture, ModelType};
use kjarni_transformers::traits::Device;

use crate::common::DownloadPolicy;

use super::builder::Seq2SeqGeneratorBuilder;
use super::resolution::{ResolvedSeq2SeqConfig, resolve_seq2seq_config};
use super::types::{Seq2SeqError, Seq2SeqOverrides, Seq2SeqResult, Seq2SeqTask, Seq2SeqToken};
use super::validation::validate_for_seq2seq;

/// Generic text-to-text generator for encoder-decoder models.
///
/// Seq2SeqGenerator provides raw access to encoder-decoder generation without
/// task-specific formatting. It's the foundation for higher-level APIs like
/// `Translator` and `Summarizer`.
///
/// # Supported Models
///
/// - T5 family (flan-t5-base, flan-t5-large)
/// - BART family (bart-large-cnn, distilbart-cnn)
///
/// # Example
///
/// ```ignore
/// use kjarni::seq2seq::Seq2SeqGenerator;
///
/// // Basic usage
/// let generator = Seq2SeqGenerator::new("flan-t5-base").await?;
/// let output = generator.generate("translate English to German: Hello").await?;
///
/// // With custom config
/// let generator = Seq2SeqGenerator::builder("flan-t5-large")
///     .num_beams(6)
///     .max_length(256)
///     .gpu()
///     .build()
///     .await?;
/// ```
pub struct Seq2SeqGenerator {
    /// The underlying encoder-decoder generator (wrapped in Arc for streaming).
    encoder_decoder: Arc<EncoderDecoderGenerator>,

    /// Model type from registry.
    model_type: ModelType,

    /// Task hint (affects default config resolution).
    task: Option<Seq2SeqTask>,

    /// User-provided overrides (stored for re-resolution with runtime overrides).
    user_overrides: Seq2SeqOverrides,

    /// Device the model is running on.
    device: Device,

    /// GPU context if using GPU.
    context: Option<Arc<WgpuContext>>,
}

impl Seq2SeqGenerator {
    // =========================================================================
    // Construction
    // =========================================================================

    /// Create a Seq2SeqGenerator with default settings.
    ///
    /// Uses CPU, downloads model if needed.
    pub async fn new(model: &str) -> Seq2SeqResult<Self> {
        Seq2SeqGeneratorBuilder::new(model).build().await
    }

    /// Create a builder for custom configuration.
    pub fn builder(model: &str) -> Seq2SeqGeneratorBuilder {
        Seq2SeqGeneratorBuilder::new(model)
    }

    /// Internal: construct from builder.
    pub(crate) async fn from_builder(builder: Seq2SeqGeneratorBuilder) -> Seq2SeqResult<Self> {
        // Resolve model type
        let model_type = ModelType::from_cli_name(&builder.model)
            .ok_or_else(|| Seq2SeqError::UnknownModel(builder.model.clone()))?;

        // Validate model for seq2seq
        let validation = validate_for_seq2seq(model_type)?;

        // Emit warnings if not suppressed
        if !builder.quiet && !builder.allow_warnings {
            for warning in &validation.warnings {
                eprintln!("warning: {}", warning);
            }
        }

        // Determine cache directory
        let cache_dir = builder.cache_dir.clone().unwrap_or_else(|| {
            dirs::cache_dir()
                .expect("no cache directory found")
                .join("kjarni")
        });

        // Check if model is downloaded
        let is_downloaded = model_type.is_downloaded(&cache_dir);

        if !is_downloaded {
            match builder.download_policy {
                DownloadPolicy::Never => {
                    return Err(Seq2SeqError::ModelNotDownloaded(builder.model.clone()));
                }
                DownloadPolicy::IfMissing | DownloadPolicy::Eager => {
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

        // Load the model based on architecture
        let model: Box<dyn EncoderDecoderLanguageModel> =
            Self::load_model(model_type, &cache_dir, device, context.clone(), load_config)
                .await
                .map_err(|e| Seq2SeqError::LoadFailed {
                    model: builder.model.clone(),
                    source: e,
                })?;

        // Create the encoder-decoder generator (takes ownership of model)
        let encoder_decoder = Arc::new(EncoderDecoderGenerator::new(model).map_err(|e| {
            Seq2SeqError::LoadFailed {
                model: builder.model.clone(),
                source: e,
            }
        })?);

        Ok(Self {
            encoder_decoder,
            model_type,
            task: builder.task,
            user_overrides: builder.overrides,
            device,
            context,
        })
    }

    /// Load the appropriate model based on architecture.
    async fn load_model(
        model_type: ModelType,
        cache_dir: &std::path::Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: ModelLoadConfig,
    ) -> anyhow::Result<Box<dyn EncoderDecoderLanguageModel>> {
        let info = model_type.info();

        match info.architecture {
            ModelArchitecture::T5 => {
                use kjarni_models::models::t5::T5Model;
                let model = T5Model::from_registry(
                    model_type,
                    Some(cache_dir.to_path_buf()),
                    device,
                    context,
                    Some(load_config),
                )
                .await?;
                Ok(Box::new(model) as Box<dyn EncoderDecoderLanguageModel>)
            }

            ModelArchitecture::Bart => {
                let model = BartModel::from_registry(
                    model_type,
                    Some(cache_dir.to_path_buf()),
                    device,
                    context,
                    Some(load_config),
                )
                .await?;
                Ok(Box::new(model) as Box<dyn EncoderDecoderLanguageModel>)
            }

            _ => Err(anyhow!(
                "Architecture {:?} is not supported for seq2seq generation. Use T5 or BART.",
                info.architecture
            )),
        }
    }

    // =========================================================================
    // Generation
    // =========================================================================

    /// Generate output from input text.
    ///
    /// The input should be the fully-formatted prompt (e.g., "translate English to German: Hello").
    /// For task-specific formatting, use `Translator` or `Summarizer` instead.
    pub async fn generate(&self, input: &str) -> Seq2SeqResult<String> {
        self.generate_with_config(input, &Seq2SeqOverrides::default())
            .await
    }

    /// Generate with custom overrides for this call only.
    ///
    /// Runtime overrides are merged with user overrides (from builder).
    /// Runtime takes precedence.
pub async fn generate_with_config(
    &self,
    input: &str,
    runtime_overrides: &Seq2SeqOverrides,
) -> Seq2SeqResult<String> {
    // Get task-specific model defaults based on INPUT content
    // (T5 detects "translate X to Y:" prefix and returns appropriate config)
    let model_defaults = self.encoder_decoder.model.get_generation_config_for_input(input);
    
    // Resolve with user and runtime overrides
    let config = resolve_seq2seq_config(
        model_defaults,
        self.task,
        &self.user_overrides,
        runtime_overrides,
    );

    // DEBUG: Print the resolved config
    eprintln!("DEBUG input prefix: '{}'", &input[..input.len().min(60)]);
    eprintln!("DEBUG resolved config: {:?}", config.as_ref());

    // Generate
    let output = self
        .encoder_decoder
        .generate(input, Some(config.as_ref()))
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

    /// Stream with custom overrides.
    pub async fn stream_with_config(
    &self,
    input: &str,
    runtime_overrides: Seq2SeqOverrides,
) -> Seq2SeqResult<std::pin::Pin<Box<dyn Stream<Item = Seq2SeqResult<Seq2SeqToken>> + Send>>>
{
    // Get task-specific model defaults based on INPUT content
    let model_defaults = self.encoder_decoder.model.get_generation_config_for_input(input);
    
    // Resolve with user and runtime overrides
    let config = resolve_seq2seq_config(
        model_defaults,
        self.task,
        &self.user_overrides,
        &runtime_overrides,
    );

    // Clone/own everything we need for the spawned task
    let encoder_decoder = self.encoder_decoder.clone();
    let config = config.into_inner();
    let input = input.to_string();

    let (tx, rx) = mpsc::channel::<Seq2SeqResult<Seq2SeqToken>>(32);

    tokio::spawn(async move {
        let stream = encoder_decoder.generate_stream(&input, Some(&config));

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
                break; // Receiver dropped
            }
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Ok(Box::pin(stream))
}

    /// Stream as simple strings (convenience method).
    pub async fn stream_text(
        &self,
        input: &str,
    ) -> Seq2SeqResult<std::pin::Pin<Box<dyn Stream<Item = Seq2SeqResult<String>> + Send>>> {
        let inner_stream = self.stream(input).await?;
        let mapped = inner_stream.map(|result| result.map(|token| token.text));
        Ok(Box::pin(mapped))
    }

    // =========================================================================
    // Config Access
    // =========================================================================

    /// Get the model's default generation config.
    pub fn get_model_defaults(&self) -> GenerationConfig {
        self.encoder_decoder.model.get_default_generation_config()
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the model type.
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Get the model's CLI name.
    pub fn model_name(&self) -> &str {
        self.model_type.cli_name()
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
    ///
    /// For advanced use cases that need direct access.
    pub fn encoder_decoder(&self) -> &EncoderDecoderGenerator {
        &self.encoder_decoder
    }
}
