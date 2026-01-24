// =============================================================================
// kjarni/src/generator/model.rs
// =============================================================================

//! Core Generator implementation.

use std::sync::Arc;

use anyhow::anyhow;
use futures::{StreamExt, Stream, pin_mut};
use tokio::sync::mpsc;

use kjarni_transformers::{
    models::{ModelArchitecture, ModelType},
    traits::Device,
    WgpuContext,
};

use crate::common::DownloadPolicy;
use crate::generation::{
    resolve_generation_config, GenerationOverrides, ResolvedGenerationConfig,
};
use crate::{DecoderGenerator, DecoderLanguageModel, TokenType};

use super::builder::GeneratorBuilder;
use super::types::{GeneratedToken, GeneratorError, GeneratorResult};
use super::validation::validate_for_generation;

/// Raw text generator for decoder language models.
///
/// Generator provides direct access to language model text generation
/// without chat templates, system prompts, or conversation management.
///
/// # When to Use Generator vs Chat
///
/// | Use Case | Generator | Chat |
/// |----------|-----------|------|
/// | Text completion (GPT-2 style) | ✅ | ❌ |
/// | Custom prompt formats | ✅ | ❌ |
/// | Base models | ✅ | ❌ |
/// | Conversational AI | ❌ | ✅ |
/// | Multi-turn dialogue | ❌ | ✅ |
/// | Instruction-tuned models | ⚠️ | ✅ |
///
/// # Example
///
/// ```ignore
/// use kjarni::generator::Generator;
///
/// // Simple completion
/// let generator = Generator::new("gpt2").await?;
/// let completion = generator.generate("The quick brown fox").await?;
/// println!("{}", completion);
///
/// // With custom configuration
/// let generator = Generator::builder("llama3.2-1b")
///     .temperature(0.8)
///     .max_tokens(200)
///     .top_p(0.9)
///     .build()
///     .await?;
///
/// let text = generator.generate("Once upon a time").await?;
///
/// // Streaming
/// use futures::StreamExt;
///
/// let mut stream = generator.stream("In a galaxy far away").await?;
/// while let Some(token) = stream.next().await {
///     print!("{}", token?.text);
/// }
/// ```
pub struct Generator {
    /// The underlying decoder generator.
    pub(crate) decoder: Arc<DecoderGenerator>,

    /// Model type from registry.
    model_type: ModelType,

    /// Resolved generation config (from model defaults + user overrides).
    generation_config: ResolvedGenerationConfig,

    /// User-provided overrides (stored for reuse).
    user_overrides: GenerationOverrides,

    /// Device the model is running on.
    device: Device,

    /// GPU context if using GPU.
    context: Option<Arc<WgpuContext>>,
}

impl Generator {
    // =========================================================================
    // Construction
    // =========================================================================

    /// Create a Generator with default settings.
    ///
    /// Uses CPU, downloads model if needed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let generator = Generator::new("gpt2").await?;
    /// ```
    pub async fn new(model: &str) -> GeneratorResult<Self> {
        GeneratorBuilder::new(model).build().await
    }

    /// Create a builder for custom configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let generator = Generator::builder("llama3.2-1b")
    ///     .gpu()
    ///     .temperature(0.9)
    ///     .build()
    ///     .await?;
    /// ```
    pub fn builder(model: &str) -> GeneratorBuilder {
        GeneratorBuilder::new(model)
    }

    /// Internal: construct from builder.
    pub(crate) async fn from_builder(builder: GeneratorBuilder) -> GeneratorResult<Self> {
        // Step 1: Resolve model type
        let model_type = ModelType::from_cli_name(&builder.model)
            .ok_or_else(|| GeneratorError::UnknownModel(builder.model.clone()))?;

        // Step 2: Validate model for generation
        let validation = validate_for_generation(model_type)?;

        // Step 3: Emit warnings if not suppressed
        if !builder.quiet && !builder.allow_warnings {
            for warning in &validation.warnings {
                eprintln!("⚠️  {}", warning);
            }
        }

        // Step 4: Resolve cache directory
        let cache_dir = builder.cache_dir.clone().unwrap_or_else(|| {
            dirs::cache_dir()
                .expect("No cache directory found")
                .join("kjarni")
        });

        // Step 5: Check download status and download if needed
        let is_downloaded = model_type.is_downloaded(&cache_dir);

        if !is_downloaded {
            match builder.download_policy {
                DownloadPolicy::Never => {
                    return Err(GeneratorError::ModelNotDownloaded(builder.model.clone()));
                }
                DownloadPolicy::IfMissing | DownloadPolicy::Eager => {
                    if !builder.quiet {
                        eprintln!("Downloading model '{}'...", builder.model);
                    }
                    kjarni_transformers::models::download_model_files(
                        &model_type.cache_dir(&cache_dir),
                        &model_type.info().paths,
                        kjarni_transformers::models::registry::WeightsFormat::SafeTensors,
                        builder.quiet,
                    )
                    .await
                    .map_err(|e| GeneratorError::DownloadFailed {
                        model: builder.model.clone(),
                        source: e,
                    })?;
                }
            }
        }

        // Step 6: Resolve device
        let device = builder.device.to_device();

        // Step 7: Create GPU context if needed
        let context = if device == Device::Wgpu {
            if let Some(ctx) = builder.context {
                Some(ctx)
            } else {
                Some(
                    WgpuContext::new()
                        .await
                        .map_err(|_| GeneratorError::GpuUnavailable)?,
                )
            }
        } else {
            None
        };

        // Step 8: Load the model
        let load_config = builder
            .load_config
            .map(|c| c.into_inner())
            .unwrap_or_default();

        let model: Arc<dyn DecoderLanguageModel> =
            Self::load_model(model_type, &cache_dir, device, context.clone(), load_config)
                .await
                .map_err(|e| GeneratorError::LoadFailed {
                    model: builder.model.clone(),
                    source: e,
                })?;

        // Step 9: Create the decoder generator
        let decoder = Arc::new(
            DecoderGenerator::new(model.clone()).map_err(|e| GeneratorError::LoadFailed {
                model: builder.model.clone(),
                source: e,
            })?,
        );

        // Step 10: Resolve generation config
        let model_defaults = model.get_default_generation_config();
        let generation_config = resolve_generation_config(
            model_defaults,
            &builder.generation_overrides,
            &GenerationOverrides::default(),
        );

        Ok(Self {
            decoder,
            model_type,
            generation_config,
            user_overrides: builder.generation_overrides,
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
        load_config: kjarni_transformers::models::base::ModelLoadConfig,
    ) -> anyhow::Result<Arc<dyn DecoderLanguageModel + Send + Sync>> {
        let info = model_type.info();

        match info.architecture {
            ModelArchitecture::Llama => {
                use kjarni_models::models::llama::LlamaModel;
                let model = LlamaModel::from_registry(
                    model_type,
                    Some(cache_dir.to_path_buf()),
                    device,
                    context,
                    Some(load_config),
                )
                .await?;
                Ok(Arc::new(model) as Arc<dyn DecoderLanguageModel + Send + Sync>)
            }

            ModelArchitecture::Qwen2 => {
                use kjarni_models::models::qwen::QwenModel;
                let model = QwenModel::from_registry(
                    model_type,
                    Some(cache_dir.to_path_buf()),
                    device,
                    context,
                    Some(load_config),
                )
                .await?;
                Ok(Arc::new(model) as Arc<dyn DecoderLanguageModel + Send + Sync>)
            }

            ModelArchitecture::Mistral => {
                use kjarni_models::models::mistral::MistralModel;
                let model = MistralModel::from_registry(
                    model_type,
                    Some(cache_dir.to_path_buf()),
                    device,
                    context,
                    Some(load_config),
                )
                .await?;
                Ok(Arc::new(model) as Arc<dyn DecoderLanguageModel + Send + Sync>)
            }

            ModelArchitecture::GPT => {
                use kjarni_models::models::gpt2::Gpt2Model;
                let model = Gpt2Model::from_registry(
                    model_type,
                    Some(cache_dir.to_path_buf()),
                    device,
                    context,
                    Some(load_config),
                )
                .await?;
                Ok(Arc::new(model) as Arc<dyn DecoderLanguageModel + Send + Sync>)
            }

            ModelArchitecture::Phi3 => {
                Err(anyhow!("Phi3 model loading not yet implemented"))
            }

            _ => Err(anyhow!(
                "Architecture {:?} is not supported for text generation",
                info.architecture
            )),
        }
    }

    // =========================================================================
    // Generation - Stateless
    // =========================================================================

    /// Generate text from a prompt.
    ///
    /// Uses the configured generation parameters.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let completion = generator.generate("The meaning of life is").await?;
    /// println!("{}", completion);
    /// ```
    pub async fn generate(&self, prompt: &str) -> GeneratorResult<String> {
        self.generate_with_config(prompt, &GenerationOverrides::default())
            .await
    }

    /// Generate with custom overrides for this call only.
    ///
    /// Overrides take precedence over builder defaults.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let creative = generator.generate_with_config(
    ///     "Write a poem about",
    ///     &GenerationOverrides {
    ///         temperature: Some(1.0),
    ///         max_new_tokens: Some(200),
    ///         ..Default::default()
    ///     }
    /// ).await?;
    /// ```
    pub async fn generate_with_config(
        &self,
        prompt: &str,
        runtime_overrides: &GenerationOverrides,
    ) -> GeneratorResult<String> {
        let config = resolve_generation_config(
            self.generation_config.inner.clone(),
            &self.user_overrides,
            runtime_overrides,
        );

        let stream = self
            .decoder
            .generate_stream(prompt, config.as_ref(), None)
            .await?;
        
        pin_mut!(stream);

        let mut output = String::new();
        while let Some(token_result) = stream.next().await {
            let token = token_result?;

            // Skip prompt tokens - only include generated tokens
            if token.token_type == TokenType::Prompt {
                continue;
            }

            output.push_str(&token.text);
        }

        Ok(output)
    }

    // =========================================================================
    // Generation - Streaming
    // =========================================================================

    /// Stream generated tokens.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use futures::StreamExt;
    ///
    /// let mut stream = generator.stream("Once upon a time").await?;
    /// while let Some(result) = stream.next().await {
    ///     let token = result?;
    ///     print!("{}", token.text);
    ///     std::io::stdout().flush()?;
    /// }
    /// ```
    pub async fn stream(
        &self,
        prompt: &str,
    ) -> GeneratorResult<std::pin::Pin<Box<dyn Stream<Item = GeneratorResult<GeneratedToken>> + Send>>>
    {
        self.stream_with_config(prompt, GenerationOverrides::default())
            .await
    }

    /// Stream with custom overrides.
    pub async fn stream_with_config(
        &self,
        prompt: &str,
        runtime_overrides: GenerationOverrides,
    ) -> GeneratorResult<std::pin::Pin<Box<dyn Stream<Item = GeneratorResult<GeneratedToken>> + Send>>>
    {
        let config = resolve_generation_config(
            self.generation_config.inner.clone(),
            &self.user_overrides,
            &runtime_overrides,
        );

        // Clone Arc to move into spawned task
        let decoder = self.decoder.clone();
        let config = config.into_inner();
        let prompt = prompt.to_string();

        // Create channel for tokens
        let (tx, rx) = mpsc::channel::<GeneratorResult<GeneratedToken>>(32);

        // Spawn task to process stream
        tokio::spawn(async move {
            let stream = match decoder.generate_stream(&prompt, &config, None).await {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(Err(GeneratorError::GenerationFailed(e))).await;
                    return;
                }
            };

            pin_mut!(stream);

            while let Some(result) = stream.next().await {
                let msg = match result {
                    Ok(token) => {
                        // Skip prompt tokens
                        if token.token_type == TokenType::Prompt {
                            continue;
                        }

                        Ok(GeneratedToken {
                            text: token.text,
                            id: token.id,
                            is_special: token.token_type == TokenType::Special,
                        })
                    }
                    Err(e) => Err(GeneratorError::GenerationFailed(e)),
                };

                if tx.send(msg).await.is_err() {
                    // Receiver dropped, stop generating
                    break;
                }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }

    /// Stream as simple strings (convenience method).
    pub async fn stream_text(
        &self,
        prompt: &str,
    ) -> GeneratorResult<std::pin::Pin<Box<dyn Stream<Item = GeneratorResult<String>> + Send>>>
    {
        let inner_stream = self.stream(prompt).await?;

        let mapped = inner_stream.map(|result| result.map(|token| token.text));

        Ok(Box::pin(mapped))
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
        self.decoder.model.context_size()
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.decoder.model.vocab_size()
    }

    /// Get the current generation config.
    pub fn generation_config(&self) -> &ResolvedGenerationConfig {
        &self.generation_config
    }

    /// Get a reference to the underlying decoder generator.
    ///
    /// For advanced use cases that need direct access.
    pub fn decoder(&self) -> &DecoderGenerator {
        &self.decoder
    }

    /// Get the GPU context if using GPU.
    pub fn gpu_context(&self) -> Option<&Arc<WgpuContext>> {
        self.context.as_ref()
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Generate text with default settings.
///
/// # Example
///
/// ```ignore
/// let text = kjarni::generator::generate("gpt2", "The quick brown").await?;
/// ```
pub async fn generate(model: &str, prompt: &str) -> GeneratorResult<String> {
    Generator::new(model).await?.generate(prompt).await
}

/// Generate text with custom configuration.
pub async fn generate_with_config(
    model: &str,
    prompt: &str,
    config: GenerationOverrides,
) -> GeneratorResult<String> {
    Generator::builder(model)
        .generation_config(config)
        .build()
        .await?
        .generate(prompt)
        .await
}