//! Core Generator implementation.

use std::sync::Arc;

use anyhow::anyhow;
use futures::{Stream, StreamExt, pin_mut};
use tokio::sync::mpsc;

use kjarni_transformers::WgpuContext;
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::models::{ModelArchitecture, ModelType};
use kjarni_transformers::tensor::DType;
use kjarni_transformers::traits::Device;

use crate::common::DownloadPolicy;
use crate::generation::{GenerationOverrides, ResolvedGenerationConfig, resolve_generation_config};
use crate::{DecoderGenerator, DecoderLanguageModel, TokenType};

use super::builder::GeneratorBuilder;
use super::types::{GeneratedToken, GeneratorError, GeneratorResult};
use super::validation::validate_for_generation;
use kjarni_transformers::models::get_default_cache_dir;

/// Raw text generator for decoder language models.
///
/// Provides direct access to language model text generation without chat
/// templates, system prompts, or conversation management.
pub struct Generator {
    pub(crate) decoder: Arc<DecoderGenerator>,
    model_type: Option<ModelType>,
    model_name: String,
    generation_config: ResolvedGenerationConfig,
    user_overrides: GenerationOverrides,
    device: Device,
    context: Option<Arc<WgpuContext>>,
}

impl Generator {
    /// Creates a Generator with default settings.
    pub async fn new(model: &str) -> GeneratorResult<Self> {
        GeneratorBuilder::new(model).build().await
    }

    /// Creates a builder for custom configuration.
    pub fn builder(model: &str) -> GeneratorBuilder {
        GeneratorBuilder::new(model)
    }

    pub(crate) async fn from_builder(builder: GeneratorBuilder) -> GeneratorResult<Self> {
        let cache_dir = builder
            .cache_dir
            .clone()
            .unwrap_or_else(get_default_cache_dir);

        let mut load_config = builder
            .load_config
            .clone()
            .map(|c| c.into_inner())
            .unwrap_or_default();
        // The builder's `quiet` has to reach the loader, which downloads GGUF
        // weights itself and prints progress unless its own config says not to.
        // A caller's load config started from defaults, so `quiet` was dropped.
        load_config.quiet |= builder.quiet;

        let device = builder.device.to_device();
        let context = if device == Device::Wgpu {
            if let Some(ctx) = builder.context.clone() {
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

        let (model, model_type, model_name): (Arc<dyn DecoderLanguageModel>, _, _) = match &builder
            .model_path
        {
            Some(path) => {
                let name = folder_name(path);
                if !path.exists() {
                    return Err(GeneratorError::InvalidConfig(format!(
                        "model path not found: {}",
                        path.display()
                    )));
                }
                let family = DecoderFamily::of_folder(path)
                    .map_err(|reason| GeneratorError::InvalidModel(name.clone(), reason))?;
                if !builder.quiet {
                    eprintln!("Loading {} from {}...", family.name(), path.display());
                }
                let model = family
                    .load(path, device, context.clone(), load_config)
                    .map_err(|e| GeneratorError::LoadFailed {
                        model: name.clone(),
                        source: e,
                    })?;
                (model, None, name)
            }
            None => {
                let model_type = ModelType::from_cli_name(&builder.model)
                    .ok_or_else(|| GeneratorError::UnknownModel(builder.model.clone()))?;
                validate_for_generation(model_type)?;

                // A quantized request is served from the GGUF, which the loader fetches;
                // downloading the safetensors here first cost a model's full-precision
                // size (2.5GB for Llama 3.2 1B) for weights that were never read.
                let wants_gguf = load_config.use_gguf
                    || matches!(
                        load_config.target_dtype,
                        Some(DType::Q4_K) | Some(DType::Q6_K) | Some(DType::Q8_0)
                    );
                let is_downloaded = wants_gguf || model_type.is_downloaded(&cache_dir);

                if !is_downloaded {
                    match builder.download_policy {
                        DownloadPolicy::Never => {
                            return Err(GeneratorError::ModelNotDownloaded(builder.model.clone()));
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
                            .map_err(|e| {
                                GeneratorError::DownloadFailed {
                                    model: builder.model.clone(),
                                    source: e,
                                }
                            })?;
                        }
                    }
                }

                let model =
                    Self::load_model(model_type, &cache_dir, device, context.clone(), load_config)
                        .await
                        .map_err(|e| GeneratorError::LoadFailed {
                            model: builder.model.clone(),
                            source: e,
                        })?;
                (model, Some(model_type), builder.model.clone())
            }
        };

        let mut decoder_inner =
            DecoderGenerator::new(model.clone()).map_err(|e| GeneratorError::LoadFailed {
                model: model_name.clone(),
                source: e,
            })?;

        // The draft is loaded before the generator is shared, because attaching
        // one needs &mut. It must share the target's vocabulary; a mismatch is
        // reported here rather than producing proposals that can never be scored.
        let speculation = if let Some(ref draft_name) = builder.draft_model {
            let draft_type = ModelType::from_cli_name(draft_name)
                .ok_or_else(|| GeneratorError::UnknownModel(draft_name.clone()))?;
            let draft_model = Self::load_model(
                draft_type,
                &cache_dir,
                device,
                context.clone(),
                Default::default(),
            )
            .await
            .map_err(|e| GeneratorError::LoadFailed {
                model: draft_name.clone(),
                source: e,
            })?;
            decoder_inner.load_draft_model(draft_model).map_err(|e| {
                GeneratorError::LoadFailed {
                    model: draft_name.clone(),
                    source: e,
                }
            })?;
            Some(kjarni_transformers::common::SpeculationParams {
                num_tokens: builder.draft_tokens,
                prompt_lookup: builder.prompt_lookup,
            })
        } else if builder.prompt_lookup.is_some() {
            Some(kjarni_transformers::common::SpeculationParams {
                num_tokens: builder.draft_tokens,
                prompt_lookup: builder.prompt_lookup,
            })
        } else {
            None
        };

        // Capacity caps the prefix that can be reused, and the KV cache behind it
        // is allocated eagerly, so this is a memory decision the caller makes
        // rather than the model's full context window.
        if let Some(tokens) = builder.prefix_cache_tokens {
            decoder_inner
                .enable_prefix_cache(tokens)
                .map_err(|e| GeneratorError::LoadFailed {
                    model: model_name.clone(),
                    source: e,
                })?;
        }

        let decoder = Arc::new(decoder_inner);

        let model_defaults = model.get_default_generation_config();
        let mut generation_config = resolve_generation_config(
            model_defaults,
            &builder.generation_overrides,
            &GenerationOverrides::default(),
        );
        generation_config.inner.speculation = speculation;

        Ok(Self {
            decoder,
            model_type,
            model_name,
            generation_config,
            user_overrides: builder.generation_overrides,
            device,
            context,
        })
    }

    async fn load_model(
        model_type: ModelType,
        cache_dir: &std::path::Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: ModelLoadConfig,
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
                use kjarni_models::models::phi::PhiModel;
                let model = PhiModel::from_registry(
                    model_type,
                    Some(cache_dir.to_path_buf()),
                    device,
                    context,
                    Some(load_config),
                )
                .await?;
                Ok(Arc::new(model) as Arc<dyn DecoderLanguageModel + Send + Sync>)
            }
            _ => Err(anyhow!(
                "architecture {:?} is not supported for text generation",
                info.architecture
            )),
        }
    }

    /// Generates text from a prompt.
    pub async fn generate(&self, prompt: &str) -> GeneratorResult<String> {
        self.generate_with_config(prompt, &GenerationOverrides::default())
            .await
    }

    /// Generates with custom overrides for this call only.
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

            if token.token_type == TokenType::Prompt {
                continue;
            }

            output.push_str(&token.text);
        }

        Ok(output)
    }

    /// Streams generated tokens.
    pub async fn stream(
        &self,
        prompt: &str,
    ) -> GeneratorResult<
        std::pin::Pin<Box<dyn Stream<Item = GeneratorResult<GeneratedToken>> + Send>>,
    > {
        self.stream_with_config(prompt, GenerationOverrides::default())
            .await
    }

    /// Streams with custom overrides.
    pub async fn stream_with_config(
        &self,
        prompt: &str,
        runtime_overrides: GenerationOverrides,
    ) -> GeneratorResult<
        std::pin::Pin<Box<dyn Stream<Item = GeneratorResult<GeneratedToken>> + Send>>,
    > {
        let config = resolve_generation_config(
            self.generation_config.inner.clone(),
            &self.user_overrides,
            &runtime_overrides,
        );

        let decoder = self.decoder.clone();
        let config = config.into_inner();
        let prompt = prompt.to_string();

        let (tx, rx) = mpsc::channel::<GeneratorResult<GeneratedToken>>(32);

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
                    break;
                }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }

    /// Streams as simple strings.
    pub async fn stream_text(
        &self,
        prompt: &str,
    ) -> GeneratorResult<std::pin::Pin<Box<dyn Stream<Item = GeneratorResult<String>> + Send>>>
    {
        let inner_stream = self.stream(prompt).await?;
        let mapped = inner_stream.map(|result| result.map(|token| token.text));
        Ok(Box::pin(mapped))
    }

    /// Whether a BOS token is put before prompts that do not start with one.
    pub(crate) fn set_add_bos_token(&mut self, add: bool) {
        self.generation_config.inner.add_bos_token = add;
    }

    /// The registry model, or `None` for one loaded from a folder.
    pub fn model_type(&self) -> Option<ModelType> {
        self.model_type
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn context_size(&self) -> usize {
        self.decoder.model.context_size()
    }

    pub fn vocab_size(&self) -> usize {
        self.decoder.model.vocab_size()
    }

    pub fn generation_config(&self) -> &ResolvedGenerationConfig {
        &self.generation_config
    }

    /// Returns a reference to the underlying decoder generator.
    pub fn decoder(&self) -> &DecoderGenerator {
        &self.decoder
    }

    pub fn gpu_context(&self) -> Option<&Arc<WgpuContext>> {
        self.context.as_ref()
    }
}

#[allow(
    dead_code,
    reason = "not referenced yet; kept until the path that needs it lands"
)]
/// Generates text with default settings.
pub async fn generate(model: &str, prompt: &str) -> GeneratorResult<String> {
    Generator::new(model).await?.generate(prompt).await
}

#[expect(
    dead_code,
    reason = "not referenced yet; kept until the path that needs it lands"
)]
/// Generates text with custom configuration.
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

/// Load the decoder in a model folder, its architecture read from its
/// `config.json`: Llama, Qwen2, Mistral, Phi-3 or GPT-2. A folder that is not
/// a registry model uses its own Jinja chat template.
pub async fn load_decoder(
    path: &std::path::Path,
    device: Device,
    load_config: Option<ModelLoadConfig>,
) -> GeneratorResult<Arc<dyn DecoderLanguageModel + Send + Sync>> {
    let name = folder_name(path);
    if !path.exists() {
        return Err(GeneratorError::InvalidConfig(format!(
            "model path not found: {}",
            path.display()
        )));
    }
    let family = DecoderFamily::of_folder(path)
        .map_err(|reason| GeneratorError::InvalidModel(name.clone(), reason))?;
    let context = if device == Device::Wgpu {
        Some(
            WgpuContext::new()
                .await
                .map_err(|_| GeneratorError::GpuUnavailable)?,
        )
    } else {
        None
    };
    family
        .load(path, device, context, load_config.unwrap_or_default())
        .map_err(|e| GeneratorError::LoadFailed {
            model: name,
            source: e,
        })
}

/// The decoder architectures a model folder can hold, by its `config.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecoderFamily {
    Llama,
    Qwen2,
    Mistral,
    Phi3,
    Gpt2,
}

impl DecoderFamily {
    /// The family named by `config.json`: its `architectures`, or failing
    /// that its `model_type`. A file path, such as a `.gguf`, is read next
    /// to the `config.json` in its folder.
    pub(crate) fn of_folder(path: &std::path::Path) -> Result<Self, String> {
        let dir = if path.is_file() {
            path.parent().unwrap_or(path)
        } else {
            path
        };
        let file = dir.join("config.json");
        let text = std::fs::read_to_string(&file)
            .map_err(|e| format!("cannot read {}: {e}", file.display()))?;
        let config: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| format!("cannot parse {}: {e}", file.display()))?;
        let architecture = config["architectures"]
            .get(0)
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let model_type = config["model_type"].as_str().unwrap_or("");
        match (architecture, model_type) {
            ("LlamaForCausalLM", _) | ("", "llama") => Ok(Self::Llama),
            ("Qwen2ForCausalLM", _) | ("", "qwen2") => Ok(Self::Qwen2),
            ("MistralForCausalLM", _) | ("", "mistral") => Ok(Self::Mistral),
            ("Phi3ForCausalLM", _) | ("", "phi3") => Ok(Self::Phi3),
            ("GPT2LMHeadModel", _) | ("", "gpt2") => Ok(Self::Gpt2),
            _ => Err(format!(
                "it is a {} model; text generation supports LlamaForCausalLM, \
                 Qwen2ForCausalLM, MistralForCausalLM, Phi3ForCausalLM and GPT2LMHeadModel",
                if architecture.is_empty() {
                    model_type
                } else {
                    architecture
                }
            )),
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Llama => "Llama",
            Self::Qwen2 => "Qwen2",
            Self::Mistral => "Mistral",
            Self::Phi3 => "Phi-3",
            Self::Gpt2 => "GPT-2",
        }
    }

    pub(crate) fn load(
        self,
        path: &std::path::Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: ModelLoadConfig,
    ) -> anyhow::Result<Arc<dyn DecoderLanguageModel + Send + Sync>> {
        use kjarni_models::models::{
            gpt2::Gpt2Model, llama::LlamaModel, mistral::MistralModel, phi::PhiModel,
            qwen::QwenModel,
        };
        let cfg = Some(load_config);
        Ok(match self {
            Self::Llama => Arc::new(LlamaModel::from_pretrained(
                path, device, context, cfg, None,
            )?),
            Self::Qwen2 => Arc::new(QwenModel::from_pretrained(
                path, device, context, cfg, None,
            )?),
            Self::Mistral => Arc::new(MistralModel::from_pretrained(
                path, device, context, cfg, None,
            )?),
            Self::Phi3 => Arc::new(PhiModel::from_pretrained(path, device, context, cfg, None)?),
            Self::Gpt2 => Arc::new(Gpt2Model::from_pretrained(
                path,
                ModelType::Gpt2,
                device,
                context,
                cfg,
            )?),
        })
    }
}

pub(crate) use crate::common::folder_name;

#[cfg(test)]
mod family_tests {
    use super::*;

    fn folder(config: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), config).unwrap();
        dir
    }

    #[test]
    fn the_architecture_names_the_family() {
        for (arch, family) in [
            ("LlamaForCausalLM", DecoderFamily::Llama),
            ("Qwen2ForCausalLM", DecoderFamily::Qwen2),
            ("MistralForCausalLM", DecoderFamily::Mistral),
            ("Phi3ForCausalLM", DecoderFamily::Phi3),
            ("GPT2LMHeadModel", DecoderFamily::Gpt2),
        ] {
            // The architecture wins over a model_type that says otherwise.
            let dir = folder(&format!(
                r#"{{"architectures": ["{arch}"], "model_type": "bert"}}"#
            ));
            assert_eq!(DecoderFamily::of_folder(dir.path()), Ok(family), "{arch}");
        }
    }

    #[test]
    fn without_architectures_the_model_type_names_the_family() {
        for (mt, family) in [
            ("llama", DecoderFamily::Llama),
            ("qwen2", DecoderFamily::Qwen2),
            ("mistral", DecoderFamily::Mistral),
            ("phi3", DecoderFamily::Phi3),
            ("gpt2", DecoderFamily::Gpt2),
        ] {
            let dir = folder(&format!(r#"{{"model_type": "{mt}"}}"#));
            assert_eq!(DecoderFamily::of_folder(dir.path()), Ok(family), "{mt}");
        }
    }

    #[test]
    fn a_file_in_the_folder_is_read_by_its_folder() {
        let dir = folder(r#"{"architectures": ["Qwen2ForCausalLM"]}"#);
        let gguf = dir.path().join("model.gguf");
        std::fs::write(&gguf, b"").unwrap();
        assert_eq!(DecoderFamily::of_folder(&gguf), Ok(DecoderFamily::Qwen2));
    }

    #[test]
    fn other_models_are_named_in_the_error() {
        let dir = folder(r#"{"architectures": ["XLMRobertaForTokenClassification"]}"#);
        let err = DecoderFamily::of_folder(dir.path()).unwrap_err();
        assert!(
            err.starts_with("it is a XLMRobertaForTokenClassification model;"),
            "{err}"
        );

        let dir = folder(r#"{"model_type": "t5"}"#);
        let err = DecoderFamily::of_folder(dir.path()).unwrap_err();
        assert!(err.starts_with("it is a t5 model;"), "{err}");
    }

    #[test]
    fn a_missing_or_broken_config_is_an_error() {
        let empty = tempfile::tempdir().unwrap();
        let err = DecoderFamily::of_folder(empty.path()).unwrap_err();
        assert!(
            err.starts_with("cannot read") && err.contains("config.json"),
            "{err}"
        );

        let dir = folder("{not json");
        let err = DecoderFamily::of_folder(dir.path()).unwrap_err();
        assert!(err.starts_with("cannot parse"), "{err}");
    }

    #[test]
    fn a_folder_model_is_called_by_its_folder_name() {
        assert_eq!(
            folder_name(std::path::Path::new("/models/my-finetune")),
            "my-finetune"
        );
        assert_eq!(
            folder_name(std::path::Path::new("/models/my-finetune/")),
            "my-finetune"
        );
    }
}
