//! Core Chat model implementation.
//!
//! The Chat struct wraps a decoder language model and provides
//! a high-level API for conversational text generation.

use std::sync::Arc;

use crate::generation::resolved::ResolvedGenerationConfig;
use crate::generation::{overrides::GenerationOverrides, resolve_generation_config};
use crate::{DecoderGenerator, DecoderLanguageModel, TokenType};
use anyhow::anyhow;
use futures_util::StreamExt;
use kjarni_transformers::{
    models::{ModelArchitecture, ModelType}, traits::Device, ChatTemplate,
    Conversation,
    WgpuContext,
};

use super::builder::ChatBuilder;
use super::conversation::ChatConversation;
use super::types::{ChatDevice, ChatError, ChatMode, ChatResult, History, Role};
use super::validation::validate_for_chat;
use crate::common::DownloadPolicy;

/// High-level chat interface for conversational AI.
///
/// Chat wraps a decoder language model and provides simple methods
/// for text generation with conversation support.
///
/// # Example
///
/// ```ignore
/// use kjarni::chat::Chat;
///
/// // Simple one-shot
/// let chat = Chat::new("llama3.2-1b").await?;
/// let response = chat.send("Hello!").await?;
///
/// // With history
/// let mut history = History::with_system("You are helpful.");
/// history.push_user("What is Rust?");
/// let response = chat.send_with_history(&history, "Tell me more.").await?;
/// ```
pub struct Chat {

    // Generator instance
    generator: Arc<DecoderGenerator>,

    /// Model type from registry.
    model_type: ModelType,

    /// Default system prompt.
    system_prompt: Option<String>,

    /// Generation defaults (resolved from model + user config).
    generation_config: ResolvedGenerationConfig,

    /// User-provided overrides (stored for reuse).
    user_overrides: GenerationOverrides,

    /// Chat mode (affects generation defaults).
    mode: ChatMode,

    /// Device the model is running on.
    device: Device,

    /// GPU context if using GPU.
    context: Option<Arc<WgpuContext>>,
}

impl Chat {
    /// Create a Chat with default settings.
    ///
    /// Uses CPU, downloads model if needed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let chat = Chat::new("llama3.2-1b").await?;
    /// ```
    pub async fn new(model: &str) -> ChatResult<Self> {
        ChatBuilder::new(model).build().await
    }

    /// Internal: construct from builder.
    pub(crate) async fn from_builder(builder: ChatBuilder) -> ChatResult<Self> {
        // Step 1: Resolve model type
        let model_type = ModelType::from_cli_name(&builder.model)
            .ok_or_else(|| ChatError::UnknownModel(builder.model.clone()))?;

        // Step 2: Validate model for chat
        let validation = validate_for_chat(model_type)?;

        // Step 3: Emit warnings if not suppressed
        if !builder.quiet && !builder.allow_suboptimal {
            for warning in &validation.warnings {
                eprintln!("{}", warning);
            }
        }

        // Step 4: Check download status
        let cache_dir = builder.cache_dir.clone().unwrap_or_else(|| {
            dirs::cache_dir()
                .expect("No cache directory found")
                .join("kjarni")
        });

        let is_downloaded = model_type.is_downloaded(&cache_dir);

        if !is_downloaded {
            match builder.download_policy {
                DownloadPolicy::Never => {
                    return Err(ChatError::ModelNotDownloaded(builder.model.clone()));
                }
                DownloadPolicy::IfMissing | DownloadPolicy::Eager => {
                    if !builder.quiet {
                        eprintln!("Downloading model '{}'...", builder.model);
                    }
                    kjarni_transformers::models::download_model_files(
                        &model_type.cache_dir(&cache_dir),
                        &model_type.info().paths,
                        kjarni_transformers::models::registry::WeightsFormat::SafeTensors,
                    )
                        .await
                        .map_err(|e| ChatError::DownloadFailed {
                            model: builder.model.clone(),
                            source: e,
                        })?;
                }
            }
        }

        // Step 5: Resolve device
        let device = match builder.device.resolve() {
            ChatDevice::Cpu => Device::Cpu,
            ChatDevice::Gpu | ChatDevice::Auto => Device::Wgpu,
        };

        // Step 6: Create GPU context if needed
        let context = if device == Device::Wgpu {
            if let Some(ctx) = builder.context {
                Some(ctx)
            } else {
                Some(
                    WgpuContext::new()
                        .await
                        .map_err(|_| ChatError::GpuUnavailable)?,
                )
            }
        } else {
            None
        };

        // Load the model
        let model: Arc<dyn DecoderLanguageModel> =
            Self::load_model(model_type, &cache_dir, device, context.clone())
                .await
                .map_err(|e| ChatError::LoadFailed {
                    model: builder.model.clone(),
                    source: e,
                })?;

        // Create generator
        let generator = Arc::new(DecoderGenerator::new(model.clone())
            .map_err(|e| ChatError::LoadFailed { 
                model: builder.model.clone(), 
                source: e 
            })?);

        // Verify chat template exists
        if model.chat_template().is_none() {
            return Err(ChatError::InvalidConfig(
                "Model does not have a chat template configured".to_string(),
            ));
        }

        // Step 9: Resolve generation config
        let model_defaults = model.get_default_generation_config();

        // Apply mode-specific adjustments
        let mut mode_overrides = builder.generation_overrides.clone();
        if mode_overrides.temperature.is_none() {
            mode_overrides.temperature = Some(builder.mode.default_temperature());
        }
        if mode_overrides.max_new_tokens.is_none() {
            mode_overrides.max_new_tokens = Some(builder.mode.default_max_tokens());
        }

        let generation_config = resolve_generation_config(
            model_defaults,
            &mode_overrides,
            &GenerationOverrides::default(),
        );
        
        Ok(Self {
            generator,
            model_type,
            system_prompt: builder.system_prompt,
            generation_config,
            user_overrides: builder.generation_overrides,
            mode: builder.mode,
            device,
            context,
        })
    }

    /// Get a reference to the inner model.
    fn inner_model(&self) -> &dyn DecoderLanguageModel {
        self.generator.model.as_ref()
    }

    /// Load the appropriate model based on architecture.
    async fn load_model(
        model_type: ModelType,
        cache_dir: &std::path::Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
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
                    None,
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
                    None,
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
                    None,
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
                    None,
                )
                    .await?;
                Ok(Arc::new(model) as Arc<dyn DecoderLanguageModel + Send + Sync>)
            }

            // Phi3 - you may need to add this to kjarni-models
            ModelArchitecture::Phi3 => Err(anyhow!("Phi3 model loading not yet implemented")),

            // These should have been caught by validation
            _ => Err(anyhow!(
                "Architecture {:?} is not supported for chat",
                info.architecture
            )),
        }
    }

    // =========================================================================
    // Stateless Generation
    // =========================================================================

    /// Get the chat template from the inner model.
    fn get_chat_template(&self) -> ChatResult<&dyn ChatTemplate> {
        self.inner_model().chat_template().ok_or_else(|| {
            ChatError::InvalidConfig("Model does not have a chat template".to_string())
        })
    }

    /// Send a message and get a response.
    ///
    /// This is stateless - no conversation history is maintained.
    /// For multi-turn conversations, use `send_with_history` or `conversation()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let response = chat.send("What is the capital of France?").await?;
    /// println!("{}", response);
    /// ```
    pub async fn send(&self, message: &str) -> ChatResult<String> {
        let mut conversation = self.create_conversation();
        conversation.push_user(message);

        let prompt = self.get_chat_template()?.apply(&conversation);
        self.generate(&prompt, &GenerationOverrides::default())
            .await
    }

    /// Send a message with conversation history.
    ///
    /// The history is used for context but not modified.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut history = History::with_system("You are a helpful assistant.");
    /// history.push_user("Hello");
    /// history.push_assistant("Hi! How can I help?");
    ///
    /// let response = chat.send_with_history(&history, "What's the weather?").await?;
    /// ```
    pub async fn send_with_history(&self, history: &History, message: &str) -> ChatResult<String> {
        let mut conversation = self.history_to_conversation(history);
        conversation.push_user(message);

        let prompt = self.get_chat_template()?.apply(&conversation);
        self.generate(&prompt, &GenerationOverrides::default())
            .await
    }

    /// Send with custom generation overrides.
    ///
    /// Overrides take precedence over builder defaults.
    pub async fn send_with_config(
        &self,
        message: &str,
        overrides: &GenerationOverrides,
    ) -> ChatResult<String> {
        let mut conversation = self.create_conversation();
        conversation.push_user(message);

        let prompt = self.get_chat_template()?.apply(&conversation);
        self.generate(&prompt, overrides).await
    }

    // =========================================================================
    // Streaming Generation
    // =========================================================================

    /// Stream a response token by token.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use futures_util::StreamExt;
    ///
    /// let mut stream = chat.stream("Tell me a story.").await?;
    /// while let Some(token) = stream.next().await {
    ///     print!("{}", token?);
    /// }
    /// ```
    pub async fn stream(
        &self,
        message: &str,
    ) -> ChatResult<std::pin::Pin<Box<dyn futures_util::Stream<Item=ChatResult<String>> + Send>>>
    {
        let mut conversation = self.create_conversation();
        conversation.push_user(message);

        let prompt = self.get_chat_template()?.apply(&conversation);
        self.generate_stream(prompt, GenerationOverrides::default())
            .await
    }

    /// Stream with conversation history.
    pub async fn stream_with_history(
        &self,
        history: &History,
        message: &str,
    ) -> ChatResult<std::pin::Pin<Box<dyn futures_util::Stream<Item=ChatResult<String>> + Send>>>
    {
        let mut conversation = self.history_to_conversation(history);
        conversation.push_user(message);

        let prompt = self.get_chat_template()?.apply(&conversation);
        self.generate_stream(prompt, GenerationOverrides::default())
            .await
    }

    // =========================================================================
    // Stateful Conversation
    // =========================================================================

    /// Create a stateful conversation.
    ///
    /// The conversation maintains history automatically.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut convo = chat.conversation();
    /// convo.send("Hello!").await?;
    /// convo.send("Tell me about Rust.").await?;
    ///
    /// // Inspect history
    /// for msg in convo.history() {
    ///     println!("{}: {}", msg.role, msg.content);
    /// }
    /// ```
    pub fn conversation(&self) -> ChatConversation<'_> {
        ChatConversation::new(self)
    }

    /// Create a stateful conversation with a custom system prompt.
    pub fn conversation_with_system(&self, system: impl Into<String>) -> ChatConversation<'_> {
        ChatConversation::with_system(self, system.into())
    }

    // =========================================================================
    // Internal Methods
    // =========================================================================

    /// Create a Conversation with the default system prompt.
    fn create_conversation(&self) -> Conversation {
        match &self.system_prompt {
            Some(prompt) => Conversation::with_system(prompt),
            None => {
                // Try to get default from template
                if let Some(template) = self.inner_model().chat_template() {
                    if let Some(default) = template.default_system_prompt() {
                        return Conversation::with_system(default);
                    }
                }
                Conversation::new()
            }
        }
    }

    /// Convert History to Conversation.
    fn history_to_conversation(&self, history: &History) -> Conversation {
        let mut conversation = Conversation::new();

        for msg in history.messages() {
            match msg.role {
                Role::System => {
                    // Set system prompt (overrides)
                    conversation = Conversation::with_system(&msg.content);
                }
                Role::User => conversation.push_user(&msg.content),
                Role::Assistant => conversation.push_assistant(&msg.content),
            }
        }

        // If no system in history, use default
        if history.messages().first().map(|m| m.role) != Some(Role::System) {
            if let Some(system) = &self.system_prompt {
                let mut with_system = Conversation::with_system(system);
                for msg in conversation.messages() {
                    match msg.role {
                        kjarni_transformers::Role::User => with_system.push_user(&msg.content),
                        kjarni_transformers::Role::Assistant => {
                            with_system.push_assistant(&msg.content)
                        }
                        _ => {}
                    }
                }
                return with_system;
            }
        }

        conversation
    }

    /// Generate a complete response.
    async fn generate(
        &self,
        prompt: &str,
        runtime_overrides: &GenerationOverrides,
    ) -> ChatResult<String> {
        let config = resolve_generation_config(
            self.generation_config.inner.clone(),
            &self.user_overrides,
            runtime_overrides,
        );

        let stream = self.generator
            .generate_stream(prompt, config.as_ref(), None)
            .await?;

        futures_util::pin_mut!(stream);

        let mut response = String::new();
        while let Some(token_result) = stream.next().await {
            let token = token_result?;

            // Skip prompt tokens
            if token.token_type == TokenType::Prompt {
                continue;
            }


            response.push_str(&token.text);
        }

        Ok(response.trim().to_string())
    }

    /// Generate a streaming response using a channel-based approach.
    ///
    /// This spawns a task to produce tokens, avoiding lifetime issues.
    async fn generate_stream(
        &self,
        prompt: String,
        runtime_overrides: GenerationOverrides,
    ) -> ChatResult<std::pin::Pin<Box<dyn futures_util::Stream<Item=ChatResult<String>> + Send>>>
    {
        use tokio::sync::mpsc;

        let config = resolve_generation_config(
            self.generation_config.inner.clone(),
            &self.user_overrides,
            &runtime_overrides,
        );

        // Clone the Arc to move into the spawned task
        let generator = self.generator.clone();
        let config = config.into_inner();

        // Create a channel to send tokens
        let (tx, rx) = mpsc::channel::<ChatResult<String>>(32);

        // Spawn a task to process the stream
        tokio::spawn(async move {

            // Get the stream
            let stream = match generator.generate_stream(&prompt, &config, None).await {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(Err(ChatError::GenerationFailed(e))).await;
                    return;
                }
            };

            futures_util::pin_mut!(stream);

            while let Some(result) = stream.next().await {
                let msg = match result {
                    Ok(token) => {
                        // Skip prompt tokens
                        if token.token_type == TokenType::Prompt {
                            continue;
                        }

                        Ok(token.text)
                    }
                    Err(e) => Err(ChatError::GenerationFailed(e)),
                };

                if tx.send(msg).await.is_err() {
                    // Receiver dropped, stop generating
                    break;
                }
            }
        });

        // Convert receiver to stream
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
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

    /// Get the current chat mode.
    pub fn mode(&self) -> ChatMode {
        self.mode
    }

    /// Get the device the model is running on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get the context window size.
    pub fn context_size(&self) -> usize {
        self.generator.model.context_size()
    }

    /// Get the system prompt.
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }
}
