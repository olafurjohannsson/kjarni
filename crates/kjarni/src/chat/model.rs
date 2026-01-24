// =============================================================================
// kjarni/src/chat/model.rs
// =============================================================================

//! Core Chat implementation using Generator.

use futures::{Stream, StreamExt};
use tokio::sync::mpsc;

use kjarni_transformers::{
    ChatTemplate, Conversation, models::ModelType, traits::Device,
};

use crate::generation::{GenerationOverrides, ResolvedGenerationConfig};
use crate::generator::{Generator, GeneratorBuilder};

use super::builder::ChatBuilder;
use super::conversation::ChatConversation;
use super::types::{ChatError, ChatMode, ChatResult, History, Role};
use super::validation::validate_for_chat;

/// High-level chat interface for conversational AI.
///
/// Chat wraps a `Generator` and adds chat-specific functionality:
/// - Chat template formatting
/// - System prompts
/// - Conversation history management
/// - Chat modes (default, creative, reasoning)
///
/// # Example
///
/// ```ignore
/// use kjarni::chat::Chat;
///
/// // Simple one-shot message
/// let chat = Chat::new("llama3.2-1b-instruct").await?;
/// let response = chat.send("What is Rust?").await?;
/// println!("{}", response);
///
/// // With system prompt and history
/// let chat = Chat::builder("llama3.2-1b-instruct")
///     .system("You are a helpful coding assistant.")
///     .mode(ChatMode::Reasoning)
///     .build()
///     .await?;
///
/// // Multi-turn conversation
/// let mut convo = chat.conversation();
/// let r1 = convo.send("What is a closure in Rust?").await?;
/// let r2 = convo.send("Can you give me an example?").await?;
/// ```
pub struct Chat {
    /// Inner generator for text generation.
    generator: Generator,

    /// Model type from registry.
    model_type: ModelType,

    /// Default system prompt.
    system_prompt: Option<String>,

    /// Resolved generation config.
    generation_config: ResolvedGenerationConfig,

    /// User-provided overrides.
    user_overrides: GenerationOverrides,

    /// Chat mode.
    mode: ChatMode,
}

impl Chat {
    // =========================================================================
    // Construction
    // =========================================================================

    /// Create a Chat with default settings.
    ///
    /// Uses CPU, downloads model if needed.
    pub async fn new(model: &str) -> ChatResult<Self> {
        ChatBuilder::new(model).build().await
    }

    /// Create a builder for custom configuration.
    pub fn builder(model: &str) -> ChatBuilder {
        ChatBuilder::new(model)
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
                eprintln!("⚠️  {}", warning);
            }
        }

        // Step 4: Build the inner generator
        let mut gen_builder = GeneratorBuilder::new(&builder.model)
            .device(builder.device)
            .download_policy(builder.download_policy);

        if builder.quiet {
            gen_builder = gen_builder.quiet();
        }

        if let Some(cache_dir) = builder.cache_dir {
            gen_builder = gen_builder.cache_dir(cache_dir);
        }

        if let Some(context) = builder.context {
            gen_builder = gen_builder.with_context(context);
        }

        if let Some(load_config) = builder.load_config {
            gen_builder = gen_builder.with_load_config(|_| {
                // Reconstruct from the stored config
                crate::common::LoadConfigBuilder::from_config(load_config.clone())
            });
        }

        // Apply mode-specific defaults if not overridden
        let mut mode_overrides = builder.generation_overrides.clone();
        if mode_overrides.temperature.is_none() {
            mode_overrides.temperature = Some(builder.mode.default_temperature());
        }
        if mode_overrides.max_new_tokens.is_none() {
            mode_overrides.max_new_tokens = Some(builder.mode.default_max_tokens());
        }

        gen_builder = gen_builder.generation_config(mode_overrides.clone());

        let generator = gen_builder.build().await.map_err(|e| match e {
            crate::generator::GeneratorError::UnknownModel(m) => ChatError::UnknownModel(m),
            crate::generator::GeneratorError::ModelNotDownloaded(m) => {
                ChatError::ModelNotDownloaded(m)
            }
            crate::generator::GeneratorError::DownloadFailed { model, source } => {
                ChatError::DownloadFailed { model, source }
            }
            crate::generator::GeneratorError::LoadFailed { model, source } => {
                ChatError::LoadFailed { model, source }
            }
            crate::generator::GeneratorError::GpuUnavailable => ChatError::GpuUnavailable,
            crate::generator::GeneratorError::GenerationFailed(e) => ChatError::GenerationFailed(e),
            crate::generator::GeneratorError::InvalidModel(m, r) => ChatError::InvalidModel(m, r),
            crate::generator::GeneratorError::InvalidConfig(s) => ChatError::InvalidConfig(s),
        })?;

        // Step 5: Verify chat template exists
        if generator.decoder().model.chat_template().is_none() {
            return Err(ChatError::NoChatTemplate(builder.model));
        }

        // Step 6: Get resolved generation config
        let generation_config = generator.generation_config().clone();

        Ok(Self {
            generator,
            model_type,
            system_prompt: builder.system_prompt,
            generation_config,
            user_overrides: builder.generation_overrides,
            mode: builder.mode,
        })
    }

    // =========================================================================
    // Chat Template Access
    // =========================================================================

    /// Get the chat template from the model.
    fn chat_template(&self) -> &dyn ChatTemplate {
        self.generator
            .decoder()
            .model
            .chat_template()
            .expect("Chat template verified during construction")
    }

    /// Format a conversation using the chat template.
    pub(crate) fn format_prompt(&self, conversation: &Conversation) -> String {
        self.chat_template().apply(conversation)
    }

    // =========================================================================
    // Conversation Building
    // =========================================================================

    /// Create a Conversation with the configured system prompt.
    fn create_conversation(&self) -> Conversation {
        match &self.system_prompt {
            Some(prompt) => Conversation::with_system(prompt),
            None => {
                // Try to get default from template
                if let Some(default) = self.chat_template().default_system_prompt() {
                    return Conversation::with_system(default);
                }
                Conversation::new()
            }
        }
    }

    /// Convert History to Conversation.
    pub(crate) fn history_to_conversation(&self, history: &History) -> Conversation {
        let mut conversation = Conversation::new();
        let mut has_system = false;

        for msg in history.messages() {
            match msg.role {
                Role::System => {
                    conversation = Conversation::with_system(&msg.content);
                    has_system = true;
                }
                Role::User => conversation.push_user(&msg.content),
                Role::Assistant => conversation.push_assistant(&msg.content),
            }
        }

        // If no system in history, use default
        if !has_system {
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

    // =========================================================================
    // Stateless Sending
    // =========================================================================

    /// Send a message and get a response.
    ///
    /// This is stateless - no conversation history is maintained.
    /// For multi-turn conversations, use `conversation()`.
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

        let prompt = self.format_prompt(&conversation);
        self.generate(&prompt, &GenerationOverrides::default())
            .await
    }

    /// Send with conversation history.
    ///
    /// The history is used for context but not modified.
    pub async fn send_with_history(&self, history: &History, message: &str) -> ChatResult<String> {
        let mut conversation = self.history_to_conversation(history);
        conversation.push_user(message);

        let prompt = self.format_prompt(&conversation);
        self.generate(&prompt, &GenerationOverrides::default())
            .await
    }

    /// Send with custom generation overrides.
    pub async fn send_with_config(
        &self,
        message: &str,
        overrides: &GenerationOverrides,
    ) -> ChatResult<String> {
        let mut conversation = self.create_conversation();
        conversation.push_user(message);

        let prompt = self.format_prompt(&conversation);
        self.generate(&prompt, overrides).await
    }

    // =========================================================================
    // Streaming
    // =========================================================================

    /// Stream a response token by token.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use futures::StreamExt;
    ///
    /// let mut stream = chat.stream("Tell me a story.").await?;
    /// while let Some(token) = stream.next().await {
    ///     print!("{}", token?);
    /// }
    /// ```
    pub async fn stream(
        &self,
        message: &str,
    ) -> ChatResult<std::pin::Pin<Box<dyn Stream<Item = ChatResult<String>> + Send>>>
    {
        let mut conversation = self.create_conversation();
        conversation.push_user(message);

        let prompt = self.format_prompt(&conversation);
        self.generate_stream(prompt, GenerationOverrides::default())
            .await
    }

    /// Stream with conversation history.
    pub async fn stream_with_history(
        &self,
        history: &History,
        message: &str,
    ) -> ChatResult<std::pin::Pin<Box<dyn Stream<Item = ChatResult<String>> + Send>>>
    {
        let mut conversation = self.history_to_conversation(history);
        conversation.push_user(message);

        let prompt = self.format_prompt(&conversation);
        self.generate_stream(prompt, GenerationOverrides::default())
            .await
    }

    /// Stream with custom overrides.
    pub async fn stream_with_config(
        &self,
        message: &str,
        overrides: GenerationOverrides,
    ) -> ChatResult<std::pin::Pin<Box<dyn Stream<Item = ChatResult<String>> + Send>>>
    {
        let mut conversation = self.create_conversation();
        conversation.push_user(message);

        let prompt = self.format_prompt(&conversation);
        self.generate_stream(prompt, overrides).await
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
    // Internal Generation
    // =========================================================================

    /// Generate using the inner generator.
    pub(crate) async fn generate(
        &self,
        prompt: &str,
        runtime_overrides: &GenerationOverrides,
    ) -> ChatResult<String> {
        let response = self
            .generator
            .generate_with_config(prompt, runtime_overrides)
            .await
            .map_err(|e| ChatError::GenerationFailed(e.into()))?;

        let mut cleaned = response.trim().to_string();

        // Strip stop sequences defined by the chat template
        for stop_seq in self.chat_template().stop_sequences() {
            if let Some(stripped) = cleaned.strip_suffix(&stop_seq) {
                cleaned = stripped.trim().to_string();
            }
        }

        Ok(cleaned)
    }

    /// Generate streaming using channel approach.
    pub(crate) async fn generate_stream(
        &self,
        prompt: String,
        runtime_overrides: GenerationOverrides,
    ) -> ChatResult<std::pin::Pin<Box<dyn Stream<Item = ChatResult<String>> + Send>>>
    {
        let inner_stream = self
            .generator
            .stream_with_config(&prompt, runtime_overrides)
            .await
            .map_err(|e| ChatError::GenerationFailed(e.into()))?;

        // Map GeneratorResult<GeneratedToken> to ChatResult<String>
        let mapped = inner_stream.map(|result| {
            result
                .map(|token| token.text)
                .map_err(|e| ChatError::GenerationFailed(e.into()))
        });

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

    /// Get the current chat mode.
    pub fn mode(&self) -> ChatMode {
        self.mode
    }

    /// Get the device the model is running on.
    pub fn device(&self) -> Device {
        self.generator.device()
    }

    /// Get the context window size.
    pub fn context_size(&self) -> usize {
        self.generator.context_size()
    }

    /// Get the system prompt.
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    /// Get a reference to the inner generator.
    ///
    /// For advanced use cases that need direct access.
    pub fn generator(&self) -> &Generator {
        &self.generator
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Send a single message with default settings.
///
/// # Example
///
/// ```ignore
/// let response = kjarni::chat::send("llama3.2-1b-instruct", "Hello!").await?;
/// ```
pub async fn send(model: &str, message: &str) -> ChatResult<String> {
    Chat::new(model).await?.send(message).await
}
