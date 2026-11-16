//! Core Chat implementation using Generator.

use futures::{Stream, StreamExt};

use kjarni_transformers::{ChatTemplate, Conversation, models::ModelType, traits::Device};

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

    /// The registry model, or `None` for one loaded from a folder.
    model_type: Option<ModelType>,

    /// The registry name, or the folder's.
    model_name: String,

    /// Default system prompt.
    system_prompt: Option<String>,

    #[expect(
        dead_code,
        reason = "not referenced yet; kept until the path that needs it lands"
    )]
    /// Resolved generation config.
    generation_config: ResolvedGenerationConfig,

    #[expect(
        dead_code,
        reason = "not referenced yet; kept until the path that needs it lands"
    )]
    /// User-provided overrides.
    user_overrides: GenerationOverrides,

    /// Chat mode.
    mode: ChatMode,

    /// Whether replies reason first; `None` leaves it to the template.
    thinking: Option<bool>,
}

impl Chat {
    /// Create a Chat with default settings.
    pub async fn new(model: &str) -> ChatResult<Self> {
        ChatBuilder::new(model).build().await
    }

    /// Create a builder for custom configuration.
    pub fn builder(model: &str) -> ChatBuilder {
        ChatBuilder::new(model)
    }

    /// Internal: construct from builder.
    pub async fn from_builder(builder: ChatBuilder) -> ChatResult<Self> {
        // A folder is checked by loading it: whether it can chat is whether
        // it brings a chat template.
        let model_type = match &builder.model_path {
            Some(_) => None,
            None => {
                let model_type = ModelType::from_cli_name(&builder.model)
                    .ok_or_else(|| ChatError::UnknownModel(builder.model.clone()))?;
                let validation = validate_for_chat(model_type)?;
                if !builder.quiet && !builder.allow_suboptimal {
                    for warning in &validation.warnings {
                        eprintln!("Warning: {}", warning);
                    }
                }
                Some(model_type)
            }
        };

        // Build the inner generator
        let mut gen_builder = GeneratorBuilder::new(&builder.model)
            .device(builder.device)
            .download_policy(builder.download_policy);
        if let Some(path) = &builder.model_path {
            gen_builder = gen_builder.model_path(path);
        }

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

        if let Some(ref draft) = builder.draft_model {
            gen_builder = gen_builder.draft(draft, builder.draft_tokens);
        }

        if let Some(tokens) = builder.prefix_cache_tokens {
            gen_builder = gen_builder.prefix_cache_tokens(tokens);
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

        let mut generator = generator;
        let model_name = generator.model_name().to_string();
        let renders_special = match generator.decoder().model.chat_template() {
            Some(t) => t.renders_special_tokens(),
            None => return Err(ChatError::NoChatTemplate(model_name)),
        };
        if renders_special {
            generator.set_add_bos_token(false);
        }

        // Get resolved generation config
        let generation_config = generator.generation_config().clone();

        Ok(Self {
            generator,
            model_type,
            model_name,
            system_prompt: builder.system_prompt,
            generation_config,
            user_overrides: builder.generation_overrides,
            mode: builder.mode,
            thinking: builder.thinking,
        })
    }

    /// Get the chat template from the model.
    fn chat_template(&self) -> &dyn ChatTemplate {
        self.generator
            .decoder()
            .model
            .chat_template()
            .expect("Chat template verified during construction")
    }

    /// Format a conversation using the chat template.
    pub fn format_prompt(&self, conversation: &Conversation) -> String {
        self.chat_template().apply(conversation)
    }

    /// [`format_prompt`](Self::format_prompt), or why the model's template
    /// refuses the conversation.
    /// The chat's own thinking setting applies unless the conversation has one.
    pub(crate) fn render_prompt(&self, conversation: &Conversation) -> ChatResult<String> {
        let own;
        let conversation = if conversation.thinking().is_none() && self.thinking.is_some() {
            let mut c = conversation.clone();
            c.set_thinking(self.thinking);
            own = c;
            &own
        } else {
            conversation
        };
        self.chat_template()
            .try_apply(conversation)
            .map_err(ChatError::InvalidConfig)
    }

    /// Whether replies reason before answering, as the builder set it.
    pub fn thinking(&self) -> Option<bool> {
        self.thinking
    }

    /// The prompt for `conversation` as the model's template renders it, tools
    /// and tool results included. Fails when the conversation offers tools and
    /// the template has no format for them.
    pub fn render_conversation(&self, conversation: &Conversation) -> ChatResult<String> {
        if !conversation.tools().is_empty() && !self.supports_tools() {
            return Err(ChatError::InvalidConfig(format!(
                "{} has no tool-calling format; use a model such as qwen3.5-4b",
                self.model_name
            )));
        }
        self.render_prompt(conversation)
    }

    /// `call` as this model writes calls, for putting an assistant message's
    /// earlier calls back into its history.
    pub fn format_tool_call(&self, call: &kjarni_transformers::ToolCall) -> String {
        self.chat_template().format_tool_call(call)
    }

    /// Whether the model's chat template has a format for tools and their
    /// results.
    pub fn supports_tools(&self) -> bool {
        self.chat_template().supports_tools()
    }

    /// The number of tokens `text` encodes to.
    pub fn count_tokens(&self, text: &str) -> ChatResult<usize> {
        self.generator
            .count_tokens(text)
            .map_err(|e| ChatError::GenerationFailed(e.into()))
    }

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

    /// The prompt for a whole conversation, as the model's chat template
    /// renders it, ready for [`generate`](Self::generate) or
    /// [`generate_stream`](Self::generate_stream).
    ///
    /// System messages are joined into one at the start, because templates take
    /// a single system prompt. Without any, the chat's own system prompt or the
    /// template's default is used, as [`send`](Self::send) does.
    pub fn render_messages(&self, messages: &[super::types::Message]) -> ChatResult<String> {
        self.render_messages_with_tools(messages, &[])
    }

    /// [`render_messages`](Self::render_messages) with `tools` offered to the
    /// model. Fails when there are tools and the template has no format for
    /// them.
    pub fn render_messages_with_tools(
        &self,
        messages: &[super::types::Message],
        tools: &[kjarni_transformers::ToolSpec],
    ) -> ChatResult<String> {
        self.render_messages_thinking(messages, tools, None)
    }

    /// [`render_messages_with_tools`](Self::render_messages_with_tools), with
    /// `thinking` deciding whether the reply reasons first; `None` keeps the
    /// chat's setting.
    pub fn render_messages_thinking(
        &self,
        messages: &[super::types::Message],
        tools: &[kjarni_transformers::ToolSpec],
        thinking: Option<bool>,
    ) -> ChatResult<String> {
        let system: Vec<&str> = messages
            .iter()
            .filter(|m| matches!(m.role, Role::System))
            .map(|m| m.content.as_str())
            .collect();
        let mut conversation = if system.is_empty() {
            self.create_conversation()
        } else {
            Conversation::with_system(system.join("\n\n"))
        };
        for message in messages {
            match message.role {
                Role::User => conversation.push_user(&message.content),
                Role::Assistant => conversation.push_assistant(&message.content),
                Role::Tool => {
                    conversation.push(kjarni_transformers::Message::tool(&message.content))
                }
                Role::System => {}
            }
        }
        conversation.set_tools(tools.to_vec());
        conversation.set_thinking(thinking);
        self.render_conversation(&conversation)
    }

    /// What [`generate_stream`](Self::generate_stream) sends before the first
    /// generated token for this prompt: the opening of the reply when the
    /// template writes it into the prompt, such as a reasoning model's
    /// `<think>`, and otherwise nothing.
    pub fn reply_prefix(&self, prompt: &str) -> &str {
        reply_prefix(self.chat_template(), prompt)
    }

    /// Convert History to Conversation.
    pub fn history_to_conversation(&self, history: &History) -> Conversation {
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
                Role::Tool => conversation.push(kjarni_transformers::Message::tool(&msg.content)),
            }
        }

        // If no system in history, use default
        if !has_system && let Some(system) = &self.system_prompt {
            let mut with_system = Conversation::with_system(system);
            for msg in conversation.messages() {
                match msg.role {
                    kjarni_transformers::Role::User => with_system.push_user(&msg.content),
                    kjarni_transformers::Role::Assistant => {
                        with_system.push_assistant(&msg.content)
                    }
                    kjarni_transformers::Role::Tool => with_system.push(msg.clone()),
                    _ => {}
                }
            }
            return with_system;
        }

        conversation
    }

    /// Send a message and get a response.
    pub async fn send(&self, message: &str) -> ChatResult<String> {
        let mut conversation = self.create_conversation();
        conversation.push_user(message);

        let prompt = self.render_prompt(&conversation)?;
        self.generate(&prompt, &GenerationOverrides::default())
            .await
    }

    /// Send with conversation history.
    ///
    /// The history is used for context but not modified.
    pub async fn send_with_history(&self, history: &History, message: &str) -> ChatResult<String> {
        let mut conversation = self.history_to_conversation(history);
        conversation.push_user(message);

        let prompt = self.render_prompt(&conversation)?;
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

        let prompt = self.render_prompt(&conversation)?;
        self.generate(&prompt, overrides).await
    }

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
    ) -> ChatResult<std::pin::Pin<Box<dyn Stream<Item = ChatResult<String>> + Send>>> {
        let mut conversation = self.create_conversation();
        conversation.push_user(message);

        let prompt = self.render_prompt(&conversation)?;
        self.generate_stream(prompt, GenerationOverrides::default())
            .await
    }

    /// Stream with conversation history.
    pub async fn stream_with_history(
        &self,
        history: &History,
        message: &str,
    ) -> ChatResult<std::pin::Pin<Box<dyn Stream<Item = ChatResult<String>> + Send>>> {
        let mut conversation = self.history_to_conversation(history);
        conversation.push_user(message);

        let prompt = self.render_prompt(&conversation)?;
        self.generate_stream(prompt, GenerationOverrides::default())
            .await
    }

    /// Stream with custom overrides.
    pub async fn stream_with_config(
        &self,
        message: &str,
        overrides: GenerationOverrides,
    ) -> ChatResult<std::pin::Pin<Box<dyn Stream<Item = ChatResult<String>> + Send>>> {
        let mut conversation = self.create_conversation();
        conversation.push_user(message);

        let prompt = self.render_prompt(&conversation)?;
        self.generate_stream(prompt, overrides).await
    }

    /// Create a stateful conversation.
    ///
    /// The conversation maintains history automatically.
    pub fn conversation(&self) -> ChatConversation<'_> {
        ChatConversation::new(self)
    }

    /// Create a stateful conversation with a custom system prompt.
    pub fn conversation_with_system(&self, system: impl Into<String>) -> ChatConversation<'_> {
        ChatConversation::with_system(self, system.into())
    }

    /// Generate using the inner generator.
    pub async fn generate(
        &self,
        prompt: &str,
        runtime_overrides: &GenerationOverrides,
    ) -> ChatResult<String> {
        let response = self
            .generator
            .generate_with_config(prompt, runtime_overrides)
            .await
            .map_err(|e| ChatError::GenerationFailed(e.into()))?;

        let prefix = reply_prefix(self.chat_template(), prompt).to_string();
        let mut cleaned = response.trim().to_string();

        // Strip stop sequences defined by the chat template
        for stop_seq in self.chat_template().stop_sequences() {
            if let Some(stripped) = cleaned.strip_suffix(&stop_seq) {
                cleaned = stripped.trim().to_string();
            }
        }

        Ok(prefix + &cleaned)
    }

    /// Generate streaming using channel approach.
    pub async fn generate_stream(
        &self,
        prompt: String,
        runtime_overrides: GenerationOverrides,
    ) -> ChatResult<std::pin::Pin<Box<dyn Stream<Item = ChatResult<String>> + Send>>> {
        let prefix = reply_prefix(self.chat_template(), &prompt).to_string();
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

        let opening = (!prefix.is_empty()).then_some(Ok(prefix));
        Ok(Box::pin(futures::stream::iter(opening).chain(mapped)))
    }

    /// The registry model, or `None` for one loaded from a folder.
    pub fn model_type(&self) -> Option<ModelType> {
        self.model_type
    }

    /// The model's registry name, or its folder's name.
    pub fn model_name(&self) -> &str {
        &self.model_name
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

/// What the reply to `prompt` starts with that the model will not generate. A prompt not
/// built by the template may not end with the prefix, and then has none.
pub(crate) fn reply_prefix<'a>(template: &'a dyn ChatTemplate, prompt: &str) -> &'a str {
    template.reply_prefix_for(prompt)
}

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
