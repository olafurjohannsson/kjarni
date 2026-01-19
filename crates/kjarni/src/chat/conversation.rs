// =============================================================================
// kjarni/src/chat/conversation.rs
// =============================================================================

//! Stateful conversation management.

use super::model::Chat;
use super::types::{ChatResult, History, Role};
use crate::generation::GenerationOverrides;

/// A stateful conversation that maintains history automatically.
///
/// # Example
///
/// ```ignore
/// let chat = Chat::new("llama3.2-1b-instruct").await?;
/// let mut convo = chat.conversation();
///
/// // Blocking send (adds to history automatically)
/// let response = convo.send("What is Rust?").await?;
///
/// // Streaming (must manually add response to history)
/// convo.push_user("Tell me more.");
/// let mut stream = convo.stream_next().await?;
/// let mut response = String::new();
/// while let Some(token) = stream.next().await {
///     let text = token?;
///     print!("{}", text);
///     response.push_str(&text);
/// }
/// convo.push_assistant(&response);
/// ```
pub struct ChatConversation<'a> {
    chat: &'a Chat,
    history: History,
}

impl<'a> ChatConversation<'a> {
    /// Create a new conversation.
    pub(crate) fn new(chat: &'a Chat) -> Self {
        let history = if let Some(system) = chat.system_prompt() {
            History::with_system(system)
        } else {
            History::new()
        };

        Self { chat, history }
    }

    /// Create a conversation with a custom system prompt.
    pub(crate) fn with_system(chat: &'a Chat, system: String) -> Self {
        Self {
            chat,
            history: History::with_system(system),
        }
    }

    // =========================================================================
    // Blocking Send (Auto-manages history)
    // =========================================================================

    /// Send a message and get a response.
    ///
    /// Both the user message and assistant response are added to history.
    pub async fn send(&mut self, message: &str) -> ChatResult<String> {
        // Add user message to history
        self.history.push_user(message);

        // Convert full history to conversation and generate
        let conversation = self.chat.history_to_conversation(&self.history);
        let prompt = self.chat.format_prompt(&conversation);

        // Debug: uncomment to see what's being sent
        // eprintln!("DEBUG prompt:\n{}", prompt);

        let response = self
            .chat
            .generate(&prompt, &GenerationOverrides::default())
            .await?;

        // Add assistant response
        self.history.push_assistant(&response);

        Ok(response)
    }

    /// Send with custom generation overrides.
    pub async fn send_with_config(
        &mut self,
        message: &str,
        overrides: &GenerationOverrides,
    ) -> ChatResult<String> {
        self.history.push_user(message);

        let conversation = self.chat.history_to_conversation(&self.history);
        let prompt = self.chat.format_prompt(&conversation);
        let response = self.chat.generate(&prompt, overrides).await?;

        self.history.push_assistant(&response);

        Ok(response)
    }

    // =========================================================================
    // Streaming (Manual history management)
    // =========================================================================

    /// Add a user message to history (for streaming workflow).
    pub fn push_user(&mut self, message: &str) {
        self.history.push_user(message);
    }

    /// Add an assistant response to history (for streaming workflow).
    pub fn push_assistant(&mut self, response: &str) {
        self.history.push_assistant(response);
    }

    /// Stream the next response based on current history.
    ///
    /// **Important:** You must call `push_user()` before this, and
    /// `push_assistant()` after collecting the full response.
    ///
    /// # Example
    ///
    /// ```ignore
    /// convo.push_user("Hello!");
    /// let mut stream = convo.stream_next().await?;
    /// let mut response = String::new();
    /// while let Some(token) = stream.next().await {
    ///     response.push_str(&token?);
    /// }
    /// convo.push_assistant(&response);
    /// ```
    pub async fn stream_next(
        &self,
    ) -> ChatResult<std::pin::Pin<Box<dyn futures_util::Stream<Item = ChatResult<String>> + Send>>>
    {
        // Format prompt from current history
        let conversation = self.chat.history_to_conversation(&self.history);
        let prompt = self.chat.format_prompt(&conversation);

        // Get stream from chat (internal method)
        self.chat
            .generate_stream(prompt, GenerationOverrides::default())
            .await
    }

    /// Convenience: Push user message and stream response.
    ///
    /// Returns the stream. You must still call `push_assistant()` after.
    pub async fn stream(
        &mut self,
        message: &str,
    ) -> ChatResult<std::pin::Pin<Box<dyn futures_util::Stream<Item = ChatResult<String>> + Send>>>
    {
        self.push_user(message);
        self.stream_next().await
    }

    // =========================================================================
    // History Management
    // =========================================================================

    /// Get the conversation history.
    pub fn history(&self) -> &History {
        &self.history
    }

    /// Get mutable access to history.
    pub fn history_mut(&mut self) -> &mut History {
        &mut self.history
    }

    /// Clear the conversation history.
    ///
    /// If `keep_system` is true, the system prompt is preserved.
    pub fn clear(&mut self, keep_system: bool) {
        self.history.clear(keep_system);
    }

    /// Get the number of messages in history.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Check if history is empty.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
}
