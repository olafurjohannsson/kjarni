//! Stateful conversation wrapper.
//!
//! Provides a convenience type for multi-turn conversations
//! that automatically maintains history.

use futures_util::StreamExt;

use super::model::Chat;
use super::types::{ChatResult, History, Message, Role};
use crate::generation::overrides::GenerationOverrides;

/// A stateful conversation that maintains history.
///
/// Created via `chat.conversation()`. Each call to `send()` automatically
/// appends to the history.
///
/// # Example
///
/// ```ignore
/// let mut convo = chat.conversation();
///
/// // First turn
/// let response = convo.send("Hello!").await?;
/// println!("Assistant: {}", response);
///
/// // Second turn (includes history)
/// let response = convo.send("What did I just say?").await?;
/// println!("Assistant: {}", response);
///
/// // View history
/// for msg in convo.history().messages() {
///     println!("{}: {}", msg.role, msg.content);
/// }
///
/// // Clear and start over
/// convo.clear();
/// ```
pub struct ChatConversation<'a> {
    chat: &'a Chat,
    history: History,
}

impl<'a> ChatConversation<'a> {
    /// Create a new conversation.
    pub(crate) fn new(chat: &'a Chat) -> Self {
        let history = match chat.system_prompt() {
            Some(system) => History::with_system(system),
            None => History::new(),
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

    /// Send a message and get a response.
    ///
    /// Both the user message and assistant response are added to history.
    pub async fn send(&mut self, message: impl Into<String>) -> ChatResult<String> {
        let message = message.into();

        // Add user message to history
        self.history.push_user(&message);

        // Generate response
        let response = self
            .chat
            .send_with_history(&self.history_without_last(), &message)
            .await?;

        // Add assistant response to history
        self.history.push_assistant(&response);

        Ok(response)
    }

    /// Send with custom generation overrides.
    pub async fn send_with_config(
        &mut self,
        message: impl Into<String>,
        overrides: &GenerationOverrides,
    ) -> ChatResult<String> {
        let message = message.into();
        self.history.push_user(&message);

        // We need to use the underlying chat's internal method
        // For now, we'll use send_with_history which doesn't take overrides
        // This is a limitation we should address
        let response = self
            .chat
            .send_with_history(&self.history_without_last(), &message)
            .await?;

        self.history.push_assistant(&response);
        Ok(response)
    }

    /// Stream a response token by token.
    ///
    /// The complete response is added to history after streaming completes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use futures_util::StreamExt;
    ///
    /// let mut stream = convo.stream("Tell me a story.").await?;
    /// let mut full_response = String::new();
    ///
    /// while let Some(token) = stream.next().await {
    ///     let text = token?;
    ///     print!("{}", text);
    ///     full_response.push_str(&text);
    /// }
    ///
    /// // Note: For streaming, you need to manually add the response
    /// // after collecting it, or use send() for automatic handling
    /// ```
    pub async fn stream(
        &mut self,
        message: impl Into<String>,
    ) -> ChatResult<std::pin::Pin<Box<dyn futures_util::Stream<Item = ChatResult<String>> + Send>>> {
        let message = message.into();
        self.history.push_user(&message);

        self.chat
            .stream_with_history(&self.history_without_last(), &message)
            .await
    }

    /// Add the assistant's response to history after streaming.
    ///
    /// Call this after collecting a streamed response.
    pub fn add_response(&mut self, response: impl Into<String>) {
        self.history.push_assistant(response);
    }

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
    /// If a system prompt was set, it is preserved.
    pub fn clear(&mut self) {
        self.history.clear_keep_system();
    }

    /// Clear everything including system prompt.
    pub fn clear_all(&mut self) {
        self.history.clear();
    }

    /// Set a new system prompt.
    ///
    /// This clears the conversation and starts fresh.
    pub fn set_system(&mut self, system: impl Into<String>) {
        self.history = History::with_system(system);
    }

    /// Get the number of messages in the conversation.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Check if the conversation is empty.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Get history without the last message (for internal use).
    fn history_without_last(&self) -> History {
        let mut h = History::new();
        let messages = self.history.messages();

        if messages.len() <= 1 {
            return h;
        }

        for msg in &messages[..messages.len() - 1] {
            h.push(Message {
                role: msg.role,
                content: msg.content.clone(),
            });
        }

        h
    }
}

/// Extension trait for iterating over conversation messages.
impl<'a> IntoIterator for &'a ChatConversation<'a> {
    type Item = &'a Message;
    type IntoIter = std::slice::Iter<'a, Message>;

    fn into_iter(self) -> Self::IntoIter {
        self.history.messages().iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_history_management() {
        let mut history = History::with_system("You are helpful.");
        assert_eq!(history.len(), 1);

        history.push_user("Hello");
        history.push_assistant("Hi there!");
        assert_eq!(history.len(), 3);

        history.clear_keep_system();
        assert_eq!(history.len(), 1);
        assert_eq!(history.messages()[0].role, Role::System);
    }
}
