//! Chat conversation types and templates
//!
//! Provides the core abstractions for multi-turn conversations with LLMs.

use std::fmt;

/// A role in a conversation
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    /// System instructions (sets behavior)
    System,
    /// User messages (human input)
    User,
    /// Assistant responses (model output)
    Assistant,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
        }
    }
}

/// A single message in a conversation
#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into() }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }
}

/// A conversation history
#[derive(Clone, Debug, Default)]
pub struct Conversation {
    messages: Vec<Message>,
}

impl Conversation {
    /// Create an empty conversation
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a conversation with a system prompt
    pub fn with_system(system_prompt: impl Into<String>) -> Self {
        Self {
            messages: vec![Message::system(system_prompt)],
        }
    }

    /// Add a user message
    pub fn push_user(&mut self, content: impl Into<String>) {
        self.messages.push(Message::user(content));
    }

    /// Add an assistant message
    pub fn push_assistant(&mut self, content: impl Into<String>) {
        self.messages.push(Message::assistant(content));
    }

    /// Add any message
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Get all messages
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Get the last message
    pub fn last(&self) -> Option<&Message> {
        self.messages.last()
    }

    /// Get the system prompt if set
    pub fn system_prompt(&self) -> Option<&str> {
        self.messages.first().and_then(|m| {
            if m.role == Role::System {
                Some(m.content.as_str())
            } else {
                None
            }
        })
    }

    /// Clear all messages (optionally keep system prompt)
    pub fn clear(&mut self, keep_system: bool) {
        if keep_system {
            if let Some(system) = self.system_prompt().map(|s| s.to_string()) {
                self.messages.clear();
                self.messages.push(Message::system(system));
            } else {
                self.messages.clear();
            }
        } else {
            self.messages.clear();
        }
    }

    /// Count messages (excluding system)
    pub fn turn_count(&self) -> usize {
        self.messages.iter().filter(|m| m.role != Role::System).count()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Get message count
    pub fn len(&self) -> usize {
        self.messages.len()
    }
}

/// Trait for formatting conversations into model-specific prompts
pub trait ChatTemplate: Send + Sync {
    /// Format a conversation into a prompt string for the model
    fn apply(&self, conversation: &Conversation) -> String;

    /// Get stop tokens/sequences that indicate end of assistant response
    fn stop_sequences(&self) -> Vec<String> {
        vec![]
    }

    /// Get the model's default system prompt (if any)
    fn default_system_prompt(&self) -> Option<&str> {
        None
    }

    /// Validate if a conversation is properly formatted for this template
    fn validate(&self, conversation: &Conversation) -> Result<(), String> {
        // Default: no validation
        Ok(())
    }
}

/// A no-op template that just concatenates messages (for base models)
#[derive(Clone, Debug, Default)]
pub struct RawTemplate;

impl ChatTemplate for RawTemplate {
    fn apply(&self, conversation: &Conversation) -> String {
        conversation
            .messages()
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}