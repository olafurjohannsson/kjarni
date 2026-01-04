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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_display() {
        assert_eq!(Role::System.to_string(), "system");
        assert_eq!(Role::User.to_string(), "user");
        assert_eq!(Role::Assistant.to_string(), "assistant");
    }

    #[test]
    fn message_constructors() {
        let m1 = Message::system("sys");
        assert_eq!(m1.role, Role::System);
        assert_eq!(m1.content, "sys");

        let m2 = Message::user("hi");
        assert_eq!(m2.role, Role::User);
        assert_eq!(m2.content, "hi");

        let m3 = Message::assistant("ok");
        assert_eq!(m3.role, Role::Assistant);
        assert_eq!(m3.content, "ok");
    }

    #[test]
    fn conversation_basic_operations() {
        let mut convo = Conversation::new();
        assert!(convo.is_empty());
        assert_eq!(convo.len(), 0);
        assert!(convo.system_prompt().is_none());

        convo.push_user("hello");
        assert_eq!(convo.len(), 1);
        assert_eq!(convo.turn_count(), 1);
        assert_eq!(convo.last().unwrap().role, Role::User);

        convo.push_assistant("hi there");
        assert_eq!(convo.len(), 2);
        assert_eq!(convo.turn_count(), 2);
        assert_eq!(convo.last().unwrap().role, Role::Assistant);

        // push generic message
        convo.push(Message::system("system message"));
        assert_eq!(convo.len(), 3);
        assert_eq!(convo.last().unwrap().role, Role::System);
    }

    #[test]
    fn conversation_with_system_prompt() {
        let convo = Conversation::with_system("init prompt");
        assert_eq!(convo.len(), 1);
        assert_eq!(convo.system_prompt(), Some("init prompt"));
    }

    #[test]
    fn conversation_clear_keep_system() {
        let mut convo = Conversation::with_system("init");
        convo.push_user("hello");
        convo.push_assistant("hi");

        convo.clear(true);
        assert_eq!(convo.len(), 1);
        assert_eq!(convo.system_prompt(), Some("init"));

        convo.clear(false);
        assert!(convo.is_empty());
    }

    #[test]
    fn conversation_turn_count() {
        let mut convo = Conversation::new();
        convo.push(Message::system("sys"));
        convo.push_user("u1");
        convo.push_assistant("a1");
        convo.push_user("u2");

        assert_eq!(convo.turn_count(), 3); // excludes system
    }

    #[test]
    fn raw_template_apply() {
        let mut convo = Conversation::new();
        convo.push(Message::system("sys"));
        convo.push_user("u1");
        convo.push_assistant("a1");

        let tmpl = RawTemplate::default();
        let output = tmpl.apply(&convo);
        let expected = "sys\n\nu1\n\na1";
        assert_eq!(output, expected);
    }

    #[test]
    fn conversation_last_message() {
        let mut convo = Conversation::new();
        assert!(convo.last().is_none());

        convo.push_user("hello");
        assert_eq!(convo.last().unwrap().content, "hello");
    }
}
