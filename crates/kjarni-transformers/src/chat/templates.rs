//! Chat conversation types and templates

use std::fmt;

use super::tools::ToolSpec;

/// A role in a conversation
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    /// System instructions (sets behavior)
    System,
    /// User messages (human input)
    User,
    /// Assistant responses (model output)
    Assistant,
    /// The result of a tool call, fed back to the model
    Tool,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
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
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }

    /// A tool's result. An assistant message carries its calls in its content,
    /// as the model wrote them.
    pub fn tool(content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
        }
    }
}

/// A conversation history
#[derive(Clone, Debug, Default)]
pub struct Conversation {
    messages: Vec<Message>,
    tools: Vec<ToolSpec>,
    thinking: Option<bool>,
    preserve_thinking: bool,
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
            ..Self::default()
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

    /// Offer `tools` to the model. Only a template whose
    /// [`supports_tools`](ChatTemplate::supports_tools) is true renders them.
    pub fn set_tools(&mut self, tools: Vec<ToolSpec>) {
        self.tools = tools;
    }

    /// The tools offered to the model.
    pub fn tools(&self) -> &[ToolSpec] {
        &self.tools
    }

    /// Whether the model should reason before it answers, for templates of
    /// models that can do either; `None` leaves it to the template. Others
    /// ignore it.
    pub fn set_thinking(&mut self, thinking: Option<bool>) {
        self.thinking = thinking;
    }

    pub fn thinking(&self) -> Option<bool> {
        self.thinking
    }

    /// Whether assistant turns before the last user message keep their think
    /// blocks, for templates that otherwise drop them. Keeping them makes
    /// each prompt start with the previous one, so a prefix cache can reuse
    /// all of it; the rendering then differs from the model's own template
    /// in those earlier turns. Others ignore it.
    pub fn set_preserve_thinking(&mut self, preserve: bool) {
        self.preserve_thinking = preserve;
    }

    pub fn preserve_thinking(&self) -> bool {
        self.preserve_thinking
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
        self.messages
            .iter()
            .filter(|m| m.role != Role::System)
            .count()
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

    /// [`apply`](Self::apply), or why the conversation cannot be formatted:
    /// a model's own Jinja template can refuse one, such as one whose roles do
    /// not alternate.
    fn try_apply(&self, conversation: &Conversation) -> Result<String, String> {
        Ok(self.apply(conversation))
    }

    /// Whether the template renders [`Conversation::tools`] and tool messages
    /// in the format the model was tuned on.
    fn supports_tools(&self) -> bool {
        false
    }

    /// `call` written as the model writes calls, for an assistant message in
    /// the history. Qwen 3.5's XML format unless the template says otherwise.
    fn format_tool_call(&self, call: &super::tools::ToolCall) -> String {
        call.to_qwen_xml()
    }

    /// Get stop tokens/sequences that indicate end of assistant response
    fn stop_sequences(&self) -> Vec<String> {
        vec![]
    }

    /// Whether the prompt is complete as rendered, special tokens included,
    /// so that nothing is added when it is tokenized. True of a model's own
    /// Jinja template, which transformers tokenizes with
    /// `add_special_tokens=False`: a template that writes no BOS gets none.
    fn renders_special_tokens(&self) -> bool {
        false
    }

    /// Get the model's default system prompt (if any)
    fn default_system_prompt(&self) -> Option<&str> {
        None
    }

    /// Text the generation prompt already opens the reply with, which the model therefore
    /// does not generate but which belongs to the reply.
    fn reply_prefix(&self) -> &str {
        ""
    }

    /// [`reply_prefix`](Self::reply_prefix) when `prompt` ends with it, as a
    /// prompt this template rendered does, and otherwise nothing.
    fn reply_prefix_for(&self, prompt: &str) -> &str {
        let prefix = self.reply_prefix();
        if prompt.ends_with(prefix) { prefix } else { "" }
    }

    /// Validate if a conversation is properly formatted for this template
    fn validate(&self, _conversation: &Conversation) -> Result<(), String> {
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

        let tmpl = RawTemplate;
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
