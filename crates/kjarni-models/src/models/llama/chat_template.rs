//! Llama 3 Instruct chat template
//!
//! Formats conversations according to Llama 3's expected format.

use kjarni_transformers::{ChatTemplate, Conversation, Role, Message};

/// Chat template for Llama 3 Instruct models
///
/// Format:
/// ```text
/// <|begin_of_text|><|start_header_id|>system<|end_header_id|>
///
/// {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
///
/// {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
///
/// {assistant_message}<|eot_id|>
/// ```
#[derive(Clone, Debug, Default)]
pub struct Llama3ChatTemplate {
    /// Whether to add <|begin_of_text|> at the start
    pub add_bos: bool,
    /// Whether to add the assistant header at the end (for generation)
    pub add_generation_prompt: bool,
}

impl Llama3ChatTemplate {
    pub fn new() -> Self {
        Self {
            add_bos: true,
            add_generation_prompt: true,
        }
    }

    /// Create template for generation (adds assistant header at end)
    pub fn for_generation() -> Self {
        Self {
            add_bos: true,
            add_generation_prompt: true,
        }
    }

    /// Create template for training/fine-tuning (no trailing header)
    pub fn for_training() -> Self {
        Self {
            add_bos: true,
            add_generation_prompt: false,
        }
    }

    // Special tokens
    const BEGIN_OF_TEXT: &'static str = "<|begin_of_text|>";
    const END_OF_TEXT: &'static str = "<|end_of_text|>";
    const START_HEADER: &'static str = "<|start_header_id|>";
    const END_HEADER: &'static str = "<|end_header_id|>";
    const EOT: &'static str = "<|eot_id|>";

    fn format_message(&self, role: &Role, content: &str) -> String {
        let role_str = match role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        };

        format!(
            "{}{}{}\n\n{}{}",
            Self::START_HEADER,
            role_str,
            Self::END_HEADER,
            content,
            Self::EOT
        )
    }
}

impl ChatTemplate for Llama3ChatTemplate {
    fn apply(&self, conversation: &Conversation) -> String {
        let mut prompt = String::new();

        // Add BOS token
        if self.add_bos {
            prompt.push_str(Self::BEGIN_OF_TEXT);
        }

        // Format each message
        for message in conversation.messages() {
            prompt.push_str(&self.format_message(&message.role, &message.content));
        }

        // Add assistant header for generation
        if self.add_generation_prompt {
            prompt.push_str(Self::START_HEADER);
            prompt.push_str("assistant");
            prompt.push_str(Self::END_HEADER);
            prompt.push_str("\n\n");
        }

        prompt
    }

    fn stop_sequences(&self) -> Vec<String> {
        vec![
            Self::EOT.to_string(),
            Self::END_OF_TEXT.to_string(),
        ]
    }

    fn default_system_prompt(&self) -> Option<&str> {
        Some("You are a helpful, harmless, and honest assistant.")
    }

    fn validate(&self, conversation: &Conversation) -> Result<(), String> {
        let messages = conversation.messages();
        
        if messages.is_empty() {
            return Err("Conversation is empty".to_string());
        }

        // Check that messages alternate properly (after optional system)
        let mut expect_user = true;
        let start_idx = if messages.first().map(|m| m.role == Role::System).unwrap_or(false) {
            1
        } else {
            0
        };

        for (i, msg) in messages.iter().enumerate().skip(start_idx) {
            match (&msg.role, expect_user) {
                (Role::User, true) => expect_user = false,
                (Role::Assistant, false) => expect_user = true,
                (Role::System, _) if i > 0 => {
                    return Err("System message can only appear at the start".to_string());
                }
                (role, _) => {
                    let expected = if expect_user { "user" } else { "assistant" };
                    return Err(format!(
                        "Expected {} message at position {}, got {:?}",
                        expected, i, role
                    ));
                }
            }
        }

        Ok(())
    }
}

/// Chat template for Llama 2 Instruct models (legacy format)
///
/// Format:
/// ```text
/// [INST] <<SYS>>
/// {system_message}
/// <</SYS>>
///
/// {user_message} [/INST] {assistant_message} </s><s>[INST] {user_message} [/INST]
/// ```
#[derive(Clone, Debug, Default)]
pub struct Llama2ChatTemplate {
    pub add_bos: bool,
    pub add_generation_prompt: bool,
}

impl Llama2ChatTemplate {
    pub fn new() -> Self {
        Self {
            add_bos: true,
            add_generation_prompt: true,
        }
    }

    const BOS: &'static str = "<s>";
    const EOS: &'static str = "</s>";
    const INST_START: &'static str = "[INST]";
    const INST_END: &'static str = "[/INST]";
    const SYS_START: &'static str = "<<SYS>>\n";
    const SYS_END: &'static str = "\n<</SYS>>\n\n";
}

impl ChatTemplate for Llama2ChatTemplate {
    fn apply(&self, conversation: &Conversation) -> String {
        let mut prompt = String::new();
        let messages = conversation.messages();

        if messages.is_empty() {
            return prompt;
        }

        let mut msg_iter = messages.iter().peekable();

        // Handle system message
        let system_content = if let Some(first) = msg_iter.peek() {
            if first.role == Role::System {
                let sys = msg_iter.next().unwrap();
                Some(sys.content.as_str())
            } else {
                None
            }
        } else {
            None
        };

        let mut is_first_user = true;

        while let Some(msg) = msg_iter.next() {
            match msg.role {
                Role::User => {
                    if self.add_bos || !is_first_user {
                        prompt.push_str(Self::BOS);
                    }
                    prompt.push_str(Self::INST_START);
                    prompt.push(' ');

                    // Include system in first user message
                    if is_first_user {
                        if let Some(sys) = system_content {
                            prompt.push_str(Self::SYS_START);
                            prompt.push_str(sys);
                            prompt.push_str(Self::SYS_END);
                        }
                        is_first_user = false;
                    }

                    prompt.push_str(&msg.content);
                    prompt.push(' ');
                    prompt.push_str(Self::INST_END);
                }
                Role::Assistant => {
                    prompt.push(' ');
                    prompt.push_str(&msg.content);
                    prompt.push(' ');
                    prompt.push_str(Self::EOS);
                }
                Role::System => {
                    // System messages after first are ignored in Llama 2
                }
            }
        }

        prompt
    }

    fn stop_sequences(&self) -> Vec<String> {
        vec![Self::EOS.to_string()]
    }

    fn default_system_prompt(&self) -> Option<&str> {
        Some("You are a helpful, respectful and honest assistant.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama3_simple_conversation() {
        let template = Llama3ChatTemplate::for_generation();
        let mut conv = Conversation::new();
        conv.push_user("Hello!");

        let prompt = template.apply(&conv);
        
        assert!(prompt.contains("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.contains("<|eot_id|>"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_llama3_with_system() {
        let template = Llama3ChatTemplate::for_generation();
        let mut conv = Conversation::with_system("You are a pirate.");
        conv.push_user("Hello!");

        let prompt = template.apply(&conv);

        assert!(prompt.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(prompt.contains("You are a pirate."));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>"));
    }

    #[test]
    fn test_llama3_multi_turn() {
        let template = Llama3ChatTemplate::for_generation();
        let mut conv = Conversation::new();
        conv.push_user("What is 2+2?");
        conv.push_assistant("2+2 equals 4.");
        conv.push_user("And 3+3?");

        let prompt = template.apply(&conv);

        // Should have two user sections and one assistant section
        assert_eq!(prompt.matches("<|start_header_id|>user<|end_header_id|>").count(), 2);
        assert_eq!(prompt.matches("<|start_header_id|>assistant<|end_header_id|>").count(), 2); // 1 response + 1 generation prompt
    }

    #[test]
    fn test_llama3_validation() {
        let template = Llama3ChatTemplate::new();
        
        // Valid conversation
        let mut conv = Conversation::new();
        conv.push_user("Hi");
        assert!(template.validate(&conv).is_ok());

        // Invalid: assistant first
        let mut conv = Conversation::new();
        conv.push_assistant("Hi");
        assert!(template.validate(&conv).is_err());

        // Invalid: system not first
        let mut conv = Conversation::new();
        conv.push_user("Hi");
        conv.push(Message::system("System"));
        assert!(template.validate(&conv).is_err());
    }
}