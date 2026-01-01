use crate::{ChatTemplate, Conversation, Role, Message};

/// ChatML Template (Used by Qwen 2/2.5, Yi, etc.)
///
/// Format:
/// <|im_start|>system
/// {content}<|im_end|>
/// <|im_start|>user
/// {content}<|im_end|>
/// <|im_start|>assistant
/// {content}<|im_end|>
#[derive(Clone, Debug, Default)]
pub struct ChatMLTemplate {
    pub add_generation_prompt: bool,
}

impl ChatMLTemplate {
    pub fn new() -> Self {
        Self { add_generation_prompt: true }
    }
}

impl ChatTemplate for ChatMLTemplate {
    fn apply(&self, conversation: &Conversation) -> String {
        let mut prompt = String::new();

        for message in conversation.messages() {
            let role = match message.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            
            prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, message.content));
        }

        if self.add_generation_prompt {
            prompt.push_str("<|im_start|>assistant\n");
        }

        prompt
    }

    fn stop_sequences(&self) -> Vec<String> {
        vec![
            "<|im_end|>".to_string(),
            "<|endoftext|>".to_string(),
        ]
    }

    fn default_system_prompt(&self) -> Option<&str> {
        Some("You are a helpful assistant.")
    }
}

