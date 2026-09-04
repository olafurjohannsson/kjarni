//! Phi-3 chat template.
//!
//! Matches the `chat_template` in `microsoft/Phi-3.5-mini-instruct`'s
//! `tokenizer_config.json`: each turn is a role tag, a newline, the content and
//! `<|end|>`, and a bare `<|assistant|>` opens the model's turn. A system turn is
//! emitted only when it has content, which is what the upstream template does.
//!
//! Without this the Phi3 architecture fell through to no template at all, so an
//! instruct model was handed a raw prompt and continued it rather than answering.

use crate::{ChatTemplate, Conversation, Role};

#[derive(Debug, Clone)]
pub struct Phi3ChatTemplate {
    pub add_generation_prompt: bool,
}

impl Phi3ChatTemplate {
    pub fn new() -> Self {
        Self {
            add_generation_prompt: true,
        }
    }
}

impl Default for Phi3ChatTemplate {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatTemplate for Phi3ChatTemplate {
    fn apply(&self, conversation: &Conversation) -> String {
        let mut prompt = String::new();

        for message in conversation.messages() {
            // Upstream skips a system turn with empty content rather than
            // emitting an empty block.
            if message.role == Role::System && message.content.is_empty() {
                continue;
            }
            let role = match message.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            prompt.push_str(&format!("<|{}|>\n{}<|end|>\n", role, message.content));
        }

        if self.add_generation_prompt {
            prompt.push_str("<|assistant|>\n");
        }

        prompt
    }

    fn stop_sequences(&self) -> Vec<String> {
        vec!["<|end|>".to_string(), "<|endoftext|>".to_string()]
    }

    fn default_system_prompt(&self) -> Option<&str> {
        Some("You are a helpful assistant.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Conversation;

    #[test]
    fn phi3_generation_prompt_only() {
        let t = Phi3ChatTemplate::new();
        assert_eq!(t.apply(&Conversation::new()), "<|assistant|>\n");
    }

    #[test]
    fn phi3_matches_upstream_layout() {
        let t = Phi3ChatTemplate::new();
        let mut c = Conversation::with_system("Be terse.");
        c.push_user("Hi");
        assert_eq!(
            t.apply(&c),
            "<|system|>\nBe terse.<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>\n"
        );
    }

    #[test]
    fn phi3_skips_empty_system() {
        let t = Phi3ChatTemplate::new();
        let mut c = Conversation::with_system("");
        c.push_user("Hi");
        assert_eq!(t.apply(&c), "<|user|>\nHi<|end|>\n<|assistant|>\n");
    }
}
