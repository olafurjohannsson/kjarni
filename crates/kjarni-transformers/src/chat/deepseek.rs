//! DeepSeek-R1 chat template.
//!
//! Matches the `chat_template` in `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`'s
//! `tokenizer_config.json`, without its tool-call branches: the system prompt
//! straight after the BOS token, `<｜User｜>` before each user turn,
//! `<｜Assistant｜>` before each reply, and a model turn opened with `<think>`.
//! A reply already in the history keeps only what follows its `</think>`.
//!
//! The distilled R1 models are Llama underneath but were tuned on this format,
//! not Llama 3's, so the template is chosen by model rather than architecture.

use crate::{ChatTemplate, Conversation, Role};

#[derive(Debug, Clone)]
pub struct DeepSeekR1ChatTemplate {
    pub add_generation_prompt: bool,
}

impl DeepSeekR1ChatTemplate {
    const BOS: &'static str = "<｜begin▁of▁sentence｜>";
    const EOS: &'static str = "<｜end▁of▁sentence｜>";

    pub fn new() -> Self {
        Self {
            add_generation_prompt: true,
        }
    }
}

impl Default for DeepSeekR1ChatTemplate {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatTemplate for DeepSeekR1ChatTemplate {
    fn apply(&self, conversation: &Conversation) -> String {
        let messages = conversation.messages();
        let mut prompt = String::from(Self::BOS);
        // Upstream keeps the last system message and puts it first.
        if let Some(system) = messages.iter().rev().find(|m| m.role == Role::System) {
            prompt.push_str(&system.content);
        }
        for message in messages {
            match message.role {
                Role::System => {}
                Role::User => {
                    prompt.push_str("<｜User｜>");
                    prompt.push_str(&message.content);
                }
                Role::Assistant => {
                    let content = match message.content.rsplit_once("</think>") {
                        Some((_, answer)) => answer,
                        None => &message.content,
                    };
                    prompt.push_str("<｜Assistant｜>");
                    prompt.push_str(content);
                    prompt.push_str(Self::EOS);
                }
            }
        }
        if self.add_generation_prompt {
            prompt.push_str("<｜Assistant｜><think>\n");
        }
        prompt
    }

    fn stop_sequences(&self) -> Vec<String> {
        vec![Self::EOS.to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_upstream_template() {
        let mut conv = Conversation::with_system("Be brief.");
        conv.push_user("What is 2+2?");
        conv.push_assistant("<think>\nSimple sum.\n</think>\n\n4");
        conv.push_user("And 3+3?");
        assert_eq!(
            DeepSeekR1ChatTemplate::new().apply(&conv),
            "<｜begin▁of▁sentence｜>Be brief.<｜User｜>What is 2+2?<｜Assistant｜>\n\n4\
             <｜end▁of▁sentence｜><｜User｜>And 3+3?<｜Assistant｜><think>\n"
        );
    }

    #[test]
    fn no_system_prompt_is_invented() {
        let mut conv = Conversation::new();
        conv.push_user("Hi");
        let template = DeepSeekR1ChatTemplate::new();
        assert_eq!(template.default_system_prompt(), None);
        assert_eq!(
            template.apply(&conv),
            "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜><think>\n"
        );
    }
}
