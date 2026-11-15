use crate::{ChatTemplate, Conversation, Role};

/// Qwen 3.5's chat template in its default, non-thinking mode, for text.
///
/// ChatML, with three differences from Qwen 2.5's: no default system turn,
/// every message trimmed, and a generation prompt that closes an empty
/// `<think>` block so the model answers directly. Reasoning in an earlier
/// assistant turn is dropped from the history, as the model's own template
/// drops it; a turn after the last user message keeps it in a think block.
#[derive(Clone, Debug, Default)]
pub struct Qwen3_5ChatTemplate;

impl Qwen3_5ChatTemplate {
    pub fn new() -> Self {
        Self
    }
}

/// The reasoning and the answer of an assistant message that wrote its
/// reasoning inline, between `<think>` and `</think>`.
fn split_reasoning(content: &str) -> (&str, &str) {
    match content.split_once("</think>") {
        Some((before, after)) => {
            let reasoning = before.trim_end_matches('\n');
            let reasoning = reasoning
                .rsplit_once("<think>")
                .map_or(reasoning, |(_, r)| r)
                .trim_start_matches('\n');
            // The answer is what follows the last `</think>`.
            let answer = content.rsplit_once("</think>").map_or(after, |(_, a)| a);
            (reasoning, answer.trim_start_matches('\n'))
        }
        None => ("", content),
    }
}

impl ChatTemplate for Qwen3_5ChatTemplate {
    fn apply(&self, conversation: &Conversation) -> String {
        let messages = conversation.messages();
        let last_user = messages.iter().rposition(|m| m.role == Role::User);
        let mut prompt = String::new();
        for (i, message) in messages.iter().enumerate() {
            let content = message.content.trim();
            match message.role {
                Role::System => {
                    prompt.push_str(&format!("<|im_start|>system\n{content}<|im_end|>\n"));
                }
                Role::User => {
                    prompt.push_str(&format!("<|im_start|>user\n{content}<|im_end|>\n"));
                }
                Role::Assistant => {
                    let (reasoning, answer) = split_reasoning(content);
                    if last_user.is_some_and(|u| i > u) {
                        prompt.push_str(&format!(
                            "<|im_start|>assistant\n<think>\n{}\n</think>\n\n{answer}<|im_end|>\n",
                            reasoning.trim()
                        ));
                    } else {
                        prompt.push_str(&format!("<|im_start|>assistant\n{answer}<|im_end|>\n"));
                    }
                }
            }
        }
        prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
        prompt
    }

    fn stop_sequences(&self) -> Vec<String> {
        vec!["<|im_end|>".to_string(), "<|endoftext|>".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Each expected prompt is what transformers 5.16.1 renders with
    // Qwen/Qwen3.5-0.8B's chat_template.jinja and add_generation_prompt=True.

    #[test]
    fn a_single_user_turn() {
        let mut c = Conversation::new();
        c.push_user("Hello!");
        assert_eq!(
            Qwen3_5ChatTemplate::new().apply(&c),
            "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn a_system_turn_and_trimmed_content() {
        let mut c = Conversation::with_system("  Be brief. ");
        c.push_user(" What is Rust? \n");
        assert_eq!(
            Qwen3_5ChatTemplate::new().apply(&c),
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nWhat is Rust?<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn earlier_turns_have_no_think_block() {
        let mut c = Conversation::new();
        c.push_user("Hi");
        c.push_assistant("Hello! How can I help?");
        c.push_user("Tell me a joke.");
        assert_eq!(
            Qwen3_5ChatTemplate::new().apply(&c),
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello! How can I help?<|im_end|>\n\
             <|im_start|>user\nTell me a joke.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn reasoning_in_an_earlier_turn_is_dropped() {
        let mut c = Conversation::new();
        c.push_user("Hi");
        c.push_assistant("<think>\nThe user greets.\n</think>\n\nHello!");
        c.push_user("Bye");
        assert_eq!(
            Qwen3_5ChatTemplate::new().apply(&c),
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n\
             <|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn a_turn_after_the_last_user_message_keeps_a_think_block() {
        let mut c = Conversation::new();
        c.push_user("Hi");
        c.push_assistant("Hello");
        // transformers renders this one without a generation prompt; the
        // template here always adds one, so compare up to it.
        let prompt = Qwen3_5ChatTemplate::new().apply(&c);
        let expected = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello<|im_end|>\n";
        assert!(prompt.starts_with(expected), "{prompt:?}");
    }

    #[test]
    fn splits_inline_reasoning() {
        assert_eq!(
            split_reasoning("<think>\nhmm\n</think>\n\nyes"),
            ("hmm", "yes")
        );
        assert_eq!(split_reasoning("plain"), ("", "plain"));
    }
}
