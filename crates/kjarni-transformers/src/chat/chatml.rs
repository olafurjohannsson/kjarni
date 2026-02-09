use crate::{ChatTemplate, Conversation, Role};

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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::Conversation;

    #[test]
    fn chatml_empty_conversation() {
        let template = ChatMLTemplate::new();
        let convo = Conversation::new();
        let prompt = template.apply(&convo);

        // Since add_generation_prompt = true by default, assistant header is added
        assert_eq!(prompt, "<|im_start|>assistant\n");
    }

    #[test]
    fn chatml_single_user_message() {
        let template = ChatMLTemplate::new();
        let mut convo = Conversation::new();
        convo.push_user("Hello!");

        let prompt = template.apply(&convo);
        assert!(prompt.contains("<|im_start|>user\n"));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.contains("<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn chatml_with_system_message() {
        let template = ChatMLTemplate::new();
        let mut convo = Conversation::with_system("You are a helpful assistant.");
        convo.push_user("Hello!");

        let prompt = template.apply(&convo);
        assert!(prompt.starts_with("<|im_start|>system\n"));
        assert!(prompt.contains("You are a helpful assistant."));
        assert!(prompt.contains("<|im_start|>user\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn chatml_multi_turn() {
        let template = ChatMLTemplate::new();
        let mut convo = Conversation::new();
        convo.push_user("Hi");
        convo.push_assistant("Hello!");
        convo.push_user("How are you?");
        convo.push_assistant("I am fine.");

        let prompt = template.apply(&convo);
        assert!(prompt.matches("<|im_start|>user\n").count() == 2);
        assert!(prompt.matches("<|im_start|>assistant\n").count() == 3); 
        // 2 assistant messages + trailing generation header
        assert!(prompt.matches("<|im_start|>system\n").count() == 0);
    }

    #[test]
    fn chatml_stop_sequences() {
        let template = ChatMLTemplate::new();
        let stops = template.stop_sequences();
        assert!(stops.contains(&"<|im_end|>".to_string()));
        assert!(stops.contains(&"<|endoftext|>".to_string()));
    }

    #[test]
    fn chatml_default_system_prompt() {
        let template = ChatMLTemplate::new();
        assert_eq!(template.default_system_prompt(), Some("You are a helpful assistant."));
    }

    #[test]
    fn chatml_disable_generation_prompt() {
        let mut template = ChatMLTemplate::new();
        template.add_generation_prompt = false;

        let mut convo = Conversation::new();
        convo.push_user("Hello!");
        let prompt = template.apply(&convo);

        // Should not have trailing assistant header
        assert!(!prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn chatml_system_message_only() {
        let template = ChatMLTemplate::new();
        let convo = Conversation::with_system("System only message");

        let prompt = template.apply(&convo);
        assert!(prompt.starts_with("<|im_start|>system\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }
}
