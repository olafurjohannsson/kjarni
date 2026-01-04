use crate::{ChatTemplate, Conversation, Role};


/// Mistral Instruct Template (v0.1, v0.2, v0.3)
///
/// Format:
/// <s>[INST] {system} {user} [/INST] {assistant} </s>[INST] {user} [/INST]
#[derive(Clone, Debug, Default)]
pub struct MistralChatTemplate {
    pub add_bos: bool,
}

impl MistralChatTemplate {
    pub fn new() -> Self {
        Self { add_bos: true }
    }
}

impl ChatTemplate for MistralChatTemplate {
    fn apply(&self, conversation: &Conversation) -> String {
        let mut prompt = String::new();
        let messages = conversation.messages();

        if messages.is_empty() {
            return prompt;
        }

        if self.add_bos {
            prompt.push_str("<s>");
        }

        let mut msg_iter = messages.iter().peekable();

        // Handle System Prompt (Mistral merges it into the first User message)
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
                    prompt.push_str("[INST] ");
                    
                    if is_first_user {
                        if let Some(sys) = system_content {
                            prompt.push_str(sys);
                            prompt.push_str("\n\n");
                        }
                        is_first_user = false;
                    }

                    prompt.push_str(&msg.content);
                    prompt.push_str(" [/INST]");
                }
                Role::Assistant => {
                    prompt.push(' ');
                    prompt.push_str(&msg.content);
                    prompt.push_str("</s>");
                    // If there is another user message coming, we usually don't add BOS again 
                    // in standard Mistral format, just straight to [INST].
                }
                Role::System => {
                    // System messages appearing later are usually ignored or concatenated
                }
            }
        }

        prompt
    }

    fn stop_sequences(&self) -> Vec<String> {
        vec!["</s>".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Conversation, Message, Role};

    #[test]
    fn mistral_empty_conversation() {
        let template = MistralChatTemplate::new();
        let convo = Conversation::new();
        let prompt = template.apply(&convo);
        assert_eq!(prompt, ""); // empty conversation -> empty string
    }

    #[test]
    fn mistral_single_user_no_system() {
        let template = MistralChatTemplate::new();
        let mut convo = Conversation::new();
        convo.push_user("Hello there");

        let prompt = template.apply(&convo);
        // <s>[INST] {user} [/INST]
        assert_eq!(prompt, "<s>[INST] Hello there [/INST]");
    }

    #[test]
    fn mistral_system_and_user() {
        let template = MistralChatTemplate::new();
        let mut convo = Conversation::new();
        convo.push(Message::system("You are a helpful assistant."));
        convo.push_user("What is 2 + 2?");

        let prompt = template.apply(&convo);
        // System merged into first user message
        let expected = "<s>[INST] You are a helpful assistant.\n\nWhat is 2 + 2? [/INST]";
        assert_eq!(prompt, expected);
    }

    #[test]
    fn mistral_user_and_assistant() {
        let template = MistralChatTemplate::new();
        let mut convo = Conversation::new();
        convo.push_user("Tell me a joke.");
        convo.push_assistant("Why did the chicken cross the road?");

        let prompt = template.apply(&convo);
        let expected = "<s>[INST] Tell me a joke. [/INST] Why did the chicken cross the road?</s>";
        assert_eq!(prompt, expected);
    }

    #[test]
    fn mistral_multiple_turns_with_system() {
        let template = MistralChatTemplate::new();
        let mut convo = Conversation::new();
        convo.push(Message::system("Assistant is friendly."));
        convo.push_user("Hello!");
        convo.push_assistant("Hi there!");
        convo.push_user("How are you?");
        convo.push_assistant("I'm good, thank you!");

        let prompt = template.apply(&convo);
        let expected = "<s>[INST] Assistant is friendly.\n\nHello! [/INST] Hi there!</s>[INST] How are you? [/INST] I'm good, thank you!</s>";
        assert_eq!(prompt, expected);
    }

    #[test]
    fn mistral_stop_sequences() {
        let template = MistralChatTemplate::new();
        let stops = template.stop_sequences();
        assert_eq!(stops, vec!["</s>".to_string()]);
    }

    #[test]
    fn mistral_add_bos_false() {
        let mut template = MistralChatTemplate::new();
        template.add_bos = false;

        let mut convo = Conversation::new();
        convo.push_user("Hello!");
        let prompt = template.apply(&convo);

        // No <s> at beginning
        assert_eq!(prompt, "[INST] Hello! [/INST]");
    }
}
