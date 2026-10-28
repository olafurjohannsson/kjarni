use crate::{ChatTemplate, Conversation, Role};

/// Mistral Instruct v0.3 template (v0.1 and v0.2 put a space before `[/INST]`)
///
/// Format:
/// <s>[INST] {user}[/INST] {assistant}</s>[INST] {system}\n\n{user}[/INST]
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

        // Mistral v0.3's template puts the system prompt in the last user turn,
        // and closes each with `[/INST]` straight after the text. This had a
        // space before `[/INST]` and the system prompt in the first turn, so
        // every prompt tokenized differently from the model's own template.
        // Only when that turn is the conversation's last message, as upstream's
        // `loop.last`; a conversation ending on a reply carries no system prompt.
        let last_user = messages
            .last()
            .filter(|m| m.role == Role::User)
            .map(|_| messages.len() - 1);

        for (i, msg) in msg_iter.enumerate() {
            let i = i + usize::from(system_content.is_some());
            match msg.role {
                Role::User => {
                    prompt.push_str("[INST] ");
                    if Some(i) == last_user
                        && let Some(sys) = system_content
                    {
                        prompt.push_str(sys);
                        prompt.push_str("\n\n");
                    }
                    prompt.push_str(&msg.content);
                    prompt.push_str("[/INST]");
                }
                Role::Assistant => {
                    prompt.push(' ');
                    prompt.push_str(&msg.content);
                    prompt.push_str("</s>");
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

    #[test]
    fn matches_mistral_v0_3_chat_template() {
        // transformers' rendering of the model's own template.
        let template = MistralChatTemplate::new();
        let mut conv = Conversation::new();
        conv.push_user("Hi");
        assert_eq!(template.apply(&conv), "<s>[INST] Hi[/INST]");

        let mut conv = Conversation::with_system("Be brief.");
        conv.push_user("Q1");
        conv.push_assistant("A1");
        conv.push_user("Q2");
        assert_eq!(
            template.apply(&conv),
            "<s>[INST] Q1[/INST] A1</s>[INST] Be brief.\n\nQ2[/INST]"
        );
    }

    use crate::{Conversation, Message};

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
        assert_eq!(prompt, "<s>[INST] Hello there[/INST]");
    }

    #[test]
    fn mistral_system_and_user() {
        let template = MistralChatTemplate::new();
        let mut convo = Conversation::new();
        convo.push(Message::system("You are a helpful assistant."));
        convo.push_user("What is 2 + 2?");

        let prompt = template.apply(&convo);
        // System merged into first user message
        let expected = "<s>[INST] You are a helpful assistant.\n\nWhat is 2 + 2?[/INST]";
        assert_eq!(prompt, expected);
    }

    #[test]
    fn mistral_user_and_assistant() {
        let template = MistralChatTemplate::new();
        let mut convo = Conversation::new();
        convo.push_user("Tell me a joke.");
        convo.push_assistant("Why did the chicken cross the road?");

        let prompt = template.apply(&convo);
        let expected = "<s>[INST] Tell me a joke.[/INST] Why did the chicken cross the road?</s>";
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
        // Ending on a reply, the system prompt has no user turn to go in.
        let expected = "<s>[INST] Hello![/INST] Hi there!</s>[INST] How are you?[/INST] I'm good, thank you!</s>";
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
        assert_eq!(prompt, "[INST] Hello![/INST]");
    }
}
