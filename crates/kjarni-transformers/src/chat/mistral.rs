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