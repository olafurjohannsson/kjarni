//! Llama 3 Instruct chat template
//!
//! Formats conversations according to Llama 3's expected format.

use crate::{ChatTemplate, Conversation, Role};

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
    /// The date block Llama 3.1 and 3.2 open every system turn with. `None`
    /// is Llama 3.0's plain format.
    pub date: Option<KnowledgeDate>,
}

/// What `Today Date:` says in a Llama 3.1 or 3.2 system turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KnowledgeDate {
    /// Llama 3.1's template: always "26 Jul 2024".
    Fixed(&'static str),
    /// Llama 3.2's template: the day the prompt is made, as transformers
    /// renders it with `strftime_now("%d %b %Y")`.
    Today,
}

impl KnowledgeDate {
    fn render(self) -> String {
        match self {
            Self::Fixed(date) => date.to_string(),
            Self::Today => today_dd_mon_yyyy(),
        }
    }
}

/// Today's date as "27 Oct 2026", in UTC. In the browser there is no clock to
/// read here, so it falls back to the date Llama 3.2's template itself uses
/// when it has none.
fn today_dd_mon_yyyy() -> String {
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(now) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        return format_civil_date((now.as_secs() / 86_400) as i64);
    }
    "26 Jul 2024".to_string()
}

/// A day count since 1970-01-01 as "dd Mon yyyy" (Howard Hinnant's
/// `civil_from_days`).
// The browser build never reads the clock, so only the tests call this there.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn format_civil_date(days: i64) -> String {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{day:02} {} {year}", MONTHS[(month - 1) as usize])
}

impl Llama3ChatTemplate {
    pub fn new() -> Self {
        Self {
            add_bos: true,
            add_generation_prompt: true,
            date: None,
        }
    }

    /// Create template for generation (adds assistant header at end)
    pub fn for_generation() -> Self {
        Self {
            add_bos: true,
            add_generation_prompt: true,
            date: None,
        }
    }

    /// Create template for training/fine-tuning (no trailing header)
    pub fn for_training() -> Self {
        Self {
            add_bos: true,
            add_generation_prompt: false,
            date: None,
        }
    }

    /// Llama 3.1 Instruct, as its `chat_template` renders without tools.
    pub fn llama3_1() -> Self {
        Self {
            date: Some(KnowledgeDate::Fixed("26 Jul 2024")),
            ..Self::for_generation()
        }
    }

    /// Llama 3.2 Instruct, as its `chat_template` renders without tools.
    pub fn llama3_2() -> Self {
        Self {
            date: Some(KnowledgeDate::Today),
            ..Self::for_generation()
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

        if let Some(date) = self.date {
            // Llama 3.1 and 3.2 always open with a system turn that carries the
            // knowledge cutoff and today's date, with any system prompt after
            // it, and trim every message. The model was tuned on exactly this;
            // without the date block the same question got a different answer
            // from the one transformers gives.
            let messages = conversation.messages();
            let (system, rest) = match messages.first() {
                Some(first) if first.role == Role::System => (first.content.trim(), &messages[1..]),
                _ => ("", messages),
            };
            prompt.push_str(&self.format_message(
                &Role::System,
                &format!(
                    "Cutting Knowledge Date: December 2023\nToday Date: {}\n\n{system}",
                    date.render()
                ),
            ));
            for message in rest {
                prompt.push_str(&self.format_message(&message.role, message.content.trim()));
            }
        } else {
            for message in conversation.messages() {
                prompt.push_str(&self.format_message(&message.role, &message.content));
            }
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
        vec![Self::EOT.to_string(), Self::END_OF_TEXT.to_string()]
    }

    fn default_system_prompt(&self) -> Option<&str> {
        // Llama 3.1 and 3.2 have no default; their system turn is the date
        // block alone.
        if self.date.is_some() {
            return None;
        }
        Some("You are a helpful, harmless, and honest assistant.")
    }

    fn validate(&self, conversation: &Conversation) -> Result<(), String> {
        let messages = conversation.messages();

        if messages.is_empty() {
            return Err("Conversation is empty".to_string());
        }

        // Check that messages alternate properly (after optional system)
        let mut expect_user = true;
        let start_idx = if messages
            .first()
            .map(|m| m.role == Role::System)
            .unwrap_or(false)
        {
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

        for msg in msg_iter {
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
    use crate::Message;

    use super::*;

    #[test]
    fn llama3_2_matches_its_chat_template() {
        let template = Llama3ChatTemplate {
            date: Some(KnowledgeDate::Fixed("27 Oct 2026")),
            ..Llama3ChatTemplate::for_generation()
        };
        let mut conv = Conversation::with_system("You are a helpful assistant.");
        conv.push_user("Write a haiku about autumn leaves.");
        // transformers' rendering of the model's own template.
        assert_eq!(
            template.apply(&conv),
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: \
             December 2023\nToday Date: 27 Oct 2026\n\nYou are a helpful assistant.<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\nWrite a haiku about autumn leaves.<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        // No system prompt still opens with the date block, and none is invented.
        assert_eq!(template.default_system_prompt(), None);
        let mut bare = Conversation::new();
        bare.push_user("Hi");
        assert!(template.apply(&bare).starts_with(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: \
             December 2023\nToday Date: 27 Oct 2026\n\n<|eot_id|>"
        ));
    }

    #[test]
    fn civil_dates_render_like_strftime() {
        assert_eq!(format_civil_date(0), "01 Jan 1970");
        assert_eq!(format_civil_date(19_930), "26 Jul 2024");
        assert_eq!(format_civil_date(20_753), "27 Oct 2026");
        assert_eq!(format_civil_date(11_016), "29 Feb 2000");
    }

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
        assert_eq!(
            prompt
                .matches("<|start_header_id|>user<|end_header_id|>")
                .count(),
            2
        );
        assert_eq!(
            prompt
                .matches("<|start_header_id|>assistant<|end_header_id|>")
                .count(),
            2
        ); // 1 response + 1 generation prompt
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
    #[test]
    fn llama3_empty_conversation() {
        let template = Llama3ChatTemplate::new();
        let convo = Conversation::new();
        let prompt = template.apply(&convo);
        assert!(prompt.starts_with("<|begin_of_text|>")); // BOS still added
    }

    #[test]
    fn llama3_single_user_message() {
        let template = Llama3ChatTemplate::for_generation();
        let mut convo = Conversation::new();
        convo.push_user("Hello!");
        let prompt = template.apply(&convo);

        assert!(prompt.contains("Hello!"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn llama3_with_system_message() {
        let template = Llama3ChatTemplate::for_generation();
        let mut convo = Conversation::with_system("You are helpful.");
        convo.push_user("Hello!");
        let prompt = template.apply(&convo);

        assert!(prompt.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>"));
    }

    #[test]
    fn llama3_multi_turn() {
        let template = Llama3ChatTemplate::for_generation();
        let mut convo = Conversation::new();
        convo.push_user("Hi");
        convo.push_assistant("Hello!");
        convo.push_user("How are you?");
        convo.push_assistant("I am good.");

        let prompt = template.apply(&convo);

        assert_eq!(
            prompt
                .matches("<|start_header_id|>user<|end_header_id|>")
                .count(),
            2
        );
        assert_eq!(
            prompt
                .matches("<|start_header_id|>assistant<|end_header_id|>")
                .count(),
            3
        );
        // 2 responses + trailing assistant header
    }

    #[test]
    fn llama3_validation_ok() {
        let template = Llama3ChatTemplate::new();
        let mut convo = Conversation::new();
        convo.push_user("Hi");
        convo.push_assistant("Hello!");
        assert!(template.validate(&convo).is_ok());
    }

    #[test]
    fn llama3_validation_failures() {
        let template = Llama3ChatTemplate::new();

        // Assistant first
        let mut convo = Conversation::new();
        convo.push_assistant("Hi");
        assert!(template.validate(&convo).is_err());

        // System not first
        let mut convo = Conversation::new();
        convo.push_user("Hi");
        convo.push(Message::system("System"));
        assert!(template.validate(&convo).is_err());
    }

    #[test]
    fn llama3_stop_sequences() {
        let template = Llama3ChatTemplate::new();
        let stops = template.stop_sequences();
        assert!(stops.contains(&"<|eot_id|>".to_string()));
        assert!(stops.contains(&"<|end_of_text|>".to_string()));
    }

    #[test]
    fn llama2_empty_conversation() {
        let template = Llama2ChatTemplate::new();
        let convo = Conversation::new();
        let prompt = template.apply(&convo);
        assert_eq!(prompt, "");
    }

    #[test]
    fn llama2_single_user_message_with_system() {
        let template = Llama2ChatTemplate::new();
        let mut convo = Conversation::with_system("You are helpful.");
        convo.push_user("Hello!");
        let prompt = template.apply(&convo);

        assert!(prompt.contains("[INST]"));
        assert!(prompt.contains("<<SYS>>"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.contains("[/INST]"));
    }

    #[test]
    fn llama2_multi_turn() {
        let template = Llama2ChatTemplate::new();
        let mut convo = Conversation::new();
        convo.push_user("Hi");
        convo.push_assistant("Hello!");
        convo.push_user("How are you?");
        convo.push_assistant("I am fine.");

        let prompt = template.apply(&convo);
        assert!(prompt.contains("[INST]"));
        assert!(prompt.matches("[/INST]").count() >= 2);
        assert!(prompt.matches("</s>").count() >= 2);
    }

    #[test]
    fn llama2_stop_sequences() {
        let template = Llama2ChatTemplate::new();
        let stops = template.stop_sequences();
        assert_eq!(stops, vec!["</s>"]);
    }

    #[test]
    fn llama2_default_system_prompt() {
        let template = Llama2ChatTemplate::new();
        assert_eq!(
            template.default_system_prompt(),
            Some("You are a helpful, respectful and honest assistant.")
        );
    }
}
