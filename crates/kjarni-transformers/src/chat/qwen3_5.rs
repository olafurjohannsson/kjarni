use crate::{ChatTemplate, Conversation, Role};

/// Qwen 3.5's chat template, for text.
///
/// ChatML, with three differences from Qwen 2.5's: no default system turn,
/// every message trimmed, and a generation prompt that opens the reply with a
/// `<think>` block. Unless the conversation asks for thinking, the block is
/// closed empty so the model answers directly; Qwen's own template thinks
/// unless told not to, and Kjarni answers directly unless told to think. Reasoning in an earlier
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

/// The system turn's opening when tools are offered, before the definitions.
const TOOLS_HEADER: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";

/// What the template tells the model about calling them, after the definitions.
const TOOLS_INSTRUCTIONS: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

impl ChatTemplate for Qwen3_5ChatTemplate {
    fn apply(&self, conversation: &Conversation) -> String {
        let messages = conversation.messages();
        let tools = conversation.tools();
        let last_user = messages.iter().rposition(|m| m.role == Role::User);
        let mut prompt = String::new();

        // The system turn comes first, and only the first system message is
        // rendered, as the template requires.
        let system = messages
            .first()
            .filter(|m| m.role == Role::System)
            .map(|m| m.content.trim());
        if !tools.is_empty() {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(TOOLS_HEADER);
            for tool in tools {
                prompt.push('\n');
                prompt.push_str(&tool.to_prompt_json());
            }
            prompt.push_str(TOOLS_INSTRUCTIONS);
            if let Some(system) = system.filter(|s| !s.is_empty()) {
                prompt.push_str("\n\n");
                prompt.push_str(system);
            }
            prompt.push_str("<|im_end|>\n");
        } else if let Some(system) = system {
            prompt.push_str(&format!("<|im_start|>system\n{system}<|im_end|>\n"));
        }

        for (i, message) in messages.iter().enumerate() {
            let content = message.content.trim();
            match message.role {
                Role::System => {}
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
                // Consecutive results share one user turn.
                Role::Tool => {
                    let follows_tool = i > 0 && messages[i - 1].role == Role::Tool;
                    if !follows_tool {
                        prompt.push_str("<|im_start|>user");
                    }
                    prompt.push_str(&format!("\n<tool_response>\n{content}\n</tool_response>"));
                    let precedes_tool = messages.get(i + 1).is_some_and(|m| m.role == Role::Tool);
                    if !precedes_tool {
                        prompt.push_str("<|im_end|>\n");
                    }
                }
            }
        }
        if conversation.thinking() == Some(true) {
            prompt.push_str("<|im_start|>assistant\n<think>\n");
        } else {
            prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
        }
        prompt
    }

    /// A thinking reply starts inside the block the prompt opened. A direct
    /// one's prompt ends with the block closed, so it has no prefix.
    fn reply_prefix(&self) -> &str {
        "<think>\n"
    }

    fn supports_tools(&self) -> bool {
        true
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

    /// transformers with enable_thinking=True.
    #[test]
    fn a_thinking_reply_opens_the_think_block_and_starts_inside_it() {
        let mut c = Conversation::new();
        c.push_user("Hello!");
        c.set_thinking(Some(true));
        let t = Qwen3_5ChatTemplate::new();
        let prompt = t.apply(&c);
        assert_eq!(
            prompt,
            "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert_eq!(t.reply_prefix_for(&prompt), "<think>\n");
        // Asking for no thinking is the default.
        c.set_thinking(Some(false));
        let direct = t.apply(&c);
        assert!(direct.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert_eq!(t.reply_prefix_for(&direct), "");
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

    fn read_file_tool() -> Vec<crate::ToolSpec> {
        vec![crate::ToolSpec::new(
            "read_file",
            "Read a file.",
            serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
        )]
    }

    fn tool_turns() -> Conversation {
        let mut c = Conversation::new();
        c.set_tools(read_file_tool());
        c.push_user("What's in notes.txt?");
        c.push_assistant(
            "<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n</function>\n</tool_call>",
        );
        c.push(crate::Message::tool("buy milk"));
        c.push(crate::Message::tool("buy eggs"));
        c
    }

    // The expected prompts below are what transformers 4.57.3 renders with
    // Qwen/Qwen3.5-0.8B's chat_template.jinja, the same tool given as
    // {"type": "function", "function": {"name", "description", "parameters"}},
    // and the call given as the assistant message's tool_calls.

    #[test]
    fn tools_and_the_system_prompt_share_the_system_turn() {
        let mut c = Conversation::with_system("Be brief.");
        c.set_tools(read_file_tool());
        c.push_user("What's in notes.txt?");
        assert_eq!(
            Qwen3_5ChatTemplate::new().apply(&c),
            "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"read_file\", \"description\": \"Read a file.\", \"parameters\": {\"properties\": {\"path\": {\"type\": \"string\"}}, \"required\": [\"path\"], \"type\": \"object\"}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>\n\nBe brief.<|im_end|>\n<|im_start|>user\nWhat's in notes.txt?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn consecutive_tool_results_share_one_user_turn() {
        assert_eq!(
            Qwen3_5ChatTemplate::new().apply(&tool_turns()),
            "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"read_file\", \"description\": \"Read a file.\", \"parameters\": {\"properties\": {\"path\": {\"type\": \"string\"}}, \"required\": [\"path\"], \"type\": \"object\"}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT><|im_end|>\n<|im_start|>user\nWhat's in notes.txt?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nbuy milk\n</tool_response>\n<tool_response>\nbuy eggs\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn a_finished_tool_exchange_loses_its_think_blocks() {
        let mut c = tool_turns();
        c.push_assistant("Buy milk and eggs.");
        c.push_user("Thanks!");
        assert_eq!(
            Qwen3_5ChatTemplate::new().apply(&c),
            "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"read_file\", \"description\": \"Read a file.\", \"parameters\": {\"properties\": {\"path\": {\"type\": \"string\"}}, \"required\": [\"path\"], \"type\": \"object\"}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT><|im_end|>\n<|im_start|>user\nWhat's in notes.txt?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nbuy milk\n</tool_response>\n<tool_response>\nbuy eggs\n</tool_response><|im_end|>\n<|im_start|>assistant\nBuy milk and eggs.<|im_end|>\n<|im_start|>user\nThanks!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
