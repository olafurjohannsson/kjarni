//! Saved messages as a client shows them: an assistant message split into
//! its reasoning, its text and the tool calls it made.

use kjarni::Message;
use kjarni::agent::{ToolSpec, parse_tool_calls};
use serde_json::{Value as Json, json};

/// `(reasoning, rest)`: the text of a leading `<think>` block, if the reply
/// has one, and what follows it. A reply that opens a block and never closes
/// it is all reasoning.
pub fn split_reasoning(text: &str) -> (Option<&str>, &str) {
    let trimmed = text.trim_start();
    let Some(inside) = trimmed.strip_prefix("<think>") else {
        // Templates that open the block in the prompt leave only the close.
        return match text.split_once("</think>") {
            Some((thought, rest)) => (Some(thought.trim()), rest.trim_start()),
            None => (None, text),
        };
    };
    match inside.split_once("</think>") {
        Some((thought, rest)) => (Some(thought.trim()), rest.trim_start()),
        None => (Some(inside.trim()), ""),
    }
}

/// One message, for a client to render.
pub fn message(m: &Message, specs: &[ToolSpec]) -> Json {
    let role = m.role.to_string();
    if role != "assistant" {
        return json!({ "role": role, "text": m.content });
    }
    let (reasoning, rest) = split_reasoning(&m.content);
    let parsed = parse_tool_calls(rest, specs);
    let calls: Vec<Json> = parsed
        .tool_calls
        .iter()
        .map(|c| json!({ "name": c.name, "arguments": Json::Object(c.arguments.clone()) }))
        .collect();
    json!({
        "role": role,
        "text": parsed.content,
        "reasoning": reasoning.filter(|r| !r.is_empty()),
        "tool_calls": calls,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_is_split_from_the_reply() {
        assert_eq!(
            split_reasoning("<think>\nhmm\n</think>\n\nHello."),
            (Some("hmm"), "Hello.")
        );
        assert_eq!(split_reasoning("hmm</think>Hi"), (Some("hmm"), "Hi"));
        assert_eq!(split_reasoning("Just text"), (None, "Just text"));
        assert_eq!(
            split_reasoning("<think>still going"),
            (Some("still going"), "")
        );
    }

    #[test]
    fn an_assistant_message_lists_its_calls() {
        let specs = vec![ToolSpec::new(
            "read_file",
            "Reads a file.",
            json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        )];
        let m = Message::assistant(
            "<think>look first</think>Let me look.\n<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.md\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(
            message(&m, &specs),
            json!({
                "role": "assistant",
                "text": "Let me look.",
                "reasoning": "look first",
                "tool_calls": [{"name": "read_file", "arguments": {"path": "notes.md"}}],
            })
        );
        assert_eq!(
            message(&Message::tool("contents"), &specs),
            json!({"role": "tool", "text": "contents"})
        );
    }
}
