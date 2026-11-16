//! Tools a chat model can call: their definitions, as a template puts them in
//! the prompt, and the calls read back out of what the model wrote.
//!
//! Two formats are read. Qwen 3.5's own template asks for
//!
//! ```text
//! <tool_call>
//! <function=read_file>
//! <parameter=path>
//! src/main.rs
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! and Qwen 2.5, Qwen 3 and fine-tunes built on their templates write a JSON
//! object in the same tags: `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.

use serde_json::{Map, Value as Json};

use super::pyjson::PyJson;

/// A function the model may call.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// A JSON Schema object describing the arguments.
    pub parameters: Json,
}

impl ToolSpec {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Json) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    /// The definition in OpenAI's shape, `{"type": "function", "function": {...}}`,
    /// which is what chat templates are given as `tools`.
    pub fn to_json(&self) -> Json {
        let mut function = Map::new();
        function.insert("name".into(), Json::String(self.name.clone()));
        function.insert("description".into(), Json::String(self.description.clone()));
        function.insert("parameters".into(), self.parameters.clone());
        let mut tool = Map::new();
        tool.insert("type".into(), Json::String("function".into()));
        tool.insert("function".into(), Json::Object(function));
        Json::Object(tool)
    }

    /// The definition as a template's `tool | tojson` writes it. The outer keys
    /// come in the order Python callers write them (`type`, then `function`
    /// with `name`, `description`, `parameters`); keys inside `parameters` are
    /// sorted.
    pub fn to_prompt_json(&self) -> String {
        format!(
            "{{\"type\": \"function\", \"function\": {{\"name\": {}, \"description\": {}, \"parameters\": {}}}}}",
            PyJson::dumps(&Json::String(self.name.clone())),
            PyJson::dumps(&Json::String(self.description.clone())),
            PyJson::dumps(&self.parameters),
        )
    }

    /// The JSON Schema type declared for argument `name`, if any.
    fn parameter_type(&self, name: &str) -> Option<&str> {
        self.parameters
            .get("properties")?
            .get(name)?
            .get("type")?
            .as_str()
    }
}

/// One call the model asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Map<String, Json>,
}

impl ToolCall {
    /// The arguments as a JSON object string, as OpenAI's API carries them.
    pub fn arguments_json(&self) -> String {
        Json::Object(self.arguments.clone()).to_string()
    }

    /// The call in Qwen 3.5's format, as its template writes an assistant
    /// message's `tool_calls`.
    pub fn to_qwen_xml(&self) -> String {
        let mut out = format!("<tool_call>\n<function={}>\n", self.name);
        for (key, value) in &self.arguments {
            let value = match value {
                Json::String(s) => s.clone(),
                Json::Object(_) | Json::Array(_) => PyJson::dumps(value),
                other => other.to_string(),
            };
            out.push_str(&format!("<parameter={key}>\n{value}\n</parameter>\n"));
        }
        out.push_str("</function>\n</tool_call>");
        out
    }
}

impl ToolCall {
    /// The call as Qwen 2.5 and Qwen 3 templates write it:
    /// `<tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>`.
    pub fn to_json_block(&self) -> String {
        format!(
            "<tool_call>\n{{\"name\": {}, \"arguments\": {}}}\n</tool_call>",
            PyJson::dumps(&Json::String(self.name.clone())),
            PyJson::dumps(&Json::Object(self.arguments.clone()))
        )
    }
}

/// What a model wrote, split into its text and the calls it made.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParsedReply {
    /// The text outside the tool-call blocks, trimmed.
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    /// Blocks that looked like calls but could not be read, as written.
    pub malformed: Vec<String>,
}

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// Splits `text` into its prose and its tool calls. A block left unclosed at
/// the end, as when generation stops on a stop sequence, is read to the end.
/// `tools` gives each argument's declared type: a `string` argument stays
/// text even when it looks like a number, and any other is read as JSON when
/// it parses as JSON.
pub fn parse_tool_calls(text: &str, tools: &[ToolSpec]) -> ParsedReply {
    let mut reply = ParsedReply::default();
    let mut prose = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(OPEN) {
        prose.push_str(&rest[..start]);
        let after = &rest[start + OPEN.len()..];
        let (body, next) = match after.find(CLOSE) {
            Some(end) => (&after[..end], &after[end + CLOSE.len()..]),
            None => (after, ""),
        };
        match parse_block(body, tools) {
            Some(call) => reply.tool_calls.push(call),
            None => reply.malformed.push(body.trim().to_string()),
        }
        rest = next;
    }
    prose.push_str(rest);
    reply.content = prose.trim().to_string();
    reply
}

fn parse_block(body: &str, tools: &[ToolSpec]) -> Option<ToolCall> {
    let body = body.trim();
    if body.starts_with("<function=") {
        parse_xml(body, tools)
    } else {
        parse_json(body)
    }
}

fn parse_xml(body: &str, tools: &[ToolSpec]) -> Option<ToolCall> {
    let after = body.strip_prefix("<function=")?;
    let name_end = after.find('>')?;
    let name = after[..name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let spec = tools.iter().find(|t| t.name == name);
    let mut inner = &after[name_end + 1..];
    if let Some(end) = inner.rfind("</function>") {
        inner = &inner[..end];
    }
    let mut arguments = Map::new();
    while let Some(start) = inner.find("<parameter=") {
        let after = &inner[start + "<parameter=".len()..];
        let key_end = after.find('>')?;
        let key = after[..key_end].trim().to_string();
        let value_and_rest = &after[key_end + 1..];
        let (raw, next) = match value_and_rest.find("</parameter>") {
            Some(end) => (
                &value_and_rest[..end],
                &value_and_rest[end + "</parameter>".len()..],
            ),
            // An unclosed parameter runs to the next one, or to the end.
            None => match value_and_rest.find("<parameter=") {
                Some(next) => (&value_and_rest[..next], &value_and_rest[next..]),
                None => (value_and_rest, ""),
            },
        };
        // The template writes one newline either side of the value.
        let raw = raw.strip_prefix('\n').unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        let declared = spec.and_then(|s| s.parameter_type(&key));
        arguments.insert(key, typed_value(raw, declared));
        inner = next;
    }
    Some(ToolCall { name, arguments })
}

/// An XML parameter's value as JSON: text for a `string` argument, and
/// otherwise the JSON it parses as, or text when it does not parse.
fn typed_value(raw: &str, declared: Option<&str>) -> Json {
    if declared == Some("string") {
        return Json::String(raw.to_string());
    }
    match serde_json::from_str::<Json>(raw.trim()) {
        Ok(v) if declared.is_some() || !v.is_string() => v,
        _ => Json::String(raw.to_string()),
    }
}

fn parse_json(body: &str) -> Option<ToolCall> {
    let value: Json = serde_json::from_str(body).ok()?;
    let name = value.get("name")?.as_str()?.to_string();
    let arguments = match value.get("arguments").or_else(|| value.get("parameters")) {
        None | Some(Json::Null) => Map::new(),
        Some(Json::Object(map)) => map.clone(),
        // Some models write the arguments as a JSON string, as OpenAI's API does.
        Some(Json::String(s)) => match serde_json::from_str(s).ok()? {
            Json::Object(map) => map,
            _ => return None,
        },
        Some(_) => return None,
    };
    Some(ToolCall { name, arguments })
}

/// How far into `text` it is safe to show while streaming: everything before
/// a `<tool_call>` tag, or before a trailing fragment that could still grow
/// into one.
pub fn displayable_prefix_len(text: &str) -> usize {
    if let Some(start) = text.find(OPEN) {
        return start;
    }
    // Hold back a trailing "<", "<to", ... that may be the start of the tag.
    for keep in (1..OPEN.len()).rev() {
        if text.len() >= keep && text.is_char_boundary(text.len() - keep) {
            let tail = &text[text.len() - keep..];
            if OPEN.starts_with(tail) {
                return text.len() - keep;
            }
        }
    }
    text.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn read_file() -> ToolSpec {
        ToolSpec::new(
            "read_file",
            "Read a file.",
            json!({"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}),
        )
    }

    #[test]
    fn reads_qwen3_5_xml_calls() {
        let text = "Let me look.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n007.txt\n</parameter>\n<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>";
        let reply = parse_tool_calls(text, &[read_file()]);
        assert_eq!(reply.content, "Let me look.");
        assert_eq!(
            reply.tool_calls,
            vec![ToolCall {
                name: "read_file".into(),
                arguments: json!({"path": "007.txt", "limit": 20})
                    .as_object()
                    .unwrap()
                    .clone(),
            }]
        );
        assert!(reply.malformed.is_empty());
    }

    #[test]
    fn multi_line_values_keep_their_inner_newlines() {
        let spec = ToolSpec::new(
            "write_file",
            "",
            json!({"type": "object", "properties": {"content": {"type": "string"}}}),
        );
        let text = "<tool_call>\n<function=write_file>\n<parameter=content>\nline 1\n\nline 3\n</parameter>\n</function>\n</tool_call>";
        let call = &parse_tool_calls(text, &[spec]).tool_calls[0];
        assert_eq!(call.arguments["content"], "line 1\n\nline 3");
    }

    #[test]
    fn reads_json_calls_with_object_or_string_arguments() {
        let text = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": \"{\\\"path\\\": \\\"b.rs\\\"}\"}\n</tool_call>";
        let reply = parse_tool_calls(text, &[]);
        assert_eq!(reply.content, "");
        let paths: Vec<_> = reply
            .tool_calls
            .iter()
            .map(|c| c.arguments["path"].as_str().unwrap())
            .collect();
        assert_eq!(paths, ["a.rs", "b.rs"]);
    }

    #[test]
    fn an_unclosed_block_at_the_end_is_read() {
        let text = "<tool_call>\n<function=read_file>\n<parameter=path>\nx.md\n</parameter>\n</function>\n";
        let reply = parse_tool_calls(text, &[read_file()]);
        assert_eq!(reply.tool_calls[0].arguments["path"], "x.md");
    }

    #[test]
    fn unreadable_blocks_are_reported_not_dropped_silently() {
        let reply = parse_tool_calls("ok <tool_call>{not json}</tool_call>", &[]);
        assert_eq!(reply.content, "ok");
        assert!(reply.tool_calls.is_empty());
        assert_eq!(reply.malformed, ["{not json}"]);
    }

    #[test]
    fn plain_text_has_no_calls() {
        let reply = parse_tool_calls("  Paris is the capital.  ", &[]);
        assert_eq!(reply.content, "Paris is the capital.");
        assert!(reply.tool_calls.is_empty());
    }

    #[test]
    fn untyped_values_are_json_when_they_parse() {
        assert_eq!(typed_value("42", None), json!(42));
        assert_eq!(typed_value("[1, 2]", None), json!([1, 2]));
        assert_eq!(typed_value("hello", None), json!("hello"));
        assert_eq!(typed_value("42", Some("string")), json!("42"));
        assert_eq!(typed_value("true", Some("boolean")), json!(true));
    }

    #[test]
    fn writes_calls_as_the_template_does() {
        let call = ToolCall {
            name: "read_file".into(),
            arguments: json!({"limit": 5, "path": "a.rs"})
                .as_object()
                .unwrap()
                .clone(),
        };
        assert_eq!(
            call.to_qwen_xml(),
            "<tool_call>\n<function=read_file>\n<parameter=limit>\n5\n</parameter>\n<parameter=path>\na.rs\n</parameter>\n</function>\n</tool_call>"
        );
        // And reads back as itself.
        assert_eq!(
            parse_tool_calls(&call.to_qwen_xml(), &[read_file()]).tool_calls,
            [call]
        );
    }

    #[test]
    fn writes_json_calls_and_reads_them_back() {
        let call = ToolCall {
            name: "read_file".into(),
            arguments: json!({"path": "a.rs"}).as_object().unwrap().clone(),
        };
        assert_eq!(
            call.to_json_block(),
            "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\"}}\n</tool_call>"
        );
        assert_eq!(
            parse_tool_calls(&call.to_json_block(), &[]).tool_calls,
            [call]
        );
    }

    #[test]
    fn prompt_json_matches_python_json_dumps() {
        // json.dumps({"type": "function", "function": {"name": "read_file",
        // "description": "Read a file.", "parameters": {"properties": {...},
        // "required": ["path"], "type": "object"}}}, ensure_ascii=False)
        assert_eq!(
            read_file().to_prompt_json(),
            "{\"type\": \"function\", \"function\": {\"name\": \"read_file\", \"description\": \"Read a file.\", \"parameters\": {\"properties\": {\"limit\": {\"type\": \"integer\"}, \"path\": {\"type\": \"string\"}}, \"required\": [\"path\"], \"type\": \"object\"}}}"
        );
    }

    #[test]
    fn streaming_holds_back_a_partial_tag() {
        assert_eq!(displayable_prefix_len("Hello"), 5);
        assert_eq!(displayable_prefix_len("Hello <tool"), 6);
        assert_eq!(displayable_prefix_len("Hello <tool_call>\n<fun"), 6);
        assert_eq!(displayable_prefix_len("a < b"), 5);
        assert_eq!(displayable_prefix_len("þ<"), "þ".len());
    }
}
