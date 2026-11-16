//! A model's own chat template, the Jinja `chat_template` its tokenizer
//! config carries, rendered as transformers' `apply_chat_template` renders it.
//!
//! The registry's models have hand-written templates; this is for a model
//! folder Kjarni has no template for. The environment is set up as
//! transformers 4.57 sets up its own: `trim_blocks` and `lstrip_blocks`, loop
//! controls, a `tojson` that writes what Python's `json.dumps` writes, and the
//! `raise_exception` and `strftime_now` globals. The special tokens are passed
//! in by name, as `apply_chat_template` passes `special_tokens_map`.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use minijinja::value::Kwargs;
use minijinja::{Environment, Error, ErrorKind, Value};
use serde_json::Value as Json;

use super::pyjson::PyJson;
use crate::{ChatTemplate, Conversation, Role};

const TEMPLATE_NAME: &str = "chat_template";

/// A chat template written in Jinja, as Hugging Face models ship them.
pub struct JinjaChatTemplate {
    env: Environment<'static>,
    special_tokens: serde_json::Map<String, Json>,
    stop: Vec<String>,
    /// Whether the source reads `tools`, as templates written for tool use do.
    reads_tools: bool,
    /// Whether calls are written in Qwen 3.5's XML form rather than as JSON.
    xml_calls: bool,
}

impl std::fmt::Debug for JinjaChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JinjaChatTemplate")
            .field("special_tokens", &self.special_tokens)
            .finish_non_exhaustive()
    }
}

impl JinjaChatTemplate {
    /// The template of the model in `dir`: `chat_template.jinja` if there is
    /// one, else the `chat_template` of `tokenizer_config.json`. `None` when
    /// the model has no template, as base models do not.
    pub fn from_dir(dir: &Path) -> Result<Option<Self>> {
        let config = read_object(&dir.join("tokenizer_config.json"))?;
        let jinja = dir.join("chat_template.jinja");
        let source = if jinja.exists() {
            std::fs::read_to_string(&jinja)
                .with_context(|| format!("reading {}", jinja.display()))?
        } else {
            match config.get("chat_template") {
                None | Some(Json::Null) => return Ok(None),
                Some(Json::String(s)) => s.clone(),
                // Several named templates; transformers uses the one named "default".
                Some(Json::Array(named)) => named
                    .iter()
                    .find(|t| t.get("name").and_then(Json::as_str) == Some("default"))
                    .and_then(|t| t.get("template"))
                    .and_then(Json::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        anyhow!(
                            "{} has several chat templates and none named \"default\"",
                            dir.display()
                        )
                    })?,
                Some(_) => bail!("chat_template in {} is not a string", dir.display()),
            }
        };

        let mut special = config;
        for (k, v) in read_object(&dir.join("special_tokens_map.json"))? {
            special.insert(k, v);
        }
        Self::new(&source, &special).map(Some)
    }

    /// A template from its source and the tokenizer config holding the
    /// special tokens (`bos_token`, `eos_token` and the rest).
    pub fn new(source: &str, tokenizer_config: &serde_json::Map<String, Json>) -> Result<Self> {
        let mut special_tokens = serde_json::Map::new();
        for name in [
            "bos_token",
            "eos_token",
            "unk_token",
            "sep_token",
            "pad_token",
            "cls_token",
            "mask_token",
        ] {
            if let Some(content) = tokenizer_config.get(name).and_then(token_content) {
                special_tokens.insert(name.into(), Json::String(content));
            }
        }
        if let Some(Json::Array(extra)) = tokenizer_config.get("additional_special_tokens") {
            let extra: Vec<Json> = extra
                .iter()
                .filter_map(token_content)
                .map(Json::String)
                .collect();
            special_tokens.insert("additional_special_tokens".into(), Json::Array(extra));
        }
        let stop = special_tokens
            .get("eos_token")
            .and_then(Json::as_str)
            .map(|s| vec![s.to_owned()])
            .unwrap_or_default();

        let reads_tools = source.contains("tools");
        let xml_calls = source.contains("<function=");
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.add_template_owned(TEMPLATE_NAME, without_generation_tags(source))
            .map_err(|e| anyhow!("the chat template does not parse: {e:#}"))?;

        Ok(Self {
            env,
            special_tokens,
            stop,
            reads_tools,
            xml_calls,
        })
    }

    /// The prompt for `conversation`, ending with the assistant's turn opened
    /// when `add_generation_prompt` is set. An error is the template's own
    /// `raise_exception`, such as for roles that do not alternate.
    pub fn render(
        &self,
        conversation: &Conversation,
        add_generation_prompt: bool,
    ) -> Result<String> {
        let messages: Vec<Value> = conversation
            .messages()
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                Value::from_iter([
                    ("role", Value::from(role)),
                    ("content", Value::from(m.content.as_str())),
                ])
            })
            .collect();
        let tools = if conversation.tools().is_empty() {
            Value::from(())
        } else {
            let specs: Vec<Json> = conversation.tools().iter().map(|t| t.to_json()).collect();
            Value::from_serialize(&specs)
        };
        let template = self.env.get_template(TEMPLATE_NAME)?;
        let ctx = minijinja::context! {
            messages => messages,
            tools => tools,
            documents => Value::from(()),
            add_generation_prompt => add_generation_prompt,
            ..Value::from_serialize(&self.special_tokens)
        };
        template
            .render(ctx)
            .map_err(|e| anyhow!("the chat template failed: {}", describe(&e)))
    }
}

impl ChatTemplate for JinjaChatTemplate {
    fn apply(&self, conversation: &Conversation) -> String {
        self.try_apply(conversation).unwrap_or_else(|e| {
            log::error!("{e}");
            String::new()
        })
    }

    fn try_apply(&self, conversation: &Conversation) -> std::result::Result<String, String> {
        self.render(conversation, true).map_err(|e| e.to_string())
    }

    fn stop_sequences(&self) -> Vec<String> {
        self.stop.clone()
    }

    fn renders_special_tokens(&self) -> bool {
        true
    }

    fn supports_tools(&self) -> bool {
        self.reads_tools
    }

    fn format_tool_call(&self, call: &crate::ToolCall) -> String {
        if self.xml_calls {
            call.to_qwen_xml()
        } else {
            call.to_json_block()
        }
    }

    fn validate(&self, conversation: &Conversation) -> std::result::Result<(), String> {
        self.try_apply(conversation).map(|_| ())
    }
}

/// The message of a template error, and of what caused it.
fn describe(e: &Error) -> String {
    let mut out = e
        .detail()
        .map(str::to_owned)
        .unwrap_or_else(|| e.to_string());
    let mut source = std::error::Error::source(e);
    while let Some(s) = source {
        out.push_str(&format!(": {s}"));
        source = s.source();
    }
    out
}

/// A special token's text: a plain string, or an object with `content`.
fn token_content(v: &Json) -> Option<String> {
    match v {
        Json::String(s) => Some(s.clone()),
        Json::Object(o) => o.get("content").and_then(Json::as_str).map(str::to_owned),
        _ => None,
    }
}

fn read_object(path: &Path) -> Result<serde_json::Map<String, Json>> {
    if !path.exists() {
        return Ok(Default::default());
    }
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    match serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))? {
        Json::Object(map) => Ok(map),
        _ => bail!("{} is not a JSON object", path.display()),
    }
}

/// transformers' `{% generation %}` block only marks which tokens are the
/// assistant's, and renders as its contents.
fn without_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(open) = rest.find("{%") {
        let Some(close) = rest[open..].find("%}") else {
            break;
        };
        let tag = &rest[open + 2..open + close];
        let name = tag.trim_matches(|c: char| c == '-' || c == '+' || c.is_whitespace());
        out.push_str(&rest[..open]);
        if name != "generation" && name != "endgeneration" {
            out.push_str(&rest[open..open + close + 2]);
        }
        rest = &rest[open + close + 2..];
    }
    out.push_str(rest);
    out
}

fn raise_exception(message: String) -> std::result::Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

/// Python's `datetime.now().strftime(format)`.
fn strftime_now(format: String) -> String {
    chrono::Local::now().format(&format).to_string()
}

/// transformers' `tojson`: `json.dumps(x, ensure_ascii=False, indent=None,
/// separators=None, sort_keys=False)`, Python's spacing included.
fn tojson(value: Value, kwargs: Kwargs) -> std::result::Result<Value, Error> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let indent: Option<Value> = kwargs.get("indent")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    let separators: Option<Vec<String>> = kwargs.get("separators")?;
    kwargs.assert_all_used()?;

    let indent = match indent {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => match v.as_usize() {
            Some(n) => Some(" ".repeat(n)),
            None => Some(v.to_string()),
        },
    };
    let (item_sep, key_sep) = match separators.as_deref() {
        Some([a, b]) => (a.clone(), b.clone()),
        Some(_) => {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "tojson separators must be a pair",
            ));
        }
        // Python's defaults: no space before a newline when indenting.
        None if indent.is_some() => (",".into(), ": ".into()),
        None => (", ".into(), ": ".into()),
    };
    let json = serde_json::to_value(&value)
        .map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    let style = PyJson {
        ensure_ascii: ensure_ascii.unwrap_or(false),
        indent,
        item_sep,
        key_sep,
        sort_keys: sort_keys.unwrap_or(false),
    };
    let mut out = String::new();
    style.write(&json, 0, &mut out);
    // Safe: the output goes into a prompt, never through HTML escaping.
    Ok(Value::from_safe_string(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config(v: Json) -> serde_json::Map<String, Json> {
        v.as_object().unwrap().clone()
    }

    fn convo(turns: &[(Role, &str)]) -> Conversation {
        let mut c = Conversation::new();
        for (role, text) in turns {
            c.push(crate::Message {
                role: role.clone(),
                content: (*text).into(),
            });
        }
        c
    }

    /// Qwen2.5's template; expected output from transformers 4.57's
    /// apply_chat_template(add_generation_prompt=True).
    #[test]
    fn renders_a_chatml_template_like_transformers() {
        let source = "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}";
        let t =
            JinjaChatTemplate::new(source, &config(json!({"eos_token": "<|im_end|>"}))).unwrap();
        let got = t
            .render(
                &convo(&[(Role::System, "Be brief."), (Role::User, "Hæ!")]),
                true,
            )
            .unwrap();
        assert_eq!(
            got,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHæ!<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(t.stop_sequences(), ["<|im_end|>"]);
    }

    #[test]
    fn special_tokens_reach_the_template_by_name() {
        let t = JinjaChatTemplate::new(
            "{{ bos_token }}{% for m in messages %}[{{ m['content'] }}]{% endfor %}{{ eos_token }}",
            &config(json!({"bos_token": {"content": "<s>", "lstrip": false}, "eos_token": "</s>"})),
        )
        .unwrap();
        assert_eq!(
            t.render(&convo(&[(Role::User, "a")]), false).unwrap(),
            "<s>[a]</s>"
        );
    }

    #[test]
    fn raise_exception_is_an_error_with_its_message() {
        let t = JinjaChatTemplate::new(
            "{% if messages[0]['role'] == 'assistant' %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}",
            &Default::default(),
        )
        .unwrap();
        let err = t
            .render(&convo(&[(Role::Assistant, "x")]), true)
            .unwrap_err();
        assert!(err.to_string().contains("roles must alternate"), "{err}");
        assert!(t.validate(&convo(&[(Role::Assistant, "x")])).is_err());
    }

    /// Expected strings from Python's json.dumps(x, ensure_ascii=False) with
    /// the same arguments.
    #[test]
    fn tojson_writes_what_python_json_dumps_writes() {
        let t = JinjaChatTemplate::new(
            "{{ {'a': 1, 'b': [1.5, 'þ\\n\"'], 'c': none, 'd': true} | tojson }}|{{ {'b': 1, 'a': 2} | tojson(indent=2, sort_keys=true) }}|{{ 'é' | tojson(ensure_ascii=true) }}",
            &Default::default(),
        )
        .unwrap();
        assert_eq!(
            t.render(&Conversation::new(), false).unwrap(),
            "{\"a\": 1, \"b\": [1.5, \"þ\\n\\\"\"], \"c\": null, \"d\": true}|{\n  \"a\": 2,\n  \"b\": 1\n}|\"\\u00e9\""
        );
    }

    #[test]
    fn generation_blocks_render_as_their_contents() {
        assert_eq!(
            without_generation_tags("a{% generation %}b{%- endgeneration -%}c{% if x %}"),
            "abc{% if x %}"
        );
    }

    #[test]
    fn python_string_methods_work() {
        let t = JinjaChatTemplate::new(
            "{{ messages[0]['content'].strip() }}|{{ 'a</think>b'.split('</think>')[-1] }}|{{ messages[::-1][0].role }}",
            &Default::default(),
        )
        .unwrap();
        assert_eq!(
            t.render(
                &convo(&[(Role::User, "  hi  "), (Role::Assistant, "x")]),
                false
            )
            .unwrap(),
            "hi|b|assistant"
        );
    }
}
