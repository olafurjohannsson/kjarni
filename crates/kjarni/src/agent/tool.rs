//! What a tool is, and what it may touch.

use async_trait::async_trait;
use serde_json::{Map, Value as Json};

pub use kjarni_transformers::chat::tools::displayable_prefix_len;
pub use kjarni_transformers::{ParsedReply, ToolCall, ToolSpec, parse_tool_calls};

/// The kind of access a tool needs, which decides whether a call is run
/// straight away or waits for the user's approval.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Access {
    /// Reads files or the agent's memory; changes nothing.
    Read,
    /// Writes files or memory.
    Write,
    /// Runs a program.
    Execute,
    /// Reaches the network.
    Network,
}

impl Access {
    pub fn as_str(self) -> &'static str {
        match self {
            Access::Read => "read",
            Access::Write => "write",
            Access::Execute => "execute",
            Access::Network => "network",
        }
    }
}

/// A function the agent can call.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The definition the model sees.
    fn spec(&self) -> ToolSpec;

    /// What the tool touches.
    fn access(&self) -> Access;

    /// The call in a few words, for approval prompts and UIs:
    /// `read_file src/main.rs`, `run_command cargo test`.
    fn summary(&self, args: &Map<String, Json>) -> String {
        let name = self.spec().name;
        match args.values().next() {
            Some(Json::String(s)) => format!("{name} {}", first_line(s)),
            Some(other) => format!("{name} {other}"),
            None => name,
        }
    }

    /// The file the call writes, for front ends that list a conversation's
    /// changes. None for tools that write no file.
    fn writes(&self, _args: &Map<String, Json>) -> Option<std::path::PathBuf> {
        None
    }

    /// Runs the call. An error is shown to the model as the call's result, so
    /// it says what went wrong in words the model can act on.
    async fn call(&self, args: &Map<String, Json>) -> anyhow::Result<String>;
}

/// The first line of `s`, shortened to 80 characters.
pub(crate) fn first_line(s: &str) -> String {
    let line = s.lines().next().unwrap_or("");
    let mut out: String = line.chars().take(80).collect();
    if out.len() < line.len() || s.lines().nth(1).is_some() {
        out.push('…');
    }
    out
}

/// Argument `key` as a string, or an error naming it.
pub(crate) fn string_arg<'a>(args: &'a Map<String, Json>, key: &str) -> anyhow::Result<&'a str> {
    match args.get(key) {
        Some(Json::String(s)) => Ok(s),
        Some(other) => anyhow::bail!("argument '{key}' must be a string, got {other}"),
        None => anyhow::bail!("missing required argument '{key}'"),
    }
}

/// Optional argument `key` as a whole number.
pub(crate) fn usize_arg(args: &Map<String, Json>, key: &str) -> anyhow::Result<Option<usize>> {
    match args.get(key) {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Number(n)) => n
            .as_u64()
            .map(|n| Some(n as usize))
            .ok_or_else(|| anyhow::anyhow!("argument '{key}' must be a whole number")),
        Some(Json::String(s)) => s
            .trim()
            .parse()
            .map(Some)
            .map_err(|_| anyhow::anyhow!("argument '{key}' must be a whole number, got '{s}'")),
        Some(other) => anyhow::bail!("argument '{key}' must be a whole number, got {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(v: Json) -> Map<String, Json> {
        v.as_object().unwrap().clone()
    }

    struct Echo;

    #[async_trait]
    impl Tool for Echo {
        fn spec(&self) -> ToolSpec {
            ToolSpec::new("echo", "Echoes.", json!({"type": "object"}))
        }
        fn access(&self) -> Access {
            Access::Read
        }
        async fn call(&self, _: &Map<String, Json>) -> anyhow::Result<String> {
            Ok(String::new())
        }
    }

    #[test]
    fn the_default_summary_names_the_tool_and_its_first_argument() {
        assert_eq!(
            Echo.summary(&args(json!({"path": "src/main.rs"}))),
            "echo src/main.rs"
        );
        assert_eq!(Echo.summary(&args(json!({"n": 3}))), "echo 3");
        assert_eq!(Echo.summary(&Map::new()), "echo");
        assert_eq!(
            Echo.summary(&args(json!({"command": "cargo build\ncargo test"}))),
            "echo cargo build…"
        );
    }

    #[test]
    fn first_line_marks_what_it_cuts() {
        assert_eq!(first_line("short"), "short");
        assert_eq!(first_line(""), "");
        let long = "é".repeat(100);
        assert_eq!(first_line(&long), format!("{}…", "é".repeat(80)));
    }

    #[test]
    fn string_args_are_required_and_typed() {
        let a = args(json!({"path": "a.txt", "n": 1}));
        assert_eq!(string_arg(&a, "path").unwrap(), "a.txt");
        assert_eq!(
            string_arg(&a, "n").unwrap_err().to_string(),
            "argument 'n' must be a string, got 1"
        );
        assert_eq!(
            string_arg(&a, "text").unwrap_err().to_string(),
            "missing required argument 'text'"
        );
    }

    #[test]
    fn whole_numbers_come_as_numbers_or_digit_strings() {
        let a = args(json!({"a": 5, "b": " 12 ", "c": null, "d": -1, "e": "ten", "f": [1]}));
        assert_eq!(usize_arg(&a, "a").unwrap(), Some(5));
        assert_eq!(usize_arg(&a, "b").unwrap(), Some(12));
        assert_eq!(usize_arg(&a, "c").unwrap(), None);
        assert_eq!(usize_arg(&a, "missing").unwrap(), None);
        assert!(usize_arg(&a, "d").is_err());
        assert_eq!(
            usize_arg(&a, "e").unwrap_err().to_string(),
            "argument 'e' must be a whole number, got 'ten'"
        );
        assert!(usize_arg(&a, "f").is_err());
    }

    #[test]
    fn access_names() {
        let names: Vec<_> = [
            Access::Read,
            Access::Write,
            Access::Execute,
            Access::Network,
        ]
        .map(Access::as_str)
        .into();
        assert_eq!(names, ["read", "write", "execute", "network"]);
    }
}

/// A file's content before or after a call wrote it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FileState {
    /// There was no file.
    Missing,
    Text(String),
    /// Larger than [`FileState::MAX_BYTES`], or not UTF-8 text.
    Unreadable,
}

impl FileState {
    /// Files larger than this are not read for a change list.
    pub const MAX_BYTES: u64 = 1 << 20;

    pub fn read(path: &std::path::Path) -> Self {
        match std::fs::metadata(path) {
            Err(_) => Self::Missing,
            Ok(m) if !m.is_file() || m.len() > Self::MAX_BYTES => Self::Unreadable,
            Ok(_) => match std::fs::read(path) {
                Err(_) => Self::Unreadable,
                Ok(bytes) => String::from_utf8(bytes).map_or(Self::Unreadable, Self::Text),
            },
        }
    }
}

/// A file a call changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileChange {
    pub path: std::path::PathBuf,
    pub before: FileState,
    pub after: FileState,
}
