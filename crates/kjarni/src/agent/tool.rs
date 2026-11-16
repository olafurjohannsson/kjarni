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
