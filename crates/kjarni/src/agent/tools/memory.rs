//! The agent's own memory, as tools.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value as Json, json};

use crate::agent::memory::Memory;
use crate::agent::tool::{Access, Tool, ToolSpec, string_arg, usize_arg};

pub struct Remember(pub Arc<Memory>);

#[async_trait]
impl Tool for Remember {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "remember",
            "Save a fact for future conversations: something about the user, their preferences or their projects that will matter later. One short, self-contained sentence.",
            json!({
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact, e.g. 'The user's main project is in ~/dev/kjarni.'"}
                },
                "required": ["fact"]
            }),
        )
    }

    fn access(&self) -> Access {
        Access::Write
    }

    async fn call(&self, args: &Map<String, Json>) -> anyhow::Result<String> {
        let entry = self.0.remember(string_arg(args, "fact")?).await?;
        Ok(format!("Remembered (#{}).", entry.id))
    }
}

pub struct Recall(pub Arc<Memory>);

#[async_trait]
impl Tool for Recall {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "recall",
            "Search the facts saved in earlier conversations.",
            json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "limit": {"type": "integer", "description": "Most facts to return. Default 5."}
                },
                "required": ["query"]
            }),
        )
    }

    fn access(&self) -> Access {
        Access::Read
    }

    async fn call(&self, args: &Map<String, Json>) -> anyhow::Result<String> {
        let query = string_arg(args, "query")?;
        let limit = usize_arg(args, "limit")?.unwrap_or(5);
        let hits = self.0.recall(query, limit).await?;
        if hits.is_empty() {
            return Ok("Nothing saved matches.".into());
        }
        Ok(hits
            .iter()
            .map(|(_, e)| format!("#{}: {}", e.id, e.text))
            .collect::<Vec<_>>()
            .join("\n"))
    }
}

pub struct Forget(pub Arc<Memory>);

#[async_trait]
impl Tool for Forget {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "forget",
            "Delete a saved fact by its number, as recall shows it.",
            json!({
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "The fact's number."}
                },
                "required": ["id"]
            }),
        )
    }

    fn access(&self) -> Access {
        Access::Write
    }

    async fn call(&self, args: &Map<String, Json>) -> anyhow::Result<String> {
        let id = usize_arg(args, "id")?
            .ok_or_else(|| anyhow::anyhow!("missing required argument 'id'"))?;
        if self.0.forget(id as u64)? {
            Ok(format!("Forgot #{id}."))
        } else {
            Ok(format!("There is no fact #{id}."))
        }
    }
}
