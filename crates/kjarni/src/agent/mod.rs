//! An agent: a chat model that calls tools, remembers between conversations,
//! and keeps its conversations on disk.
//!
//! ```ignore
//! use std::sync::Arc;
//! use kjarni::agent::{Agent, AgentConfig, NoEvents, tools};
//! use kjarni::chat::Chat;
//!
//! let chat = Chat::new("qwen3.5-4b").await?;
//! let workspace = Arc::new(tools::Workspace::new("."));
//! let mut agent = Agent::new(Arc::new(chat), tools::builtin(workspace, None), AgentConfig::default());
//! let reply = agent.send("What is in Cargo.toml?", &mut NoEvents).await?;
//! println!("{}", reply.text);
//! ```
//!
//! A front end implements [`AgentEvents`] to show text as it is generated,
//! each tool call and its result, and to ask the user before a call that
//! writes, runs a program or reaches the network.

mod agent;
pub mod memory;
pub mod model;
pub mod notes;
pub mod session;
pub mod tool;
pub mod tools;

pub use agent::{
    Agent, AgentConfig, AgentEvents, AgentReply, Approval, DEFAULT_INSTRUCTIONS, NoEvents,
    ToolOutcome,
};
pub use memory::{Embed, Memory, MemoryEntry};
pub use model::{AgentModel, TextStream};
pub use notes::{ProjectNotes, project_notes};
pub use session::{Restored, Session, SessionInfo};
pub use tool::{
    Access, ParsedReply, Tool, ToolCall, ToolSpec, displayable_prefix_len, parse_tool_calls,
};

/// Where the agent keeps its memory and sessions: `$KJARNI_AGENT_DIR`, or
/// `kjarni/agent` in the user's data folder.
pub fn data_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("KJARNI_AGENT_DIR") {
        return dir.into();
    }
    dirs::data_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("kjarni")
        .join("agent")
}

#[cfg(test)]
mod tests;
