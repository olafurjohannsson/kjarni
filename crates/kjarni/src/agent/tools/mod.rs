//! The tools the agent comes with.

pub mod documents;
pub mod files;
pub mod memory;
pub mod shell;
pub mod web;

use std::sync::Arc;

pub use documents::SearchDocuments;
pub use files::{EditFile, ListDirectory, ReadFile, SearchFiles, Workspace, WriteFile};
pub use memory::{Forget, Recall, Remember};
pub use shell::RunCommand;
pub use web::FetchUrl;

use super::memory::Memory;
use super::tool::Tool;

/// The built-in tools, working in `workspace`, with the memory tools when
/// there is a memory. `search_documents` loads its models on its first call.
pub fn builtin(workspace: Arc<Workspace>, memory: Option<Arc<Memory>>) -> Vec<Arc<dyn Tool>> {
    let mut tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ListDirectory(workspace.clone())),
        Arc::new(ReadFile(workspace.clone())),
        Arc::new(SearchFiles(workspace.clone())),
        Arc::new(SearchDocuments::reranked(workspace.clone())),
        Arc::new(WriteFile(workspace.clone())),
        Arc::new(EditFile(workspace.clone())),
        Arc::new(RunCommand(workspace)),
        Arc::new(FetchUrl::new()),
    ];
    if let Some(memory) = memory {
        tools.push(Arc::new(Remember(memory.clone())));
        tools.push(Arc::new(Recall(memory.clone())));
        tools.push(Arc::new(Forget(memory)));
    }
    tools
}
