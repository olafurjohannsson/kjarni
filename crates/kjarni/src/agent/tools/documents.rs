//! Finding passages in the user's documents by meaning.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value as Json, json};

use crate::agent::tool::{Access, Tool, ToolSpec, string_arg, usize_arg};
use crate::agent::tools::files::Workspace;
use crate::folder_search::FolderSearch;

/// The cross-encoder that orders the candidates a search finds.
pub const RERANK_MODEL: &str = "minilm-l6-v2-cross-encoder";

/// Passages returned when the model gives no limit, and at most.
const DEFAULT_RESULTS: usize = 5;
const MAX_RESULTS: usize = 20;

/// Characters of each passage shown to the model.
const PASSAGE_CHARS: usize = 700;

/// Searches the documents under a folder by meaning and keywords, reranked,
/// indexing the folder on its first search.
pub struct SearchDocuments {
    workspace: Arc<Workspace>,
    search: Arc<FolderSearch>,
}

impl SearchDocuments {
    /// Searches with `search`, so indexes and loaded models can be shared.
    pub fn new(workspace: Arc<Workspace>, search: Arc<FolderSearch>) -> Self {
        Self { workspace, search }
    }

    /// Searches with [`FolderSearch::new`] and [`RERANK_MODEL`].
    pub fn reranked(workspace: Arc<Workspace>) -> Self {
        Self::new(
            workspace,
            Arc::new(FolderSearch::new().reranker(RERANK_MODEL)),
        )
    }
}

#[async_trait]
impl Tool for SearchDocuments {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "search_documents",
            "Find the passages in the user's notes, documents and code that best answer a question, by meaning as well as by words. \
             Use it for questions about what the files say; use search_files for an exact name or string. \
             Reads text and code files (md, txt, html, json, source code and the like), not PDF or Word. \
             The first search of a folder indexes it, which can take a minute; later searches are fast.",
            json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to find, as a question or a description."},
                    "path": {"type": "string", "description": "Folder to search, subfolders included. Default: the working directory."},
                    "limit": {"type": "integer", "description": "Passages to return. Default 5."}
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
        let folder = match args.get("path") {
            Some(Json::String(p)) if !p.trim().is_empty() => self.workspace.resolve(p),
            _ => self.workspace.root().to_path_buf(),
        };
        let limit = usize_arg(args, "limit")?
            .unwrap_or(DEFAULT_RESULTS)
            .clamp(1, MAX_RESULTS);

        let found = self.search.search(&folder, query, limit).await?;
        let root = self
            .workspace
            .root()
            .canonicalize()
            .unwrap_or_else(|_| self.workspace.root().to_path_buf());

        let mut out = String::new();
        if let Some(update) = &found.update {
            let folder = if found.indexed_folder == root {
                "the working folder".into()
            } else {
                shown(&root, &found.indexed_folder)
            };
            out.push_str(&format!("({}.)\n\n", update.describe(&folder)));
        }
        if found.results.is_empty() {
            out.push_str("No passages found.");
        }
        for (i, hit) in found.results.iter().enumerate() {
            out.push_str(&format!(
                "{}. {}\n{}\n\n",
                i + 1,
                shown(&root, &hit.source).display(),
                passage(&hit.text)
            ));
        }
        Ok(out.trim_end().to_string())
    }
}

/// `path` relative to the working folder when it is inside it.
fn shown(root: &Path, path: &Path) -> std::path::PathBuf {
    match path.strip_prefix(root) {
        Ok(rel) => rel.to_path_buf(),
        Err(_) => path.to_path_buf(),
    }
}

/// A passage trimmed to [`PASSAGE_CHARS`].
fn passage(text: &str) -> String {
    let text = text.trim();
    if text.chars().count() <= PASSAGE_CHARS {
        return text.to_string();
    }
    let cut: String = text.chars().take(PASSAGE_CHARS).collect();
    format!("{}…", cut.trim_end())
}
