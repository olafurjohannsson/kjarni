//! Reading, searching, writing and editing files under the working directory.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, bail};
use async_trait::async_trait;
use serde_json::{Map, Value as Json, json};

use crate::agent::tool::{Access, Tool, ToolSpec, first_line, string_arg, usize_arg};

/// The folder relative paths are resolved against.
#[derive(Clone, Debug)]
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `path` as given if absolute, else under the root. A leading `~/` is the
    /// home folder.
    pub fn resolve(&self, path: &str) -> PathBuf {
        let path = path.trim();
        if let Some(rest) = path.strip_prefix("~/")
            && let Some(home) = dirs::home_dir()
        {
            return home.join(rest);
        }
        let p = Path::new(path);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.root.join(p)
        }
    }

    /// How a path is shown back to the model: relative to the root when it is
    /// under it.
    fn display(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| path.display().to_string())
    }
}

/// Folders a search never descends into, besides hidden ones.
const SKIPPED_DIRS: &[&str] = &["target", "node_modules", "__pycache__"];

/// Lines `read_file` returns when no limit is given.
const DEFAULT_READ_LINES: usize = 400;

pub struct ReadFile(pub Arc<Workspace>);

#[async_trait]
impl Tool for ReadFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "read_file",
            "Read a text file. Returns its lines numbered from 1. Long files are returned in parts: use offset and limit to read further.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to the working directory or absolute."},
                    "offset": {"type": "integer", "description": "First line to return, from 1. Default 1."},
                    "limit": {"type": "integer", "description": "Number of lines to return. Default 400."}
                },
                "required": ["path"]
            }),
        )
    }

    fn access(&self) -> Access {
        Access::Read
    }

    async fn call(&self, args: &Map<String, Json>) -> anyhow::Result<String> {
        let path = self.0.resolve(string_arg(args, "path")?);
        let offset = usize_arg(args, "offset")?.unwrap_or(1).max(1);
        let limit = usize_arg(args, "limit")?
            .unwrap_or(DEFAULT_READ_LINES)
            .max(1);
        let bytes = tokio::fs::read(&path)
            .await
            .with_context(|| format!("cannot read {}", self.0.display(&path)))?;
        if bytes.iter().take(8000).any(|&b| b == 0) {
            bail!("{} is a binary file, not text", self.0.display(&path));
        }
        let text = String::from_utf8_lossy(&bytes);
        let lines: Vec<&str> = text.lines().collect();
        if lines.is_empty() {
            return Ok(format!("{} is empty.", self.0.display(&path)));
        }
        let start = (offset - 1).min(lines.len());
        let end = (start + limit).min(lines.len());
        let mut out = String::new();
        for (i, line) in lines[start..end].iter().enumerate() {
            out.push_str(&format!("{}\t{line}\n", start + i + 1));
        }
        if end < lines.len() || start > 0 {
            out.push_str(&format!("[lines {}-{} of {}]", start + 1, end, lines.len()));
        }
        Ok(out)
    }
}

pub struct ListDirectory(pub Arc<Workspace>);

/// Entries a listing shows at most.
const MAX_ENTRIES: usize = 300;

#[async_trait]
impl Tool for ListDirectory {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "list_directory",
            "List the files and folders in a folder. Folders end with /.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Folder path. Default: the working directory."}
                }
            }),
        )
    }

    fn access(&self) -> Access {
        Access::Read
    }

    fn summary(&self, args: &Map<String, Json>) -> String {
        let path = args.get("path").and_then(Json::as_str).unwrap_or(".");
        format!("list_directory {path}")
    }

    async fn call(&self, args: &Map<String, Json>) -> anyhow::Result<String> {
        let path = match args.get("path").and_then(Json::as_str) {
            Some(p) if !p.trim().is_empty() => self.0.resolve(p),
            _ => self.0.root().to_path_buf(),
        };
        let mut reader = tokio::fs::read_dir(&path)
            .await
            .with_context(|| format!("cannot list {}", self.0.display(&path)))?;
        let mut entries = Vec::new();
        while let Some(entry) = reader.next_entry().await? {
            let name = entry.file_name().to_string_lossy().into_owned();
            let meta = entry.metadata().await.ok();
            let line = match meta {
                Some(m) if m.is_dir() => format!("{name}/"),
                Some(m) => format!("{name}  ({})", human_size(m.len())),
                None => name,
            };
            entries.push(line);
        }
        entries.sort_by_key(|e| (!e.ends_with('/'), e.to_lowercase()));
        if entries.is_empty() {
            return Ok(format!("{} is empty.", self.0.display(&path)));
        }
        let total = entries.len();
        entries.truncate(MAX_ENTRIES);
        let mut out = entries.join("\n");
        if total > MAX_ENTRIES {
            out.push_str(&format!("\n[{} more not shown]", total - MAX_ENTRIES));
        }
        Ok(out)
    }
}

fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{size:.1} {}", UNITS[unit])
    }
}

pub struct SearchFiles(pub Arc<Workspace>);

/// Matching lines a search returns at most.
const MAX_MATCHES: usize = 60;

#[async_trait]
impl Tool for SearchFiles {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "search_files",
            "Find lines containing some text in the files under a folder, ignoring case. Returns path:line: text for each match.",
            json!({
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to look for."},
                    "path": {"type": "string", "description": "Folder to search. Default: the working directory."}
                },
                "required": ["text"]
            }),
        )
    }

    fn access(&self) -> Access {
        Access::Read
    }

    async fn call(&self, args: &Map<String, Json>) -> anyhow::Result<String> {
        let needle = string_arg(args, "text")?.to_lowercase();
        if needle.trim().is_empty() {
            bail!("argument 'text' is empty");
        }
        let root = match args.get("path").and_then(Json::as_str) {
            Some(p) if !p.trim().is_empty() => self.0.resolve(p),
            _ => self.0.root().to_path_buf(),
        };
        let workspace = self.0.clone();
        tokio::task::spawn_blocking(move || search(&workspace, &root, &needle)).await?
    }
}

fn search(workspace: &Workspace, root: &Path, needle: &str) -> anyhow::Result<String> {
    if !root.exists() {
        bail!("{} does not exist", workspace.display(root));
    }
    let mut matches = Vec::new();
    let mut more = 0usize;
    // Hidden folders (.git, .venv, an agent's own data) are skipped as
    // ripgrep skips them, unless the search starts in one.
    let walker = walkdir::WalkDir::new(root).into_iter().filter_entry(|e| {
        let name = e.file_name().to_string_lossy();
        !(e.depth() > 0
            && e.file_type().is_dir()
            && (name.starts_with('.') || SKIPPED_DIRS.contains(&name.as_ref())))
    });
    for entry in walker.filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        if meta.len() > 2 * 1024 * 1024 {
            continue;
        }
        let Ok(bytes) = std::fs::read(entry.path()) else {
            continue;
        };
        if bytes.iter().take(8000).any(|&b| b == 0) {
            continue;
        }
        let text = String::from_utf8_lossy(&bytes);
        for (i, line) in text.lines().enumerate() {
            if line.to_lowercase().contains(needle) {
                if matches.len() < MAX_MATCHES {
                    let shown: String = line.trim().chars().take(200).collect();
                    matches.push(format!(
                        "{}:{}: {shown}",
                        workspace.display(entry.path()),
                        i + 1
                    ));
                } else {
                    more += 1;
                }
            }
        }
    }
    if matches.is_empty() {
        return Ok(format!("No matches for '{needle}'."));
    }
    let mut out = matches.join("\n");
    if more > 0 {
        out.push_str(&format!("\n[{more} more matches not shown]"));
    }
    Ok(out)
}

pub struct WriteFile(pub Arc<Workspace>);

#[async_trait]
impl Tool for WriteFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "write_file",
            "Create a file, or replace a file's whole content. Missing folders are created.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path."},
                    "content": {"type": "string", "description": "The complete new content."}
                },
                "required": ["path", "content"]
            }),
        )
    }

    fn access(&self) -> Access {
        Access::Write
    }

    fn summary(&self, args: &Map<String, Json>) -> String {
        let path = args.get("path").and_then(Json::as_str).unwrap_or("?");
        let lines = args
            .get("content")
            .and_then(Json::as_str)
            .map_or(0, |c| c.lines().count());
        format!("write_file {path} ({lines} lines)")
    }

    async fn call(&self, args: &Map<String, Json>) -> anyhow::Result<String> {
        let path = self.0.resolve(string_arg(args, "path")?);
        let content = string_arg(args, "content")?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let existed = path.exists();
        tokio::fs::write(&path, content)
            .await
            .with_context(|| format!("cannot write {}", self.0.display(&path)))?;
        Ok(format!(
            "{} {} ({} bytes).",
            if existed { "Replaced" } else { "Created" },
            self.0.display(&path),
            content.len()
        ))
    }
}

pub struct EditFile(pub Arc<Workspace>);

#[async_trait]
impl Tool for EditFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "edit_file",
            "Replace one exact piece of text in a file. old_text must appear exactly once; include a few surrounding lines to make it unique.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path."},
                    "old_text": {"type": "string", "description": "The text to replace, exactly as it is in the file."},
                    "new_text": {"type": "string", "description": "The text to put in its place."}
                },
                "required": ["path", "old_text", "new_text"]
            }),
        )
    }

    fn access(&self) -> Access {
        Access::Write
    }

    fn summary(&self, args: &Map<String, Json>) -> String {
        let path = args.get("path").and_then(Json::as_str).unwrap_or("?");
        let old = args.get("old_text").and_then(Json::as_str).unwrap_or("");
        format!("edit_file {path}: {}", first_line(old))
    }

    async fn call(&self, args: &Map<String, Json>) -> anyhow::Result<String> {
        let path = self.0.resolve(string_arg(args, "path")?);
        let old = string_arg(args, "old_text")?;
        let new = string_arg(args, "new_text")?;
        if old.is_empty() {
            bail!("old_text is empty; use write_file to create a file");
        }
        let text = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("cannot read {}", self.0.display(&path)))?;
        let count = text.matches(old).count();
        match count {
            0 => bail!(
                "old_text was not found in {}; read the file and copy the text exactly",
                self.0.display(&path)
            ),
            1 => {}
            n => bail!(
                "old_text appears {n} times in {}; include more surrounding lines so it appears once",
                self.0.display(&path)
            ),
        }
        let updated = text.replacen(old, new, 1);
        tokio::fs::write(&path, &updated).await?;
        Ok(format!("Edited {}.", self.0.display(&path)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: Json) -> Map<String, Json> {
        v.as_object().unwrap().clone()
    }

    #[tokio::test]
    async fn reads_a_window_of_lines() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "a\nb\nc\nd\n").unwrap();
        let tool = ReadFile(Arc::new(Workspace::new(dir.path())));
        let out = tool
            .call(&args(json!({"path": "f.txt", "offset": 2, "limit": 2})))
            .await
            .unwrap();
        assert_eq!(out, "2\tb\n3\tc\n[lines 2-3 of 4]");
        let whole = tool.call(&args(json!({"path": "f.txt"}))).await.unwrap();
        assert_eq!(whole, "1\ta\n2\tb\n3\tc\n4\td\n");
    }

    #[tokio::test]
    async fn binary_files_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("x.bin"), [0u8, 1, 2]).unwrap();
        let tool = ReadFile(Arc::new(Workspace::new(dir.path())));
        let err = tool
            .call(&args(json!({"path": "x.bin"})))
            .await
            .unwrap_err();
        assert_eq!(err.to_string(), "x.bin is a binary file, not text");
    }

    #[tokio::test]
    async fn edits_need_exactly_one_match() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.rs");
        std::fs::write(&path, "let a = 1;\nlet b = 1;\n").unwrap();
        let tool = EditFile(Arc::new(Workspace::new(dir.path())));
        let twice = tool
            .call(&args(
                json!({"path": "m.rs", "old_text": "= 1;", "new_text": "= 2;"}),
            ))
            .await
            .unwrap_err();
        assert!(
            twice
                .to_string()
                .starts_with("old_text appears 2 times in m.rs")
        );
        tool.call(&args(
            json!({"path": "m.rs", "old_text": "let b = 1;", "new_text": "let b = 2;"}),
        ))
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "let a = 1;\nlet b = 2;\n"
        );
        let missing = tool
            .call(&args(
                json!({"path": "m.rs", "old_text": "let c", "new_text": ""}),
            ))
            .await
            .unwrap_err();
        assert!(missing.to_string().starts_with("old_text was not found"));
    }

    #[tokio::test]
    async fn search_skips_build_folders_and_reports_paths() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::create_dir_all(dir.path().join("target")).unwrap();
        std::fs::create_dir_all(dir.path().join(".agentdata")).unwrap();
        std::fs::write(
            dir.path().join("src/lib.rs"),
            "fn main() {}\n// TODO: tidy\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("target/out.rs"), "// todo: generated\n").unwrap();
        std::fs::write(dir.path().join(".agentdata/log.jsonl"), "todo\n").unwrap();
        let tool = SearchFiles(Arc::new(Workspace::new(dir.path())));
        let out = tool.call(&args(json!({"text": "todo"}))).await.unwrap();
        assert_eq!(
            out,
            format!(
                "{}:2: // TODO: tidy",
                Path::new("src").join("lib.rs").display()
            )
        );
        // Searching inside a hidden folder by name still works.
        let out = tool
            .call(&args(json!({"text": "todo", "path": ".agentdata"})))
            .await
            .unwrap();
        assert_eq!(
            out,
            format!(
                "{}:1: todo",
                Path::new(".agentdata").join("log.jsonl").display()
            )
        );
    }

    #[tokio::test]
    async fn writing_creates_folders() {
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteFile(Arc::new(Workspace::new(dir.path())));
        let out = tool
            .call(&args(json!({"path": "a/b/c.txt", "content": "hi"})))
            .await
            .unwrap();
        assert_eq!(
            out,
            format!("Created {} (2 bytes).", Path::new("a/b/c.txt").display())
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a/b/c.txt")).unwrap(),
            "hi"
        );
    }
}
