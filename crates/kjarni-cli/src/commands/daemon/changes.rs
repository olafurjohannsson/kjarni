//! The files a conversation changed, each with its content from before the
//! conversation first wrote it and after the last write, saved beside the
//! conversation so the list outlives the daemon.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::Result;
use kjarni::agent::{FileChange, FileState};
use serde::{Deserialize, Serialize};
use serde_json::{Value as Json, json};
use similar::{ChangeTag, TextDiff};

/// Unchanged lines shown around each changed part of a diff.
const CONTEXT_LINES: usize = 3;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum Content {
    Missing,
    Text { text: String },
    Unreadable,
}

impl From<&FileState> for Content {
    fn from(state: &FileState) -> Self {
        match state {
            FileState::Missing => Content::Missing,
            FileState::Text(text) => Content::Text { text: text.clone() },
            FileState::Unreadable => Content::Unreadable,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Changed {
    path: PathBuf,
    /// Before the conversation's first write.
    before: Content,
    /// After its last write.
    after: Content,
    writes: u32,
    /// Seconds since the epoch of the last write.
    updated: u64,
}

pub struct Changes {
    file: PathBuf,
    files: Mutex<Vec<Changed>>,
}

impl Changes {
    /// The changes saved in `file`; none when it is missing or unreadable.
    pub fn open(file: PathBuf) -> Self {
        let files = std::fs::read_to_string(&file)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        Self {
            file,
            files: Mutex::new(files),
        }
    }

    /// Adds a write, keeping the content from before the first write to the
    /// same file.
    pub fn record(&self, change: &FileChange) -> Result<()> {
        let mut files = self.files.lock().expect("changes lock");
        let updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        match files.iter_mut().find(|f| f.path == change.path) {
            Some(f) => {
                f.after = (&change.after).into();
                f.writes += 1;
                f.updated = updated;
            }
            None => files.push(Changed {
                path: change.path.clone(),
                before: (&change.before).into(),
                after: (&change.after).into(),
                writes: 1,
                updated,
            }),
        }
        let tmp = self.file.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string(&*files)?)?;
        std::fs::rename(&tmp, &self.file)?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.files.lock().expect("changes lock").len()
    }

    /// The changed files in the order first written, each with its diff.
    /// Paths under `workdir` are shown relative to it.
    pub fn list(&self, workdir: &Path) -> Json {
        let files = self.files.lock().expect("changes lock");
        Json::Array(files.iter().map(|f| describe(f, workdir)).collect())
    }
}

fn describe(f: &Changed, workdir: &Path) -> Json {
    let shown = f
        .path
        .strip_prefix(workdir)
        .unwrap_or(&f.path)
        .display()
        .to_string();
    let status = match (&f.before, &f.after) {
        (Content::Missing, Content::Missing) => "unchanged",
        (Content::Missing, _) => "created",
        (_, Content::Missing) => "deleted",
        (a, b) if a == b => "unchanged",
        _ => "modified",
    };
    let mut out = json!({
        "path": shown,
        "full_path": f.path.display().to_string(),
        "status": status,
        "writes": f.writes,
        "updated": f.updated,
    });
    let text = |c: &Content| match c {
        Content::Missing => Some(String::new()),
        Content::Text { text } => Some(text.clone()),
        Content::Unreadable => None,
    };
    match (text(&f.before), text(&f.after)) {
        (Some(before), Some(after)) => {
            let (hunks, added, removed) = diff(&before, &after);
            out["hunks"] = hunks;
            out["added"] = added.into();
            out["removed"] = removed.into();
        }
        _ => out["unreadable"] = true.into(),
    }
    out
}

/// Line hunks from `before` to `after` with a few lines of context, and the
/// lines added and removed.
fn diff(before: &str, after: &str) -> (Json, usize, usize) {
    let d = TextDiff::from_lines(before, after);
    let (mut added, mut removed) = (0, 0);
    let hunks: Vec<Json> = d
        .grouped_ops(CONTEXT_LINES)
        .iter()
        .map(|group| {
            let mut lines = Vec::new();
            for op in group {
                for change in d.iter_changes(op) {
                    let sign = match change.tag() {
                        ChangeTag::Insert => {
                            added += 1;
                            "+"
                        }
                        ChangeTag::Delete => {
                            removed += 1;
                            "-"
                        }
                        ChangeTag::Equal => " ",
                    };
                    let text = change.value();
                    lines.push(json!({
                        "sign": sign,
                        "text": text.strip_suffix('\n').unwrap_or(text),
                    }));
                }
            }
            let first = &group[0];
            json!({
                "old_start": first.old_range().start + 1,
                "new_start": first.new_range().start + 1,
                "lines": lines,
            })
        })
        .collect();
    (Json::Array(hunks), added, removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn change(path: &Path, before: FileState, after: FileState) -> FileChange {
        FileChange {
            path: path.to_path_buf(),
            before,
            after,
        }
    }

    fn text(s: &str) -> FileState {
        FileState::Text(s.into())
    }

    #[test]
    fn later_writes_keep_the_first_before_and_the_list_survives_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().join("work");
        let saved = dir.path().join("c1.changes.json");
        let changes = Changes::open(saved.clone());
        let plan = work.join("plan.md");
        changes
            .record(&change(&plan, text("a\nb\nc\n"), text("a\nB\nc\n")))
            .unwrap();
        changes
            .record(&change(&plan, text("a\nB\nc\n"), text("a\nB\nc\nd\n")))
            .unwrap();
        changes
            .record(&change(
                &work.join("new.txt"),
                FileState::Missing,
                text("hi\n"),
            ))
            .unwrap();

        let list = Changes::open(saved).list(&work);

        assert_eq!(list[0]["path"], "plan.md");
        assert_eq!(list[0]["status"], "modified");
        assert_eq!(list[0]["writes"], 2);
        assert_eq!(
            (list[0]["added"].as_u64(), list[0]["removed"].as_u64()),
            (Some(2), Some(1))
        );
        assert_eq!(
            list[0]["hunks"],
            json!([{
                "old_start": 1,
                "new_start": 1,
                "lines": [
                    {"sign": " ", "text": "a"},
                    {"sign": "-", "text": "b"},
                    {"sign": "+", "text": "B"},
                    {"sign": " ", "text": "c"},
                    {"sign": "+", "text": "d"},
                ],
            }])
        );
        assert_eq!(list[1]["path"], "new.txt");
        assert_eq!(list[1]["status"], "created");
        assert_eq!(list[1]["added"], 1);
    }

    #[test]
    fn far_apart_edits_are_separate_hunks_with_three_lines_around_each() {
        let before: String = (1..=20).map(|i| format!("{i}\n")).collect();
        let after: String = (1..=20)
            .map(|i| match i {
                2 => "two\n".to_string(),
                19 => "nineteen\n".to_string(),
                _ => format!("{i}\n"),
            })
            .collect();
        let (hunks, added, removed) = diff(&before, &after);
        assert_eq!((added, removed), (2, 2));
        assert_eq!(hunks.as_array().unwrap().len(), 2);
        assert_eq!(hunks[1]["old_start"], 16);
        assert_eq!(hunks[1]["lines"].as_array().unwrap().len(), 6);
    }

    #[test]
    fn a_file_written_back_to_what_it_was_is_unchanged_and_an_unreadable_one_has_no_diff() {
        let dir = tempfile::tempdir().unwrap();
        let changes = Changes::open(dir.path().join("c.changes.json"));
        let a = dir.path().join("a.txt");
        changes.record(&change(&a, text("x"), text("y"))).unwrap();
        changes.record(&change(&a, text("y"), text("x"))).unwrap();
        let b = dir.path().join("b.bin");
        changes
            .record(&change(&b, FileState::Unreadable, text("now text")))
            .unwrap();

        let list = changes.list(dir.path());

        assert_eq!(list[0]["status"], "unchanged");
        assert_eq!(list[0]["hunks"], json!([]));
        assert_eq!(list[1]["status"], "modified");
        assert_eq!(list[1]["unreadable"], true);
        assert!(list[1].get("hunks").is_none());
    }
}
