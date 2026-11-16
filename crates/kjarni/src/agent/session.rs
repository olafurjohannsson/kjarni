//! Conversations saved to disk as they happen, so one can be picked up later.
//!
//! A session is a JSON Lines file: one line per message, and a `summary` line
//! wherever earlier messages were folded into a summary to save context.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use kjarni_transformers::{Message, Role};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Entry {
    Message {
        role: String,
        content: String,
    },
    /// The first `replaces` messages so far were folded into `text`.
    Summary {
        text: String,
        replaces: usize,
    },
}

/// A saved conversation.
#[derive(Clone, Debug)]
pub struct Session {
    id: String,
    path: PathBuf,
}

/// What a listing shows of a session.
#[derive(Clone, Debug)]
pub struct SessionInfo {
    pub id: String,
    /// Seconds since the Unix epoch.
    pub modified: u64,
    /// The first thing the user said, shortened.
    pub title: String,
}

/// A conversation read back from disk.
#[derive(Clone, Debug, Default)]
pub struct Restored {
    pub messages: Vec<Message>,
    pub summary: Option<String>,
}

impl Session {
    /// A new, empty session in `dir`, named for the time it was created.
    pub fn create(dir: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
        let base = timestamp_id(now.as_secs());
        let mut id = base.clone();
        let mut n = 1;
        while dir.join(format!("{id}.jsonl")).exists() {
            n += 1;
            id = format!("{base}-{n}");
        }
        let path = dir.join(format!("{id}.jsonl"));
        std::fs::File::create(&path)?;
        Ok(Self { id, path })
    }

    /// Session `id` in `dir`.
    pub fn open(dir: &Path, id: &str) -> anyhow::Result<Self> {
        let path = dir.join(format!("{id}.jsonl"));
        anyhow::ensure!(
            path.exists(),
            "there is no session '{id}' in {}",
            dir.display()
        );
        Ok(Self {
            id: id.to_string(),
            path,
        })
    }

    /// The session changed most recently.
    pub fn latest(dir: &Path) -> anyhow::Result<Option<Self>> {
        Ok(Self::list(dir)?.into_iter().next().map(|info| Self {
            path: dir.join(format!("{}.jsonl", info.id)),
            id: info.id,
        }))
    }

    /// The sessions in `dir`, most recently changed first.
    pub fn list(dir: &Path) -> anyhow::Result<Vec<SessionInfo>> {
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut sessions = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let Some(id) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let modified = std::fs::metadata(&path)?
                .modified()?
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            let restored = Self {
                id: id.to_string(),
                path: path.clone(),
            }
            .load()
            .unwrap_or_default();
            let title = restored
                .messages
                .iter()
                .find(|m| m.role == Role::User)
                .map(|m| crate::agent::tool::first_line(&m.content))
                .or(restored
                    .summary
                    .as_deref()
                    .map(crate::agent::tool::first_line))
                .unwrap_or_default();
            if title.is_empty() {
                continue;
            }
            sessions.push(SessionInfo {
                id: id.to_string(),
                modified,
                title,
            });
        }
        sessions.sort_by(|a, b| b.modified.cmp(&a.modified).then(b.id.cmp(&a.id)));
        Ok(sessions)
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn append_message(&self, message: &Message) -> anyhow::Result<()> {
        self.append(&Entry::Message {
            role: message.role.to_string(),
            content: message.content.clone(),
        })
    }

    pub(crate) fn append_summary(&self, text: &str, replaces: usize) -> anyhow::Result<()> {
        self.append(&Entry::Summary {
            text: text.to_string(),
            replaces,
        })
    }

    fn append(&self, entry: &Entry) -> anyhow::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.path)
            .with_context(|| format!("opening {}", self.path.display()))?;
        writeln!(file, "{}", serde_json::to_string(entry)?)?;
        Ok(())
    }

    /// The conversation as saved: its messages since the last summary, and
    /// that summary.
    pub fn load(&self) -> anyhow::Result<Restored> {
        let text = std::fs::read_to_string(&self.path)
            .with_context(|| format!("reading {}", self.path.display()))?;
        let mut restored = Restored::default();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            // A line cut short by a crash is skipped.
            let Ok(entry) = serde_json::from_str::<Entry>(line) else {
                continue;
            };
            match entry {
                Entry::Message { role, content } => {
                    let role = match role.as_str() {
                        "user" => Role::User,
                        "assistant" => Role::Assistant,
                        "tool" => Role::Tool,
                        "system" => Role::System,
                        _ => continue,
                    };
                    restored.messages.push(Message { role, content });
                }
                Entry::Summary { text, replaces } => {
                    let n = replaces.min(restored.messages.len());
                    restored.messages.drain(..n);
                    restored.summary = Some(text);
                }
            }
        }
        Ok(restored)
    }
}

/// `2026-11-15_213005`, in UTC.
fn timestamp_id(secs: u64) -> String {
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    let rem = secs % 86_400;
    format!(
        "{y:04}-{m:02}-{d:02}_{:02}{:02}{:02}",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Today's date in UTC, `2026-11-15`.
pub(crate) fn today() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

/// The calendar date `days` after 1970-01-01 (Howard Hinnant's algorithm).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_are_civil() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // 2026-11-15 is day 20772.
        assert_eq!(civil_from_days(20_772), (2026, 11, 15));
        assert_eq!(civil_from_days(11_016), (2000, 2, 29));
        assert_eq!(
            timestamp_id(20_772 * 86_400 + 21 * 3600 + 30 * 60 + 5),
            "2026-11-15_213005"
        );
    }

    #[test]
    fn a_session_reads_back_with_its_summary_applied() {
        let dir = tempfile::tempdir().unwrap();
        let session = Session::create(dir.path()).unwrap();
        session
            .append_message(&Message::user("Fix the build"))
            .unwrap();
        session
            .append_message(&Message::assistant("<tool_call>...</tool_call>"))
            .unwrap();
        session
            .append_message(&Message::tool("exit code 0"))
            .unwrap();
        session
            .append_summary("The user asked to fix the build; it passes.", 3)
            .unwrap();
        session.append_message(&Message::user("Thanks")).unwrap();

        let restored = Session::open(dir.path(), session.id())
            .unwrap()
            .load()
            .unwrap();
        assert_eq!(
            restored.summary.as_deref(),
            Some("The user asked to fix the build; it passes.")
        );
        assert_eq!(restored.messages.len(), 1);
        assert_eq!(restored.messages[0].content, "Thanks");

        let listed = Session::list(dir.path()).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].title, "Thanks");
        assert_eq!(
            Session::latest(dir.path()).unwrap().unwrap().id(),
            session.id()
        );
    }

    #[test]
    fn empty_sessions_are_not_listed() {
        let dir = tempfile::tempdir().unwrap();
        Session::create(dir.path()).unwrap();
        assert!(Session::list(dir.path()).unwrap().is_empty());
    }
}
