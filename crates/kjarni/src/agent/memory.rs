//! Long-term memory: short facts the agent keeps between sessions, found
//! again by meaning.
//!
//! Each fact is a line of `memory.jsonl` with its embedding. Recall ranks by
//! cosine similarity to the query, or by shared words when no embedder is
//! loaded.

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Turns text into an embedding.
#[async_trait]
pub trait Embed: Send + Sync {
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>>;
}

#[async_trait]
impl Embed for crate::embedder::Embedder {
    async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        crate::embedder::Embedder::embed(self, text)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// One remembered fact.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MemoryEntry {
    pub id: u64,
    pub text: String,
    /// Seconds since the Unix epoch.
    pub created: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

/// The agent's long-term memory, kept in one file.
pub struct Memory {
    path: PathBuf,
    entries: Mutex<Vec<MemoryEntry>>,
    embedder: Option<Arc<dyn Embed>>,
}

impl Memory {
    /// The memory kept in `dir/memory.jsonl`, created when first written.
    pub fn open(dir: &Path, embedder: Option<Arc<dyn Embed>>) -> anyhow::Result<Self> {
        let path = dir.join("memory.jsonl");
        let mut entries = Vec::new();
        if path.exists() {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                // A line that does not parse is skipped, not fatal: the rest
                // of the memory is still worth having.
                if let Ok(entry) = serde_json::from_str::<MemoryEntry>(line) {
                    entries.push(entry);
                }
            }
        }
        Ok(Self {
            path,
            entries: Mutex::new(entries),
            embedder,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> usize {
        self.entries.lock().map_or(0, |e| e.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// All facts, oldest first.
    pub fn entries(&self) -> Vec<MemoryEntry> {
        self.entries.lock().map(|e| e.clone()).unwrap_or_default()
    }

    /// The `n` newest facts, oldest of them first.
    pub fn recent(&self, n: usize) -> Vec<MemoryEntry> {
        let entries = self.entries();
        let start = entries.len().saturating_sub(n);
        entries[start..].to_vec()
    }

    /// Keeps `text`. A fact already kept, word for word, is not kept twice.
    pub async fn remember(&self, text: &str) -> anyhow::Result<MemoryEntry> {
        let text = text.trim();
        anyhow::ensure!(!text.is_empty(), "there is nothing to remember");
        if let Some(existing) = self.entries().into_iter().find(|e| e.text == text) {
            return Ok(existing);
        }
        let embedding = match &self.embedder {
            Some(e) => Some(e.embed(text).await?),
            None => None,
        };
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let entry = {
            let mut entries = self
                .entries
                .lock()
                .map_err(|_| anyhow::anyhow!("memory lock poisoned"))?;
            let id = entries.iter().map(|e| e.id).max().map_or(1, |m| m + 1);
            let entry = MemoryEntry {
                id,
                text: text.to_string(),
                created,
                embedding,
            };
            entries.push(entry.clone());
            entry
        };
        self.append(&entry)?;
        Ok(entry)
    }

    /// Up to `k` facts most related to `query`, best first, with their scores.
    pub async fn recall(&self, query: &str, k: usize) -> anyhow::Result<Vec<(f32, MemoryEntry)>> {
        let entries = self.entries();
        if entries.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let query_embedding = match &self.embedder {
            Some(e) => Some(e.embed(query).await?),
            None => None,
        };
        let query_words = words(query);
        let mut scored: Vec<(f32, MemoryEntry)> = entries
            .into_iter()
            .map(|entry| {
                let score = match (&query_embedding, &entry.embedding) {
                    (Some(q), Some(e)) if q.len() == e.len() => {
                        crate::utils::cosine_similarity(q, e)
                    }
                    _ => keyword_score(&query_words, &entry.text),
                };
                (score, entry)
            })
            .filter(|(score, _)| *score > 0.0)
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(k);
        Ok(scored)
    }

    /// Removes fact `id`. Returns whether there was one.
    pub fn forget(&self, id: u64) -> anyhow::Result<bool> {
        let removed = {
            let mut entries = self
                .entries
                .lock()
                .map_err(|_| anyhow::anyhow!("memory lock poisoned"))?;
            let before = entries.len();
            entries.retain(|e| e.id != id);
            before != entries.len()
        };
        if removed {
            self.rewrite()?;
        }
        Ok(removed)
    }

    fn append(&self, entry: &MemoryEntry) -> anyhow::Result<()> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("opening {}", self.path.display()))?;
        writeln!(file, "{}", serde_json::to_string(entry)?)?;
        Ok(())
    }

    fn rewrite(&self) -> anyhow::Result<()> {
        let entries = self.entries();
        let mut text = String::new();
        for entry in &entries {
            text.push_str(&serde_json::to_string(entry)?);
            text.push('\n');
        }
        let tmp = self.path.with_extension("jsonl.tmp");
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }
}

/// Lowercase words of three letters or more.
fn words(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 3)
        .map(str::to_lowercase)
        .collect()
}

/// The share of the query's words that `text` contains.
fn keyword_score(query: &HashSet<String>, text: &str) -> f32 {
    if query.is_empty() {
        return 0.0;
    }
    let text = words(text);
    query.iter().filter(|w| text.contains(*w)).count() as f32 / query.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An embedder that maps text to counts of a few marker words, so that
    /// similarity is predictable.
    struct MarkerEmbedder;

    #[async_trait]
    impl Embed for MarkerEmbedder {
        async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
            let t = text.to_lowercase();
            Ok(["coffee", "rust", "dog"]
                .iter()
                .map(|w| t.matches(w).count() as f32 + 0.01)
                .collect())
        }
    }

    #[tokio::test]
    async fn remembers_across_reopening_and_recalls_by_meaning() {
        let dir = tempfile::tempdir().unwrap();
        {
            let memory = Memory::open(dir.path(), Some(Arc::new(MarkerEmbedder))).unwrap();
            memory
                .remember("The user drinks coffee black.")
                .await
                .unwrap();
            memory
                .remember("The user's dog is called Loki.")
                .await
                .unwrap();
            memory.remember("The user writes Rust.").await.unwrap();
            // Kept once.
            memory.remember("The user writes Rust.").await.unwrap();
        }
        let memory = Memory::open(dir.path(), Some(Arc::new(MarkerEmbedder))).unwrap();
        assert_eq!(memory.len(), 3);
        let hits = memory
            .recall("what pet does the dog owner have", 1)
            .await
            .unwrap();
        assert_eq!(hits[0].1.text, "The user's dog is called Loki.");
    }

    #[tokio::test]
    async fn without_an_embedder_recall_matches_words() {
        let dir = tempfile::tempdir().unwrap();
        let memory = Memory::open(dir.path(), None).unwrap();
        memory.remember("Deploys go out on Fridays.").await.unwrap();
        memory
            .remember("The staging server is called kestrel.")
            .await
            .unwrap();
        let hits = memory.recall("which server is staging?", 5).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].1.text, "The staging server is called kestrel.");
    }

    #[tokio::test]
    async fn forgetting_removes_the_fact_from_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let memory = Memory::open(dir.path(), None).unwrap();
        let a = memory.remember("first").await.unwrap();
        memory.remember("second").await.unwrap();
        assert!(memory.forget(a.id).unwrap());
        assert!(!memory.forget(a.id).unwrap());
        let reopened = Memory::open(dir.path(), None).unwrap();
        let texts: Vec<_> = reopened.entries().into_iter().map(|e| e.text).collect();
        assert_eq!(texts, ["second"]);
        // A new id follows the highest kept.
        assert_eq!(reopened.remember("third").await.unwrap().id, 3);
    }
}
