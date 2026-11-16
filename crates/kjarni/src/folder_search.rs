//! Searching a folder of documents by meaning, with the index kept out of
//! sight: the first search of a folder indexes it into Kjarni's cache, and
//! later searches embed only the files that changed since.
//!
//! A folder inside one already indexed is searched through that index, so
//! indexing `~/notes` once covers `~/notes/work` too. A folder with more
//! documents than [`FolderSearch::max_documents`] is refused with the
//! subfolders that hold them, so a caller (often a model) can pick a smaller
//! one; indexing a whole home folder would take hours and fill the cache.
//!
//! ```no_run
//! # async fn demo() -> anyhow::Result<()> {
//! use kjarni::folder_search::FolderSearch;
//!
//! let search = FolderSearch::new().reranker("minilm-l6-v2-cross-encoder");
//! let found = search.search("/home/me/notes".as_ref(), "when does the ferry go", 5).await?;
//! for hit in &found.results {
//!     println!("{} {}", hit.source.display(), hit.text);
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, bail};
use kjarni_rag::TEXT_EXTENSIONS;
use tokio::sync::{Mutex, OnceCell};

use crate::indexer::{Indexer, SyncStats};
use crate::searcher::{SearchOptions, Searcher};
use crate::{SearchMode, registry};

/// The embedding model folders are indexed with. Small and fast, which
/// matters more here than the last point of accuracy: a folder is indexed
/// while someone waits for an answer.
pub const DEFAULT_MODEL: &str = "minilm-l6-v2";

/// Documents a folder may hold before a search asks for a smaller folder.
pub const DEFAULT_MAX_DOCUMENTS: usize = 10_000;

/// Never indexed: build output, dependencies and virtual environments, which
/// in a code repository outnumber the sources many times over. Hidden folders
/// such as .git are skipped as well.
pub const EXCLUDED: &[&str] = &[
    "target",
    "node_modules",
    "dist",
    "build",
    "out",
    "bin",
    "obj",
    "vendor",
    "venv",
    "__pycache__",
    "*.min.js",
];

/// Files larger than this are not indexed, as the indexer's own default.
const MAX_FILE_BYTES: u64 = 10 * 1024 * 1024;

/// Searches folders, indexing each on first use. Models load on first use too.
pub struct FolderSearch {
    model: String,
    reranker: Option<String>,
    cache_dir: PathBuf,
    max_documents: usize,
    indexer: OnceCell<Indexer>,
    searcher: OnceCell<Searcher>,
    reranking_searcher: OnceCell<Searcher>,
    /// One index update at a time, so two searches never write one index.
    updating: Mutex<()>,
}

/// One passage found.
#[derive(Clone, Debug)]
pub struct FolderHit {
    /// The file it came from, as an absolute path.
    pub source: PathBuf,
    pub text: String,
    pub score: f32,
}

/// What a search found, and what it did to the index first.
#[derive(Clone, Debug)]
pub struct FolderResults {
    pub results: Vec<FolderHit>,
    /// The folder whose index was searched: the folder itself, or one it is in.
    pub indexed_folder: PathBuf,
    /// Set when the index was built or brought up to date for this search.
    pub update: Option<IndexUpdate>,
}

/// An index built or updated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IndexUpdate {
    Built {
        files: usize,
    },
    Updated {
        added: usize,
        changed: usize,
        removed: usize,
    },
}

impl IndexUpdate {
    /// The update in a sentence, naming the folder.
    pub fn describe(&self, folder: &Path) -> String {
        match self {
            IndexUpdate::Built { files } => {
                format!("Indexed {} ({files} files)", folder.display())
            }
            IndexUpdate::Updated {
                added,
                changed,
                removed,
            } => format!(
                "Updated the index of {}: {added} new, {changed} changed, {removed} removed",
                folder.display()
            ),
        }
    }
}

impl Default for FolderSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl FolderSearch {
    /// Indexes with [`DEFAULT_MODEL`] into `folder-indexes` in the model cache,
    /// with no reranker.
    pub fn new() -> Self {
        Self {
            model: DEFAULT_MODEL.to_string(),
            reranker: None,
            cache_dir: registry::cache_dir().join("folder-indexes"),
            max_documents: DEFAULT_MAX_DOCUMENTS,
            indexer: OnceCell::new(),
            searcher: OnceCell::new(),
            reranking_searcher: OnceCell::new(),
            updating: Mutex::new(()),
        }
    }

    /// Reorders the best candidates with this cross-encoder, by default; it
    /// loads on the first search that reranks.
    pub fn reranker(mut self, model: &str) -> Self {
        self.reranker = Some(model.to_string());
        self
    }

    /// Where indexes are kept.
    pub fn cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = dir.into();
        self
    }

    /// Documents a folder may hold before a search asks for a smaller one.
    pub fn max_documents(mut self, n: usize) -> Self {
        self.max_documents = n;
        self
    }

    /// The `top_k` passages in `folder` and its subfolders best matching
    /// `query`, by meaning and keywords together, reranked when a reranker is
    /// set.
    pub async fn search(
        &self,
        folder: &Path,
        query: &str,
        top_k: usize,
    ) -> anyhow::Result<FolderResults> {
        self.search_with(
            folder,
            query,
            top_k,
            SearchMode::Hybrid,
            self.reranker.is_some(),
        )
        .await
    }

    /// [`search`](Self::search) with the mode and reranking chosen per call.
    /// `rerank` does nothing without a reranker.
    pub async fn search_with(
        &self,
        folder: &Path,
        query: &str,
        top_k: usize,
        mode: SearchMode,
        rerank: bool,
    ) -> anyhow::Result<FolderResults> {
        let folder = folder
            .canonicalize()
            .with_context(|| format!("cannot open {}", folder.display()))?;
        if !folder.is_dir() {
            bail!("{} is not a folder", folder.display());
        }

        let (indexed_folder, index, update) = self.index_for(&folder).await?;
        let rerank = rerank && self.reranker.is_some();
        let mut options = SearchOptions::new().mode(mode).top_k(top_k).rerank(rerank);
        if indexed_folder != folder {
            options = options.source(&within(&folder));
        }
        let searcher = self.searcher(rerank).await?;
        let index = index.to_string_lossy().into_owned();
        let results = searcher
            .search_with_options(&index, query, &options)
            .await?
            .into_iter()
            .map(|r| FolderHit {
                source: PathBuf::from(r.metadata.get("source").cloned().unwrap_or_default()),
                text: r.text,
                score: r.score,
            })
            .collect();
        Ok(FolderResults {
            results,
            indexed_folder,
            update,
        })
    }

    /// The index to search for `folder` (canonical): that of the nearest
    /// indexed folder containing it, else its own, built now. Brought up to
    /// date with the files first.
    async fn index_for(
        &self,
        folder: &Path,
    ) -> anyhow::Result<(PathBuf, PathBuf, Option<IndexUpdate>)> {
        let indexed = folder
            .ancestors()
            .skip(1)
            .find(|a| within_is_literal(folder) && self.index_path(a).exists())
            .unwrap_or(folder)
            .to_path_buf();
        let index = self.index_path(&indexed);
        let stamp_file = index.with_extension("stamp");

        let _one_at_a_time = self.updating.lock().await;
        let scan = scan(&indexed, &self.model);
        let current = std::fs::read_to_string(&stamp_file).ok();
        if index.exists() && current.as_deref() == Some(scan.stamp.as_str()) {
            return Ok((indexed, index, None));
        }
        if scan.documents > self.max_documents {
            bail!(too_many(&indexed, &scan, self.max_documents));
        }

        let indexer = self.indexer().await?;
        std::fs::create_dir_all(&self.cache_dir)?;
        let index_str = index.to_string_lossy().into_owned();
        let folder_str = indexed.to_string_lossy().into_owned();
        let existed = index.exists();
        let stats: SyncStats = match indexer.sync(&index_str, &[folder_str.as_str()]).await {
            Ok(stats) => stats,
            // An index this version cannot update, such as one from another
            // model, is rebuilt.
            Err(_) if existed => {
                std::fs::remove_dir_all(&index).context("removing the outdated index")?;
                indexer.sync(&index_str, &[folder_str.as_str()]).await?
            }
            Err(e) => return Err(e.into()),
        };
        std::fs::write(&stamp_file, &scan.stamp)?;

        let update = if !existed {
            Some(IndexUpdate::Built {
                files: stats.files_added,
            })
        } else if stats.files_added + stats.files_updated + stats.files_removed == 0 {
            None
        } else {
            Some(IndexUpdate::Updated {
                added: stats.files_added,
                changed: stats.files_updated,
                removed: stats.files_removed,
            })
        };
        Ok((indexed, index, update))
    }

    /// Where the index of `folder` lives.
    pub fn index_path(&self, folder: &Path) -> PathBuf {
        self.cache_dir.join(folder_key(folder))
    }

    async fn indexer(&self) -> anyhow::Result<&Indexer> {
        self.indexer
            .get_or_try_init(|| async {
                let mut builder = Indexer::builder(&self.model).quiet(true);
                for pattern in EXCLUDED {
                    builder = builder.exclude(pattern);
                }
                Ok::<_, anyhow::Error>(builder.build().await?)
            })
            .await
    }

    async fn searcher(&self, rerank: bool) -> anyhow::Result<&Searcher> {
        let (cell, reranker) = match (&self.reranker, rerank) {
            (Some(model), true) => (&self.reranking_searcher, Some(model)),
            _ => (&self.searcher, None),
        };
        cell.get_or_try_init(|| async {
            let mut builder = Searcher::builder(&self.model).quiet(true);
            if let Some(model) = reranker {
                builder = builder.reranker(model);
            }
            Ok::<_, anyhow::Error>(builder.build().await?)
        })
        .await
    }
}

/// A stable name for a folder's index, the same across runs.
pub fn folder_key(folder: &Path) -> String {
    format!("{:016x}", fnv1a(folder.as_os_str().as_encoded_bytes()))
}

/// What a walk of a folder found.
#[derive(Clone, Debug)]
pub struct Scan {
    /// Changes whenever a file is added, removed or modified: the model, the
    /// number of files, their total size and the newest modification time.
    pub stamp: String,
    /// Files the indexer would read: text and code, at most 10 MB each.
    pub documents: usize,
    /// Those documents per top-level subfolder, by name. Files directly in
    /// the folder are not counted here.
    pub documents_by_subfolder: BTreeMap<String, usize>,
}

/// Walks `folder` as the indexer would, skipping [`EXCLUDED`] and hidden
/// folders.
pub fn scan(folder: &Path, model: &str) -> Scan {
    let (mut files, mut bytes, mut newest) = (0u64, 0u64, 0u128);
    let mut documents = 0;
    let mut documents_by_subfolder = BTreeMap::new();
    let walker = walkdir::WalkDir::new(folder).into_iter().filter_entry(|e| {
        e.depth() == 0 || {
            let name = e.file_name().to_string_lossy();
            !name.starts_with('.') && !EXCLUDED.iter().any(|p| wildcard(p, &name))
        }
    });
    for entry in walker.filter_map(|e| e.ok()) {
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        files += 1;
        bytes += meta.len();
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok());
        newest = newest.max(mtime.map_or(0, |d| d.as_nanos()));

        let text = entry
            .path()
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| TEXT_EXTENSIONS.contains(&e.to_lowercase().as_str()));
        if text && meta.len() <= MAX_FILE_BYTES {
            documents += 1;
            if entry.depth() > 1
                && let Ok(rel) = entry.path().strip_prefix(folder)
                && let Some(first) = rel.components().next()
            {
                *documents_by_subfolder
                    .entry(first.as_os_str().to_string_lossy().into_owned())
                    .or_insert(0) += 1;
            }
        }
    }
    Scan {
        stamp: format!("{model} {files} {bytes} {newest}"),
        documents,
        documents_by_subfolder,
    }
}

/// The refusal for a folder over the limit, naming the subfolders that hold
/// the most documents.
fn too_many(folder: &Path, scan: &Scan, max: usize) -> String {
    let mut biggest: Vec<(&String, &usize)> = scan.documents_by_subfolder.iter().collect();
    biggest.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    let list: Vec<String> = biggest
        .iter()
        .take(8)
        .map(|(name, n)| format!("{} ({n})", folder.join(name).display()))
        .collect();
    let mut text = format!(
        "{} holds {} documents, more than the {max} a search indexes at once. \
         Search a folder inside it instead",
        folder.display(),
        scan.documents
    );
    if list.is_empty() {
        text.push('.');
    } else {
        text.push_str(&format!("; the largest are {}.", list.join(", ")));
    }
    text
}

/// The source pattern for files under `folder`.
fn within(folder: &Path) -> String {
    format!("{}/**", folder.to_string_lossy().replace('\\', "/"))
}

/// Whether `folder`'s path matches only itself as a glob pattern. A parent's
/// index is reused only then, since the search filters its sources with one.
/// A `?` matches itself among others, so Windows' `\\?\` prefix is fine.
fn within_is_literal(folder: &Path) -> bool {
    !folder.to_string_lossy().contains(['*', '[', ']', '{', '}'])
}

/// `*` matches any run of characters, as in the indexer's own exclusions.
pub fn wildcard(pattern: &str, name: &str) -> bool {
    match pattern.split_once('*') {
        None => pattern == name,
        Some((head, tail)) => {
            name.len() >= head.len() + tail.len() && name.starts_with(head) && name.ends_with(tail)
        }
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests;
