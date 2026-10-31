//! Index writer with automatic segmentation
//!
//! `manifest.json` is the single source of truth for what an index contains. Segments
//! are written beside the live ones and stay invisible until a commit atomically
//! replaces the manifest, so a reader sees either all of a commit or none of it.

use anyhow::{Context, Result, anyhow, bail};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::Write;
use std::path::{Path, PathBuf};

use super::config::{INDEX_VERSION, IndexConfig};
use super::segment::{Segment, SegmentBuilder};

const MANIFEST_FILE: &str = "manifest.json";
const LOCK_FILE: &str = "write.lock";
const SEGMENTS_DIR: &str = "segments";
const TEMP_DIR: &str = "temp";

/// Merge once this many segments are under half the segment size.
const MERGE_THRESHOLD: usize = 8;

/// The list of live segments, in document ID order.
#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Manifest {
    /// Tells apart two manifests with the same segment names, such as those of an index
    /// and of a new index later built at the same path.
    pub(crate) written_at_ns: u64,
    pub(crate) segments: Vec<String>,
}

pub(crate) fn read_manifest(root: &Path) -> Result<Manifest> {
    let path = root.join(MANIFEST_FILE);
    let text = fs::read_to_string(&path)
        .with_context(|| format!("could not read {}; rebuild the index", path.display()))?;
    Ok(serde_json::from_str(&text)?)
}

/// Rejects an index in any format but the one this build reads.
pub(crate) fn check_version(config: &IndexConfig) -> Result<()> {
    if config.version > INDEX_VERSION {
        bail!(
            "index format version {} is newer than this kjarni reads ({INDEX_VERSION})",
            config.version
        );
    }
    if config.version < INDEX_VERSION {
        bail!(
            "index format version {} is no longer supported; rebuild the index",
            config.version
        );
    }
    Ok(())
}

/// Replaces a file in one step, so a crash leaves the old contents or the new.
fn write_atomic(path: &Path, contents: &str) -> Result<()> {
    let temp = path.with_extension("tmp");
    // Closed before the rename, which Windows refuses on an open file.
    {
        let mut file = File::create(&temp)?;
        file.write_all(contents.as_bytes())?;
        file.sync_all()?;
    }
    fs::rename(&temp, path)?;
    // Directories cannot be opened for syncing on every platform.
    if let Some(parent) = path.parent()
        && let Ok(dir) = File::open(parent)
    {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// One writer per index. The operating system releases the lock when the process exits,
/// however it exits, so a crash cannot leave an index locked.
fn acquire_write_lock(root: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(root.join(LOCK_FILE))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => {
            bail!("another writer is using the index at {}", root.display())
        }
        Err(TryLockError::Error(e)) if e.kind() == std::io::ErrorKind::Unsupported => Ok(file),
        Err(TryLockError::Error(e)) => Err(e.into()),
    }
}

/// A segment in a writer's view of the index.
#[derive(Debug, Clone)]
struct SegmentEntry {
    name: String,
    doc_count: usize,
}

/// Writes documents to a segmented index
pub struct IndexWriter {
    root: PathBuf,
    config: IndexConfig,
    current_builder: Option<SegmentBuilder>,
    next_segment_id: u64,
    total_docs: usize,
    /// Segments from earlier commits, as this writer will leave them.
    committed: Vec<SegmentEntry>,
    /// Segments holding documents added by this writer.
    added: Vec<SegmentEntry>,
    /// The segments named by the manifest on disk. Anything else in the segments folder
    /// is garbage.
    published: HashSet<String>,
    _lock: File,
}

impl IndexWriter {
    /// Create a new index. Refuses a path that already holds one; `open_existing` adds
    /// to an index.
    pub fn open(root: impl AsRef<Path>, config: IndexConfig) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        if root.join("config.json").exists() {
            bail!(
                "{} already holds an index; add to it or delete it first",
                root.display()
            );
        }
        fs::create_dir_all(root.join(SEGMENTS_DIR))?;
        let lock = acquire_write_lock(&root)?;

        write_atomic(
            &root.join("config.json"),
            &serde_json::to_string_pretty(&config)?,
        )?;
        Self::write_manifest(&root, Vec::new())?;

        Ok(Self {
            root,
            config,
            current_builder: None,
            next_segment_id: 0,
            total_docs: 0,
            committed: Vec::new(),
            added: Vec::new(),
            published: HashSet::new(),
            _lock: lock,
        })
    }

    /// Open an existing index to change it
    pub fn open_existing(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let lock = acquire_write_lock(&root)?;

        let config: IndexConfig =
            serde_json::from_str(&fs::read_to_string(root.join("config.json"))?)?;
        check_version(&config)?;

        let manifest = read_manifest(&root)?;
        let published: HashSet<String> = manifest.segments.iter().cloned().collect();
        // Left behind by a writer that was killed before it committed.
        Self::remove_unpublished(&root, &published);

        let next_segment_id = Self::find_next_segment_id(&root)?;

        let mut committed = Vec::with_capacity(manifest.segments.len());
        for name in manifest.segments {
            let meta_path = root.join(SEGMENTS_DIR).join(&name).join("segment.json");
            let meta: super::segment::SegmentMeta =
                serde_json::from_str(&fs::read_to_string(&meta_path).with_context(|| {
                    format!("index segment {name} is damaged; rebuild the index")
                })?)?;
            committed.push(SegmentEntry {
                name,
                doc_count: meta.doc_count,
            });
        }
        let total_docs = committed.iter().map(|s| s.doc_count).sum();

        Ok(Self {
            root,
            config,
            current_builder: None,
            next_segment_id,
            total_docs,
            committed,
            added: Vec::new(),
            published,
            _lock: lock,
        })
    }

    fn write_manifest(root: &Path, segments: Vec<String>) -> Result<()> {
        let written_at_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos() as u64;
        let manifest = Manifest {
            written_at_ns,
            segments,
        };
        write_atomic(
            &root.join(MANIFEST_FILE),
            &serde_json::to_string_pretty(&manifest)?,
        )
    }

    fn remove_unpublished(root: &Path, published: &HashSet<String>) {
        let _ = fs::remove_dir_all(root.join(TEMP_DIR));
        let Ok(entries) = fs::read_dir(root.join(SEGMENTS_DIR)) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            if !name.to_str().is_some_and(|n| published.contains(n)) {
                // A reader on Windows can hold a retired segment open. The next writer
                // removes it instead.
                let _ = fs::remove_dir_all(entry.path());
            }
        }
    }

    fn find_next_segment_id(root: &Path) -> Result<u64> {
        let mut max_id = 0u64;
        for entry in fs::read_dir(root.join(SEGMENTS_DIR))? {
            let entry = entry?;
            if let Some(name) = entry.file_name().to_str()
                && let Some(id_str) = name.strip_prefix("seg_")
                && let Ok(id) = id_str.parse::<u64>()
            {
                max_id = max_id.max(id + 1);
            }
        }

        Ok(max_id)
    }

    fn new_builder(&self) -> Result<SegmentBuilder> {
        let temp_dir = self
            .root
            .join(TEMP_DIR)
            .join(format!("seg_{}", self.next_segment_id));
        SegmentBuilder::new(
            &temp_dir,
            self.config.dimension,
            self.config.max_docs_per_segment,
        )
    }

    fn finish_builder(&mut self, builder: SegmentBuilder) -> Result<SegmentEntry> {
        let name = segment_dir_name(self.next_segment_id);
        let segment_dir = self.root.join(SEGMENTS_DIR).join(&name);
        let meta = builder.flush(&segment_dir, self.next_segment_id)?;
        self.next_segment_id += 1;

        log::info!(
            "Flushed segment {} with {} documents",
            meta.id,
            meta.doc_count
        );

        Ok(SegmentEntry {
            name,
            doc_count: meta.doc_count,
        })
    }

    /// Add a document to the index
    pub fn add(
        &mut self,
        text: &str,
        embedding: &[f32],
        metadata: Option<&HashMap<String, String>>,
    ) -> Result<()> {
        // Create builder if needed
        if self.current_builder.is_none() {
            self.current_builder = Some(self.new_builder()?);
        }

        let builder = self
            .current_builder
            .as_mut()
            .ok_or_else(|| anyhow!("no segment builder"))?;
        builder.add(text, embedding, metadata)?;
        self.total_docs += 1;

        // Flush if full
        if builder.is_full() {
            self.flush_current_segment()?;
        }

        Ok(())
    }

    /// Flush current segment to disk
    fn flush_current_segment(&mut self) -> Result<()> {
        if let Some(builder) = self.current_builder.take()
            && !builder.is_empty()
        {
            let entry = self.finish_builder(builder)?;
            self.added.push(entry);
        }
        Ok(())
    }

    /// Removes documents from earlier commits whose metadata matches, and returns how
    /// many. Documents added through this writer are kept, so replacing a file is:
    /// add its new chunks, then remove its old ones. Takes effect on commit.
    pub fn remove_where(
        &mut self,
        matches: impl Fn(&HashMap<String, String>) -> bool,
    ) -> Result<usize> {
        self.flush_current_segment()?;

        let mut kept = Vec::with_capacity(self.committed.len());
        let mut removed = 0;
        for entry in std::mem::take(&mut self.committed) {
            let (rewritten, count) = self.rewrite(std::slice::from_ref(&entry), &matches)?;
            removed += count;
            if count == 0 {
                kept.push(entry);
            } else {
                kept.extend(rewritten);
            }
        }
        self.committed = kept;
        self.total_docs -= removed;
        Ok(removed)
    }

    /// Copies segments into new ones, leaving out matching documents. Writes nothing
    /// when a single segment has no match. Returns the new segments and the number left
    /// out.
    fn rewrite(
        &mut self,
        sources: &[SegmentEntry],
        matches: &dyn Fn(&HashMap<String, String>) -> bool,
    ) -> Result<(Vec<SegmentEntry>, usize)> {
        let mut rewritten = Vec::new();
        let mut removed = 0;
        let mut builder: Option<SegmentBuilder> = None;

        for source in sources {
            let segment = Segment::open(&self.root.join(SEGMENTS_DIR).join(&source.name))?;
            let metadata = segment.read_all_metadata()?;
            if sources.len() == 1 && !metadata.iter().any(matches) {
                return Ok((Vec::new(), 0));
            }

            for (doc_id, meta) in metadata.iter().enumerate() {
                if matches(meta) {
                    removed += 1;
                    continue;
                }
                let embedding = segment
                    .get_embedding(doc_id)
                    .ok_or_else(|| anyhow!("segment {} is missing a vector", source.name))?;
                let text = segment.get_document(doc_id)?;

                if builder.is_none() {
                    builder = Some(self.new_builder()?);
                }
                let current = builder
                    .as_mut()
                    .ok_or_else(|| anyhow!("no segment builder"))?;
                current.add(&text, embedding, Some(meta))?;
                if current.is_full()
                    && let Some(full) = builder.take()
                {
                    rewritten.push(self.finish_builder(full)?);
                }
            }
        }

        if let Some(last) = builder.take()
            && !last.is_empty()
        {
            rewritten.push(self.finish_builder(last)?);
        }
        Ok((rewritten, removed))
    }

    /// Combines small segments once enough have built up. Every small commit adds a
    /// segment, and each one costs every later search.
    fn merge_small_segments(&mut self) -> Result<()> {
        let small_size = self.config.max_docs_per_segment / 2;
        let (small, mut large): (Vec<_>, Vec<_>) = std::mem::take(&mut self.committed)
            .into_iter()
            .partition(|s| s.doc_count < small_size);

        if small.len() < MERGE_THRESHOLD {
            large.extend(small);
            self.committed = large;
            return Ok(());
        }

        let (merged, _) = self.rewrite(&small, &|_| false)?;
        large.extend(merged);
        self.committed = large;
        Ok(())
    }

    /// Commit all pending writes. Readers see nothing from this writer until here, and
    /// dropping the writer without committing discards everything it did.
    pub fn commit(mut self) -> Result<()> {
        self.flush_current_segment()?;
        let mut added = std::mem::take(&mut self.added);
        self.committed.append(&mut added);
        self.merge_small_segments()?;

        let segments: Vec<String> = self.committed.iter().map(|s| s.name.clone()).collect();
        Self::write_manifest(&self.root, segments.clone())?;
        self.published = segments.into_iter().collect();

        log::info!(
            "Index committed: {} documents in {} segments",
            self.total_docs,
            self.published.len()
        );

        Ok(())
    }

    /// Get current document count
    pub fn len(&self) -> usize {
        self.total_docs
    }

    /// Whether the index holds no documents yet.
    pub fn is_empty(&self) -> bool {
        self.total_docs == 0
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// The model the index's vectors came from, if recorded
    pub fn embedding_model(&self) -> Option<&str> {
        self.config.embedding_model.as_deref()
    }
}

impl Drop for IndexWriter {
    fn drop(&mut self) {
        // Close the builder's files first; Windows will not remove them while open.
        self.current_builder = None;
        Self::remove_unpublished(&self.root, &self.published);
    }
}

fn segment_dir_name(id: u64) -> String {
    format!("seg_{id:06}")
}

/// Whether a path holds a Kjarni index.
pub fn is_index_dir(path: &Path) -> bool {
    path.join("config.json").is_file() && path.join("segments").is_dir()
}

/// A new index built beside its final path and moved there once complete, so a
/// failure or cancellation leaves whatever was at the target untouched.
pub struct IndexBuild {
    target: PathBuf,
    build_dir: PathBuf,
}

impl IndexBuild {
    pub fn new(target: impl AsRef<Path>) -> Result<Self> {
        let target = target.as_ref().to_path_buf();
        let build_dir = Self::sibling(&target, "building")?;
        if build_dir.exists() {
            fs::remove_dir_all(&build_dir)?;
        }
        Ok(Self { target, build_dir })
    }

    /// Where to write the index while it is being built.
    pub fn path(&self) -> &Path {
        &self.build_dir
    }

    /// Moves the finished index to the target, replacing what is there.
    pub fn install(self) -> Result<()> {
        if self.target.exists() {
            let old = Self::sibling(&self.target, "old")?;
            if old.exists() {
                fs::remove_dir_all(&old)?;
            }
            fs::rename(&self.target, &old)?;
            if let Err(e) = fs::rename(&self.build_dir, &self.target) {
                fs::rename(&old, &self.target)?;
                return Err(e.into());
            }
            fs::remove_dir_all(&old)?;
        } else {
            fs::rename(&self.build_dir, &self.target)?;
        }
        Ok(())
    }

    /// A hidden name, so a directory walk over the parent does not read it.
    fn sibling(target: &Path, suffix: &str) -> Result<PathBuf> {
        let absolute = std::path::absolute(target)?;
        let name = absolute
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("{} is not a usable index path", target.display()))?
            .to_string_lossy()
            .into_owned();
        Ok(absolute.with_file_name(format!(".{name}.{suffix}")))
    }
}

impl Drop for IndexBuild {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.build_dir);
    }
}
