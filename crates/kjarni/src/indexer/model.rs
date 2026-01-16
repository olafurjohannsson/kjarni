// kjarni/src/indexer/model.rs

use crate::embedder::Embedder;
use crate::indexer::{IndexInfo, IndexStats, IndexerBuilder, IndexerError, IndexerResult};
use anyhow::Result;
use kjarni_rag::{
    DocumentLoader, IndexConfig, IndexReader, IndexWriter, LoaderConfig, Progress,
    ProgressCallback, ProgressStage, SplitterConfig,
};
use kjarni_transformers::Device;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::watch;

pub struct Indexer {
    embedder: Embedder,
    loader_config: LoaderConfig,
    chunk_size: usize,
    chunk_overlap: usize,
    max_docs_per_segment: usize,
    batch_size: usize,
    quiet: bool,
    progress_callback: Option<ProgressCallback>,
    cancel_token: Option<watch::Receiver<bool>>,
}

fn calculate_index_size(path: &str) -> Result<u64> {
    let mut total = 0u64;
    for entry in walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            total += entry.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }
    Ok(total)
}

impl Indexer {
    /// Create with builder pattern
    pub fn builder(model: &str) -> IndexerBuilder {
        IndexerBuilder::new(model)
    }

    pub(crate) async fn from_builder(builder: IndexerBuilder) -> IndexerResult<Self> {
        // Build embedder
        let mut embedder_builder = Embedder::builder(&builder.model);

        match builder.device {
            Device::Wgpu => embedder_builder = embedder_builder.gpu(),
            Device::Cpu => embedder_builder = embedder_builder.cpu(),
        }

        if let Some(ref cache_dir) = builder.cache_dir {
            embedder_builder = embedder_builder.cache_dir(cache_dir);
        }

        let embedder = embedder_builder
            .build()
            .await
            .map_err(IndexerError::EmbedderError)?;

        // Build loader config
        let splitter_config = SplitterConfig {
            chunk_size: builder.chunk_size,
            chunk_overlap: builder.chunk_overlap,
            ..Default::default()
        };

        let loader_config = LoaderConfig {
            splitter: splitter_config,
            recursive: builder.recursive,
            extensions: builder.extensions.clone(),
            exclude_patterns: builder.exclude_patterns.clone(),
            include_hidden: builder.include_hidden,
            max_file_size: builder.max_file_size,
        };

        Ok(Self {
            embedder,
            loader_config,
            chunk_size: builder.chunk_size,
            chunk_overlap: builder.chunk_overlap,
            max_docs_per_segment: builder.max_docs_per_segment,
            batch_size: builder.batch_size,
            quiet: builder.quiet,
            progress_callback: builder.progress_callback,
            cancel_token: builder.cancel_token,
        })
    }

    /// Create with default settings
    pub async fn new(model: &str) -> IndexerResult<Self> {
        Self::builder(model).build().await
    }

    /// Get index info (static - doesn't need embedder)
    pub fn info(index_path: &str) -> IndexerResult<IndexInfo> {
        if !Path::new(index_path).exists() {
            return Err(IndexerError::IndexNotFound(index_path.to_string()));
        }

        let reader = IndexReader::open(index_path).map_err(IndexerError::IndexingFailed)?;

        let size_bytes = calculate_index_size(index_path).unwrap_or(0);

        // Read config for model info
        let config_path = Path::new(index_path).join("config.json");
        let embedding_model = if config_path.exists() {
            std::fs::read_to_string(&config_path)
                .ok()
                .and_then(|s| serde_json::from_str::<IndexConfig>(&s).ok())
                .and_then(|c| c.embedding_model)
        } else {
            None
        };

        Ok(IndexInfo {
            path: index_path.to_string(),
            document_count: reader.len(),
            segment_count: reader.segment_count(),
            dimension: reader.dimension(),
            size_bytes,
            embedding_model,
            created_at: None,
        })
    }

    /// Delete an index
    pub fn delete(index_path: &str) -> IndexerResult<()> {
        if !Path::new(index_path).exists() {
            return Err(IndexerError::IndexNotFound(index_path.to_string()));
        }
        std::fs::remove_dir_all(index_path).map_err(|e| IndexerError::IndexingFailed(e.into()))
    }

    // =========================================================================
    // CREATE METHODS
    // =========================================================================

    /// Create a new index from files/directories
    pub async fn create(&self, index_path: &str, inputs: &[&str]) -> IndexerResult<IndexStats> {
        self.create_impl::<fn(ProgressStage, usize, usize, Option<&str>), fn() -> bool>(
            index_path, inputs, false, None, None,
        )
        .await
    }

    /// Create with force overwrite option
    pub async fn create_with_options(
        &self,
        index_path: &str,
        inputs: &[&str],
        force: bool,
    ) -> IndexerResult<IndexStats> {
        self.create_impl::<fn(ProgressStage, usize, usize, Option<&str>), fn() -> bool>(
            index_path, inputs, force, None, None,
        )
        .await
    }

    /// Create with callback and cancellation support (for FFI)
    pub async fn create_with_callback<F, C>(
        &self,
        index_path: &str,
        inputs: &[&str],
        force: bool,
        on_progress: Option<F>,
        is_cancelled: Option<C>,
    ) -> IndexerResult<IndexStats>
    where
        F: Fn(ProgressStage, usize, usize, Option<&str>),
        C: Fn() -> bool,
    {
        self.create_impl(index_path, inputs, force, on_progress, is_cancelled)
            .await
    }

    async fn create_impl<F, C>(
        &self,
        index_path: &str,
        inputs: &[&str],
        force: bool,
        on_progress: Option<F>,
        is_cancelled: Option<C>,
    ) -> IndexerResult<IndexStats>
    where
        F: Fn(ProgressStage, usize, usize, Option<&str>),
        C: Fn() -> bool,
    {
        if inputs.is_empty() {
            return Err(IndexerError::NoInputs);
        }

        // Check if exists
        if Path::new(index_path).exists() {
            if force {
                std::fs::remove_dir_all(index_path)
                    .map_err(|e| IndexerError::IndexingFailed(e.into()))?;
            } else {
                return Err(IndexerError::IndexExists(index_path.to_string()));
            }
        }

        let start = std::time::Instant::now();
        let dimension = self.embedder.dimension();

        // Helper to report progress
        let report = |stage, current, total, msg: Option<&str>| {
            if let Some(ref cb) = on_progress {
                cb(stage, current, total, msg);
            }
        };

        // Helper to check cancellation
        let check_cancelled = || is_cancelled.as_ref().map(|f| f()).unwrap_or(false);

        // Create index writer
        let config = IndexConfig {
            dimension,
            max_docs_per_segment: self.max_docs_per_segment,
            embedding_model: Some(self.embedder.model_name().to_string()),
            ..Default::default()
        };

        let mut writer =
            IndexWriter::open(index_path, config).map_err(IndexerError::IndexingFailed)?;

        // Collect files
        report(ProgressStage::Scanning, 0, 0, Some("Discovering files..."));
        let files = self.collect_files(inputs)?;
        let total_files = files.len();

        if check_cancelled() {
            return Err(IndexerError::Cancelled);
        }

        let loader = DocumentLoader::new(self.loader_config.clone());

        let mut total_docs = 0usize;
        let mut total_chunks = 0usize;
        let mut files_processed = 0usize;
        let mut files_skipped = 0usize;

        let mut batch_texts: Vec<String> = Vec::with_capacity(self.batch_size);
        let mut batch_metadata: Vec<HashMap<String, String>> = Vec::with_capacity(self.batch_size);

        for (file_idx, file_path) in files.iter().enumerate() {
            if check_cancelled() {
                return Err(IndexerError::Cancelled);
            }

            report(
                ProgressStage::Loading,
                file_idx,
                total_files,
                Some(&file_path.to_string_lossy()),
            );

            match loader.load_file(file_path) {
                Ok(chunks) => {
                    total_chunks += chunks.len();
                    files_processed += 1;

                    for chunk in chunks {
                        batch_texts.push(chunk.text);
                        batch_metadata.push(chunk.metadata.to_hashmap());

                        if batch_texts.len() >= self.batch_size {
                            if check_cancelled() {
                                return Err(IndexerError::Cancelled);
                            }

                            report(ProgressStage::Embedding, total_docs, 0, None);

                            let added = self
                                .flush_batch(&mut writer, &mut batch_texts, &mut batch_metadata)
                                .await?;
                            total_docs += added;
                        }
                    }
                }
                Err(e) => {
                    files_skipped += 1;
                    if !self.quiet {
                        eprintln!("Warning: Failed to load {}: {}", file_path.display(), e);
                    }
                }
            }
        }

        // Flush remaining
        if !batch_texts.is_empty() {
            report(ProgressStage::Embedding, total_docs, 0, None);
            let added = self
                .flush_batch(&mut writer, &mut batch_texts, &mut batch_metadata)
                .await?;
            total_docs += added;
        }

        report(
            ProgressStage::Committing,
            total_docs,
            total_docs,
            Some("Finalizing index..."),
        );

        writer.commit().map_err(IndexerError::IndexingFailed)?;

        let size_bytes = calculate_index_size(index_path).unwrap_or(0);

        Ok(IndexStats {
            documents_indexed: total_docs,
            chunks_created: total_chunks,
            dimension,
            size_bytes,
            files_processed,
            files_skipped,
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }

    // =========================================================================
    // ADD METHODS
    // =========================================================================

    /// Add documents to existing index
    pub async fn add(&self, index_path: &str, inputs: &[&str]) -> IndexerResult<usize> {
        self.add_impl::<fn(ProgressStage, usize, usize, Option<&str>), fn() -> bool>(
            index_path, inputs, None, None,
        )
        .await
    }

    /// Add with callback support (for FFI)
    pub async fn add_with_callback<F, C>(
        &self,
        index_path: &str,
        inputs: &[&str],
        on_progress: Option<F>,
        is_cancelled: Option<C>,
    ) -> IndexerResult<usize>
    where
        F: Fn(ProgressStage, usize, usize, Option<&str>),
        C: Fn() -> bool,
    {
        self.add_impl(index_path, inputs, on_progress, is_cancelled)
            .await
    }

    async fn add_impl<F, C>(
        &self,
        index_path: &str,
        inputs: &[&str],
        on_progress: Option<F>,
        is_cancelled: Option<C>,
    ) -> IndexerResult<usize>
    where
        F: Fn(ProgressStage, usize, usize, Option<&str>),
        C: Fn() -> bool,
    {
        if inputs.is_empty() {
            return Ok(0);
        }

        if !Path::new(index_path).exists() {
            return Err(IndexerError::IndexNotFound(index_path.to_string()));
        }

        // Helper to report progress
        let report = |stage, current, total, msg: Option<&str>| {
            if let Some(ref cb) = on_progress {
                cb(stage, current, total, msg);
            }
        };

        // Helper to check cancellation
        let check_cancelled = || is_cancelled.as_ref().map(|f| f()).unwrap_or(false);

        // Open existing index for appending
        let mut writer =
            IndexWriter::open_existing(index_path).map_err(IndexerError::IndexingFailed)?;

        // Validate dimension matches
        if writer.dimension() != self.embedder.dimension() {
            return Err(IndexerError::DimensionMismatch {
                index_dim: writer.dimension(),
                model_dim: self.embedder.dimension(),
            });
        }

        // Collect files
        report(ProgressStage::Scanning, 0, 0, Some("Discovering files..."));
        let files = self.collect_files(inputs)?;
        let total_files = files.len();

        if check_cancelled() {
            return Err(IndexerError::Cancelled);
        }

        let loader = DocumentLoader::new(self.loader_config.clone());

        let mut total_docs = 0usize;
        let mut batch_texts: Vec<String> = Vec::with_capacity(self.batch_size);
        let mut batch_metadata: Vec<HashMap<String, String>> = Vec::with_capacity(self.batch_size);

        for (file_idx, file_path) in files.iter().enumerate() {
            if check_cancelled() {
                return Err(IndexerError::Cancelled);
            }

            report(
                ProgressStage::Loading,
                file_idx,
                total_files,
                Some(&file_path.to_string_lossy()),
            );

            match loader.load_file(file_path) {
                Ok(chunks) => {
                    for chunk in chunks {
                        batch_texts.push(chunk.text);
                        batch_metadata.push(chunk.metadata.to_hashmap());

                        if batch_texts.len() >= self.batch_size {
                            if check_cancelled() {
                                return Err(IndexerError::Cancelled);
                            }

                            report(ProgressStage::Embedding, total_docs, 0, None);

                            let added = self
                                .flush_batch(&mut writer, &mut batch_texts, &mut batch_metadata)
                                .await?;
                            total_docs += added;
                        }
                    }
                }
                Err(e) => {
                    if !self.quiet {
                        eprintln!("Warning: Failed to load {}: {}", file_path.display(), e);
                    }
                }
            }
        }

        // Flush remaining
        if !batch_texts.is_empty() {
            report(ProgressStage::Embedding, total_docs, 0, None);
            let added = self
                .flush_batch(&mut writer, &mut batch_texts, &mut batch_metadata)
                .await?;
            total_docs += added;
        }

        report(
            ProgressStage::Committing,
            total_docs,
            total_docs,
            Some("Finalizing..."),
        );

        writer.commit().map_err(IndexerError::IndexingFailed)?;

        Ok(total_docs)
    }

    // =========================================================================
    // INTERNAL HELPERS
    // =========================================================================

    async fn flush_batch(
        &self,
        writer: &mut IndexWriter,
        texts: &mut Vec<String>,
        metadata: &mut Vec<HashMap<String, String>>,
    ) -> IndexerResult<usize> {
        let count = texts.len();

        let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = self
            .embedder
            .embed_batch(&texts_ref)
            .await
            .map_err(IndexerError::EmbedderError)?;

        for ((text, meta), embedding) in texts.drain(..).zip(metadata.drain(..)).zip(embeddings) {
            writer
                .add(&text, &embedding, Some(&meta))
                .map_err(IndexerError::IndexingFailed)?;
        }

        Ok(count)
    }

    fn collect_files(&self, inputs: &[&str]) -> IndexerResult<Vec<PathBuf>> {
        use walkdir::WalkDir;

        let mut files = Vec::new();

        for input in inputs {
            let path = Path::new(input);

            if !path.exists() {
                return Err(IndexerError::PathNotFound(input.to_string()));
            }

            if path.is_file() {
                if self.is_supported_file(path) {
                    files.push(path.to_path_buf());
                }
            } else if path.is_dir() {
                let walker = if self.loader_config.recursive {
                    WalkDir::new(path)
                } else {
                    WalkDir::new(path).max_depth(1)
                };

                for entry in walker.into_iter().filter_map(|e| e.ok()) {
                    let entry_path = entry.path();

                    if !entry_path.is_file() {
                        continue;
                    }

                    // Skip hidden files
                    if !self.loader_config.include_hidden {
                        if let Some(name) = entry_path.file_name().and_then(|n| n.to_str()) {
                            if name.starts_with('.') {
                                continue;
                            }
                        }
                    }

                    // Check exclude patterns
                    let path_str = entry_path.to_string_lossy();
                    let excluded = self
                        .loader_config
                        .exclude_patterns
                        .iter()
                        .any(|pattern| glob_match::glob_match(pattern, &path_str));
                    if excluded {
                        continue;
                    }

                    // Check file size
                    if let Some(max_size) = self.loader_config.max_file_size {
                        if let Ok(metadata) = entry_path.metadata() {
                            if metadata.len() as usize > max_size {
                                continue;
                            }
                        }
                    }

                    if self.is_supported_file(entry_path) {
                        files.push(entry_path.to_path_buf());
                    }
                }
            }
        }

        Ok(files)
    }

    fn is_supported_file(&self, path: &Path) -> bool {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());

        match ext {
            Some(ext) => {
                if self.loader_config.extensions.is_empty() {
                    kjarni_rag::TEXT_EXTENSIONS.contains(&ext.as_str())
                } else {
                    self.loader_config.extensions.iter().any(|e| e == &ext)
                }
            }
            None => false,
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancel_token
            .as_ref()
            .map(|t| *t.borrow())
            .unwrap_or(false)
    }

    fn report_progress(&self, progress: Progress) {
        if let Some(ref callback) = self.progress_callback {
            callback(&progress, None);
        }
    }

    // =========================================================================
    // ACCESSORS
    // =========================================================================

    pub fn model_name(&self) -> &str {
        self.embedder.model_name()
    }

    pub fn dimension(&self) -> usize {
        self.embedder.dimension()
    }

    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn is_quiet(&self) -> bool {
        self.quiet
    }
}