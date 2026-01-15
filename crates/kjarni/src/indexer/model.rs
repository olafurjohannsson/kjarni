// kjarni/src/indexer/model.rs

use crate::embedder::Embedder;
use kjarni_rag::{DocumentLoader, IndexConfig, IndexReader, IndexWriter, LoaderConfig, SplitterConfig};
use std::path::Path;
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

impl Indexer {
    /// Create with builder pattern
    pub fn builder(model: &str) -> IndexerBuilder {
        IndexerBuilder::new(model)
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
        
        let reader = IndexReader::open(index_path)
            .map_err(|e| IndexerError::IndexingFailed(e))?;
        
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
            created_at: None, // TODO: read from config
        })
    }
    
    /// Delete an index
    pub fn delete(index_path: &str) -> IndexerResult<()> {
        if !Path::new(index_path).exists() {
            return Err(IndexerError::IndexNotFound(index_path.to_string()));
        }
        std::fs::remove_dir_all(index_path)
            .map_err(|e| IndexerError::IndexingFailed(e.into()))
    }
    
    /// Create a new index from files/directories
    pub async fn create(
        &self,
        index_path: &str,
        inputs: &[&str],
    ) -> IndexerResult<IndexStats> {
        self.create_with_options(index_path, inputs, false).await
    }
    
    /// Create with force overwrite option
    pub async fn create_with_options(
        &self,
        index_path: &str,
        inputs: &[&str],
        force: bool,
    ) -> IndexerResult<IndexStats> {
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
        
        // Create index writer
        let config = IndexConfig {
            dimension,
            max_docs_per_segment: self.max_docs_per_segment,
            embedding_model: Some(self.embedder.model_name().to_string()),
            ..Default::default()
        };
        
        let mut writer = IndexWriter::open(index_path, config)
            .map_err(|e| IndexerError::IndexingFailed(e))?;
        
        // Process files
        let stats = self.process_inputs(&mut writer, inputs).await?;
        
        // Commit
        writer.commit().map_err(|e| IndexerError::IndexingFailed(e))?;
        
        let size_bytes = calculate_index_size(index_path).unwrap_or(0);
        
        Ok(IndexStats {
            documents_indexed: stats.0,
            chunks_created: stats.1,
            dimension,
            size_bytes,
            files_processed: stats.2,
            files_skipped: stats.3,
            elapsed_ms: start.elapsed().as_millis() as u64,
        })
    }
    
    /// Add documents to existing index
    pub async fn add(
        &self,
        index_path: &str,
        inputs: &[&str],
    ) -> IndexerResult<usize> {
        if inputs.is_empty() {
            return Err(IndexerError::NoInputs);
        }
        
        if !Path::new(index_path).exists() {
            return Err(IndexerError::IndexNotFound(index_path.to_string()));
        }
        
        // Open existing
        let mut writer = IndexWriter::open_existing(index_path)
            .map_err(|e| IndexerError::IndexingFailed(e))?;
        
        // Validate dimension
        if writer.dimension() != self.embedder.dimension() {
            return Err(IndexerError::DimensionMismatch {
                index_dim: writer.dimension(),
                model_dim: self.embedder.dimension(),
            });
        }
        
        let (docs_added, _, _, _) = self.process_inputs(&mut writer, inputs).await?;
        
        writer.commit().map_err(|e| IndexerError::IndexingFailed(e))?;
        
        Ok(docs_added)
    }
    
    // Internal: process input files
    async fn process_inputs(
        &self,
        writer: &mut IndexWriter,
        inputs: &[&str],
    ) -> IndexerResult<(usize, usize, usize, usize)> {
        // Returns (docs_indexed, chunks_created, files_processed, files_skipped)
        
        let loader = DocumentLoader::new(self.loader_config.clone());
        
        let mut total_docs = 0;
        let mut total_chunks = 0;
        let mut files_processed = 0;
        let mut files_skipped = 0;
        
        let mut batch_texts: Vec<String> = Vec::with_capacity(self.batch_size);
        let mut batch_metadata: Vec<HashMap<String, String>> = Vec::with_capacity(self.batch_size);
        
        // Collect all files
        self.report_progress(Progress {
            stage: ProgressStage::Scanning,
            current: 0,
            total: None,
            message: Some("Discovering files...".to_string()),
        });
        
        let files = self.collect_files(inputs)?;
        let total_files = files.len();
        
        for (file_idx, file_path) in files.iter().enumerate() {
            // Check cancellation
            if self.is_cancelled() {
                return Err(IndexerError::Cancelled);
            }
            
            self.report_progress(Progress {
                stage: ProgressStage::Loading,
                current: file_idx,
                total: Some(total_files),
                message: Some(format!("Loading {}", file_path.display())),
            });
            
            match loader.load_file(file_path) {
                Ok(chunks) => {
                    total_chunks += chunks.len();
                    files_processed += 1;
                    
                    for chunk in chunks {
                        batch_texts.push(chunk.text);
                        batch_metadata.push(chunk.metadata.to_hashmap());
                        
                        if batch_texts.len() >= self.batch_size {
                            let added = self.flush_batch(
                                writer,
                                &mut batch_texts,
                                &mut batch_metadata,
                            ).await?;
                            total_docs += added;
                            
                            self.report_progress(Progress {
                                stage: ProgressStage::Embedding,
                                current: total_docs,
                                total: None,
                                message: None,
                            });
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
            let added = self.flush_batch(writer, &mut batch_texts, &mut batch_metadata).await?;
            total_docs += added;
        }
        
        self.report_progress(Progress {
            stage: ProgressStage::Committing,
            current: total_docs,
            total: Some(total_docs),
            message: Some("Finalizing index...".to_string()),
        });
        
        Ok((total_docs, total_chunks, files_processed, files_skipped))
    }
    
    async fn flush_batch(
        &self,
        writer: &mut IndexWriter,
        texts: &mut Vec<String>,
        metadata: &mut Vec<HashMap<String, String>>,
    ) -> IndexerResult<usize> {
        let count = texts.len();
        
        let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = self.embedder.embed_batch(&texts_ref).await
            .map_err(|e| IndexerError::EmbedderError(e))?;
        
        for ((text, meta), embedding) in texts.drain(..).zip(metadata.drain(..)).zip(embeddings) {
            writer.add(&text, &embedding, Some(&meta))
                .map_err(|e| IndexerError::IndexingFailed(e))?;
        }
        
        Ok(count)
    }
    
    fn collect_files(&self, inputs: &[&str]) -> IndexerResult<Vec<PathBuf>> {
        // Walk directories, filter by extensions, apply exclude patterns
        // ...implementation...
    }
    
    fn is_cancelled(&self) -> bool {
        self.cancel_token.as_ref().map(|t| *t.borrow()).unwrap_or(false)
    }
    
    fn report_progress(&self, progress: Progress) {
        if let Some(ref callback) = self.progress_callback {
            callback(progress);
        }
    }
    
    // Accessors
    pub fn model_name(&self) -> &str { self.embedder.model_name() }
    pub fn dimension(&self) -> usize { self.embedder.dimension() }
    pub fn chunk_size(&self) -> usize { self.chunk_size }
}