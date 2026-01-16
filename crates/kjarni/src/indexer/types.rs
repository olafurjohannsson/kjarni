// kjarni/src/indexer/types.rs

use std::collections::HashMap;

/// Statistics returned after indexing
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Number of documents indexed
    pub documents_indexed: usize,
    /// Number of chunks created
    pub chunks_created: usize,
    /// Embedding dimension used
    pub dimension: usize,
    /// Total index size in bytes
    pub size_bytes: u64,
    /// Number of files processed
    pub files_processed: usize,
    /// Number of files skipped (errors, unsupported)
    pub files_skipped: usize,
    /// Time taken in milliseconds
    pub elapsed_ms: u64,
}

/// Information about an existing index
#[derive(Debug, Clone)]
pub struct IndexInfo {
    /// Path to the index
    pub path: String,
    /// Number of documents
    pub document_count: usize,
    /// Number of segments
    pub segment_count: usize,
    /// Embedding dimension
    pub dimension: usize,
    /// Total size in bytes
    pub size_bytes: u64,
    /// Embedding model used (if recorded)
    pub embedding_model: Option<String>,
    /// When the index was created
    pub created_at: Option<u64>,
}

/// Error types for indexing operations
#[derive(Debug, thiserror::Error)]
pub enum IndexerError {
    #[error("No input paths specified")]
    NoInputs,

    #[error("Path not found: {0}")]
    PathNotFound(String),

    #[error("Index already exists at {0}. Use force=true to overwrite.")]
    IndexExists(String),

    #[error("Index not found at {0}")]
    IndexNotFound(String),

    #[error("Failed to load embedder: {0}")]
    EmbedderError(#[from] crate::embedder::EmbedderError),

    #[error("Indexing failed: {0}")]
    IndexingFailed(#[from] anyhow::Error),

    #[error("Dimension mismatch: index has {index_dim}, model produces {model_dim}")]
    DimensionMismatch { index_dim: usize, model_dim: usize },

    #[error("Operation cancelled")]
    Cancelled,
}

pub type IndexerResult<T> = Result<T, IndexerError>;