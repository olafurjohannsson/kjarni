// kjarni-rag/src/progress.rs (NEW)

/// Progress update during indexing/searching
#[derive(Debug, Clone)]
pub struct Progress {
    pub stage: ProgressStage,
    pub current: usize,
    pub total: Option<usize>,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressStage {
    /// Discovering files
    Scanning,
    /// Loading and chunking documents
    Loading,
    /// Generating embeddings
    Embedding,
    /// Writing to index
    Writing,
    /// Committing index
    Committing,
    /// Search in progress
    Searching,
    /// Reranking results
    Reranking,
}

/// Progress callback type
pub type ProgressCallback = Box<dyn Fn(Progress) + Send + Sync>;