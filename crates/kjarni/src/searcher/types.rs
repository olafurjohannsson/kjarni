
use kjarni_rag::{MetadataFilter, SearchMode};

/// Search options
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// Search mode (default: Hybrid)
    pub mode: Option<SearchMode>,
    /// Number of results (default: 10)
    pub top_k: Option<usize>,
    /// Use reranker if available (default: true if configured)
    pub rerank: Option<bool>,
    /// Minimum score threshold
    pub threshold: Option<f32>,
    /// Metadata filter
    pub filter: Option<MetadataFilter>,
}

impl SearchOptions {
    pub fn new() -> Self { Self::default() }
    
    pub fn mode(mut self, m: SearchMode) -> Self { self.mode = Some(m); self }
    pub fn top_k(mut self, k: usize) -> Self { self.top_k = Some(k); self }
    pub fn rerank(mut self, r: bool) -> Self { self.rerank = Some(r); self }
    pub fn threshold(mut self, t: f32) -> Self { self.threshold = Some(t); self }
    pub fn filter(mut self, f: MetadataFilter) -> Self { self.filter = Some(f); self }
    
    /// Filter by source pattern
    pub fn source(mut self, pattern: &str) -> Self {
        let mut f = self.filter.unwrap_or_default();
        f = f.source(pattern);
        self.filter = Some(f);
        self
    }
}

/// Error types for search
#[derive(Debug, thiserror::Error)]
pub enum SearcherError {
    #[error("Index not found: {0}")]
    IndexNotFound(String),
    
    #[error("Failed to load embedder: {0}")]
    EmbedderError(#[from] crate::embedder::EmbedderError),
    
    #[error("Failed to load reranker: {0}")]
    RerankerError(#[from] crate::reranker::RerankerError),
    
    #[error("Dimension mismatch: index expects {index_dim}, model produces {model_dim}")]
    DimensionMismatch { index_dim: usize, model_dim: usize },
    
    #[error("Search failed: {0}")]
    SearchFailed(#[from] anyhow::Error),
}

pub type SearcherResult<T> = Result<T, SearcherError>;

