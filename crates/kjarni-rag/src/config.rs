use serde::{Deserialize, Serialize};

/// Index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Embedding dimension
    pub dimension: usize,
    /// Maximum documents per segment before flushing
    pub max_docs_per_segment: usize,
    /// Maximum memory for in-flight segment (bytes)
    pub max_segment_memory: usize,
    /// Model used to generate embeddings (for validation)
    pub embedding_model: Option<String>,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dimension: 384,
            max_docs_per_segment: 10_000,
            max_segment_memory: 100 * 1024 * 1024, // 100MB
            embedding_model: None,
        }
    }
}

impl IndexConfig {
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }

    /// Estimate memory usage for a segment
    pub fn estimate_segment_memory(&self, doc_count: usize, avg_doc_len: usize) -> usize {
        let vector_mem = doc_count * self.dimension * 4; // f32
        let text_mem = doc_count * avg_doc_len;
        let overhead = doc_count * 100; // metadata, bm25 overhead
        vector_mem + text_mem + overhead
    }
}