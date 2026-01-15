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
    pub model_name: Option<String>,     // Human-readable model name
    pub created_at: Option<u64>,        // Unix timestamp
    pub version: u32,                   // Index format version (for future migrations)
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            dimension: 384,
            max_docs_per_segment: 10_000,
            max_segment_memory: 100 * 1024 * 1024, // 100MB
            embedding_model: None,
            model_name: None,
            created_at: None,
            version: 1,
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

    /// Set the model name
    pub fn with_model_name(mut self, model_name: &str) -> Self {
        self.model_name = Some(model_name.to_string());
        self
    }

    /// Set the created at timestamp
    pub fn with_created_at(mut self, created_at: u64) -> Self {
        self.created_at = Some(created_at);
        self
    }

    /// Set the index version
    pub fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Set the embedding model
    pub fn with_embedding_model(mut self, embedding_model: &str) -> Self {
        self.embedding_model = Some(embedding_model.to_string());
        self
    }

    

    /// Estimate memory usage for a segment
    pub fn estimate_segment_memory(&self, doc_count: usize, avg_doc_len: usize) -> usize {
        let vector_mem = doc_count * self.dimension * 4; // f32
        let text_mem = doc_count * avg_doc_len;
        let overhead = doc_count * 100; // metadata, bm25 overhead
        vector_mem + text_mem + overhead
    }
}