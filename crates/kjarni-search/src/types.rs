//! Core types for RAG operations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A document chunk ready for indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique identifier
    pub id: String,
    /// The text content
    pub text: String,
    /// Metadata about this chunk
    pub metadata: ChunkMetadata,
}

impl Chunk {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            id: uuid_simple(),
            text: text.into(),
            metadata: ChunkMetadata::default(),
        }
    }

    pub fn with_metadata(mut self, metadata: ChunkMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.metadata.source = Some(source.into());
        self
    }
}

/// Metadata for a chunk
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChunkMetadata {
    /// Source file path
    pub source: Option<String>,
    /// Chunk index within source
    pub chunk_index: Option<usize>,
    /// Total chunks from this source
    pub total_chunks: Option<usize>,
    /// Page number (for PDFs)
    pub page: Option<u32>,
    /// Section/heading hierarchy
    #[serde(default)]
    pub sections: Vec<String>,
    /// Custom key-value metadata
    #[serde(default)]
    pub custom: HashMap<String, String>,
}

impl ChunkMetadata {
    /// Convert to HashMap for SearchIndex compatibility
    pub fn to_hashmap(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        if let Some(ref s) = self.source {
            map.insert("source".to_string(), s.clone());
        }
        if let Some(idx) = self.chunk_index {
            map.insert("chunk_index".to_string(), idx.to_string());
        }
        if let Some(total) = self.total_chunks {
            map.insert("total_chunks".to_string(), total.to_string());
        }
        if let Some(page) = self.page {
            map.insert("page".to_string(), page.to_string());
        }
        if !self.sections.is_empty() {
            map.insert("sections".to_string(), self.sections.join(" > "));
        }
        map.extend(self.custom.clone());
        map
    }
}

/// Search result from an index query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Relevance score
    pub score: f32,
    /// Document ID in the index
    pub document_id: usize,
    /// The chunk text
    pub text: String,
    /// Chunk metadata
    pub metadata: HashMap<String, String>,
}

/// How to search the index
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// BM25 keyword search
    Keyword,
    /// Embedding-based semantic search
    Semantic,
    /// Combined BM25 + semantic (reciprocal rank fusion)
    #[default]
    Hybrid,
}

impl std::str::FromStr for SearchMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "keyword" | "bm25" => Ok(SearchMode::Keyword),
            "semantic" | "vector" => Ok(SearchMode::Semantic),
            "hybrid" => Ok(SearchMode::Hybrid),
            _ => Err(format!("Unknown search mode: '{}'. Use: keyword, semantic, hybrid", s)),
        }
    }
}

/// Simple UUID generator (no external dependency)
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:032x}", now)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_new() {
        let chunk = Chunk::new("Hello world");
        assert_eq!(chunk.text, "Hello world");
        assert!(!chunk.id.is_empty());
    }

    #[test]
    fn test_chunk_with_source() {
        let chunk = Chunk::new("text").with_source("file.txt");
        assert_eq!(chunk.metadata.source, Some("file.txt".to_string()));
    }

    #[test]
    fn test_metadata_to_hashmap() {
        let meta = ChunkMetadata {
            source: Some("doc.txt".to_string()),
            chunk_index: Some(5),
            ..Default::default()
        };
        let map = meta.to_hashmap();
        assert_eq!(map.get("source"), Some(&"doc.txt".to_string()));
        assert_eq!(map.get("chunk_index"), Some(&"5".to_string()));
    }

    #[test]
    fn test_search_mode_parse() {
        assert_eq!("keyword".parse::<SearchMode>().unwrap(), SearchMode::Keyword);
        assert_eq!("bm25".parse::<SearchMode>().unwrap(), SearchMode::Keyword);
        assert_eq!("semantic".parse::<SearchMode>().unwrap(), SearchMode::Semantic);
        assert_eq!("hybrid".parse::<SearchMode>().unwrap(), SearchMode::Hybrid);
        assert!("invalid".parse::<SearchMode>().is_err());
    }
}