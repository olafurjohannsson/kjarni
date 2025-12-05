//! Unified search index combining BM25 and vector search

use edgesearch::{bm25::Bm25Index, hybrid::hybrid_search, types::SearchResult, vector::VectorStore};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete search index with BM25 and semantic search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchIndex {
    bm25: Bm25Index,
    vectors: VectorStore,
    documents: Vec<String>,
    metadata: Vec<HashMap<String, String>>,
}

impl SearchIndex {
    /// Create empty search index
    pub fn new() -> Self {
        Self {
            bm25: Bm25Index::new(),
            vectors: VectorStore::default(),
            documents: Vec::new(),
            metadata: Vec::new(),
        }
    }

    /// Create empty index with known embedding dimension
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            bm25: Bm25Index::new(),
            vectors: VectorStore::with_dimension(dimension),
            documents: Vec::new(),
            metadata: Vec::new(),
        }
    }

    /// Build index from documents and their embeddings
    pub fn build(
        documents: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        metadata: Vec<HashMap<String, String>>,
    ) -> Result<Self> {
        if documents.len() != embeddings.len() {
            return Err(anyhow::anyhow!(
                "Documents and embeddings length mismatch: {} vs {}",
                documents.len(),
                embeddings.len()
            ));
        }

        if !metadata.is_empty() && metadata.len() != documents.len() {
            return Err(anyhow::anyhow!(
                "Metadata length mismatch: {} vs {} documents",
                metadata.len(),
                documents.len()
            ));
        }

        // Build BM25 index
        let mut bm25 = Bm25Index::new();
        for (i, doc) in documents.iter().enumerate() {
            bm25.add_document(i, doc);
        }

        // Build vector store
        let vectors = VectorStore::new(embeddings)?;

        // Use empty metadata if not provided
        let metadata = if metadata.is_empty() {
            vec![HashMap::new(); documents.len()]
        } else {
            metadata
        };

        Ok(Self {
            bm25,
            vectors,
            documents,
            metadata,
        })
    }

    /// Add a single document to the index
    pub fn add_document(
        &mut self,
        text: String,
        embedding: Vec<f32>,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<usize> {
        let doc_id = self.documents.len();

        // Add to BM25
        self.bm25.add_document(doc_id, &text);

        // Add to vector store
        self.vectors.add(embedding)?;

        // Store document and metadata
        self.documents.push(text);
        self.metadata.push(metadata.unwrap_or_default());

        Ok(doc_id)
    }

    /// Add multiple documents to the index
    pub fn add_documents(
        &mut self,
        documents: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        metadata: Option<Vec<HashMap<String, String>>>,
    ) -> Result<Vec<usize>> {
        if documents.len() != embeddings.len() {
            return Err(anyhow::anyhow!(
                "Documents and embeddings length mismatch: {} vs {}",
                documents.len(),
                embeddings.len()
            ));
        }

        let metadata = metadata.unwrap_or_else(|| vec![HashMap::new(); documents.len()]);
        if metadata.len() != documents.len() {
            return Err(anyhow::anyhow!(
                "Metadata length mismatch: {} vs {} documents",
                metadata.len(),
                documents.len()
            ));
        }

        let mut ids = Vec::with_capacity(documents.len());
        for ((text, embedding), meta) in documents.into_iter().zip(embeddings).zip(metadata) {
            let id = self.add_document(text, embedding, Some(meta))?;
            ids.push(id);
        }

        Ok(ids)
    }

    /// Search using BM25 only
    pub fn search_keywords(&self, query: &str, limit: usize) -> Vec<SearchResult> {
        let results = self.bm25.search(query, limit);
        self.format_results(results)
    }

    /// Search using semantic similarity only
    pub fn search_semantic(&self, query_embedding: &[f32], limit: usize) -> Vec<SearchResult> {
        let results = self.vectors.search(query_embedding, limit);
        self.format_results(results)
    }

    /// Hybrid search combining BM25 and semantic search
    pub fn search_hybrid(
        &self,
        query: &str,
        query_embedding: &[f32],
        limit: usize,
    ) -> Vec<SearchResult> {
        // Fetch more results for fusion
        let keyword_results = self.bm25.search(query, limit * 2);
        let semantic_results = self.vectors.search(query_embedding, limit * 2);

        let fused = hybrid_search(keyword_results, semantic_results, limit);
        self.format_results(fused)
    }

    /// Hybrid search with configurable weights
    pub fn search_hybrid_weighted(
        &self,
        query: &str,
        query_embedding: &[f32],
        limit: usize,
        keyword_weight: f32,
        semantic_weight: f32,
    ) -> Vec<SearchResult> {
        let keyword_results = self.bm25.search(query, limit * 2);
        let semantic_results = self.vectors.search(query_embedding, limit * 2);

        let fused = hybrid_search_weighted(
            keyword_results,
            semantic_results,
            limit,
            keyword_weight,
            semantic_weight,
        );
        self.format_results(fused)
    }

    fn format_results(&self, results: Vec<(usize, f32)>) -> Vec<SearchResult> {
        results
            .into_iter()
            .filter_map(|(idx, score)| {
                if idx < self.documents.len() {
                    Some(SearchResult {
                        score,
                        document_id: idx,
                        text: self.documents[idx].clone(),
                        metadata: self.metadata.get(idx).cloned().unwrap_or_default(),
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get document by ID
    pub fn get_document(&self, id: usize) -> Option<&str> {
        self.documents.get(id).map(|s| s.as_str())
    }

    /// Get metadata by document ID
    pub fn get_metadata(&self, id: usize) -> Option<&HashMap<String, String>> {
        self.metadata.get(id)
    }

    /// Get number of indexed documents
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        self.vectors.dimension
    }

    /// Save index to JSON
    pub fn save_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Into::into)
    }

    /// Load index from JSON
    pub fn load_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(Into::into)
    }
}

impl Default for SearchIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Weighted hybrid search (Reciprocal Rank Fusion variant)
fn hybrid_search_weighted(
    keyword_results: Vec<(usize, f32)>,
    semantic_results: Vec<(usize, f32)>,
    limit: usize,
    keyword_weight: f32,
    semantic_weight: f32,
) -> Vec<(usize, f32)> {
    use std::collections::HashMap;

    const K: f32 = 60.0; // RRF constant

    let mut scores: HashMap<usize, f32> = HashMap::new();

    // Add keyword scores (RRF)
    for (rank, (doc_id, _)) in keyword_results.iter().enumerate() {
        let rrf_score = keyword_weight / (K + rank as f32 + 1.0);
        *scores.entry(*doc_id).or_insert(0.0) += rrf_score;
    }

    // Add semantic scores (RRF)
    for (rank, (doc_id, _)) in semantic_results.iter().enumerate() {
        let rrf_score = semantic_weight / (K + rank as f32 + 1.0);
        *scores.entry(*doc_id).or_insert(0.0) += rrf_score;
    }

    let mut results: Vec<(usize, f32)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_embedding(seed: f32) -> Vec<f32> {
        vec![seed, seed * 0.5, seed * 0.25, seed * 0.1]
    }

    #[test]
    fn test_new() {
        let index = SearchIndex::new();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_with_dimension() {
        let index = SearchIndex::with_dimension(384);
        assert!(index.is_empty());
        assert_eq!(index.dimension(), 384);
    }

    #[test]
    fn test_default() {
        let index = SearchIndex::default();
        assert!(index.is_empty());
    }

    #[test]
    fn test_build_basic() {
        let documents = vec![
            "rust programming language".to_string(),
            "python is great".to_string(),
        ];
        let embeddings = vec![mock_embedding(1.0), mock_embedding(2.0)];
        let metadata = vec![HashMap::new(), HashMap::new()];

        let index = SearchIndex::build(documents, embeddings, metadata).unwrap();

        assert_eq!(index.len(), 2);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_build_empty_metadata() {
        let documents = vec!["doc one".to_string(), "doc two".to_string()];
        let embeddings = vec![mock_embedding(1.0), mock_embedding(2.0)];

        let index = SearchIndex::build(documents, embeddings, vec![]).unwrap();

        assert_eq!(index.len(), 2);
        assert!(index.get_metadata(0).unwrap().is_empty());
    }

    #[test]
    fn test_build_length_mismatch_docs_embeddings() {
        let documents = vec!["one".to_string(), "two".to_string()];
        let embeddings = vec![mock_embedding(1.0)]; // Only one!

        let result = SearchIndex::build(documents, embeddings, vec![]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mismatch"));
    }

    #[test]
    fn test_build_length_mismatch_metadata() {
        let documents = vec!["one".to_string(), "two".to_string()];
        let embeddings = vec![mock_embedding(1.0), mock_embedding(2.0)];
        let metadata = vec![HashMap::new()]; // Only one!

        let result = SearchIndex::build(documents, embeddings, metadata);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_document() {
        let mut index = SearchIndex::with_dimension(4);

        let id = index
            .add_document("first document".to_string(), mock_embedding(1.0), None)
            .unwrap();
        assert_eq!(id, 0);
        assert_eq!(index.len(), 1);

        let id = index
            .add_document("second document".to_string(), mock_embedding(2.0), None)
            .unwrap();
        assert_eq!(id, 1);
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_add_document_with_metadata() {
        let mut index = SearchIndex::with_dimension(4);

        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "wikipedia".to_string());

        index
            .add_document("test doc".to_string(), mock_embedding(1.0), Some(meta))
            .unwrap();

        let retrieved = index.get_metadata(0).unwrap();
        assert_eq!(retrieved.get("source"), Some(&"wikipedia".to_string()));
    }

    #[test]
    fn test_add_documents_batch() {
        let mut index = SearchIndex::with_dimension(4);

        let docs = vec!["doc one".to_string(), "doc two".to_string()];
        let embeddings = vec![mock_embedding(1.0), mock_embedding(2.0)];

        let ids = index.add_documents(docs, embeddings, None).unwrap();

        assert_eq!(ids, vec![0, 1]);
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_get_document() {
        let documents = vec!["hello world".to_string()];
        let embeddings = vec![mock_embedding(1.0)];

        let index = SearchIndex::build(documents, embeddings, vec![]).unwrap();

        assert_eq!(index.get_document(0), Some("hello world"));
        assert_eq!(index.get_document(99), None);
    }

    #[test]
    fn test_search_keywords() {
        let documents = vec![
            "rust programming language".to_string(),
            "python scripting".to_string(),
            "rust is fast".to_string(),
        ];
        let embeddings = vec![
            mock_embedding(1.0),
            mock_embedding(2.0),
            mock_embedding(3.0),
        ];

        let index = SearchIndex::build(documents, embeddings, vec![]).unwrap();

        let results = index.search_keywords("rust", 10);

        // Should find docs 0 and 2
        assert!(!results.is_empty());
        let doc_ids: Vec<usize> = results.iter().map(|r| r.document_id).collect();
        assert!(doc_ids.contains(&0));
        assert!(doc_ids.contains(&2));
        assert!(!doc_ids.contains(&1));
    }

    #[test]
    fn test_search_semantic() {
        let documents = vec![
            "first document".to_string(),
            "second document".to_string(),
        ];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];

        let index = SearchIndex::build(documents, embeddings, vec![]).unwrap();

        // Query similar to first document
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.search_semantic(&query, 10);

        assert!(!results.is_empty());
        assert_eq!(results[0].document_id, 0); // Most similar
    }

    #[test]
    fn test_search_hybrid() {
        let documents = vec![
            "rust programming language".to_string(),
            "python scripting language".to_string(),
            "rust is very fast".to_string(),
        ];
        // Embeddings: doc 0 and 2 are similar (both about rust)
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.9, 0.1, 0.0],
        ];

        let index = SearchIndex::build(documents, embeddings, vec![]).unwrap();

        // Query: "rust" with embedding close to doc 0/2
        let query_embedding = vec![1.0, 0.0, 0.0];
        let results = index.search_hybrid("rust", &query_embedding, 10);

        assert!(!results.is_empty());
        // Docs 0 and 2 should rank highest (match both keyword and semantic)
        let top_ids: Vec<usize> = results.iter().take(2).map(|r| r.document_id).collect();
        assert!(top_ids.contains(&0) || top_ids.contains(&2));
    }

    #[test]
    fn test_search_empty_index() {
        let index = SearchIndex::new();

        let keyword_results = index.search_keywords("test", 10);
        assert!(keyword_results.is_empty());

        let semantic_results = index.search_semantic(&[1.0, 2.0, 3.0], 10);
        assert!(semantic_results.is_empty());
    }

    #[test]
    fn test_format_results_out_of_bounds() {
        let index = SearchIndex::build(
            vec!["only one doc".to_string()],
            vec![mock_embedding(1.0)],
            vec![],
        )
        .unwrap();

        // Simulate results with invalid index
        let fake_results = vec![(0, 1.0), (999, 0.5)]; // 999 is out of bounds
        let formatted = index.format_results(fake_results);

        assert_eq!(formatted.len(), 1); // Only valid one included
        assert_eq!(formatted[0].document_id, 0);
    }

    #[test]
    fn test_serde_roundtrip() {
        let documents = vec!["test document".to_string()];
        let embeddings = vec![mock_embedding(1.0)];
        let mut meta = HashMap::new();
        meta.insert("key".to_string(), "value".to_string());

        let index = SearchIndex::build(documents, embeddings, vec![meta]).unwrap();

        let json = index.save_json().unwrap();
        let restored = SearchIndex::load_json(&json).unwrap();

        assert_eq!(restored.len(), 1);
        assert_eq!(restored.get_document(0), Some("test document"));
        assert_eq!(
            restored.get_metadata(0).unwrap().get("key"),
            Some(&"value".to_string())
        );
    }

    #[test]
    fn test_hybrid_search_weighted() {
        let keyword_results = vec![(0, 1.0), (1, 0.8), (2, 0.6)];
        let semantic_results = vec![(2, 0.9), (0, 0.7), (3, 0.5)];

        // Equal weights
        let results = hybrid_search_weighted(keyword_results.clone(), semantic_results.clone(), 10, 1.0, 1.0);
        assert!(!results.is_empty());

        // Keyword-heavy
        let keyword_heavy = hybrid_search_weighted(keyword_results.clone(), semantic_results.clone(), 10, 2.0, 0.5);

        // Semantic-heavy
        let semantic_heavy = hybrid_search_weighted(keyword_results, semantic_results, 10, 0.5, 2.0);

        // With keyword weight, doc 0 (rank 1 in keywords) should score higher
        // With semantic weight, doc 2 (rank 1 in semantic) should score higher
        // These should produce different orderings
        assert_ne!(keyword_heavy[0].0, semantic_heavy[0].0);
    }
}