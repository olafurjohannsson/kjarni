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

    /// Build index from documents and their embeddings
    pub fn build(
        documents: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        metadata: Vec<HashMap<String, String>>,
    ) -> Result<Self> {
        unimplemented!()
        // if documents.len() != embeddings.len() {
        //     return Err(anyhow::anyhow!(
        //         "Documents and embeddings length mismatch: {} vs {}",
        //         documents.len(),
        //         embeddings.len()
        //     ));
        // }

        // // Build BM25 index
        // let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
        // let mut bm25 = Bm25Index::new();
        // bm25.build(&doc_refs);

        // // Build vector store
        // let vectors = VectorStore::new(embeddings)?;

        // Ok(Self {
        //     bm25,
        //     vectors,
        //     documents,
        //     metadata,
        // })
    }

    /// Search using BM25 only
    pub fn search_keywords(&self, query: &str, limit: usize) -> Vec<SearchResult> {
        let results = self.bm25.search(query, limit);
        self.format_results(results, "keyword")
    }

    /// Search using semantic similarity only
    pub fn search_semantic(&self, query_embedding: &[f32], limit: usize) -> Vec<SearchResult> {
        let results = self.vectors.search(query_embedding, limit);
        self.format_results(results, "semantic")
    }

    /// Hybrid search combining BM25 and semantic search
    pub fn search_hybrid(
        &self,
        query: &str,
        query_embedding: &[f32],
        limit: usize,
    ) -> Vec<SearchResult> {
        let keyword_results = self.bm25.search(query, limit * 2);
        let semantic_results = self.vectors.search(query_embedding, limit * 2);

        let fused = hybrid_search(keyword_results, semantic_results, limit);
        self.format_results(fused, "hybrid")
    }

    fn format_results(
        &self,
        results: Vec<(usize, f32)>,
        search_type: &str,
    ) -> Vec<SearchResult> {
        unimplemented!()
        // results
        //     .into_iter()
        //     .filter_map(|(idx, score)| {
        //         if idx < self.documents.len() {
        //             Some(SearchResult {
        //                 score,
        //                 chunk_id: format!("doc_{}", idx),
        //                 text: self.documents[idx].clone(),
        //                 metadata: self.metadata.get(idx).cloned().unwrap_or_default(),
        //             })
        //         } else {
        //             None
        //         }
        //     })
        //     .collect()
    }

    /// Get number of indexed documents
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
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