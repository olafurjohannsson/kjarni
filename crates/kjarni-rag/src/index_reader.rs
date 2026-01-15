//! Index reader with multi-segment search

use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::config::IndexConfig;
use super::segment::Segment;
use kjarni_search::hybrid::hybrid_search;
use crate::SearchResult;

/// Filter for metadata-based search
#[derive(Debug, Clone, Default)]
pub struct MetadataFilter {
    /// Required key-value matches (AND logic)
    pub must_match: HashMap<String, String>,
    /// At least one must match (OR logic)  
    pub should_match: Vec<HashMap<String, String>>,
    /// Must NOT have these values
    pub must_not_match: HashMap<String, String>,
    /// Source file patterns (glob)
    pub source_patterns: Vec<String>,
}


impl MetadataFilter {
    pub fn matches(&self, metadata: &HashMap<String, String>) -> bool {
        // Check must_match (all must be present and equal)
        for (k, v) in &self.must_match {
            if metadata.get(k) != Some(v) {
                return false;
            }
        }
        
        // Check must_not_match
        for (k, v) in &self.must_not_match {
            if metadata.get(k) == Some(v) {
                return false;
            }
        }
        
        // Check should_match (at least one group must match)
        if !self.should_match.is_empty() {
            let any_match = self.should_match.iter().any(|group| {
                group.iter().all(|(k, v)| metadata.get(k) == Some(v))
            });
            if !any_match {
                return false;
            }
        }
        
        // Check source patterns
        if !self.source_patterns.is_empty() {
            if let Some(source) = metadata.get("source") {
                let matches_pattern = self.source_patterns.iter()
                    .any(|p| glob_match(p, source));
                if !matches_pattern {
                    return false;
                }
            } else {
                return false;
            }
        }
        
        true
    }
    
    /// Builder: require source to match pattern
    pub fn source(mut self, pattern: &str) -> Self {
        self.source_patterns.push(pattern.to_string());
        self
    }
    
    /// Builder: require metadata key=value
    pub fn must(mut self, key: &str, value: &str) -> Self {
        self.must_match.insert(key.to_string(), value.to_string());
        self
    }
    
    /// Builder: exclude if key=value
    pub fn must_not(mut self, key: &str, value: &str) -> Self {
        self.must_not_match.insert(key.to_string(), value.to_string());
        self
    }
}

/// Reads and searches a segmented index
pub struct IndexReader {
    root: PathBuf,
    config: IndexConfig,
    segments: Vec<Segment>,
    total_docs: usize,
}

impl IndexReader {
    /// Search with metadata filtering
    pub fn search_semantic_filtered(
        &self,
        query_embedding: &[f32],
        limit: usize,
        filter: &MetadataFilter,
    ) -> Vec<SearchResult> {
        // Get more results, then filter
        let candidates = self.search_semantic(query_embedding, limit * 3);
        self.apply_filter(candidates, filter, limit)
    }
    
    /// Search keywords with filter
    pub fn search_keywords_filtered(
        &self,
        query: &str,
        limit: usize,
        filter: &MetadataFilter,
    ) -> Vec<SearchResult> {
        let candidates = self.search_keywords(query, limit * 3);
        self.apply_filter(candidates, filter, limit)
    }
    
    /// Hybrid search with filter
    pub fn search_hybrid_filtered(
        &self,
        query: &str,
        query_embedding: &[f32],
        limit: usize,
        filter: &MetadataFilter,
    ) -> Vec<SearchResult> {
        let candidates = self.search_hybrid(query, query_embedding, limit * 3);
        self.apply_filter(candidates, filter, limit)
    }
    
    fn apply_filter(&self, results: Vec<SearchResult>, filter: &MetadataFilter, limit: usize) -> Vec<SearchResult> {
        results.into_iter()
            .filter(|r| filter.matches(&r.metadata))
            .take(limit)
            .collect()
    }
}


impl IndexReader {
    /// Open an index for reading
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        
        let config: IndexConfig = serde_json::from_str(
            &fs::read_to_string(root.join("config.json"))?
        )?;
        
        // Load all segments
        let mut segments = Vec::new();
        let segments_dir = root.join("segments");
        
        if segments_dir.exists() {
            let mut entries: Vec<_> = fs::read_dir(&segments_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .collect();
            
            // Sort by segment ID
            entries.sort_by_key(|e| e.file_name());
            
            for entry in entries {
                match Segment::open(&entry.path()) {
                    Ok(segment) => segments.push(segment),
                    Err(e) => log::warn!("Failed to load segment {:?}: {}", entry.path(), e),
                }
            }
        }
        
        let total_docs = segments.iter().map(|s| s.len()).sum();
        
        log::info!("Opened index with {} segments, {} documents", 
            segments.len(), total_docs);
        
        Ok(Self {
            root,
            config,
            segments,
            total_docs,
        })
    }
    
    /// Semantic search across all segments
    pub fn search_semantic(&self, query_embedding: &[f32], limit: usize) -> Vec<SearchResult> {
        if query_embedding.len() != self.config.dimension {
            return vec![];
        }
        
        // Collect results from all segments
        let mut all_results: Vec<(usize, usize, f32)> = Vec::new(); // (segment_idx, doc_id, score)
        
        for (seg_idx, segment) in self.segments.iter().enumerate() {
            let segment_results = segment.search_vectors(query_embedding, limit);
            for (doc_id, score) in segment_results {
                all_results.push((seg_idx, doc_id, score));
            }
        }
        
        // Sort by score descending
        all_results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(limit);
        
        // Convert to SearchResults
        self.convert_results(all_results)
    }
    
    /// Keyword search across all segments
    pub fn search_keywords(&self, query: &str, limit: usize) -> Vec<SearchResult> {
        let mut all_results: Vec<(usize, usize, f32)> = Vec::new();
        
        for (seg_idx, segment) in self.segments.iter().enumerate() {
            let segment_results = segment.search_keywords(query, limit);
            for (doc_id, score) in segment_results {
                all_results.push((seg_idx, doc_id, score));
            }
        }
        
        all_results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(limit);
        
        self.convert_results(all_results)
    }
    
    /// Hybrid search (BM25 + semantic)
    pub fn search_hybrid(
        &self,
        query: &str,
        query_embedding: &[f32],
        limit: usize,
    ) -> Vec<SearchResult> {
        // Get more results for fusion
        let keyword_results = self.search_keywords(query, limit * 2);
        let semantic_results = self.search_semantic(query_embedding, limit * 2);
        
        // Convert to (global_id, score) format for fusion
        let keyword_indexed: Vec<(usize, f32)> = keyword_results
            .iter()
            .map(|r| (r.document_id, r.score))
            .collect();
        
        let semantic_indexed: Vec<(usize, f32)> = semantic_results
            .iter()
            .map(|r| (r.document_id, r.score))
            .collect();
        
        let fused = hybrid_search(keyword_indexed, semantic_indexed, limit);
        
        // Look up full results
        fused.into_iter()
            .filter_map(|(global_id, score)| {
                let (seg_idx, local_id) = self.global_to_local(global_id)?;
                let segment = &self.segments[seg_idx];
                
                let text = segment.get_document(local_id).ok()?;
                let metadata = segment.get_metadata(local_id).ok()?;
                
                Some(SearchResult {
                    score,
                    document_id: global_id,
                    text,
                    metadata,
                })
            })
            .collect()
    }
    
    /// Convert (segment_idx, local_doc_id, score) to SearchResults
    fn convert_results(&self, results: Vec<(usize, usize, f32)>) -> Vec<SearchResult> {
        results.into_iter()
            .filter_map(|(seg_idx, doc_id, score)| {
                let segment = &self.segments[seg_idx];
                let global_id = self.local_to_global(seg_idx, doc_id);
                
                let text = segment.get_document(doc_id).ok()?;
                let metadata = segment.get_metadata(doc_id).ok()?;
                
                Some(SearchResult {
                    score,
                    document_id: global_id,
                    text,
                    metadata,
                })
            })
            .collect()
    }
    
    /// Convert segment index + local doc ID to global ID
    fn local_to_global(&self, segment_idx: usize, local_doc_id: usize) -> usize {
        let mut offset = 0;
        for i in 0..segment_idx {
            offset += self.segments[i].len();
        }
        offset + local_doc_id
    }
    
    /// Convert global ID to (segment index, local doc ID)
    fn global_to_local(&self, global_id: usize) -> Option<(usize, usize)> {
        let mut offset = 0;
        for (seg_idx, segment) in self.segments.iter().enumerate() {
            if global_id < offset + segment.len() {
                return Some((seg_idx, global_id - offset));
            }
            offset += segment.len();
        }
        None
    }
    
    /// Get document by global ID
    pub fn get_document(&self, global_id: usize) -> Result<String> {
        let (seg_idx, local_id) = self.global_to_local(global_id)
            .ok_or_else(|| anyhow::anyhow!("Document ID out of range"))?;
        self.segments[seg_idx].get_document(local_id)
    }
    
    /// Get metadata by global ID
    pub fn get_metadata(&self, global_id: usize) -> Result<HashMap<String, String>> {
        let (seg_idx, local_id) = self.global_to_local(global_id)
            .ok_or_else(|| anyhow::anyhow!("Document ID out of range"))?;
        self.segments[seg_idx].get_metadata(local_id)
    }
    
    /// Total documents across all segments
    pub fn len(&self) -> usize {
        self.total_docs
    }
    
    pub fn is_empty(&self) -> bool {
        self.total_docs == 0
    }
    
    /// Embedding dimension
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }
    
    /// Number of segments
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}