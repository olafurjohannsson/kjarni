//! Index reader with multi-segment search

use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::config::IndexConfig;
use super::segment::Segment;
use kjarni_search::hybrid::hybrid_search;
use crate::SearchResult;

/// Reads and searches a segmented index
pub struct IndexReader {
    root: PathBuf,
    config: IndexConfig,
    segments: Vec<Segment>,
    total_docs: usize,
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