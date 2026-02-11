//! Index reader with multi-segment search

use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::config::IndexConfig;
use super::segment::Segment;
use crate::SearchResult;
use kjarni_search::hybrid::hybrid_search;

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
            let any_match = self
                .should_match
                .iter()
                .any(|group| group.iter().all(|(k, v)| metadata.get(k) == Some(v)));
            if !any_match {
                return false;
            }
        }

        // Check source patterns
        if !self.source_patterns.is_empty() {
            if let Some(source) = metadata.get("source") {
                // Extract filename for patterns without path separators
                let filename = std::path::Path::new(source)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(source);

                let matches_pattern = self.source_patterns.iter().any(|pattern| {
                    if pattern.contains('/') {
                        // Pattern has path components - match against full path
                        glob_match::glob_match(pattern, source)
                    } else {
                        // Simple pattern - match against filename only
                        glob_match::glob_match(pattern, filename)
                    }
                });

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
        self.must_not_match
            .insert(key.to_string(), value.to_string());
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

    fn apply_filter(
        &self,
        results: Vec<SearchResult>,
        filter: &MetadataFilter,
        limit: usize,
    ) -> Vec<SearchResult> {
        results
            .into_iter()
            .filter(|r| filter.matches(&r.metadata))
            .take(limit)
            .collect()
    }
}

impl IndexReader {
    /// Open an index for reading
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();

        let config: IndexConfig =
            serde_json::from_str(&fs::read_to_string(root.join("config.json"))?)?;

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

        log::info!(
            "Opened index with {} segments, {} documents",
            segments.len(),
            total_docs
        );

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
        fused
            .into_iter()
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
        results
            .into_iter()
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
        let (seg_idx, local_id) = self
            .global_to_local(global_id)
            .ok_or_else(|| anyhow::anyhow!("Document ID out of range"))?;
        self.segments[seg_idx].get_document(local_id)
    }

    /// Get metadata by global ID
    pub fn get_metadata(&self, global_id: usize) -> Result<HashMap<String, String>> {
        let (seg_idx, local_id) = self
            .global_to_local(global_id)
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

#[cfg(test)]
mod index_reader_tests {
    use std::collections::HashMap;

    use crate::index_reader::MetadataFilter;

    #[test]
    fn test_filter_default() {
        let filter = MetadataFilter::default();

        assert!(filter.must_match.is_empty());
        assert!(filter.should_match.is_empty());
        assert!(filter.must_not_match.is_empty());
        assert!(filter.source_patterns.is_empty());
    }

    #[test]
    fn test_filter_builder_must() {
        let filter = MetadataFilter::default()
            .must("category", "news")
            .must("language", "en");

        assert_eq!(filter.must_match.len(), 2);
        assert_eq!(filter.must_match.get("category"), Some(&"news".to_string()));
        assert_eq!(filter.must_match.get("language"), Some(&"en".to_string()));
    }

    #[test]
    fn test_filter_builder_must_not() {
        let filter = MetadataFilter::default()
            .must_not("status", "draft")
            .must_not("visibility", "private");

        assert_eq!(filter.must_not_match.len(), 2);
        assert_eq!(
            filter.must_not_match.get("status"),
            Some(&"draft".to_string())
        );
        assert_eq!(
            filter.must_not_match.get("visibility"),
            Some(&"private".to_string())
        );
    }

    #[test]
    fn test_filter_builder_source() {
        let filter = MetadataFilter::default()
            .source("*.txt")
            .source("docs/*.md");

        assert_eq!(filter.source_patterns.len(), 2);
        assert!(filter.source_patterns.contains(&"*.txt".to_string()));
        assert!(filter.source_patterns.contains(&"docs/*.md".to_string()));
    }

    #[test]
    fn test_filter_builder_chained() {
        let filter = MetadataFilter::default()
            .must("type", "article")
            .must_not("archived", "true")
            .source("*.md");

        assert_eq!(filter.must_match.len(), 1);
        assert_eq!(filter.must_not_match.len(), 1);
        assert_eq!(filter.source_patterns.len(), 1);
    }

    #[test]
    fn test_matches_empty_filter() {
        let filter = MetadataFilter::default();
        let metadata = HashMap::new();

        // Empty filter matches everything
        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_matches_must_match_single() {
        let filter = MetadataFilter::default().must("key", "value");

        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), "value".to_string());

        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_matches_must_match_missing_key() {
        let filter = MetadataFilter::default().must("key", "value");

        let metadata = HashMap::new();

        assert!(!filter.matches(&metadata));
    }

    #[test]
    fn test_matches_must_match_wrong_value() {
        let filter = MetadataFilter::default().must("key", "expected");

        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), "actual".to_string());

        assert!(!filter.matches(&metadata));
    }

    #[test]
    fn test_matches_must_match_multiple_all_present() {
        let filter = MetadataFilter::default()
            .must("type", "article")
            .must("language", "en");

        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "article".to_string());
        metadata.insert("language".to_string(), "en".to_string());
        metadata.insert("extra".to_string(), "ignored".to_string());

        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_matches_must_match_multiple_one_missing() {
        let filter = MetadataFilter::default()
            .must("type", "article")
            .must("language", "en");

        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "article".to_string());

        assert!(!filter.matches(&metadata));
    }

    #[test]
    fn test_matches_must_match_multiple_one_wrong() {
        let filter = MetadataFilter::default()
            .must("type", "article")
            .must("language", "en");

        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "article".to_string());
        metadata.insert("language".to_string(), "de".to_string()); // Wrong value

        assert!(!filter.matches(&metadata));
    }

    #[test]
    fn test_matches_must_not_match_key_absent() {
        let filter = MetadataFilter::default().must_not("banned", "true");

        let metadata = HashMap::new();

        // Key not present, so doesn't match the banned value
        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_matches_must_not_match_different_value() {
        let filter = MetadataFilter::default().must_not("status", "draft");

        let mut metadata = HashMap::new();
        metadata.insert("status".to_string(), "published".to_string());

        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_matches_must_not_match_same_value() {
        let filter = MetadataFilter::default().must_not("status", "draft");

        let mut metadata = HashMap::new();
        metadata.insert("status".to_string(), "draft".to_string());

        assert!(!filter.matches(&metadata));
    }

    #[test]
    fn test_matches_must_not_match_multiple() {
        let filter = MetadataFilter::default()
            .must_not("deleted", "true")
            .must_not("archived", "true");

        let mut metadata = HashMap::new();
        metadata.insert("status".to_string(), "active".to_string());

        // Neither banned key is present
        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_matches_must_not_match_one_matches() {
        let filter = MetadataFilter::default()
            .must_not("deleted", "true")
            .must_not("archived", "true");

        let mut metadata = HashMap::new();
        metadata.insert("archived".to_string(), "true".to_string());

        assert!(!filter.matches(&metadata));
    }

    #[test]
    fn test_matches_should_match_empty() {
        let filter = MetadataFilter::default();

        let metadata = HashMap::new();

        // No should_match conditions means automatic pass
        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_matches_should_match_one_group_matches() {
        let mut filter = MetadataFilter::default();

        let mut group1 = HashMap::new();
        group1.insert("type".to_string(), "article".to_string());

        let mut group2 = HashMap::new();
        group2.insert("type".to_string(), "blog".to_string());

        filter.should_match = vec![group1, group2];

        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "article".to_string());

        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_matches_should_match_second_group_matches() {
        let mut filter = MetadataFilter::default();

        let mut group1 = HashMap::new();
        group1.insert("type".to_string(), "article".to_string());

        let mut group2 = HashMap::new();
        group2.insert("type".to_string(), "blog".to_string());

        filter.should_match = vec![group1, group2];

        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "blog".to_string());

        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_matches_should_match_no_groups_match() {
        let mut filter = MetadataFilter::default();

        let mut group1 = HashMap::new();
        group1.insert("type".to_string(), "article".to_string());

        let mut group2 = HashMap::new();
        group2.insert("type".to_string(), "blog".to_string());

        filter.should_match = vec![group1, group2];

        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "video".to_string());

        assert!(!filter.matches(&metadata));
    }

    #[test]
    fn test_matches_should_match_group_with_multiple_keys() {
        let mut filter = MetadataFilter::default();

        let mut group = HashMap::new();
        group.insert("type".to_string(), "article".to_string());
        group.insert("language".to_string(), "en".to_string());

        filter.should_match = vec![group];

        let mut metadata1 = HashMap::new();
        metadata1.insert("type".to_string(), "article".to_string());
        metadata1.insert("language".to_string(), "en".to_string());
        assert!(filter.matches(&metadata1));

        let mut metadata2 = HashMap::new();
        metadata2.insert("type".to_string(), "article".to_string());
        metadata2.insert("language".to_string(), "de".to_string());
        assert!(!filter.matches(&metadata2));
    }

    #[test]
    fn test_matches_source_pattern_no_patterns() {
        let filter = MetadataFilter::default();

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "anything.txt".to_string());

        // No patterns = no restriction
        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_matches_source_pattern_simple_glob() {
        let filter = MetadataFilter::default().source("*.txt");

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "document.txt".to_string());

        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_matches_source_pattern_no_match() {
        let filter = MetadataFilter::default().source("*.txt");

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "document.md".to_string());

        assert!(!filter.matches(&metadata));
    }

    #[test]
    fn test_matches_source_pattern_multiple_one_matches() {
        let filter = MetadataFilter::default().source("*.txt").source("*.md");

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "readme.md".to_string());

        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_matches_source_pattern_with_path() {
        let filter = MetadataFilter::default().source("docs/*.md");

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "docs/readme.md".to_string());

        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_matches_source_pattern_extracts_filename() {
        let filter = MetadataFilter::default().source("*.md");

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "path/to/document.md".to_string());

        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_matches_source_missing_source_key() {
        let filter = MetadataFilter::default().source("*.txt");

        let metadata = HashMap::new();
        assert!(!filter.matches(&metadata));
    }

    #[test]
    fn test_matches_source_pattern_exact_name() {
        let filter = MetadataFilter::default().source("README.md");

        let mut metadata1 = HashMap::new();
        metadata1.insert("source".to_string(), "README.md".to_string());
        assert!(filter.matches(&metadata1));

        let mut metadata2 = HashMap::new();
        metadata2.insert("source".to_string(), "README.txt".to_string());
        assert!(!filter.matches(&metadata2));
    }

    #[test]
    fn test_matches_combined_must_and_must_not() {
        let filter = MetadataFilter::default()
            .must("category", "tech")
            .must_not("archived", "true");

        // Passes both conditions
        let mut metadata1 = HashMap::new();
        metadata1.insert("category".to_string(), "tech".to_string());
        metadata1.insert("archived".to_string(), "false".to_string());
        assert!(filter.matches(&metadata1));

        // Fails must
        let mut metadata2 = HashMap::new();
        metadata2.insert("category".to_string(), "sports".to_string());
        assert!(!filter.matches(&metadata2));

        // Fails must_not
        let mut metadata3 = HashMap::new();
        metadata3.insert("category".to_string(), "tech".to_string());
        metadata3.insert("archived".to_string(), "true".to_string());
        assert!(!filter.matches(&metadata3));
    }

    #[test]
    fn test_matches_combined_all_conditions() {
        let mut filter = MetadataFilter::default()
            .must("type", "article")
            .must_not("deleted", "true")
            .source("*.md");

        let mut should_group = HashMap::new();
        should_group.insert("priority".to_string(), "high".to_string());
        filter.should_match = vec![should_group];

        // Passes all conditions
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "article".to_string());
        metadata.insert("priority".to_string(), "high".to_string());
        metadata.insert("source".to_string(), "docs/readme.md".to_string());

        assert!(filter.matches(&metadata));

        // Fails must_not
        let mut metadata2 = metadata.clone();
        metadata2.insert("deleted".to_string(), "true".to_string());
        assert!(!filter.matches(&metadata2));

        // Fails should_match
        let mut metadata3 = HashMap::new();
        metadata3.insert("type".to_string(), "article".to_string());
        metadata3.insert("priority".to_string(), "low".to_string());
        metadata3.insert("source".to_string(), "docs/readme.md".to_string());
        assert!(!filter.matches(&metadata3));

        // Fails source pattern
        let mut metadata4 = HashMap::new();
        metadata4.insert("type".to_string(), "article".to_string());
        metadata4.insert("priority".to_string(), "high".to_string());
        metadata4.insert("source".to_string(), "docs/readme.txt".to_string());
        assert!(!filter.matches(&metadata4));
    }

    #[test]
    fn test_matches_empty_string_values() {
        let filter = MetadataFilter::default().must("key", "");

        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), "".to_string());

        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_matches_whitespace_values() {
        let filter = MetadataFilter::default().must("key", "  ");

        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), "  ".to_string());

        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_matches_case_sensitive() {
        let filter = MetadataFilter::default().must("key", "Value");

        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), "value".to_string());

        // Case sensitive - should not match
        assert!(!filter.matches(&metadata));
    }

    #[test]
    fn test_matches_unicode_values() {
        let filter = MetadataFilter::default().must("greeting", "你好");

        let mut metadata = HashMap::new();
        metadata.insert("greeting".to_string(), "你好".to_string());

        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_matches_special_characters_in_pattern() {
        let filter = MetadataFilter::default().source("file[1].txt");

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "file1.txt".to_string());
        assert!(filter.matches(&metadata));

        let filter2 = MetadataFilter::default().source("file\\[1\\].txt");
        let mut metadata2 = HashMap::new();
        metadata2.insert("source".to_string(), "file[1].txt".to_string());
        assert!(filter2.matches(&metadata2));
    }

    #[test]
    fn test_glob_asterisk() {
        let filter = MetadataFilter::default().source("test_*");

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "test_file.txt".to_string());
        assert!(filter.matches(&metadata));

        metadata.insert("source".to_string(), "other_file.txt".to_string());
        assert!(!filter.matches(&metadata));
    }

    #[test]
    fn test_glob_question_mark() {
        let filter = MetadataFilter::default().source("file?.txt");

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "file1.txt".to_string());
        assert!(filter.matches(&metadata));

        metadata.insert("source".to_string(), "file12.txt".to_string());
        assert!(!filter.matches(&metadata));
    }

    #[test]
    fn test_glob_double_asterisk() {
        let filter = MetadataFilter::default().source("**/*.md");

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "deep/nested/path/doc.md".to_string());
        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_local_to_global_concept() {
        fn local_to_global(segment_sizes: &[usize], segment_idx: usize, local_id: usize) -> usize {
            let offset: usize = segment_sizes[..segment_idx].iter().sum();
            offset + local_id
        }

        let sizes = [10, 5, 8];

        assert_eq!(local_to_global(&sizes, 0, 0), 0);
        assert_eq!(local_to_global(&sizes, 0, 9), 9);
        assert_eq!(local_to_global(&sizes, 1, 0), 10);
        assert_eq!(local_to_global(&sizes, 1, 4), 14);
        assert_eq!(local_to_global(&sizes, 2, 0), 15);
        assert_eq!(local_to_global(&sizes, 2, 7), 22);
    }

    #[test]
    fn test_global_to_local_concept() {
        fn global_to_local(segment_sizes: &[usize], global_id: usize) -> Option<(usize, usize)> {
            let mut offset = 0;
            for (seg_idx, &size) in segment_sizes.iter().enumerate() {
                if global_id < offset + size {
                    return Some((seg_idx, global_id - offset));
                }
                offset += size;
            }
            None
        }

        let sizes = [10, 5, 8];

        assert_eq!(global_to_local(&sizes, 0), Some((0, 0)));
        assert_eq!(global_to_local(&sizes, 9), Some((0, 9)));
        assert_eq!(global_to_local(&sizes, 10), Some((1, 0)));
        assert_eq!(global_to_local(&sizes, 14), Some((1, 4)));
        assert_eq!(global_to_local(&sizes, 15), Some((2, 0)));
        assert_eq!(global_to_local(&sizes, 22), Some((2, 7)));
        assert_eq!(global_to_local(&sizes, 23), None); // Out of range
    }

    #[test]
    fn test_roundtrip_conversion() {
        fn local_to_global(segment_sizes: &[usize], segment_idx: usize, local_id: usize) -> usize {
            let offset: usize = segment_sizes[..segment_idx].iter().sum();
            offset + local_id
        }

        fn global_to_local(segment_sizes: &[usize], global_id: usize) -> Option<(usize, usize)> {
            let mut offset = 0;
            for (seg_idx, &size) in segment_sizes.iter().enumerate() {
                if global_id < offset + size {
                    return Some((seg_idx, global_id - offset));
                }
                offset += size;
            }
            None
        }

        let sizes = [10, 5, 8];

        for seg_idx in 0..sizes.len() {
            for local_id in 0..sizes[seg_idx] {
                let global = local_to_global(&sizes, seg_idx, local_id);
                let (back_seg, back_local) = global_to_local(&sizes, global).unwrap();

                assert_eq!(seg_idx, back_seg);
                assert_eq!(local_id, back_local);
            }
        }
    }

    #[test]
    fn test_filter_application_concept() {
        use crate::SearchResult;

        fn apply_filter(
            results: Vec<SearchResult>,
            filter: &MetadataFilter,
            limit: usize,
        ) -> Vec<SearchResult> {
            results
                .into_iter()
                .filter(|r| filter.matches(&r.metadata))
                .take(limit)
                .collect()
        }

        let mut results = Vec::new();

        let mut meta1 = HashMap::new();
        meta1.insert("category".to_string(), "tech".to_string());
        results.push(SearchResult {
            score: 0.9,
            document_id: 0,
            text: "Tech article".to_string(),
            metadata: meta1,
        });

        let mut meta2 = HashMap::new();
        meta2.insert("category".to_string(), "sports".to_string());
        results.push(SearchResult {
            score: 0.8,
            document_id: 1,
            text: "Sports article".to_string(),
            metadata: meta2,
        });

        let mut meta3 = HashMap::new();
        meta3.insert("category".to_string(), "tech".to_string());
        results.push(SearchResult {
            score: 0.7,
            document_id: 2,
            text: "Another tech article".to_string(),
            metadata: meta3,
        });

        let filter = MetadataFilter::default().must("category", "tech");
        let filtered = apply_filter(results, &filter, 10);

        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].document_id, 0);
        assert_eq!(filtered[1].document_id, 2);
    }

    #[test]
    fn test_filter_application_respects_limit() {
        use crate::SearchResult;

        fn apply_filter(
            results: Vec<SearchResult>,
            filter: &MetadataFilter,
            limit: usize,
        ) -> Vec<SearchResult> {
            results
                .into_iter()
                .filter(|r| filter.matches(&r.metadata))
                .take(limit)
                .collect()
        }

        let results: Vec<SearchResult> = (0..10)
            .map(|i| {
                let mut meta = HashMap::new();
                meta.insert("type".to_string(), "doc".to_string());
                SearchResult {
                    score: 1.0 - (i as f32 * 0.1),
                    document_id: i,
                    text: format!("Document {}", i),
                    metadata: meta,
                }
            })
            .collect();

        let filter = MetadataFilter::default().must("type", "doc");

        let filtered = apply_filter(results, &filter, 5);
        assert_eq!(filtered.len(), 5);

        assert_eq!(filtered[0].document_id, 0);
        assert_eq!(filtered[4].document_id, 4);
    }
}
