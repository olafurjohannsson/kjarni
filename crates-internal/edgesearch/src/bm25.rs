use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// BM25 scoring parameters
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Bm25Params {
    /// Controls term frequency saturation (typically 1.2-2.0)
    /// Higher values = term frequency has more impact
    #[serde(default = "default_k1")]
    pub k1: f32,

    /// Controls length normalization (typically 0.5-0.8)
    /// 0.0 = no length normalization, 1.0 = full normalization
    #[serde(default = "default_b")]
    pub b: f32,

    /// Epsilon value to prevent log(0)
    #[serde(default = "default_epsilon")]
    pub epsilon: f32,
}

fn default_k1() -> f32 {
    1.2
}
fn default_b() -> f32 {
    0.75
}
fn default_epsilon() -> f32 {
    0.25
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self {
            k1: default_k1(),
            b: default_b(),
            epsilon: default_epsilon(),
        }
    }
}

/// BM25 index for efficient keyword search
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Bm25Index {
    /// Document frequencies for each term
    doc_frequencies: HashMap<String, usize>,

    /// Document lengths (in tokens)
    doc_lengths: Vec<usize>,

    /// Average document length
    avg_doc_length: f32,

    /// Total number of documents
    total_docs: usize,

    /// Inverted index: term -> list of (doc_id, term_frequency)
    inverted_index: HashMap<String, Vec<(usize, usize)>>,

    /// BM25 parameters
    params: Bm25Params,

    /// Token to index mapping for faster lookups
    token_to_docs: HashMap<String, HashSet<usize>>,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self {
            doc_frequencies: HashMap::new(),
            doc_lengths: Vec::new(),
            avg_doc_length: 0.0,
            total_docs: 0,
            inverted_index: HashMap::new(),
            params: Bm25Params::default(),
            token_to_docs: HashMap::new(),
        }
    }

    pub fn search(&self, query: &str, limit: usize) -> Vec<(usize, f32)> {
        if self.total_docs == 0 {
            return Vec::new();
        }

        let query_tokens = tokenize(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<usize, f32> = HashMap::new();

        for doc_id in 0..self.total_docs {
            let score = self.calculate_score(&query_tokens, doc_id);
            if score > 0.0 {
                scores.insert(doc_id, score);
            }
        }

        let mut results: Vec<(usize, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
    pub fn add_document(&mut self, doc_id: usize, text: &str) {
        let tokens = tokenize(text);
        let doc_length = tokens.len();

        // Update doc lengths
        if doc_id >= self.doc_lengths.len() {
            self.doc_lengths.resize(doc_id + 1, 0);
        }
        self.doc_lengths[doc_id] = doc_length;

        // Count term frequencies
        let mut term_counts: HashMap<String, usize> = HashMap::new();
        for token in &tokens {
            *term_counts.entry(token.clone()).or_insert(0) += 1;
        }

        // Update inverted index and doc frequencies
        for (term, count) in term_counts {
            self.inverted_index
                .entry(term.clone())
                .or_insert_with(Vec::new)
                .push((doc_id, count));

            *self.doc_frequencies.entry(term).or_insert(0) += 1;
        }

        self.total_docs = self.total_docs.max(doc_id + 1);
        self.avg_doc_length =
            self.doc_lengths.iter().sum::<usize>() as f32 / self.total_docs as f32;
    }
    /// Calculate BM25 score for a document given query terms
    fn calculate_score(&self, query_tokens: &[String], doc_id: usize) -> f32 {
        let mut score = 0.0;
        let doc_length = self.doc_lengths[doc_id] as f32;

        // Length normalization factor
        let length_norm = 1.0 - self.params.b + self.params.b * (doc_length / self.avg_doc_length);

        for term in query_tokens {
            // Get term frequency in this document
            let tf = self.get_term_frequency(term, doc_id) as f32;
            if tf == 0.0 {
                continue;
            }

            // Get document frequency
            let df = self.doc_frequencies.get(term).copied().unwrap_or(0) as f32;
            if df == 0.0 {
                continue;
            }

            // Calculate IDF component
            // Using log((N - df + 0.5) / (df + 0.5) + 1) for better scores
            let idf = ((self.total_docs as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();

            // Calculate normalized term frequency with saturation
            let normalized_tf = (tf * (self.params.k1 + 1.0)) / (tf + self.params.k1 * length_norm);

            score += idf * normalized_tf;
        }

        score
    }

    fn get_term_frequency(&self, term: &str, doc_id: usize) -> usize {
        self.inverted_index
            .get(term)
            .and_then(|postings| {
                postings
                    .iter()
                    .find(|(id, _)| *id == doc_id)
                    .map(|(_, freq)| *freq)
            })
            .unwrap_or(0)
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() >= 2)
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_params_default() {
        let params = Bm25Params::default();
        assert_eq!(params.k1, 1.2);
        assert_eq!(params.b, 0.75);
        assert_eq!(params.epsilon, 0.25);
    }

    #[test]
    fn test_bm25_index_new() {
        let index = Bm25Index::new();
        assert_eq!(index.total_docs, 0);
        assert_eq!(index.avg_doc_length, 0.0);
        assert!(index.doc_frequencies.is_empty());
        assert!(index.inverted_index.is_empty());
    }

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello World");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_filters_short() {
        let tokens = tokenize("I am a test");
        // "I", "a" filtered out (< 2 chars)
        assert_eq!(tokens, vec!["am", "test"]);
    }

    #[test]
    fn test_tokenize_handles_punctuation() {
        let tokens = tokenize("hello, world! how are you?");
        assert_eq!(tokens, vec!["hello", "world", "how", "are", "you"]);
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());

        let tokens = tokenize("   ");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_search_empty_index() {
        let index = Bm25Index::new();
        let results = index.search("test query", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_empty_query() {
        let mut index = Bm25Index::new();
        index.total_docs = 1;
        index.doc_lengths = vec![10];
        index.avg_doc_length = 10.0;

        let results = index.search("", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_index_with_documents() {
        let mut index = Bm25Index::new();

        // Manually set up index state (simulating add_document)
        index.total_docs = 3;
        index.doc_lengths = vec![5, 3, 7];
        index.avg_doc_length = 5.0;

        // Doc 0: "rust programming language"
        // Doc 1: "python programming"
        // Doc 2: "rust is fast and safe"

        index.doc_frequencies.insert("rust".to_string(), 2);
        index.doc_frequencies.insert("programming".to_string(), 2);
        index.doc_frequencies.insert("language".to_string(), 1);
        index.doc_frequencies.insert("python".to_string(), 1);
        index.doc_frequencies.insert("fast".to_string(), 1);
        index.doc_frequencies.insert("safe".to_string(), 1);

        index
            .inverted_index
            .insert("rust".to_string(), vec![(0, 1), (2, 1)]);
        index
            .inverted_index
            .insert("programming".to_string(), vec![(0, 1), (1, 1)]);
        index
            .inverted_index
            .insert("language".to_string(), vec![(0, 1)]);
        index
            .inverted_index
            .insert("python".to_string(), vec![(1, 1)]);
        index
            .inverted_index
            .insert("fast".to_string(), vec![(2, 1)]);
        index
            .inverted_index
            .insert("safe".to_string(), vec![(2, 1)]);

        // Search for "rust"
        let results = index.search("rust", 10);
        assert!(!results.is_empty());

        // Both doc 0 and doc 2 should match
        let doc_ids: Vec<usize> = results.iter().map(|(id, _)| *id).collect();
        assert!(doc_ids.contains(&0));
        assert!(doc_ids.contains(&2));
        assert!(!doc_ids.contains(&1)); // Doc 1 has no "rust"
    }

    #[test]
    fn test_bm25_score_ordering() {
        let mut index = Bm25Index::new();

        index.total_docs = 2;
        index.doc_lengths = vec![10, 10];
        index.avg_doc_length = 10.0;

        // Doc 0: term appears once
        // Doc 1: term appears 3 times (should score higher)
        index.doc_frequencies.insert("test".to_string(), 2);
        index
            .inverted_index
            .insert("test".to_string(), vec![(0, 1), (1, 3)]);

        let results = index.search("test", 10);
        assert_eq!(results.len(), 2);

        // Doc 1 (more occurrences) should rank first
        assert_eq!(results[0].0, 1);
        assert_eq!(results[1].0, 0);
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn test_bm25_idf_effect() {
        let mut index = Bm25Index::new();

        index.total_docs = 10;
        index.doc_lengths = vec![10; 10];
        index.avg_doc_length = 10.0;

        // "rare" appears in 1 doc, "common" appears in 9 docs
        index.doc_frequencies.insert("rare".to_string(), 1);
        index.doc_frequencies.insert("common".to_string(), 9);

        index
            .inverted_index
            .insert("rare".to_string(), vec![(0, 1)]);
        index.inverted_index.insert(
            "common".to_string(),
            vec![
                (0, 1),
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
                (5, 1),
                (6, 1),
                (7, 1),
                (8, 1),
            ],
        );

        // Score for rare term should be higher (higher IDF)
        let rare_score = index.calculate_score(&["rare".to_string()], 0);
        let common_score = index.calculate_score(&["common".to_string()], 0);

        assert!(
            rare_score > common_score,
            "Rare term should have higher IDF score"
        );
    }

    #[test]
    fn test_bm25_length_normalization() {
        let mut index = Bm25Index::new();

        index.total_docs = 2;
        index.doc_lengths = vec![5, 50]; // Short vs long doc
        index.avg_doc_length = 27.5;

        index.doc_frequencies.insert("test".to_string(), 2);
        index
            .inverted_index
            .insert("test".to_string(), vec![(0, 1), (1, 1)]);

        let short_doc_score = index.calculate_score(&["test".to_string()], 0);
        let long_doc_score = index.calculate_score(&["test".to_string()], 1);

        // Same term frequency, but short doc should score higher (length normalization)
        assert!(
            short_doc_score > long_doc_score,
            "Shorter doc should score higher with same TF"
        );
    }

    #[test]
    fn test_search_limit() {
        let mut index = Bm25Index::new();

        index.total_docs = 10;
        index.doc_lengths = vec![10; 10];
        index.avg_doc_length = 10.0;

        index.doc_frequencies.insert("test".to_string(), 10);
        index
            .inverted_index
            .insert("test".to_string(), (0..10).map(|i| (i, 1)).collect());

        let results = index.search("test", 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_get_term_frequency() {
        let mut index = Bm25Index::new();

        index
            .inverted_index
            .insert("hello".to_string(), vec![(0, 5), (2, 3)]);

        assert_eq!(index.get_term_frequency("hello", 0), 5);
        assert_eq!(index.get_term_frequency("hello", 2), 3);
        assert_eq!(index.get_term_frequency("hello", 1), 0); // Not present
        assert_eq!(index.get_term_frequency("nonexistent", 0), 0);
    }

    #[test]
    fn test_multi_term_query() {
        let mut index = Bm25Index::new();

        index.total_docs = 3;
        index.doc_lengths = vec![10, 10, 10];
        index.avg_doc_length = 10.0;

        // Doc 0: has "rust" only
        // Doc 1: has "fast" only
        // Doc 2: has both "rust" and "fast"
        index.doc_frequencies.insert("rust".to_string(), 2);
        index.doc_frequencies.insert("fast".to_string(), 2);

        index
            .inverted_index
            .insert("rust".to_string(), vec![(0, 1), (2, 1)]);
        index
            .inverted_index
            .insert("fast".to_string(), vec![(1, 1), (2, 1)]);

        let results = index.search("rust fast", 10);

        // Doc 2 should rank first (matches both terms)
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_bm25_params_serde() {
        let json = r#"{"k1": 1.5, "b": 0.8}"#;
        let params: Bm25Params = serde_json::from_str(json).unwrap();

        assert_eq!(params.k1, 1.5);
        assert_eq!(params.b, 0.8);
        assert_eq!(params.epsilon, 0.25); // Default
    }

    #[test]
    fn test_bm25_index_serde() {
        let mut index = Bm25Index::new();
        index.total_docs = 5;
        index.avg_doc_length = 15.0;

        let json = serde_json::to_string(&index).unwrap();
        let restored: Bm25Index = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.total_docs, 5);
        assert_eq!(restored.avg_doc_length, 15.0);
    }
    #[test]
    fn test_add_document() {
        let mut index = Bm25Index::new();

        index.add_document(0, "rust programming language");

        assert_eq!(index.total_docs, 1);
        assert_eq!(index.doc_lengths[0], 3);
        assert_eq!(index.avg_doc_length, 3.0);

        // Check doc frequencies
        assert_eq!(index.doc_frequencies.get("rust"), Some(&1));
        assert_eq!(index.doc_frequencies.get("programming"), Some(&1));
        assert_eq!(index.doc_frequencies.get("language"), Some(&1));

        // Check inverted index
        assert_eq!(index.get_term_frequency("rust", 0), 1);
    }

    #[test]
    fn test_add_multiple_documents() {
        let mut index = Bm25Index::new();

        index.add_document(0, "rust is fast");
        index.add_document(1, "python is slow");
        index.add_document(2, "rust and python");

        assert_eq!(index.total_docs, 3);
        assert_eq!(index.doc_frequencies.get("rust"), Some(&2));
        assert_eq!(index.doc_frequencies.get("python"), Some(&2));
        assert_eq!(index.doc_frequencies.get("is"), Some(&2));
        assert_eq!(index.doc_frequencies.get("fast"), Some(&1));

        // Average doc length
        let expected_avg = (3.0 + 3.0 + 3.0) / 3.0;
        assert!((index.avg_doc_length - expected_avg).abs() < 0.01);
    }

    #[test]
    fn test_add_document_repeated_terms() {
        let mut index = Bm25Index::new();

        index.add_document(0, "test test test hello");

        assert_eq!(index.get_term_frequency("test", 0), 3);
        assert_eq!(index.get_term_frequency("hello", 0), 1);
        assert_eq!(index.doc_frequencies.get("test"), Some(&1)); // Still 1 doc contains it
    }

    #[test]
    fn test_add_document_then_search() {
        let mut index = Bm25Index::new();

        index.add_document(0, "the quick brown fox");
        index.add_document(1, "the lazy dog");
        index.add_document(2, "quick quick fox jumps");

        let results = index.search("quick fox", 10);

        // Doc 2 should rank first (has "quick" twice + "fox")
        assert_eq!(results[0].0, 2);
        // Doc 0 should be second (has both terms once)
        assert_eq!(results[1].0, 0);
        // Doc 1 shouldn't appear (no matching terms)
        assert!(!results.iter().any(|(id, _)| *id == 1));
    }

    #[test]
    fn test_add_document_empty() {
        let mut index = Bm25Index::new();

        index.add_document(0, "");

        assert_eq!(index.total_docs, 1);
        assert_eq!(index.doc_lengths[0], 0);
        assert!(index.inverted_index.is_empty());
    }

    #[test]
    fn test_add_document_sparse_ids() {
        let mut index = Bm25Index::new();

        index.add_document(0, "first doc");
        index.add_document(5, "fifth doc"); // Skip IDs 1-4

        assert_eq!(index.total_docs, 6);
        assert_eq!(index.doc_lengths.len(), 6);
        assert_eq!(index.doc_lengths[0], 2);
        assert_eq!(index.doc_lengths[5], 2);
        assert_eq!(index.doc_lengths[3], 0); // Empty slot
    }
}
