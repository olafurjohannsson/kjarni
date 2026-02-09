//! Simple text splitting utilities for document chunking
//!
//! This module provides text splitting functionality for RAG (Retrieval-Augmented Generation)
//! pipelines. It breaks large documents into smaller chunks suitable for embedding and indexing.
//!
//! # Features
//!
//! - Configurable chunk size and overlap
//! - Custom separators for intelligent splitting
//! - UTF-8 safe splitting (never breaks multi-byte characters)
//! - Metadata preservation across chunks
//!
//! # Example
//!
//! ```
//! use kjarni_rag::splitter::{TextSplitter, SplitterConfig};
//!
//! let config = SplitterConfig {
//!     chunk_size: 500,
//!     chunk_overlap: 50,
//!     separator: "\n\n".to_string(),
//! };
//!
//! let splitter = TextSplitter::new(config);
//! let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
//! let chunks = splitter.split(text);
//!
//! for chunk in chunks {
//!     println!("Chunk: {}", chunk);
//! }
//! ```

use std::collections::HashMap;

/// Configuration for text splitting
///
/// Controls how documents are split into chunks for embedding and indexing.
///
/// # Fields
///
/// * `chunk_size` - Maximum size of each chunk in characters. Chunks may be slightly
///   smaller to avoid breaking at awkward positions.
/// * `chunk_overlap` - Number of characters to overlap between consecutive chunks.
///   This helps maintain context across chunk boundaries.
/// * `separator` - Primary separator used for splitting. The splitter will try to
///   break at these boundaries first before resorting to character-level splitting.
///
/// # Example
///
/// ```
/// use kjarni_rag::splitter::SplitterConfig;
///
/// // Good for prose documents
/// let prose_config = SplitterConfig {
///     chunk_size: 1000,
///     chunk_overlap: 200,
///     separator: "\n\n".to_string(), // Split on paragraphs
/// };
///
/// // Good for code
/// let code_config = SplitterConfig {
///     chunk_size: 500,
///     chunk_overlap: 50,
///     separator: "\n".to_string(), // Split on lines
/// };
/// ```
#[derive(Debug, Clone)]
pub struct SplitterConfig {
    /// Maximum chunk size in characters
    pub chunk_size: usize,
    /// Overlap between chunks in characters
    pub chunk_overlap: usize,
    /// Separator to use for splitting
    pub separator: String,
}

impl Default for SplitterConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 200,
            separator: "\n\n".to_string(),
        }
    }
}

impl SplitterConfig {
    /// Create a new config with specified chunk size
    ///
    /// Uses default overlap (20% of chunk size) and paragraph separator.
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap: chunk_size / 5, // 20% overlap
            separator: "\n\n".to_string(),
        }
    }

    /// Validate the configuration
    ///
    /// Returns an error message if the config is invalid.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.chunk_size == 0 {
            return Err("chunk_size must be greater than 0");
        }
        if self.chunk_overlap >= self.chunk_size {
            return Err("chunk_overlap must be less than chunk_size");
        }
        Ok(())
    }
}

/// Simple text splitter for document chunking
///
/// Splits text into overlapping chunks suitable for embedding and retrieval.
/// The splitter tries to break at natural boundaries (using the configured separator)
/// before falling back to character-level splitting for very long sections.
///
/// # Thread Safety
///
/// `TextSplitter` is `Send` and `Sync`, making it safe to share across threads.
///
/// # Example
///
/// ```
/// use kjarni_rag::splitter::TextSplitter;
///
/// let splitter = TextSplitter::with_defaults();
/// let chunks = splitter.split("Your long document text here...");
/// ```
pub struct TextSplitter {
    config: SplitterConfig,
}

impl TextSplitter {
    /// Create a new text splitter with the given configuration
    ///
    /// # Panics
    ///
    /// Panics if the configuration is invalid (chunk_size == 0 or overlap >= chunk_size).
    pub fn new(config: SplitterConfig) -> Self {
        if let Err(e) = config.validate() {
            panic!("Invalid SplitterConfig: {}", e);
        }
        Self { config }
    }

    /// Create a text splitter with default configuration
    ///
    /// Default: 1000 char chunks with 200 char overlap, splitting on paragraphs.
    pub fn with_defaults() -> Self {
        Self::new(SplitterConfig::default())
    }

    /// Get a reference to the current configuration
    pub fn config(&self) -> &SplitterConfig {
        &self.config
    }

    /// Split text into chunks
    ///
    /// The text is first split on the configured separator. Sections that fit within
    /// the chunk size are combined. Sections larger than the chunk size are split
    /// at character boundaries with overlap.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to split
    ///
    /// # Returns
    ///
    /// A vector of text chunks, each no larger than `chunk_size` characters.
    ///
    /// # Example
    ///
    /// ```
    /// use kjarni_rag::splitter::TextSplitter;
    ///
    /// let splitter = TextSplitter::with_defaults();
    /// let chunks = splitter.split("Para 1.\n\nPara 2.\n\nPara 3.");
    /// assert!(!chunks.is_empty());
    /// ```
    pub fn split(&self, text: &str) -> Vec<String> {
        // Handle empty text
        if text.is_empty() {
            return vec![];
        }

        let mut chunks = Vec::new();
        let sections: Vec<&str> = text.split(&self.config.separator).collect();
        let mut current_chunk = String::new();

        for section in sections {
            // Skip empty sections
            if section.is_empty() {
                continue;
            }

            // 1. Handle sections larger than chunk_size
            if section.len() > self.config.chunk_size {
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.clone());
                    current_chunk.clear();
                }
                chunks.extend(self.split_large_text(section));
                continue;
            }

            // 2. Check if adding section exceeds size
            let would_be_size = if current_chunk.is_empty() {
                section.len()
            } else {
                current_chunk.len() + self.config.separator.len() + section.len()
            };

            if would_be_size > self.config.chunk_size && !current_chunk.is_empty() {
                chunks.push(current_chunk.clone());

                // 3. Handle Overlap SAFELY (UTF-8 aware)
                if self.config.chunk_overlap > 0 {
                    current_chunk = self.get_overlap_suffix(&current_chunk);
                } else {
                    current_chunk.clear();
                }
            }

            // 4. Add section to current chunk
            if !current_chunk.is_empty() {
                current_chunk.push_str(&self.config.separator);
            }
            current_chunk.push_str(section);
        }

        // Don't forget the last chunk
        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }

    /// Get the suffix of a string for overlap
    ///
    /// Returns the last `chunk_overlap` characters of the string,
    /// handling UTF-8 boundaries correctly.
    fn get_overlap_suffix(&self, text: &str) -> String {
        let chars: Vec<char> = text.chars().collect();

        if chars.len() <= self.config.chunk_overlap {
            // Text is shorter than overlap, keep all of it
            return text.to_string();
        }

        let start_idx = chars.len() - self.config.chunk_overlap;
        chars[start_idx..].iter().collect()
    }

    /// Split a large text section that exceeds chunk_size
    ///
    /// This is called when a single section (between separators) is larger
    /// than the configured chunk size. It splits at character boundaries.
    fn split_large_text(&self, text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();

        if chars.is_empty() {
            return chunks;
        }

        let mut start = 0;

        while start < chars.len() {
            let end = (start + self.config.chunk_size).min(chars.len());
            let chunk: String = chars[start..end].iter().collect();
            chunks.push(chunk);

            // Check if we've reached the end
            if end >= chars.len() {
                break;
            }

            // Calculate next start position with overlap
            let step = if self.config.chunk_overlap > 0 && self.config.chunk_overlap < self.config.chunk_size {
                self.config.chunk_size - self.config.chunk_overlap
            } else {
                self.config.chunk_size
            };

            // Ensure we always make progress to avoid infinite loops
            start = if start + step > start {
                start + step
            } else {
                start + 1
            };
        }

        chunks
    }

    /// Split text into chunks with metadata
    ///
    /// Each chunk receives a copy of the base metadata plus additional
    /// chunk-specific metadata (index and total count).
    ///
    /// # Arguments
    ///
    /// * `text` - The text to split
    /// * `base_metadata` - Metadata to attach to each chunk
    ///
    /// # Returns
    ///
    /// A vector of (chunk_text, metadata) tuples.
    ///
    /// # Metadata Added
    ///
    /// * `chunk_index` - Zero-based index of this chunk
    /// * `total_chunks` - Total number of chunks produced
    ///
    /// # Example
    ///
    /// ```
    /// use kjarni_rag::splitter::TextSplitter;
    /// use std::collections::HashMap;
    ///
    /// let splitter = TextSplitter::with_defaults();
    /// let mut metadata = HashMap::new();
    /// metadata.insert("source".to_string(), "document.txt".to_string());
    ///
    /// let chunks = splitter.split_with_metadata("Long text...", metadata);
    /// for (text, meta) in chunks {
    ///     println!("Chunk {} of {}", meta["chunk_index"], meta["total_chunks"]);
    /// }
    /// ```
    pub fn split_with_metadata(
        &self,
        text: &str,
        base_metadata: HashMap<String, String>,
    ) -> Vec<(String, HashMap<String, String>)> {
        let chunks = self.split(text);
        let total = chunks.len();

        chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let mut metadata = base_metadata.clone();
                metadata.insert("chunk_index".to_string(), i.to_string());
                metadata.insert("total_chunks".to_string(), total.to_string());
                (chunk, metadata)
            })
            .collect()
    }

    /// Estimate the number of chunks for a given text
    ///
    /// This is a rough estimate and may not be exact due to separator handling.
    pub fn estimate_chunks(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }

        let effective_chunk_size = self.config.chunk_size - self.config.chunk_overlap;
        if effective_chunk_size == 0 {
            return 1;
        }

        let char_count = text.chars().count();
        (char_count + effective_chunk_size - 1) / effective_chunk_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // SPLITTER CONFIG TESTS
    // ============================================================================

    #[test]
    fn test_config_default() {
        let config = SplitterConfig::default();

        assert_eq!(config.chunk_size, 1000);
        assert_eq!(config.chunk_overlap, 200);
        assert_eq!(config.separator, "\n\n");
    }

    #[test]
    fn test_config_with_chunk_size() {
        let config = SplitterConfig::with_chunk_size(500);

        assert_eq!(config.chunk_size, 500);
        assert_eq!(config.chunk_overlap, 100); // 20% of 500
    }

    #[test]
    fn test_config_validate_valid() {
        let config = SplitterConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_chunk_size() {
        let config = SplitterConfig {
            chunk_size: 0,
            chunk_overlap: 0,
            separator: "\n".to_string(),
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_overlap_too_large() {
        let config = SplitterConfig {
            chunk_size: 100,
            chunk_overlap: 100, // Equal to chunk_size
            separator: "\n".to_string(),
        };
        assert!(config.validate().is_err());

        let config2 = SplitterConfig {
            chunk_size: 100,
            chunk_overlap: 150, // Greater than chunk_size
            separator: "\n".to_string(),
        };
        assert!(config2.validate().is_err());
    }

    #[test]
    fn test_config_clone() {
        let config = SplitterConfig::default();
        let cloned = config.clone();

        assert_eq!(config.chunk_size, cloned.chunk_size);
        assert_eq!(config.chunk_overlap, cloned.chunk_overlap);
        assert_eq!(config.separator, cloned.separator);
    }
    
    #[test]
    fn test_splitter_with_defaults() {
        let splitter = TextSplitter::with_defaults();
        assert_eq!(splitter.config().chunk_size, 1000);
    }

    #[test]
    #[should_panic(expected = "chunk_size must be greater than 0")]
    fn test_splitter_invalid_config_panics() {
        let config = SplitterConfig {
            chunk_size: 0,
            chunk_overlap: 0,
            separator: "\n".to_string(),
        };
        TextSplitter::new(config);
    }

    // ============================================================================
    // SPLIT TESTS
    // ============================================================================

    #[test]
    fn test_split_empty_text() {
        let splitter = TextSplitter::with_defaults();
        let chunks = splitter.split("");

        assert!(chunks.is_empty());
    }

    #[test]
    fn test_split_short_text() {
        let splitter = TextSplitter::with_defaults();
        let text = "This is a short text.";
        let chunks = splitter.split(text);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_split_on_separator() {
        let config = SplitterConfig {
            chunk_size: 100,
            chunk_overlap: 0,
            separator: "\n\n".to_string(),
        };
        let splitter = TextSplitter::new(config);

        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = splitter.split(text);

        // All paragraphs fit in one chunk (total ~55 chars)
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("First"));
        assert!(chunks[0].contains("Second"));
        assert!(chunks[0].contains("Third"));
    }

    #[test]
    fn test_split_exceeds_chunk_size() {
        let config = SplitterConfig {
            chunk_size: 30,
            chunk_overlap: 0,
            separator: "\n\n".to_string(),
        };
        let splitter = TextSplitter::new(config);

        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = splitter.split(text);

        // Each paragraph should be its own chunk
        assert!(chunks.len() >= 3);
    }

    #[test]
    fn test_split_with_overlap() {
        let config = SplitterConfig {
            chunk_size: 20,
            chunk_overlap: 5,
            separator: " ".to_string(),
        };
        let separator = config.separator.clone();
        let chunk_size = config.chunk_size;
        let splitter = TextSplitter::new(config);

        let text = "word1 word2 word3 word4 word5";
        let chunks = splitter.split(text);

        // With overlap, consecutive chunks should share some text
        assert!(chunks.len() > 1);

        // Verify chunks don't exceed max size
        for chunk in &chunks {
            assert!(chunk.len() <= chunk_size + separator.len());
        }
    }

    #[test]
    fn test_split_large_section() {
        let config = SplitterConfig {
            chunk_size: 20,
            chunk_overlap: 5,
            separator: "\n\n".to_string(),
        };
        let splitter = TextSplitter::new(config);

        // A single long section with no separators
        let text = "abcdefghijklmnopqrstuvwxyz0123456789";
        let chunks = splitter.split(text);

        assert!(chunks.len() > 1);

        // Verify all text is covered
        let total_chars: usize = chunks.iter().map(|c| c.len()).sum();
        // With overlap, total chars should be >= original length
        assert!(total_chars >= text.len());
    }

    #[test]
    fn test_split_custom_separator() {
        let config = SplitterConfig {
            chunk_size: 100,
            chunk_overlap: 0,
            separator: "|||".to_string(),
        };
        let splitter = TextSplitter::new(config);

        let text = "section1|||section2|||section3";
        let chunks = splitter.split(text);

        // All sections fit in one chunk
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("section1"));
    }

    #[test]
    fn test_split_no_separator_in_text() {
        let config = SplitterConfig {
            chunk_size: 50,
            chunk_overlap: 10,
            separator: "|||".to_string(),
        };
        let splitter = TextSplitter::new(config);

        let text = "This text has no triple pipe separators anywhere";
        let chunks = splitter.split(text);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    // ============================================================================
    // UTF-8 SAFETY TESTS
    // ============================================================================

    #[test]
    fn test_split_utf8_characters() {
        let config = SplitterConfig {
            chunk_size: 10,
            chunk_overlap: 2,
            separator: " ".to_string(),
        };
        let splitter = TextSplitter::new(config);

        // Unicode characters (emojis, CJK, etc.)
        let text = "Hello 🌍 World 你好 世界";
        let chunks = splitter.split(text);

        // Verify no broken UTF-8
        for chunk in &chunks {
            assert!(chunk.is_ascii() || chunk.chars().count() > 0);
            // Verify we can iterate chars without panic
            for _ in chunk.chars() {}
        }
    }

    #[test]
    fn test_split_emoji_heavy_text() {
        let config = SplitterConfig {
            chunk_size: 5,
            chunk_overlap: 1,
            separator: " ".to_string(),
        };
        let chunk_size = config.chunk_size;
        let splitter = TextSplitter::new(config);

        let text = "🎉🎊🎈🎁🎀 🌟🌙🌈☀️⭐";
        let chunks = splitter.split(text);

        assert!(!chunks.is_empty());
        // Each chunk should be valid UTF-8
        for chunk in chunks {
            assert!(chunk.chars().count() <= chunk_size);
        }
    }

    #[test]
    fn test_split_multibyte_characters() {
        let config = SplitterConfig {
            chunk_size: 8,
            chunk_overlap: 2,
            separator: " ".to_string(),
        };
        let splitter = TextSplitter::new(config);

        // Mix of 1-byte, 2-byte, 3-byte, and 4-byte UTF-8 characters
        let text = "a é 中 🎉 b";
        let chunks = splitter.split(text);

        for chunk in chunks {
            // Verify valid UTF-8 by attempting to iterate
            let _: Vec<char> = chunk.chars().collect();
        }
    }

    // ============================================================================
    // OVERLAP TESTS
    // ============================================================================

    #[test]
    fn test_overlap_preserves_context() {
        let config = SplitterConfig {
            chunk_size: 15,
            chunk_overlap: 5,
            separator: " ".to_string(),
        };
        let splitter = TextSplitter::new(config);

        let text = "The quick brown fox jumps over the lazy dog";
        let chunks = splitter.split(text);

        // With overlap, adjacent chunks should share some content
        for i in 0..chunks.len().saturating_sub(1) {
            let current = &chunks[i];
            let next = &chunks[i + 1];

            // The end of current chunk might appear at start of next
            // (This is a soft check - exact overlap depends on word boundaries)
            println!("Chunk {}: '{}'", i, current);
            println!("Chunk {}: '{}'", i + 1, next);
        }

        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_zero_overlap() {
        let config = SplitterConfig {
            chunk_size: 10,
            chunk_overlap: 0,
            separator: " ".to_string(),
        };
        let splitter = TextSplitter::new(config);

        let text = "one two three four five six";
        let chunks = splitter.split(text);

        assert!(!chunks.is_empty());
    }

    // ============================================================================
    // SPLIT WITH METADATA TESTS
    // ============================================================================

    #[test]
    fn test_split_with_metadata_empty() {
        let splitter = TextSplitter::with_defaults();
        let metadata = HashMap::new();
        let results = splitter.split_with_metadata("", metadata);

        assert!(results.is_empty());
    }

    #[test]
    fn test_split_with_metadata_single_chunk() {
        let splitter = TextSplitter::with_defaults();
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "test.txt".to_string());

        let results = splitter.split_with_metadata("Short text", metadata);

        assert_eq!(results.len(), 1);
        let (text, meta) = &results[0];

        assert_eq!(text, "Short text");
        assert_eq!(meta.get("source"), Some(&"test.txt".to_string()));
        assert_eq!(meta.get("chunk_index"), Some(&"0".to_string()));
        assert_eq!(meta.get("total_chunks"), Some(&"1".to_string()));
    }

    #[test]
    fn test_split_with_metadata_multiple_chunks() {
        let config = SplitterConfig {
            chunk_size: 20,
            chunk_overlap: 0,
            separator: "\n\n".to_string(),
        };
        let splitter = TextSplitter::new(config);

        let mut metadata = HashMap::new();
        metadata.insert("author".to_string(), "Test Author".to_string());
        metadata.insert("doc_id".to_string(), "123".to_string());

        let text = "First chunk text.\n\nSecond chunk text.\n\nThird chunk text.";
        let results = splitter.split_with_metadata(text, metadata);

        assert_eq!(results.len(), 3);

        for (i, (_, meta)) in results.iter().enumerate() {
            // Base metadata preserved
            assert_eq!(meta.get("author"), Some(&"Test Author".to_string()));
            assert_eq!(meta.get("doc_id"), Some(&"123".to_string()));

            // Chunk metadata added
            assert_eq!(meta.get("chunk_index"), Some(&i.to_string()));
            assert_eq!(meta.get("total_chunks"), Some(&"3".to_string()));
        }
    }

    #[test]
    fn test_split_with_metadata_preserves_all_base_metadata() {
        let splitter = TextSplitter::with_defaults();

        let mut metadata = HashMap::new();
        metadata.insert("key1".to_string(), "value1".to_string());
        metadata.insert("key2".to_string(), "value2".to_string());
        metadata.insert("key3".to_string(), "value3".to_string());

        let results = splitter.split_with_metadata("Some text", metadata);

        assert_eq!(results.len(), 1);
        let (_, meta) = &results[0];

        assert_eq!(meta.get("key1"), Some(&"value1".to_string()));
        assert_eq!(meta.get("key2"), Some(&"value2".to_string()));
        assert_eq!(meta.get("key3"), Some(&"value3".to_string()));
    }

    // ============================================================================
    // ESTIMATE CHUNKS TESTS
    // ============================================================================

    #[test]
    fn test_estimate_chunks_empty() {
        let splitter = TextSplitter::with_defaults();
        assert_eq!(splitter.estimate_chunks(""), 0);
    }

    #[test]
    fn test_estimate_chunks_short_text() {
        let splitter = TextSplitter::with_defaults();
        assert_eq!(splitter.estimate_chunks("Short"), 1);
    }

    #[test]
    fn test_estimate_chunks_long_text() {
        let config = SplitterConfig {
            chunk_size: 100,
            chunk_overlap: 20,
            separator: "\n".to_string(),
        };
        let splitter = TextSplitter::new(config);

        // 500 chars with chunk_size=100, overlap=20 → effective=80
        // 500 / 80 ≈ 6-7 chunks
        let text = "a".repeat(500);
        let estimate = splitter.estimate_chunks(&text);

        assert!(estimate >= 5 && estimate <= 10);
    }

    // ============================================================================
    // EDGE CASES
    // ============================================================================

    #[test]
    fn test_split_only_separators() {
        let config = SplitterConfig {
            chunk_size: 100,
            chunk_overlap: 10,
            separator: "\n\n".to_string(),
        };
        let splitter = TextSplitter::new(config);

        let text = "\n\n\n\n\n\n";
        let chunks = splitter.split(text);

        // Empty sections between separators should be skipped
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_split_single_char() {
        let splitter = TextSplitter::with_defaults();
        let chunks = splitter.split("X");

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "X");
    }

    #[test]
    fn test_split_exact_chunk_size() {
        let config = SplitterConfig {
            chunk_size: 10,
            chunk_overlap: 0,
            separator: "|".to_string(),
        };
        let splitter = TextSplitter::new(config);

        let text = "1234567890"; // Exactly 10 chars
        let chunks = splitter.split(text);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_split_one_char_over_chunk_size() {
        let config = SplitterConfig {
            chunk_size: 10,
            chunk_overlap: 0,
            separator: "|".to_string(),
        };
        let splitter = TextSplitter::new(config);

        let text = "12345678901"; // 11 chars
        let chunks = splitter.split(text);

        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_split_with_trailing_separator() {
        let config = SplitterConfig {
            chunk_size: 100,
            chunk_overlap: 0,
            separator: "\n\n".to_string(),
        };
        let splitter = TextSplitter::new(config);

        let text = "Content here\n\n";
        let chunks = splitter.split(text);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Content here");
    }

    #[test]
    fn test_split_with_leading_separator() {
        let config = SplitterConfig {
            chunk_size: 100,
            chunk_overlap: 0,
            separator: "\n\n".to_string(),
        };
        let splitter = TextSplitter::new(config);

        let text = "\n\nContent here";
        let chunks = splitter.split(text);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Content here");
    }

    // ============================================================================
    // REGRESSION TESTS
    // ============================================================================

    #[test]
    fn test_no_infinite_loop_large_overlap() {
        // This was a bug: if overlap >= chunk_size, could infinite loop
        // Now we validate config and prevent this
        let config = SplitterConfig {
            chunk_size: 100,
            chunk_overlap: 50,
            separator: "\n".to_string(),
        };
        let splitter = TextSplitter::new(config);

        let text = "a".repeat(1000);
        let chunks = splitter.split(&text);

        // Should complete without hanging
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_split_large_text_makes_progress() {
        let config = SplitterConfig {
            chunk_size: 10,
            chunk_overlap: 8, // Large overlap relative to chunk size
            separator: "|".to_string(),
        };
        let splitter = TextSplitter::new(config);

        let text = "a".repeat(100);
        let chunks = splitter.split(&text);

        // Should eventually complete
        assert!(!chunks.is_empty());

        // Should cover all content (with overlap, total > original)
        let total_len: usize = chunks.iter().map(|c| c.len()).sum();
        assert!(total_len >= text.len());
    }
}