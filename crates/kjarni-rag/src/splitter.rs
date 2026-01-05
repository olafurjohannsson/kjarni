//! Simple text splitting utilities
use std::collections::HashMap;

/// Configuration for text splitting
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

/// Simple text splitter
pub struct TextSplitter {
    config: SplitterConfig,
}

impl TextSplitter {
    pub fn new(config: SplitterConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(SplitterConfig::default())
    }

    pub fn split(&self, text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let sections: Vec<&str> = text.split(&self.config.separator).collect();
        let mut current_chunk = String::new();

        for section in sections {
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
            // Note: We check byte length for speed, which is safe but approximate relative to "chars" config
            if current_chunk.len() + section.len() > self.config.chunk_size
                && !current_chunk.is_empty()
            {
                chunks.push(current_chunk.clone());

                // 3. Handle Overlap SAFELY (UTF-8 aware)
                if self.config.chunk_overlap > 0 {
                    // Find the byte index that starts the last 'chunk_overlap' characters
                    let mut indices = current_chunk.char_indices().rev();

                    // We look for the character at position 'overlap' from the end
                    if let Some((idx, _)) = indices.nth(self.config.chunk_overlap - 1) {
                        current_chunk = current_chunk[idx..].to_string();
                    } else {
                        // The chunk is shorter than the overlap, keep the whole thing
                    }
                } else {
                    current_chunk.clear();
                }
            }

            if !current_chunk.is_empty() {
                current_chunk.push_str(&self.config.separator);
            }
            current_chunk.push_str(section);
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }

    fn split_large_text(&self, text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut start = 0;

        while start < chars.len() {
            let end = (start + self.config.chunk_size).min(chars.len());
            let chunk: String = chars[start..end].iter().collect();
            chunks.push(chunk);

            // FIX: Infinite loop prevention
            if end == chars.len() {
                break;
            }

            start = if self.config.chunk_overlap > 0 {
                let next_start = end.saturating_sub(self.config.chunk_overlap);
                // Ensure we always make progress to avoid infinite loops
                if next_start <= start {
                    start + 1
                } else {
                    next_start
                }
            } else {
                end
            };
        }

        chunks
    }

    /// Split text into chunks with metadata
    pub fn split_with_metadata(
        &self,
        text: &str,
        base_metadata: HashMap<String, String>,
    ) -> Vec<(String, HashMap<String, String>)> {
        unimplemented!()
        // let chunks = self.split(text);
        // chunks
        //     .into_iter()
        //     .enumerate()
        //     .map(|(i, chunk)| {
        //         let mut metadata = base_metadata.clone();
        //         metadata.insert("chunk_index".to_string(), i.to_string());
        //         metadata.insert("total_chunks".to_string(), chunks.len().to_string());
        //         (chunk, metadata)
        //     })
        //     .collect()
    }
}
