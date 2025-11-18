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

    /// Split text into chunks
    pub fn split(&self, text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        
        // First split by separator
        let sections: Vec<&str> = text.split(&self.config.separator).collect();
        
        let mut current_chunk = String::new();
        
        for section in sections {
            // If section itself is too large, split it
            if section.len() > self.config.chunk_size {
                // Finalize current chunk if it has content
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.clone());
                    current_chunk.clear();
                }
                
                // Split large section by characters
                let large_chunks = self.split_large_text(section);
                chunks.extend(large_chunks);
                continue;
            }
            
            // Check if adding this section would exceed chunk size
            if current_chunk.len() + section.len() > self.config.chunk_size && !current_chunk.is_empty() {
                chunks.push(current_chunk.clone());
                
                // Start new chunk with overlap
                if self.config.chunk_overlap > 0 && current_chunk.len() > self.config.chunk_overlap {
                    let overlap_start = current_chunk.len() - self.config.chunk_overlap;
                    current_chunk = current_chunk[overlap_start..].to_string();
                } else {
                    current_chunk.clear();
                }
            }
            
            if !current_chunk.is_empty() {
                current_chunk.push_str(&self.config.separator);
            }
            current_chunk.push_str(section);
        }
        
        // Add final chunk
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
            
            start = if self.config.chunk_overlap > 0 {
                end.saturating_sub(self.config.chunk_overlap)
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