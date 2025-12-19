//! Index writer with automatic segmentation

use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::config::IndexConfig;
use super::segment::{SegmentBuilder, SegmentMeta};

/// Writes documents to a segmented index
pub struct IndexWriter {
    root: PathBuf,
    config: IndexConfig,
    current_builder: Option<SegmentBuilder>,
    next_segment_id: u64,
    total_docs: usize,
}

impl IndexWriter {
    /// Create or open an index for writing
    pub fn open(root: impl AsRef<Path>, config: IndexConfig) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        fs::create_dir_all(root.join("segments"))?;
        
        // Save config
        fs::write(
            root.join("config.json"),
            serde_json::to_string_pretty(&config)?,
        )?;
        
        // Find next segment ID
        let next_segment_id = Self::find_next_segment_id(&root)?;
        
        Ok(Self {
            root,
            config,
            current_builder: None,
            next_segment_id,
            total_docs: 0,
        })
    }
    
    /// Open an existing index for appending
    pub fn open_existing(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        
        let config: IndexConfig = serde_json::from_str(
            &fs::read_to_string(root.join("config.json"))?
        )?;
        
        let next_segment_id = Self::find_next_segment_id(&root)?;
        
        // Count existing docs
        let mut total_docs = 0;
        for entry in fs::read_dir(root.join("segments"))? {
            let entry = entry?;
            if entry.path().is_dir() {
                let meta_path = entry.path().join("segment.json");
                if meta_path.exists() {
                    let meta: SegmentMeta = serde_json::from_str(
                        &fs::read_to_string(meta_path)?
                    )?;
                    total_docs += meta.doc_count;
                }
            }
        }
        
        Ok(Self {
            root,
            config,
            current_builder: None,
            next_segment_id,
            total_docs,
        })
    }
    
    fn find_next_segment_id(root: &Path) -> Result<u64> {
        let segments_dir = root.join("segments");
        if !segments_dir.exists() {
            return Ok(0);
        }
        
        let mut max_id = 0u64;
        for entry in fs::read_dir(segments_dir)? {
            let entry = entry?;
            if let Some(name) = entry.file_name().to_str() {
                if let Some(id_str) = name.strip_prefix("seg_") {
                    if let Ok(id) = id_str.parse::<u64>() {
                        max_id = max_id.max(id + 1);
                    }
                }
            }
        }
        
        Ok(max_id)
    }
    
    /// Add a document to the index
    pub fn add(
        &mut self,
        text: &str,
        embedding: &[f32],
        metadata: Option<&HashMap<String, String>>,
    ) -> Result<()> {
        // Create builder if needed
        if self.current_builder.is_none() {
            let temp_dir = self.root.join("temp").join(format!("seg_{}", self.next_segment_id));
            self.current_builder = Some(SegmentBuilder::new(
                &temp_dir,
                self.config.dimension,
                self.config.max_docs_per_segment,
            )?);
        }
        
        let builder = self.current_builder.as_mut().unwrap();
        builder.add(text, embedding, metadata)?;
        self.total_docs += 1;
        
        // Flush if full
        if builder.is_full() {
            self.flush_current_segment()?;
        }
        
        Ok(())
    }
    
    /// Flush current segment to disk
    fn flush_current_segment(&mut self) -> Result<Option<SegmentMeta>> {
        if let Some(builder) = self.current_builder.take() {
            if builder.is_empty() {
                return Ok(None);
            }
            
            let segment_dir = self.root
                .join("segments")
                .join(format!("seg_{:06}", self.next_segment_id));
            
            let meta = builder.flush(&segment_dir, self.next_segment_id)?;
            self.next_segment_id += 1;
            
            log::info!("Flushed segment {} with {} documents", meta.id, meta.doc_count);
            
            return Ok(Some(meta));
        }
        
        Ok(None)
    }
    
    /// Commit all pending writes
    pub fn commit(mut self) -> Result<()> {
        self.flush_current_segment()?;
        
        // Update index metadata
        let index_meta = IndexMeta {
            total_docs: self.total_docs,
            segment_count: self.next_segment_id as usize,
            dimension: self.config.dimension,
        };
        
        fs::write(
            self.root.join("index.json"),
            serde_json::to_string_pretty(&index_meta)?,
        )?;
        
        // Cleanup temp directory
        let _ = fs::remove_dir_all(self.root.join("temp"));
        
        log::info!("Index committed: {} documents in {} segments",
            self.total_docs, self.next_segment_id);
        
        Ok(())
    }
    
    /// Get current document count
    pub fn len(&self) -> usize {
        self.total_docs
    }
    
    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct IndexMeta {
    total_docs: usize,
    segment_count: usize,
    dimension: usize,
}