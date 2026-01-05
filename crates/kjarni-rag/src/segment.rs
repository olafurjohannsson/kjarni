//! Single segment of the index

use anyhow::{anyhow, Result};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::Bm25Index;

/// Segment metadata stored as JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMeta {
    pub id: u64,
    pub doc_count: usize,
    pub dimension: usize,
    pub created_at: u64,
    pub total_bytes: u64,
}

/// In-memory segment builder (accumulates until flush)
pub struct SegmentBuilder {
    pub dimension: usize,
    pub max_docs: usize,
    
    // Streaming writes - vectors go directly to temp file
    vectors_writer: BufWriter<File>,
    vectors_path: PathBuf,
    
    // Documents stored with offsets for retrieval
    docs_writer: BufWriter<File>,
    docs_path: PathBuf,
    doc_offsets: Vec<u64>,
    current_offset: u64,
    
    // BM25 index (kept in memory, serialized on flush)
    bm25: Bm25Index,
    
    // Metadata stored separately
    metadata_writer: BufWriter<File>,
    metadata_path: PathBuf,
    
    doc_count: usize,
}

impl SegmentBuilder {
    /// Create a new segment builder that streams to temp files
    pub fn new(temp_dir: &Path, dimension: usize, max_docs: usize) -> Result<Self> {
        fs::create_dir_all(temp_dir)?;
        
        let vectors_path = temp_dir.join("vectors.bin.tmp");
        let docs_path = temp_dir.join("docs.bin.tmp");
        let metadata_path = temp_dir.join("metadata.jsonl.tmp");
        
        let vectors_writer = BufWriter::with_capacity(
            64 * 1024, // 64KB buffer
            File::create(&vectors_path)?,
        );
        
        let docs_writer = BufWriter::with_capacity(
            64 * 1024,
            File::create(&docs_path)?,
        );
        
        let metadata_writer = BufWriter::with_capacity(
            32 * 1024,
            File::create(&metadata_path)?,
        );
        
        Ok(Self {
            dimension,
            max_docs,
            vectors_writer,
            vectors_path,
            docs_writer,
            docs_path,
            doc_offsets: Vec::with_capacity(max_docs),
            current_offset: 0,
            bm25: Bm25Index::new(),
            metadata_writer,
            metadata_path,
            doc_count: 0,
        })
    }
    
    /// Add a document with its embedding (streams to disk)
    pub fn add(
        &mut self,
        text: &str,
        embedding: &[f32],
        metadata: Option<&std::collections::HashMap<String, String>>,
    ) -> Result<usize> {
        if embedding.len() != self.dimension {
            return Err(anyhow!(
                "Embedding dimension mismatch: got {}, expected {}",
                embedding.len(),
                self.dimension
            ));
        }
        
        let doc_id = self.doc_count;
        
        // 1. Write embedding to vectors file (raw f32 bytes)
        for &val in embedding {
            self.vectors_writer.write_all(&val.to_le_bytes())?;
        }
        
        // 2. Write document text with newline delimiter
        self.doc_offsets.push(self.current_offset);
        let text_bytes = text.as_bytes();
        self.docs_writer.write_all(text_bytes)?;
        self.docs_writer.write_all(b"\n")?;
        self.current_offset += text_bytes.len() as u64 + 1;
        
        // 3. Write metadata as JSON line
        let meta_json = serde_json::to_string(&metadata.unwrap_or(&std::collections::HashMap::new()))?;
        self.metadata_writer.write_all(meta_json.as_bytes())?;
        self.metadata_writer.write_all(b"\n")?;
        
        // 4. Add to BM25 index (in memory)
        self.bm25.add_document(doc_id, text);
        
        self.doc_count += 1;
        Ok(doc_id)
    }
    
    /// Check if segment is full
    pub fn is_full(&self) -> bool {
        self.doc_count >= self.max_docs
    }
    
    /// Current document count
    pub fn len(&self) -> usize {
        self.doc_count
    }
    
    pub fn is_empty(&self) -> bool {
        self.doc_count == 0
    }
    
    /// Flush and finalize segment to permanent location
    pub fn flush(mut self, segment_dir: &Path, segment_id: u64) -> Result<SegmentMeta> {
        // Flush all writers
        self.vectors_writer.flush()?;
        self.docs_writer.flush()?;
        self.metadata_writer.flush()?;
        
        // Drop writers to release file handles
        drop(self.vectors_writer);
        drop(self.docs_writer);
        drop(self.metadata_writer);
        
        // Create segment directory
        fs::create_dir_all(segment_dir)?;
        
        // Move temp files to final location
        fs::rename(&self.vectors_path, segment_dir.join("vectors.bin"))?;
        fs::rename(&self.docs_path, segment_dir.join("docs.bin"))?;
        fs::rename(&self.metadata_path, segment_dir.join("metadata.jsonl"))?;
        
        // Write document offsets index
        let offsets_data = bincode::serialize(&self.doc_offsets)?;
        fs::write(segment_dir.join("docs.idx"), offsets_data)?;
        
        // Write BM25 index
        let bm25_data = bincode::serialize(&self.bm25)?;
        fs::write(segment_dir.join("bm25.bin"), bm25_data)?;
        
        // Write segment metadata
        let vectors_size = fs::metadata(segment_dir.join("vectors.bin"))?.len();
        let docs_size = fs::metadata(segment_dir.join("docs.bin"))?.len();
        
        let meta = SegmentMeta {
            id: segment_id,
            doc_count: self.doc_count,
            dimension: self.dimension,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            total_bytes: vectors_size + docs_size,
        };
        
        fs::write(
            segment_dir.join("segment.json"),
            serde_json::to_string_pretty(&meta)?,
        )?;
        
        // Cleanup temp dir if empty
        let _ = fs::remove_dir(self.vectors_path.parent().unwrap());
        
        Ok(meta)
    }
}

/// Read-only segment with memory-mapped vectors
pub struct Segment {
    pub meta: SegmentMeta,
    path: PathBuf,
    
    // Memory-mapped vectors (zero-copy access)
    vectors_mmap: Mmap,
    
    // Document offsets for retrieval
    doc_offsets: Vec<u64>,
    
    // BM25 index
    bm25: Bm25Index,
}

impl Segment {
    /// Open an existing segment
    pub fn open(path: &Path) -> Result<Self> {
        let meta: SegmentMeta = serde_json::from_str(
            &fs::read_to_string(path.join("segment.json"))?
        )?;
        
        // Memory-map vectors (doesn't load into RAM)
        let vectors_file = File::open(path.join("vectors.bin"))?;
        let vectors_mmap = unsafe { Mmap::map(&vectors_file)? };
        
        // Load document offsets (small)
        let doc_offsets: Vec<u64> = bincode::deserialize(
            &fs::read(path.join("docs.idx"))?
        )?;
        
        // Load BM25 index
        let bm25: Bm25Index = bincode::deserialize(
            &fs::read(path.join("bm25.bin"))?
        )?;
        
        Ok(Self {
            meta,
            path: path.to_path_buf(),
            vectors_mmap,
            doc_offsets,
            bm25,
        })
    }
    
    /// Get embedding by document ID (zero-copy from mmap)
    pub fn get_embedding(&self, doc_id: usize) -> Option<&[f32]> {
        if doc_id >= self.meta.doc_count {
            return None;
        }
        
        let start = doc_id * self.meta.dimension * 4;
        let end = start + self.meta.dimension * 4;
        
        if end > self.vectors_mmap.len() {
            return None;
        }
        
        // SAFETY: We're reinterpreting bytes as f32, assuming correct alignment
        // and that the data was written as little-endian f32
        let bytes = &self.vectors_mmap[start..end];
        let floats = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f32,
                self.meta.dimension,
            )
        };
        
        Some(floats)
    }
    
    /// Get document text by ID (reads from disk)
    pub fn get_document(&self, doc_id: usize) -> Result<String> {
        if doc_id >= self.meta.doc_count {
            return Err(anyhow!("Document ID out of range"));
        }
        
        let start = self.doc_offsets[doc_id];
        let end = if doc_id + 1 < self.doc_offsets.len() {
            self.doc_offsets[doc_id + 1] - 1 // -1 for newline
        } else {
            // Last document - read to end
            let file = File::open(self.path.join("docs.bin"))?;
            file.metadata()?.len() - 1
        };
        
        let file = File::open(self.path.join("docs.bin"))?;
        let mut reader = BufReader::new(file);
        
        use std::io::{Seek, SeekFrom, Read};
        reader.seek(SeekFrom::Start(start))?;
        
        let mut buffer = vec![0u8; (end - start) as usize];
        reader.read_exact(&mut buffer)?;
        
        String::from_utf8(buffer).map_err(|e| anyhow!("Invalid UTF-8: {}", e))
    }
    
    /// Get metadata by document ID
    pub fn get_metadata(&self, doc_id: usize) -> Result<std::collections::HashMap<String, String>> {
        let file = File::open(self.path.join("metadata.jsonl"))?;
        let reader = BufReader::new(file);
        
        for (i, line) in reader.lines().enumerate() {
            if i == doc_id {
                let line = line?;
                return serde_json::from_str(&line).map_err(|e| anyhow!("Invalid JSON: {}", e));
            }
        }
        
        Err(anyhow!("Document ID out of range"))
    }
    
    /// Search vectors using cosine similarity
    pub fn search_vectors(&self, query: &[f32], limit: usize) -> Vec<(usize, f32)> {
        if query.len() != self.meta.dimension {
            return vec![];
        }
        
        // Pre-compute query norm
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm < 1e-9 {
            return vec![];
        }
        
        let mut scores: Vec<(usize, f32)> = Vec::with_capacity(self.meta.doc_count);
        
        for doc_id in 0..self.meta.doc_count {
            if let Some(embedding) = self.get_embedding(doc_id) {
                let similarity = cosine_similarity_with_norm(query, embedding, query_norm);
                scores.push((doc_id, similarity));
            }
        }
        
        // Partial sort for top-k
        if scores.len() > limit {
            scores.select_nth_unstable_by(limit, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scores.truncate(limit);
        }
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }
    
    /// BM25 keyword search
    pub fn search_keywords(&self, query: &str, limit: usize) -> Vec<(usize, f32)> {
        self.bm25.search(query, limit)
    }
    
    /// Document count
    pub fn len(&self) -> usize {
        self.meta.doc_count
    }
    
    pub fn is_empty(&self) -> bool {
        self.meta.doc_count == 0
    }
}

/// Cosine similarity with pre-computed query norm
fn cosine_similarity_with_norm(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    let mut dot = 0.0f32;
    let mut b_norm_sq = 0.0f32;
    
    for i in 0..a.len() {
        dot += a[i] * b[i];
        b_norm_sq += b[i] * b[i];
    }
    
    let b_norm = b_norm_sq.sqrt();
    if b_norm < 1e-9 {
        return 0.0;
    }
    
    dot / (a_norm * b_norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_segment_roundtrip() {
        let temp = TempDir::new().unwrap();
        let build_dir = temp.path().join("build");
        let segment_dir = temp.path().join("seg_0");
        
        // Build segment
        let mut builder = SegmentBuilder::new(&build_dir, 4, 100).unwrap();
        
        builder.add("hello world", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        builder.add("goodbye world", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        
        let meta = builder.flush(&segment_dir, 0).unwrap();
        assert_eq!(meta.doc_count, 2);
        
        // Read segment
        let segment = Segment::open(&segment_dir).unwrap();
        assert_eq!(segment.len(), 2);
        
        // Check embeddings
        let emb0 = segment.get_embedding(0).unwrap();
        assert_eq!(emb0, &[1.0, 0.0, 0.0, 0.0]);
        
        let emb1 = segment.get_embedding(1).unwrap();
        assert_eq!(emb1, &[0.0, 1.0, 0.0, 0.0]);
        
        // Check documents
        assert_eq!(segment.get_document(0).unwrap(), "hello world");
        assert_eq!(segment.get_document(1).unwrap(), "goodbye world");
        
        // Search
        let results = segment.search_vectors(&[1.0, 0.0, 0.0, 0.0], 10);
        assert_eq!(results[0].0, 0); // First doc is most similar
    }
    
    #[test]
    fn test_bm25_search() {
        let temp = TempDir::new().unwrap();
        let build_dir = temp.path().join("build");
        let segment_dir = temp.path().join("seg_0");
        
        let mut builder = SegmentBuilder::new(&build_dir, 4, 100).unwrap();
        
        builder.add("rust programming language", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        builder.add("python scripting", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        builder.add("rust is fast", &[0.5, 0.5, 0.0, 0.0], None).unwrap();
        
        builder.flush(&segment_dir, 0).unwrap();
        
        let segment = Segment::open(&segment_dir).unwrap();
        let results = segment.search_keywords("rust", 10);
        
        // Should find docs 0 and 2
        let doc_ids: Vec<usize> = results.iter().map(|(id, _)| *id).collect();
        assert!(doc_ids.contains(&0));
        assert!(doc_ids.contains(&2));
        assert!(!doc_ids.contains(&1));
    }
}