//! kjarni-rag: RAG (Retrieval Augmented Generation) utilities

mod config;
mod index_reader;
mod index_writer;
mod loader;
mod progress;
mod search_index;
mod segment;
mod splitter;

pub use config::IndexConfig;
pub use index_reader::{IndexReader, MetadataFilter};
pub use index_writer::{IndexBuild, IndexWriter, is_index_dir};
pub use kjarni_search::{Chunk, ChunkMetadata, SearchMode, SearchResult};
pub use loader::{
    DocumentLoader, IMAGE_EXTENSIONS, KIND_IMAGE, KIND_KEY, LoaderConfig, SOURCE_PATH_KEY,
    TEXT_EXTENSIONS, source_path,
};
pub use progress::{CancelToken, Progress, ProgressCallback, ProgressReporter, ProgressStage};
pub use search_index::SearchIndex;
pub use segment::{Segment, SegmentBuilder, SegmentMeta};
pub use splitter::{SplitterConfig, TextSplitter, clean_markdown};

// Re-export from kjarni-search
pub use kjarni_search::bm25::Bm25Index;

#[cfg(test)]
mod tests;
