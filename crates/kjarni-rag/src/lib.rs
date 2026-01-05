pub mod search_index;
pub mod config;
pub mod splitter;
pub mod loader;
pub mod segment;
pub mod index_writer;
pub mod index_reader;

pub use search_index::SearchIndex;
pub use splitter::{TextSplitter, SplitterConfig};
pub use loader::{DocumentLoader, LoaderConfig};
pub use segment::{SegmentBuilder};
pub use index_writer::IndexWriter;
pub use index_reader::IndexReader;

pub use kjarni_search::{
    Bm25Index,
    Chunk,
    SearchResult,
    VectorStore,
    ChunkMetadata,
    SearchMode,
};


// #[cfg(test)]
// mod tests;