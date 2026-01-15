// kjarni/src/indexer/builder.rs

use crate::Indexer;
use crate::embedder::Embedder;
use crate::indexer::IndexerResult;
use kjarni_rag::{LoaderConfig, Progress, ProgressCallback, SplitterConfig};
use kjarni_transformers::Device;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::watch;

pub struct IndexerBuilder {
    // Model config
    pub(crate) model: String,
    pub(crate) device: Device,
    pub(crate) cache_dir: Option<PathBuf>,

    // Chunking config
    pub(crate) chunk_size: usize,
    pub(crate) chunk_overlap: usize,

    // File loading config
    pub(crate) extensions: Vec<String>,
    pub(crate) exclude_patterns: Vec<String>,
    pub(crate) recursive: bool,
    pub(crate) include_hidden: bool,
    pub(crate) max_file_size: Option<usize>,

    // Index config
    pub(crate) max_docs_per_segment: usize,

    // Runtime config
    pub(crate) batch_size: usize,
    pub(crate) quiet: bool,
    pub(crate) progress_callback: Option<ProgressCallback>,
    pub(crate) cancel_token: Option<watch::Receiver<bool>>,
}

impl IndexerBuilder {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            device: Device::Cpu,
            cache_dir: None,
            chunk_size: 512,
            chunk_overlap: 50,
            extensions: vec![], // Empty = use defaults
            exclude_patterns: vec![],
            recursive: true,
            include_hidden: false,
            max_file_size: Some(10 * 1024 * 1024), // 10MB default
            max_docs_per_segment: 10_000,
            batch_size: 32,
            quiet: false,
            progress_callback: None,
            cancel_token: None,
        }
    }

    // Device
    pub fn cpu(mut self) -> Self {
        self.device = Device::Cpu;
        self
    }
    pub fn gpu(mut self) -> Self {
        self.device = Device::Wgpu;
        self
    }
    pub fn device(mut self, d: &str) -> Self {
        self.device = if d == "gpu" {
            Device::Wgpu
        } else {
            Device::Cpu
        };
        self
    }

    // Cache
    pub fn cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    // Chunking
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }
    pub fn chunk_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }

    // File filtering
    pub fn extension(mut self, ext: &str) -> Self {
        self.extensions
            .push(ext.to_lowercase().trim_start_matches('.').to_string());
        self
    }
    pub fn extensions(mut self, exts: &[&str]) -> Self {
        for ext in exts {
            self.extensions
                .push(ext.to_lowercase().trim_start_matches('.').to_string());
        }
        self
    }
    pub fn exclude(mut self, pattern: &str) -> Self {
        self.exclude_patterns.push(pattern.to_string());
        self
    }
    pub fn recursive(mut self, r: bool) -> Self {
        self.recursive = r;
        self
    }
    pub fn include_hidden(mut self, h: bool) -> Self {
        self.include_hidden = h;
        self
    }
    pub fn max_file_size(mut self, bytes: usize) -> Self {
        self.max_file_size = Some(bytes);
        self
    }

    // Index config
    pub fn max_docs_per_segment(mut self, n: usize) -> Self {
        self.max_docs_per_segment = n;
        self
    }

    // Runtime
    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }
    pub fn quiet(mut self, q: bool) -> Self {
        self.quiet = q;
        self
    }

    pub fn on_progress<F>(mut self, callback: F) -> Self
    where
        F: Fn(&Progress, Option<&str>) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
        self
    }

    pub fn cancel_token(mut self, token: watch::Receiver<bool>) -> Self {
        self.cancel_token = Some(token);
        self
    }

    pub async fn build(self) -> IndexerResult<Indexer> {
        Indexer::from_builder(self).await
    }
}
