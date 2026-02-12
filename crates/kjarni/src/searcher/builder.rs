use std::path::PathBuf;
use kjarni_rag::SearchMode;
use kjarni_rag::ProgressCallback;
use kjarni_transformers::Device;

use crate::Searcher;
use crate::searcher::SearcherResult;

pub struct SearcherBuilder {
    // Embedder config
    pub(crate) model: String,
    pub(crate) device: Device,
    pub(crate) cache_dir: Option<PathBuf>,
    
    // Reranker config (optional)
    pub(crate) rerank_model: Option<String>,
    
    // Default search options
    pub(crate) default_mode: SearchMode,
    pub(crate) default_top_k: usize,
    
    // Runtime
    pub(crate) quiet: bool,
    pub(crate) progress_callback: Option<ProgressCallback>,
}

impl SearcherBuilder {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            device: Device::Cpu,
            cache_dir: None,
            rerank_model: None,
            default_mode: SearchMode::Hybrid,
            default_top_k: 10,
            quiet: false,
            progress_callback: None,
        }
    }
    
    // Device
    pub fn cpu(mut self) -> Self { self.device = Device::Cpu; self }
    pub fn gpu(mut self) -> Self { self.device = Device::Wgpu; self }
    pub fn device(mut self, d: &str) -> Self {
        self.device = if d == "gpu" { Device::Wgpu } else { Device::Cpu };
        self
    }
    
    // Reranker
    pub fn reranker(mut self, model: &str) -> Self {
        self.rerank_model = Some(model.to_string());
        self
    }
    pub fn no_reranker(mut self) -> Self {
        self.rerank_model = None;
        self
    }
    
    // Defaults
    pub fn default_mode(mut self, m: SearchMode) -> Self { self.default_mode = m; self }
    pub fn default_top_k(mut self, k: usize) -> Self { self.default_top_k = k; self }
    
    // Runtime
    pub fn quiet(mut self, q: bool) -> Self { self.quiet = q; self }
    pub fn cache_dir(mut self, d: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(d.into());
        self
    }
    
    pub async fn build(self) -> SearcherResult<Searcher> {
        Searcher::from_builder(self).await
    }
}