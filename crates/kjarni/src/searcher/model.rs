// kjarni/src/searcher/model.rs

use crate::searcher::{SearchOptions, SearcherError, SearcherResult};
use crate::searcher::types::SearchResult;
use crate::{embedder::Embedder, searcher::SearcherBuilder};
use crate::reranker::Reranker;
use kjarni_rag::{IndexReader, MetadataFilter, SearchMode};
use kjarni_transformers::Device;

pub struct Searcher {
    embedder: Embedder,
    reranker: Option<Reranker>,
    default_mode: SearchMode,
    default_top_k: usize,
    quiet: bool,
}

impl Searcher {
    pub fn builder(model: &str) -> SearcherBuilder {
        SearcherBuilder::new(model)
    }

pub(crate) async fn from_builder(builder: SearcherBuilder) -> SearcherResult<Self> {
        // Build embedder
        let mut embedder_builder = Embedder::builder(&builder.model);
        
        match builder.device {
            Device::Wgpu => embedder_builder = embedder_builder.gpu(),
            Device::Cpu => embedder_builder = embedder_builder.cpu(),
        }
        
        if let Some(ref cache_dir) = builder.cache_dir {
            embedder_builder = embedder_builder.cache_dir(cache_dir);
        }
        
        let embedder = embedder_builder.build().await
            .map_err(SearcherError::EmbedderError)?;
        
        // Build reranker if specified
        let reranker = if let Some(ref rerank_model) = builder.rerank_model {
            let mut reranker_builder = Reranker::builder(rerank_model);
            
            match builder.device {
                Device::Wgpu => reranker_builder = reranker_builder.gpu(),
                Device::Cpu => reranker_builder = reranker_builder.cpu(),
            }
            
            if let Some(ref cache_dir) = builder.cache_dir {
                reranker_builder = reranker_builder.cache_dir(cache_dir);
            }
            
            Some(reranker_builder.build().await.map_err(SearcherError::RerankerError)?)
        } else {
            None
        };
        
        Ok(Self {
            embedder,
            reranker,
            default_mode: builder.default_mode,
            default_top_k: builder.default_top_k,
            quiet: builder.quiet,
        })
    }
    
    pub async fn new(model: &str) -> SearcherResult<Self> {
        Self::builder(model).build().await
    }
    
    /// Static keyword search (no embedder needed)
    pub fn search_keywords(
        index_path: &str,
        query: &str,
        top_k: usize,
    ) -> SearcherResult<Vec<SearchResult>> {
        let reader = IndexReader::open(index_path)
            .map_err(|e| SearcherError::SearchFailed(e))?;
        
        let results = reader.search_keywords(query, top_k);
        Ok(results.into_iter().map(Into::into).collect())
    }
    
    /// Search with default options
    pub async fn search(
        &self,
        index_path: &str,
        query: &str,
    ) -> SearcherResult<Vec<SearchResult>> {
        self.search_with_options(index_path, query, &SearchOptions::default()).await
    }
    
    /// Search with custom options
    pub async fn search_with_options(
        &self,
        index_path: &str,
        query: &str,
        options: &SearchOptions,
    ) -> SearcherResult<Vec<SearchResult>> {
        let reader = IndexReader::open(index_path)
            .map_err(|e| SearcherError::SearchFailed(e))?;
        
        // Validate dimension
        if reader.dimension() != self.embedder.dimension() {
            return Err(SearcherError::DimensionMismatch {
                index_dim: reader.dimension(),
                model_dim: self.embedder.dimension(),
            });
        }
        
        let mode = options.mode.unwrap_or(self.default_mode);
        let top_k = options.top_k.unwrap_or(self.default_top_k);
        let use_reranker = options.rerank.unwrap_or(self.reranker.is_some());
        
        // Fetch more if reranking
        let fetch_k = if use_reranker && self.reranker.is_some() {
            top_k * 5
        } else {
            top_k
        };
        
        // Get results based on mode
        let mut results = match mode {
            SearchMode::Keyword => {
                if let Some(ref filter) = options.filter {
                    reader.search_keywords_filtered(query, fetch_k, filter)
                } else {
                    reader.search_keywords(query, fetch_k)
                }
            }
            SearchMode::Semantic => {
                let query_embedding = self.embedder.embed(query).await
                    .map_err(|e| SearcherError::EmbedderError(e))?;
                
                if let Some(ref filter) = options.filter {
                    reader.search_semantic_filtered(&query_embedding, fetch_k, filter)
                } else {
                    reader.search_semantic(&query_embedding, fetch_k)
                }
            }
            SearchMode::Hybrid => {
                let query_embedding = self.embedder.embed(query).await
                    .map_err(|e| SearcherError::EmbedderError(e))?;
                
                if let Some(ref filter) = options.filter {
                    reader.search_hybrid_filtered(query, &query_embedding, fetch_k, filter)
                } else {
                    reader.search_hybrid(query, &query_embedding, fetch_k)
                }
            }
        };
        
        // Rerank if enabled
        if use_reranker {
            if let Some(ref reranker) = self.reranker {
                let texts: Vec<&str> = results.iter().map(|r| r.text.as_str()).collect();
                let reranked = reranker.rerank(query, &texts).await
                    .map_err(|e| SearcherError::RerankerError(e))?;
                
                // Reconstruct with new scores
                let mut new_results = Vec::with_capacity(reranked.len().min(top_k));
                for rr in reranked.into_iter().take(top_k) {
                    let mut result = results[rr.index].clone();
                    result.score = rr.score;
                    new_results.push(result);
                }
                results = new_results;
            }
        }
        
        // Apply threshold
        if let Some(threshold) = options.threshold {
            results.retain(|r| r.score >= threshold);
        }
        
        // Truncate to top_k
        results.truncate(top_k);
        
        Ok(results.into_iter().map(Into::into).collect())
    }
    
    // Accessors
    pub fn model_name(&self) -> &str { self.embedder.model_name() }
    pub fn has_reranker(&self) -> bool { self.reranker.is_some() }
    pub fn reranker_model(&self) -> Option<&str> {
        self.reranker.as_ref().map(|r| r.model_name())
    }
    pub fn default_mode(&self) -> SearchMode { self.default_mode }
    pub fn default_top_k(&self) -> usize { self.default_top_k }
}