//! Main EdgeGPT API

use crate::model_manager::ModelManager;
use crate::sentence_encoder_api::SentenceEncoderAPI;
use crate::cross_encoder_api::CrossEncoderAPI;
use anyhow::Result;
use edgetransformers::models::ModelType;
use edgetransformers::prelude::*;
use std::sync::Arc;
use std::collections::HashMap;

// Import Search/RAG components
use edgesearch::*;
use edgerag::*;

/// Main EdgeGPT interface
pub struct EdgeGPT {
    pub(crate) models: Arc<ModelManager>,
    pub(crate) device: Device,
    pub(crate) context: Option<Arc<WgpuContext>>,
    pub(crate) sentence_model_type: ModelType,
    pub(crate) cross_encoder_model_type: ModelType,
    pub(crate) generator_model_type: ModelType, // Default: Llama3_2_1B or DistilGPT2
    pub(crate) seq2seq_model_type: ModelType,   // Default: DistilBartCnn
    pub(crate) cache_dir: Option<String>,
}

impl EdgeGPT {
    /// Create a new EdgeGPT instance with default settings
    pub fn new(device: Device, context: Option<Arc<WgpuContext>>) -> Self {
        Self {
            models: Arc::new(ModelManager::new()),
            device,
            context,
            sentence_model_type: ModelType::MiniLML6V2,
            cross_encoder_model_type: ModelType::MiniLML6V2CrossEncoder,
            seq2seq_model_type: ModelType::DistilBartCnn,
            generator_model_type: ModelType::DistilGpt2,

            cache_dir: None,
        }
    }

    /// Create a builder for custom configuration
    pub fn builder() -> EdgeGPTBuilder {
        EdgeGPTBuilder::new()
    }

    // ========================================================================
    // Text Generation (Chat / Completion)
    // ========================================================================

    /// Generate text continuation from a prompt
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        self.models.get_or_load_text_generator(
            self.generator_model_type,
            self.cache_dir.as_deref(),
            self.device,
            self.context.clone()
        ).await?;
        
        let guard = self.models.text_generator.lock().await;
        let generator = guard.as_ref().unwrap();
        
        // Use default config from the model
        let config = generator.model.get_default_generation_config();
        generator.generate(prompt, &config).await
    }

    /// Summarize a long text
    pub async fn summarize(&self, text: &str) -> Result<String> {
        self.models.get_or_load_seq2seq_generator(
            self.seq2seq_model_type,
            self.cache_dir.as_deref(),
            self.device,
            self.context.clone()
        ).await?;

        let guard = self.models.seq2seq_generator.lock().await;
        let generator = guard.as_ref().unwrap();
        
        let config = generator.model.get_default_generation_config();
        generator.generate(text, Some(&config)).await
    }

    /// Translate text (Alias for summarize/generate on translation models)
    pub async fn translate(&self, text: &str) -> Result<String> {
        // For now, translation uses the same Seq2Seq pipeline
        self.summarize(text).await
    }

    // ========================================================================
    // Sentence Encoding Methods
    // ========================================================================

    /// Encode a single sentence
    pub async fn encode(&self, text: &str) -> Result<Vec<f32>> {
        self.ensure_sentence_encoder_loaded().await?;
        let guard = self.models.sentence_encoder.lock().await;
        let encoder = guard.as_ref().unwrap();
        encoder.encode(text).await
    }

    /// Encode a batch of sentences
    pub async fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.ensure_sentence_encoder_loaded().await?;
        let guard = self.models.sentence_encoder.lock().await;
        let encoder = guard.as_ref().unwrap();
        encoder.encode_batch(texts).await
    }

    /// Encode with custom settings
    pub async fn encode_with(
        &self,
        text: &str,
        pooling: Option<&str>,
        normalize: bool,
    ) -> Result<Vec<f32>> {
        self.ensure_sentence_encoder_loaded().await?;
        let guard = self.models.sentence_encoder.lock().await;
        let encoder = guard.as_ref().unwrap();
        encoder.encode_with(text, pooling, normalize).await
    }

    /// Compute similarity between two texts
    pub async fn similarity(&self, text1: &str, text2: &str) -> Result<f32> {
        let embeddings = self.encode_batch(&[text1, text2]).await?;
        Ok(crate::utils::cosine_similarity(&embeddings[0], &embeddings[1]))
    }

    /// Find most similar texts to a query
    pub async fn find_similar(
        &self,
        query: &str,
        candidates: &[&str],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        self.ensure_sentence_encoder_loaded().await?;
        let guard = self.models.sentence_encoder.lock().await;
        let encoder = guard.as_ref().unwrap();
        let api = SentenceEncoderAPI::new(encoder); // Now takes a reference
        api.find_similar(query, candidates, top_k).await
    }

    /// Get the embedding dimension
    pub async fn embedding_dim(&self) -> Result<usize> {
        self.ensure_sentence_encoder_loaded().await?;
        let guard = self.models.sentence_encoder.lock().await;
        let encoder = guard.as_ref().unwrap();
        Ok(encoder.embedding_dim())
    }

    /// Get the maximum sequence length
    pub async fn max_seq_length(&self) -> Result<usize> {
        self.ensure_sentence_encoder_loaded().await?;
        let guard = self.models.sentence_encoder.lock().await;
        let encoder = guard.as_ref().unwrap();
        Ok(encoder.max_seq_length())
    }

    // ========================================================================
    // Cross-Encoding Methods
    // ========================================================================

    /// Score a text pair for relevance
    pub async fn predict(&self, text1: &str, text2: &str) -> Result<f32> {
        self.ensure_cross_encoder_loaded().await?;
        let guard = self.models.cross_encoder.lock().await;
        let encoder = guard.as_ref().unwrap();
        encoder.predict(text1, text2).await
    }

    /// Score multiple text pairs
    pub async fn predict_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>> {
        self.ensure_cross_encoder_loaded().await?;
        let guard = self.models.cross_encoder.lock().await;
        let encoder = guard.as_ref().unwrap();
        encoder.predict_batch(pairs).await
    }

    /// Rerank documents by relevance to a query
    pub async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
        self.ensure_cross_encoder_loaded().await?;
        let guard = self.models.cross_encoder.lock().await;
        let encoder = guard.as_ref().unwrap();
        encoder.rerank(query, documents).await
    }

    /// Rerank and return top K results
    pub async fn rerank_top_k(
        &self,
        query: &str,
        documents: &[&str],
        k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        self.ensure_cross_encoder_loaded().await?;
        let guard = self.models.cross_encoder.lock().await;
        let encoder = guard.as_ref().unwrap();
        encoder.rerank_top_k(query, documents, k).await
    }


    // ========================================================================
    // Index & Search 
    // ========================================================================

    /// Split text into chunks suitable for indexing
    pub fn split_text(&self, text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
        unimplemented!()
        // let config = SplitterConfig {
        //     chunk_size,
        //     chunk_overlap: overlap,
        //     ..Default::default()
        // };
        // let splitter = TextSplitter::new(config);
        // splitter.split(text)
    }

    /// Build a Search Index from a list of documents.
    /// This automatically computes embeddings using the loaded Sentence Encoder.
    pub async fn build_index(&self, documents: &[&str]) -> Result<SearchIndex> {
        // Ensure encoder is loaded
        self.preload_sentence_encoder().await?;
        
        // Compute embeddings
        let embeddings = self.encode_batch(documents).await?;
        
        // Convert to owned strings for the index
        let docs_owned: Vec<String> = documents.iter().map(|s| s.to_string()).collect();
        let metadata = vec![HashMap::new(); docs_owned.len()];

        // Create the index
        SearchIndex::build(docs_owned, embeddings, metadata)
    }

    /// Search an index using Hybrid Search (BM25 + Vector).
    /// This automatically computes the query embedding.
    pub async fn search(&self, index: &SearchIndex, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        // Ensure encoder is loaded
        self.preload_sentence_encoder().await?;
        
        // Compute query embedding
        let query_emb = self.encode(query).await?;
        
        // Perform Hybrid Search
        Ok(index.search_hybrid(query, &query_emb, limit))
    }

    // ========================================================================
    // Model Management
    // ========================================================================

    /// Preload the sentence encoder
    pub async fn preload_sentence_encoder(&self) -> Result<()> {
        self.ensure_sentence_encoder_loaded().await
    }

    /// Preload the cross encoder
    pub async fn preload_cross_encoder(&self) -> Result<()> {
        self.ensure_cross_encoder_loaded().await
    }

    /// Unload the sentence encoder to free memory
    pub async fn unload_sentence_encoder(&self) {
        self.models.unload_sentence_encoder().await;
    }

    /// Unload the cross encoder to free memory
    pub async fn unload_cross_encoder(&self) {
        self.models.unload_cross_encoder().await;
    }

    /// Unload all models
    pub async fn unload_all(&self) {
        self.models.unload_all().await;
    }

    // ========================================================================
    // Private Helper Methods
    // ========================================================================

    async fn ensure_sentence_encoder_loaded(&self) -> Result<()> {
        self.models
            .get_or_load_sentence_encoder(
                self.sentence_model_type,
                self.cache_dir.as_deref(),
                self.device,
                self.context.clone(),
            )
            .await
    }

    async fn ensure_cross_encoder_loaded(&self) -> Result<()> {
        self.models
            .get_or_load_cross_encoder(
                self.cross_encoder_model_type,
                self.cache_dir.as_deref(),
                self.device,
                self.context.clone(),
            )
            .await
    }

    async fn ensure_generator_loaded(&self) -> Result<()> {
        self.models.get_or_load_text_generator(
            self.generator_model_type,
            self.cache_dir.as_deref(),
            self.device,
            self.context.clone()
        ).await
    }

    async fn ensure_seq2seq_loaded(&self) -> Result<()> {
        self.models.get_or_load_seq2seq_generator(
            self.seq2seq_model_type,
            self.cache_dir.as_deref(),
            self.device,
            self.context.clone()
        ).await
    }
}

// Builder implementation remains the same...
pub struct EdgeGPTBuilder {
    device: Device,
    context: Option<Arc<WgpuContext>>,
    sentence_model: ModelType,
    cross_encoder_model: ModelType,
    seq2seq_model: ModelType,
    generator_model: ModelType,
    cache_dir: Option<String>,
}

impl EdgeGPTBuilder {
    pub fn new() -> Self {
        Self {
            device: Device::Cpu,
            context: None,
            sentence_model: ModelType::MiniLML6V2,
            cross_encoder_model: ModelType::MiniLML6V2CrossEncoder,
            generator_model: ModelType::DistilGpt2,
            seq2seq_model: ModelType::DistilBartCnn,
            cache_dir: None,
        }
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn context(mut self, context: Arc<WgpuContext>) -> Self {
        self.context = Some(context);
        self
    }

    pub fn sentence_model(mut self, model: ModelType) -> Self {
        self.sentence_model = model;
        self
    }

    pub fn cross_encoder_model(mut self, model: ModelType) -> Self {
        self.cross_encoder_model = model;
        self
    }
    
    pub fn generator_model(mut self, model: ModelType) -> Self {
        self.generator_model = model;
        self
    }

    pub fn seq2seq_model(mut self, model: ModelType) -> Self {
        self.seq2seq_model = model;
        self
    }

    pub fn cache_dir(mut self, dir: impl Into<String>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    pub fn build(self) -> EdgeGPT {
        EdgeGPT {
            models: Arc::new(ModelManager::new()),
            device: self.device,
            context: self.context,
            sentence_model_type: self.sentence_model,
            cross_encoder_model_type: self.cross_encoder_model,
            seq2seq_model_type: self.seq2seq_model,
            generator_model_type: self.generator_model,
            cache_dir: self.cache_dir,
        }
    }
}

impl Default for EdgeGPTBuilder {
    fn default() -> Self {
        Self::new()
    }
}