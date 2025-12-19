//! Python bindings for EdgeGPT using PyO3

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use crate::edge_gpt::{EdgeGPT as RustEdgeGPT, EdgeGPTBuilder};
use edgetransformers::prelude::Device;
use edgetransformers::models::ModelType;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Python wrapper for EdgeGPT
#[pyclass]
struct EdgeGPT {
    inner: Arc<RustEdgeGPT>,
    runtime: Runtime,
}

#[pymethods]
impl EdgeGPT {
    /// Create a new EdgeGPT instance
    ///
    /// Args:
    ///     device (str): "cpu" or "gpu"
    ///     cache_dir (str, optional): Directory to cache model files
    ///     sentence_model (str, optional): Sentence encoder model name
    ///     cross_encoder_model (str, optional): Cross encoder model name
    ///
    /// Returns:
    ///     EdgeGPT: A new EdgeGPT instance
    #[new]
    #[pyo3(signature = (device="cpu", cache_dir=None, sentence_model=None, cross_encoder_model=None))]
    fn new(
        device: &str,
        cache_dir: Option<String>,
        sentence_model: Option<String>,
        cross_encoder_model: Option<String>,
    ) -> PyResult<Self> {
        let device = match device {
            "cpu" => Device::Cpu,
            "gpu" => Device::Wgpu,
            _ => return Err(PyRuntimeError::new_err("Device must be 'cpu' or 'gpu'")),
        };

        let runtime = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        let mut builder = EdgeGPTBuilder::new().device(device);

        if let Some(dir) = cache_dir {
            builder = builder.cache_dir(dir);
        }

        if let Some(model_name) = sentence_model {
            let model_type = parse_model_type(&model_name)?;
            builder = builder.sentence_model(model_type);
        }

        if let Some(model_name) = cross_encoder_model {
            let model_type = parse_model_type(&model_name)?;
            builder = builder.cross_encoder_model(model_type);
        }

        let inner = Arc::new(builder.build());

        Ok(Self { inner, runtime })
    }

    /// Encode a single sentence
    ///
    /// Args:
    ///     text (str): Input text
    ///
    /// Returns:
    ///     list[float]: Embedding vector
    fn encode(&self, text: &str) -> PyResult<Vec<f32>> {
        let inner = self.inner.clone();
        let text = text.to_string();
        
        self.runtime.block_on(async move {
            inner.encode(&text).await
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Encoding failed: {}", e)))
    }

    /// Encode a batch of sentences
    ///
    /// Args:
    ///     texts (list[str]): List of input texts
    ///
    /// Returns:
    ///     list[list[float]]: List of embedding vectors
    fn encode_batch(&self, texts: Vec<&str>) -> PyResult<Vec<Vec<f32>>> {
        let inner = self.inner.clone();
        
        self.runtime.block_on(async move {
            inner.encode_batch(&texts).await
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Batch encoding failed: {}", e)))
    }

    /// Compute similarity between two texts
    ///
    /// Args:
    ///     text1 (str): First text
    ///     text2 (str): Second text
    ///
    /// Returns:
    ///     float: Cosine similarity score
    fn similarity(&self, text1: &str, text2: &str) -> PyResult<f32> {
        let inner = self.inner.clone();
        let text1 = text1.to_string();
        let text2 = text2.to_string();
        
        self.runtime.block_on(async move {
            inner.similarity(&text1, &text2).await
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Similarity computation failed: {}", e)))
    }

    /// Find the most similar texts to a query
    ///
    /// Args:
    ///     query (str): Query text
    ///     candidates (list[str]): Candidate texts
    ///     top_k (int): Number of results to return
    ///
    /// Returns:
    ///     list[tuple[int, float]]: List of (index, similarity_score) tuples
    fn find_similar(&self, query: &str, candidates: Vec<&str>, top_k: usize) -> PyResult<Vec<(usize, f32)>> {
        let inner = self.inner.clone();
        let query = query.to_string();
        
        self.runtime.block_on(async move {
            inner.find_similar(&query, &candidates, top_k).await
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Finding similar failed: {}", e)))
    }

    /// Score a text pair for relevance
    ///
    /// Args:
    ///     text1 (str): First text (typically a query)
    ///     text2 (str): Second text (typically a document)
    ///
    /// Returns:
    ///     float: Relevance score
    fn predict(&self, text1: &str, text2: &str) -> PyResult<f32> {
        let inner = self.inner.clone();
        let text1 = text1.to_string();
        let text2 = text2.to_string();
        
        self.runtime.block_on(async move {
            inner.predict(&text1, &text2).await
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Prediction failed: {}", e)))
    }

    /// Score multiple text pairs
    ///
    /// Args:
    ///     pairs (list[tuple[str, str]]): List of (text1, text2) pairs
    ///
    /// Returns:
    ///     list[float]: List of relevance scores
    fn predict_batch(&self, pairs: Vec<(&str, &str)>) -> PyResult<Vec<f32>> {
        let inner = self.inner.clone();
        
        self.runtime.block_on(async move {
            inner.predict_batch(&pairs).await
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Batch prediction failed: {}", e)))
    }

    /// Rerank documents by relevance to a query
    ///
    /// Args:
    ///     query (str): Query text
    ///     documents (list[str]): Documents to rerank
    ///
    /// Returns:
    ///     list[tuple[int, float]]: List of (index, score) tuples sorted by relevance
    fn rerank(&self, query: &str, documents: Vec<&str>) -> PyResult<Vec<(usize, f32)>> {
        let inner = self.inner.clone();
        let query = query.to_string();
        
        self.runtime.block_on(async move {
            inner.rerank(&query, &documents).await
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Reranking failed: {}", e)))
    }

    /// Rerank documents and return top K results
    ///
    /// Args:
    ///     query (str): Query text
    ///     documents (list[str]): Documents to rerank
    ///     k (int): Number of results to return
    ///
    /// Returns:
    ///     list[tuple[int, float]]: Top K (index, score) tuples
    fn rerank_top_k(&self, query: &str, documents: Vec<&str>, k: usize) -> PyResult<Vec<(usize, f32)>> {
        let inner = self.inner.clone();
        let query = query.to_string();
        
        self.runtime.block_on(async move {
            inner.rerank_top_k(&query, &documents, k).await
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Top-K reranking failed: {}", e)))
    }

    /// Get the embedding dimension
    ///
    /// Returns:
    ///     int: Embedding dimension
    fn embedding_dim(&self) -> PyResult<usize> {
        let inner = self.inner.clone();
        
        self.runtime.block_on(async move {
            inner.embedding_dim().await
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Getting embedding dim failed: {}", e)))
    }

    /// Preload the sentence encoder model
    fn preload_sentence_encoder(&self) -> PyResult<()> {
        let inner = self.inner.clone();
        
        self.runtime.block_on(async move {
            inner.preload_sentence_encoder().await
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Preloading sentence encoder failed: {}", e)))
    }

    /// Preload the cross encoder model
    fn preload_cross_encoder(&self) -> PyResult<()> {
        let inner = self.inner.clone();
        
        self.runtime.block_on(async move {
            inner.preload_cross_encoder().await
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Preloading cross encoder failed: {}", e)))
    }

    /// Unload the sentence encoder to free memory
    fn unload_sentence_encoder(&self) -> PyResult<()> {
        let inner = self.inner.clone();
        
        self.runtime.block_on(async move {
            inner.unload_sentence_encoder().await;
            Ok(())
        })
    }

    /// Unload the cross encoder to free memory
    fn unload_cross_encoder(&self) -> PyResult<()> {
        let inner = self.inner.clone();
        
        self.runtime.block_on(async move {
            inner.unload_cross_encoder().await;
            Ok(())
        })
    }

    /// Unload all models
    fn unload_all(&self) -> PyResult<()> {
        let inner = self.inner.clone();
        
        self.runtime.block_on(async move {
            inner.unload_all().await;
            Ok(())
        })
    }
}

fn parse_model_type(name: &str) -> PyResult<ModelType> {
    match name.to_lowercase().as_str() {
        "minilm" | "mini-lm" | "minilm-l6-v2" => Ok(ModelType::MiniLML6V2),
        "mpnet" | "mpnet-base-v2" => Ok(ModelType::MpnetBaseV2),
        "distilbert" | "distilbert-base-cased" => Ok(ModelType::DistilBertBaseCased),
        "cross-encoder" | "cross-encoder-minilm" => Ok(ModelType::MiniLML6V2CrossEncoder),
        "distilgpt2" => Ok(ModelType::DistilGpt2),
        "gpt2" => Ok(ModelType::Gpt2),
        "gpt2-medium" => Ok(ModelType::Gpt2Medium),
        "gpt2-large" => Ok(ModelType::Gpt2Large),
        "gpt2-xl" => Ok(ModelType::Gpt2XL),
        _ => Err(PyRuntimeError::new_err(format!("Unknown model type: {}", name))),
    }
}

/// EdgeGPT Python module
#[pymodule]
fn edgegpt(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<EdgeGPT>()?;
    Ok(())
}