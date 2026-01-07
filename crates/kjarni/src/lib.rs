// use edgemodels::sentence_encoder::SentenceEncoder;
// use edgemodels::cross_encoder::CrossEncoder;

// use anyhow::Result;
// use edgetransformers::models::ModelType;
// use edgetransformers::prelude::*;
// use std::sync::Arc;
// use tokio::sync::Mutex;

// pub mod ffi;
// pub mod model_manager;
// pub mod utils;
// pub mod edge_gpt;

// #[derive(Default)]
// struct ModelManager {
//     cross_encoder: Mutex<Option<CrossEncoder>>,
//     sentence_encoder: Mutex<Option<SentenceEncoder>>,
// }

// /// The main, user-friendly entry point for all edgeGPT capabilities.
// pub struct EdgeGPT {
//     models: Arc<ModelManager>,
//     device: Device,
//     context: Option<Arc<WgpuContext>>,
// }

// impl EdgeGPT {
//     pub fn new(device: Device, context: Option<Arc<WgpuContext>>) -> Self {
//         Self {
//             models: Arc::new(ModelManager::default()),
//             device,
//             context,
//         }
//     }
//     pub async fn encode_batch(&self, sentences: &[&str]) -> Result<Vec<Vec<f32>>> {
//         let mut model_guard = self.models.sentence_encoder.lock().await;

//         if model_guard.is_none() {
//             *model_guard = Some(
//                 SentenceEncoder::from_registry(
//                     ModelType::MiniLML6V2,
//                     None,
//                     self.device,
//                     self.context.clone(),
//                 )
//                 .await?,
//             );
//             println!("Sentence encoding model loaded.");
//         }

//         let encoder = model_guard.as_ref().unwrap();
//         encoder.encode_batch(sentences).await
//     }

// }
//! EdgeGPT - Simple, batteries-included API for edge AI
//!
//! This crate provides a high-level API for running transformer models
//! on edge devices, with Python and C bindings.


mod utils;
pub mod chat;
pub mod classifier;
pub mod embedder;
mod generation;

pub mod common;

// Re-export main API
pub use utils::*;

pub use kjarni_models::cross_encoder::CrossEncoder;
pub use kjarni_models::sentence_encoder::SentenceEncoder;
pub use kjarni_models::SequenceClassifier;

// Re-export core types
pub use kjarni_transformers::models::{ModelArchitecture, ModelTask, ModelType};
pub use kjarni_transformers::traits::Device;
pub use kjarni_transformers::models::base::DType;
// Re-export generation
pub use kjarni_transformers::common::{
    BeamSearchParams, DecodingStrategy, GenerationConfig, SamplingParams, StreamedToken, TokenType,
};

// Re-export chat
pub use kjarni_transformers::chat::{
    llama3::{Llama2ChatTemplate, Llama3ChatTemplate},
    templates::{ChatTemplate, Conversation, Message, Role},
};

pub use kjarni_transformers::decoder::generator::DecoderGenerator;
pub use kjarni_transformers::decoder::traits::DecoderLanguageModel;

// Re-export seq2seq generation (encoder-decoders)
pub use kjarni_transformers::encoder_decoder::EncoderDecoderGenerator;

pub use kjarni_transformers::cpu::encoder::traits::EncoderLanguageModel;

pub use kjarni_rag::{
    config::IndexConfig, segment::{Segment, SegmentBuilder, SegmentMeta}, DocumentLoader, IndexReader, IndexWriter, LoaderConfig,
    SearchIndex,
    SplitterConfig,
    TextSplitter,
};

pub use kjarni_search::{Bm25Index, Chunk, ChunkMetadata, SearchMode, SearchResult, VectorStore};

// Re-export commonly used types from dependencies
pub use kjarni_transformers::prelude::*;

pub mod models {
    pub use kjarni_models::models::bart::model::BartModel;
    pub use kjarni_models::models::gpt2::Gpt2Model;
    pub use kjarni_models::models::llama::{LlamaConfig, LlamaModel};
    pub use kjarni_models::models::qwen::{QwenConfig, QwenModel};
}
pub mod registry;

// FFI module (feature-gated and public)

#[cfg(any(feature = "python", feature = "c-bindings"))]
pub mod ffi;



// Prelude
pub mod prelude {
    pub use crate::utils::*;
    pub use kjarni_transformers::models::ModelType;
    pub use kjarni_transformers::prelude::*;
}


// ============================================================================
// Top-level convenience functions
// ============================================================================

/// Send a chat message (convenience wrapper).
///
/// # Example
///
/// ```ignore
/// let response = kjarni::chat("llama3.2-1b", "Hello!").await?;
/// ```
pub async fn chat_send(model: &str, message: &str) -> chat::ChatResult<String> {
    chat::send(model, message).await
}

/// Encode text to embedding (convenience wrapper).
///
/// # Example
///
/// ```ignore
/// let embedding = kjarni::embed("minilm-l6-v2", "Hello world").await?;
/// ```
pub async fn embed(model: &str, text: &str) -> anyhow::Result<Vec<f32>> {
    let encoder = crate::embedder::Embedder::new(model).await?;
    let embedding = encoder.embed(text).await?;
    Ok(embedding)
}

/// Classify text (convenience wrapper).
//// # Example
/// ```ignore
/// let result = kjarni::classify("distilbert-base-uncased-finetuned-sst-2-english", "I love programming!").await?;
/// ```
pub async fn classify(model: &str, text: &str) -> anyhow::Result<crate::classifier::ClassificationResult> {
    let classifier = crate::classifier::Classifier::new(model).await?;
    let result = classifier.classify(text).await?;
    Ok(result)
}

// ============================================================================
// Version Info
// ============================================================================

/// Get the kjarni version.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Get build information.
pub fn build_info() -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION"),
        git_hash: option_env!("GIT_HASH"),
        build_date: option_env!("BUILD_DATE"),
    }
}

/// Build metadata.
#[derive(Debug, Clone)]
pub struct BuildInfo {
    pub version: &'static str,
    pub git_hash: Option<&'static str>,
    pub build_date: Option<&'static str>,
}
