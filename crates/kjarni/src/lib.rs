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

mod config;
mod config_utils;
mod cross_encoder_api;
mod edge_gpt;
mod kjarni;
mod model_manager;
mod sentence_encoder_api;
mod utils;
// Re-export main API
pub use cross_encoder_api::CrossEncoderAPI;
pub use edge_gpt::{EdgeGPT, EdgeGPTBuilder};
pub use sentence_encoder_api::SentenceEncoderAPI;
pub use utils::*;

pub use kjarni_models::cross_encoder::CrossEncoder;
pub use kjarni_models::sentence_encoder::SentenceEncoder;
pub use kjarni_models::SequenceClassifier;

// Re-export core types
pub use kjarni_transformers::models::{ModelArchitecture, ModelType};
pub use kjarni_transformers::traits::Device;

// Re-export generation
pub use kjarni_transformers::common::{
    BeamSearchParams, DecodingStrategy, GenerationConfig, SamplingParams, StreamedToken, TokenType,
};

// Re-export chat
pub use kjarni_transformers::chat::{
    llama3::{Llama2ChatTemplate, Llama3ChatTemplate},
    templates::{ChatTemplate, Conversation, Message, Role}
};

pub use kjarni_transformers::decoder::generator::DecoderGenerator;
pub use kjarni_transformers::decoder::traits::DecoderLanguageModel;

// Re-export seq2seq generation (encoder-decoders)
pub use kjarni_transformers::encoder_decoder::EncoderDecoderGenerator;

pub use kjarni_transformers::encoder::traits::EncoderLanguageModel;

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
    pub use kjarni_models::models::llama::{
        LlamaConfig, LlamaModel,
    };
    pub use kjarni_models::models::qwen::{
        QwenModel,
        QwenConfig
    };
}
pub mod registry;

// FFI module (feature-gated and public)
#[cfg(any(feature = "python", feature = "c-bindings"))]
pub mod ffi;

// Prelude
pub mod prelude {
    pub use crate::edge_gpt::{EdgeGPT, EdgeGPTBuilder};
    pub use crate::utils::*;
    pub use kjarni_transformers::models::ModelType;
    pub use kjarni_transformers::prelude::*;
}

// mod model;