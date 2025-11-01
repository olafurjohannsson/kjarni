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

mod edge_gpt;
mod model_manager;
mod sentence_encoder_api;
mod cross_encoder_api;
mod utils;

// Re-export main API
pub use edge_gpt::{EdgeGPT, EdgeGPTBuilder};
pub use sentence_encoder_api::SentenceEncoderAPI;
pub use cross_encoder_api::CrossEncoderAPI;
pub use utils::*;

// Re-export commonly used types from dependencies
pub use edgetransformers::prelude::*;
pub use edgetransformers::models::ModelType;

// FFI module (feature-gated and public)
#[cfg(any(feature = "python", feature = "c-bindings"))]
pub mod ffi;

// Prelude 
pub mod prelude {
    pub use crate::edge_gpt::{EdgeGPT, EdgeGPTBuilder};
    pub use crate::utils::*;
    pub use edgetransformers::prelude::*;
    pub use edgetransformers::models::ModelType;
}