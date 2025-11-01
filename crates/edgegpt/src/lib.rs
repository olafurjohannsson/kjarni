use edgemodels::sentence_encoder::SentenceEncoder;
use edgemodels::cross_encoder::CrossEncoder;

use anyhow::Result;
use edgetransformers::models::ModelType;
use edgetransformers::prelude::*;
use std::sync::Arc;
use tokio::sync::Mutex;

pub mod ffi;

#[derive(Default)]
struct ModelManager {
    cross_encoder: Mutex<Option<CrossEncoder>>,
    sentence_encoder: Mutex<Option<SentenceEncoder>>,
}

/// The main, user-friendly entry point for all edgeGPT capabilities.
pub struct EdgeGPT {
    models: Arc<ModelManager>,
    device: Device,
    context: Option<Arc<WgpuContext>>,
}

impl EdgeGPT {
    pub fn new(device: Device, context: Option<Arc<WgpuContext>>) -> Self {
        Self {
            models: Arc::new(ModelManager::default()),
            device,
            context,
        }
    }
    pub async fn encode_batch(&self, sentences: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut model_guard = self.models.sentence_encoder.lock().await;

        // On-demand loading
        if model_guard.is_none() {
            println!("Loading sentence encoding model for the first time...");
            *model_guard = Some(
                SentenceEncoder::from_registry(
                    ModelType::MiniLML6V2,
                    None,
                    self.device,
                    self.context.clone(),
                )
                .await?,
            );
            println!("Sentence encoding model loaded.");
        }

        let encoder = model_guard.as_ref().unwrap();
        encoder.encode_batch(sentences).await
    }
    
}