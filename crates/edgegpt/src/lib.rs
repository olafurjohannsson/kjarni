use edgemodels::seq2seq::Seq2SeqModel;
use edgemodels::text_generator::TextGenerator;
// Add other models as you create them
// use edgemodels::sentence_encoder::SentenceEncoder;
// use edgemodels::cross_encoder::CrossEncoder;

use edgetransformers::models::ModelType;
use edgetransformers::prelude::*;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;

// A container for our models, wrapped in Mutex for safe concurrent access.
// Using Option allows us to load them on-demand.
#[derive(Default)]
struct ModelManager {
    summarizer: Mutex<Option<Seq2SeqModel>>,
    generator: Mutex<Option<TextGenerator>>,
    // translator: Mutex<Option<Seq2SeqModel>>,
    // sentence_encoder: Mutex<Option<SentenceEncoder>>,
}

/// The main, user-friendly entry point for all edgeGPT capabilities.
pub struct EdgeGPT {
    models: Arc<ModelManager>,
    device: Device,
    context: Option<Arc<WgpuContext>>,
}

impl EdgeGPT {
    /// Create a new EdgeGPT instance for a specific device.
    pub fn new(device: Device, context: Option<Arc<WgpuContext>>) -> Self {
        Self {
            models: Arc::new(ModelManager::default()),
            device,
            context,
        }
    }

    /// Summarize a piece of text.
    /// The summarization model (DistilBART) is loaded automatically on the first call.
    pub async fn summarize(&self, text: &str, max_length: usize) -> Result<String> {
        let mut model_guard = self.models.summarizer.lock().await;
        
        // On-demand loading
        if model_guard.is_none() {
            println!("Loading summarization model for the first time...");
            *model_guard = Some(
                Seq2SeqModel::from_registry(
                    ModelType::DistilBartCnn,
                    None,
                    self.device,
                    self.context.clone(),
                ).await?
            );
            println!("Summarization model loaded.");
        }

        // We know the model is Some now, so we can unwrap.
        let summarizer = model_guard.as_ref().unwrap();
        summarizer.generate(text, max_length).await
    }

    /// Generate text from a prompt.
    /// The generation model (DistilGPT2) is loaded automatically on the first call.
    pub async fn generate(&self, prompt: &str, max_new_tokens: usize) -> Result<String> {
        let mut model_guard = self.models.generator.lock().await;

        // On-demand loading
        if model_guard.is_none() {
            println!("Loading generation model for the first time...");
            *model_guard = Some(
                TextGenerator::from_registry(
                    ModelType::DistilGpt2,
                    None,
                    self.device,
                    self.context.clone(),
                ).await?
            );
             println!("Generation model loaded.");
        }

        let generator = model_guard.as_ref().unwrap();
        generator.generate(prompt, max_new_tokens).await
    }
    
    // You would add other methods here following the same pattern:
    // pub async fn translate(...) -> Result<String> { ... }
    // pub async fn encode(...) -> Result<Vec<f32>> { ... }
}