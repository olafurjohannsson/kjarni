use edgemodels::sentence_encoder::SentenceEncoder; // <-- Add this
// use edgemodels::seq2seq::Seq2SeqModel;
// use edgemodels::text_generator::TextGenerator;

use anyhow::Result;
use edgetransformers::models::ModelType;
use edgetransformers::prelude::*;
use std::sync::Arc;
use tokio::sync::Mutex;

pub mod ffi;

// A container for our models, wrapped in Mutex for safe concurrent access.
#[derive(Default)]
struct ModelManager {
    // summarizer: Mutex<Option<Seq2SeqModel>>,
    // generator: Mutex<Option<TextGenerator>>,
    sentence_encoder: Mutex<Option<SentenceEncoder>>, // <-- Add this
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

    /// Encodes a batch of sentences into embeddings.
    /// The sentence encoding model (MiniLM-L6-v2) is loaded automatically on the first call.
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

    // Summarize a piece of text.
    // ... (your existing summarize function, no changes needed)
    // pub async fn summarize(&self, text: &str, max_length: usize) -> Result<String> {
    //     let mut model_guard = self.models.summarizer.lock().await;
        
    //     if model_guard.is_none() {
    //         println!("Loading summarization model for the first time...");
    //         *model_guard = Some(
    //             Seq2SeqModel::from_registry(
    //                 ModelType::DistilBartCnn,
    //                 None,
    //                 self.device,
    //                 self.context.clone(),
    //             ).await?
    //         );
    //         println!("Summarization model loaded.");
    //     }

    //     let summarizer = model_guard.as_ref().unwrap();
    //     summarizer.generate(text, max_length).await
    // }

    // /// Generate text from a prompt.
    // // ... (your existing generate function, no changes needed)
    // pub async fn generate(&self, prompt: &str, max_new_tokens: usize) -> Result<String> {
    //     let mut model_guard = self.models.generator.lock().await;

    //     if model_guard.is_none() {
    //         println!("Loading generation model for the first time...");
    //         *model_guard = Some(
    //             TextGenerator::from_registry(
    //                 ModelType::DistilGpt2,
    //                 None,
    //                 self.device,
    //                 self.context.clone(),
    //             ).await?
    //         );
    //          println!("Generation model loaded.");
    //     }

    //     let generator = model_guard.as_ref().unwrap();
    //     generator.generate(prompt, max_new_tokens).await
    // }
}