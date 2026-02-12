//! Model validation for embedding.

use kjarni_transformers::models::{ModelArchitecture, ModelTask, ModelType};

use super::types::{EmbedderError, EmbedderResult};

/// Validate that a model can be used for embedding.
pub fn validate_for_embedding(model_type: ModelType) -> EmbedderResult<()> {
    let info = model_type.info();
    let cli_name = model_type.cli_name();

    // Check architecture
    match info.architecture {
        ModelArchitecture::Bert | ModelArchitecture::NomicBert | ModelArchitecture::Mpnet => {
            // Valid encoder architectures
        }
        _ => {
            return Err(EmbedderError::IncompatibleModel {
                model: cli_name.to_string(),
                reason: format!(
                    "Architecture '{}' is not an encoder. Embedding requires BERT-style models.",
                    info.architecture.display_name()
                ),
            });
        }
    }

    // Check task
    match info.task {
        ModelTask::Embedding => {
            // Ideal
        }
        ModelTask::ReRanking | ModelTask::Classification | ModelTask::SentimentAnalysis => {
            // Can still produce embeddings
        }
        _ => {
            return Err(EmbedderError::IncompatibleModel {
                model: cli_name.to_string(),
                reason: format!("Model is designed for {:?}, not embedding.", info.task),
            });
        }
    }

    Ok(())
}

/// Get all available embedding models.
pub fn get_embedding_models() -> Vec<&'static str> {
    ModelType::all()
        .filter(|m| validate_for_embedding(*m).is_ok())
        .map(|m| m.cli_name())
        .collect()
}