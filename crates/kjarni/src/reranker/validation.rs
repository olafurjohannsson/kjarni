//! Model validation for reranking.

use kjarni_transformers::models::{ModelArchitecture, ModelTask, ModelType};

use super::types::{RerankerError, RerankerResult};

/// Validate that a model can be used for reranking.
pub fn validate_for_reranking(model_type: ModelType) -> RerankerResult<()> {
    let info = model_type.info();
    let cli_name = model_type.cli_name();

    // Check architecture - must be encoder-based
    match info.architecture {
        ModelArchitecture::Bert | ModelArchitecture::NomicBert => {
            // Valid encoder architectures for cross-encoding
        }
        _ => {
            return Err(RerankerError::IncompatibleModel {
                model: cli_name.to_string(),
                reason: format!(
                    "Architecture '{}' is not suitable for cross-encoding. Reranking requires BERT-style encoder models.",
                    info.architecture.display_name()
                ),
            });
        }
    }

    // Check task - should be reranking or classification (can be used as cross-encoder)
    match info.task {
        ModelTask::ReRanking => {
            // Ideal - explicitly trained for reranking
        }
        ModelTask::Classification | ModelTask::SentimentAnalysis => {
            // Can work as cross-encoder if it has a head
            // (e.g., NLI models can be used for semantic similarity)
        }
        _ => {
            return Err(RerankerError::IncompatibleModel {
                model: cli_name.to_string(),
                reason: format!(
                    "Model is designed for {:?}, not reranking. Cross-encoders need a classification head.",
                    info.task
                ),
            });
        }
    }

    Ok(())
}

/// Get all available reranking models.
pub fn get_reranking_models() -> Vec<&'static str> {
    ModelType::all()
        .filter(|m| validate_for_reranking(*m).is_ok())
        .map(|m| m.cli_name())
        .collect()
}

/// Check if a model is specifically designed for reranking (vs just compatible).
pub fn is_dedicated_reranker(model_type: ModelType) -> bool {
    model_type.info().task == ModelTask::ReRanking
}