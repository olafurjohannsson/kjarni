//! Model validation for classification.

use kjarni_transformers::models::{ModelArchitecture, ModelTask, ModelType};

use super::types::{ClassifierError, ClassifierResult};

/// Validate that a model can be used for classification.
pub fn validate_for_classification(model_type: ModelType) -> ClassifierResult<()> {
    let info = model_type.info();
    let cli_name = model_type.cli_name();

    // Check architecture
    match info.architecture {
        ModelArchitecture::Bert | ModelArchitecture::NomicBert => {
            // Valid encoder architectures
        }
        _ => {
            return Err(ClassifierError::IncompatibleModel {
                model: cli_name.to_string(),
                reason: format!(
                    "Architecture '{}' is not an encoder. Classification requires BERT-style models.",
                    info.architecture.display_name()
                ),
            });
        }
    }

    // Check task
    match info.task {
        ModelTask::SentimentAnalysis
        | ModelTask::Classification
        | ModelTask::ZeroShotClassification => {
            // Ideal
        }
        ModelTask::ReRanking => {
            // Cross-encoders can be used for classification
        }
        _ => {
            return Err(ClassifierError::IncompatibleModel {
                model: cli_name.to_string(),
                reason: format!(
                    "Model is designed for {:?}, not classification.",
                    info.task
                ),
            });
        }
    }

    Ok(())
}

/// Get all available classifier models.
pub fn get_classifier_models() -> Vec<&'static str> {
    ModelType::all()
        .filter(|m| validate_for_classification(*m).is_ok())
        .map(|m| m.cli_name())
        .collect()
}