//! Model validation for summarization.

use kjarni_transformers::models::{ModelArchitecture, ModelTask, ModelType};

use super::types::{SummarizerError, SummarizerResult};

/// Validate that a model can be used for summarization.
pub fn validate_for_summarization(model_type: ModelType) -> SummarizerResult<()> {
    let info = model_type.info();
    let cli_name = model_type.cli_name();

    match info.architecture {
        ModelArchitecture::Bart => {
            // BART - ideal for summarization
            // Check task is appropriate
            match info.task {
                ModelTask::Summarization | ModelTask::Seq2Seq | ModelTask::TextToText => {}
                _ => {
                    return Err(SummarizerError::IncompatibleModel {
                        model: cli_name.to_string(),
                        reason: format!(
                            "BART model is configured for {:?}, not summarization.",
                            info.task
                        ),
                    });
                }
            }
        }
        ModelArchitecture::T5 => {
            // T5/FLAN-T5 - also works for summarization
        }
        _ => {
            return Err(SummarizerError::IncompatibleModel {
                model: cli_name.to_string(),
                reason: format!(
                    "Architecture '{}' cannot perform summarization. Use BART or T5 models.",
                    info.architecture.display_name()
                ),
            });
        }
    }

    Ok(())
}

/// Check that a model loaded from a folder can summarize: BART and T5 can,
/// Marian only translates.
pub fn validate_folder_for_summarization(
    name: &str,
    architecture: ModelArchitecture,
) -> SummarizerResult<()> {
    match architecture {
        ModelArchitecture::Bart | ModelArchitecture::T5 => Ok(()),
        other => Err(SummarizerError::IncompatibleModel {
            model: name.to_string(),
            reason: format!(
                "Architecture '{}' cannot perform summarization. Use BART or T5 models.",
                other.display_name()
            ),
        }),
    }
}

/// Get all available summarization models.
pub fn get_summarization_models() -> Vec<&'static str> {
    ModelType::all()
        .filter(|m| validate_for_summarization(*m).is_ok())
        .map(|m| m.cli_name())
        .collect()
}

/// Get recommended models for summarization.
pub fn recommended_summarization_models() -> Vec<&'static str> {
    vec![
        "distilbart-cnn", // Fast
        "bart-large-cnn", // Quality
        "flan-t5-base",   // Flexible
    ]
}
