//! Model validation for translation.

use kjarni_transformers::models::{ModelArchitecture, ModelTask, ModelType};

use super::types::{TranslatorError, TranslatorResult};

/// Validate that a model can be used for translation.
pub fn validate_for_translation(model_type: ModelType) -> TranslatorResult<()> {
    let info = model_type.info();
    let cli_name = model_type.cli_name();

    // Must be T5 (BART is primarily for summarization)
    match info.architecture {
        ModelArchitecture::T5 => {
            // T5/FLAN-T5 - ideal for translation
        }
        ModelArchitecture::Bart => {
            // BART can technically translate but isn't ideal
            return Err(TranslatorError::IncompatibleModel {
                model: cli_name.to_string(),
                reason: "BART models are optimized for summarization, not translation. Use flan-t5-base or flan-t5-large.".to_string(),
            });
        }
        _ => {
            return Err(TranslatorError::IncompatibleModel {
                model: cli_name.to_string(),
                reason: format!(
                    "Architecture '{}' cannot perform translation. Use T5 models (flan-t5-base, flan-t5-large).",
                    info.architecture.display_name()
                ),
            });
        }
    }

    Ok(())
}

/// Get all available translation models.
pub fn get_translation_models() -> Vec<&'static str> {
    ModelType::all()
        .filter(|m| validate_for_translation(*m).is_ok())
        .map(|m| m.cli_name())
        .collect()
}