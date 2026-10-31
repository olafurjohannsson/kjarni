//! Model validation for translation.

use kjarni_transformers::models::{ModelArchitecture, ModelType};

use super::types::{TranslatorError, TranslatorResult};

/// Validate that a model can be used for translation.
pub fn validate_for_translation(model_type: ModelType) -> TranslatorResult<()> {
    let info = model_type.info();
    let cli_name = model_type.cli_name();

    // Must be T5 (BART is primarily for summarization)
    match info.architecture {
        ModelArchitecture::T5 | ModelArchitecture::Marian => {
            // T5/FLAN-T5 takes the languages in a prompt; Marian has one direction per model
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

/// The source and target language of a model that translates in only one direction.
pub fn fixed_direction(model_type: ModelType) -> Option<(&'static str, &'static str)> {
    match model_type {
        ModelType::OpusMtEnDe => Some(("English", "German")),
        ModelType::OpusMtDeEn => Some(("German", "English")),
        _ => None,
    }
}

/// What the model is given: the bare text for a one-direction model, which refuses any
/// other direction, and a T5 prompt otherwise. Languages are in normalized form.
pub fn translation_input(
    model_type: ModelType,
    text: &str,
    from: &str,
    to: &str,
) -> TranslatorResult<String> {
    let Some((model_from, model_to)) = fixed_direction(model_type) else {
        return Ok(format!("translate {} to {}: {}", from, to, text));
    };
    if (from, to) != (model_from, model_to) {
        return Err(TranslatorError::IncompatibleModel {
            model: model_type.cli_name().to_string(),
            reason: format!("It only translates {model_from} to {model_to}, not {from} to {to}."),
        });
    }
    Ok(text.to_string())
}

/// Get all available translation models.
pub fn get_translation_models() -> Vec<&'static str> {
    ModelType::all()
        .filter(|m| validate_for_translation(*m).is_ok())
        .map(|m| m.cli_name())
        .collect()
}
