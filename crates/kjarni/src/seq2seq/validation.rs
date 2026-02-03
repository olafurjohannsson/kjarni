//! Model validation for seq2seq generation.

use kjarni_transformers::models::{ModelArchitecture, ModelTask, ModelType};

use super::types::{Seq2SeqError, Seq2SeqResult};

/// Validation result with optional warnings.
#[derive(Debug)]
pub struct Seq2SeqValidation {
    /// Warnings that don't prevent usage but should be shown.
    pub warnings: Vec<String>,
}

/// Validate that a model can be used for seq2seq generation.
///
/// Returns `Ok(Seq2SeqValidation)` if valid, with any warnings.
/// Returns `Err(Seq2SeqError::IncompatibleModel)` if the model cannot be used.
pub fn validate_for_seq2seq(model_type: ModelType) -> Seq2SeqResult<Seq2SeqValidation> {
    let info = model_type.info();
    let cli_name = model_type.cli_name();
    let mut warnings = Vec::new();

    // Check architecture - must be encoder-decoder
    match info.architecture {
        ModelArchitecture::T5 => {
            // T5/FLAN-T5 - ideal for seq2seq
        }
        ModelArchitecture::Bart => {
            // BART - also ideal for seq2seq
        }
        ModelArchitecture::Whisper => {
            // Whisper is encoder-decoder but for speech
            return Err(Seq2SeqError::IncompatibleModel {
                model: cli_name.to_string(),
                reason: "Whisper is for speech-to-text, not text-to-text. Use Transcriber instead."
                    .to_string(),
            });
        }
        _ => {
            return Err(Seq2SeqError::IncompatibleModel {
                model: cli_name.to_string(),
                reason: format!(
                    "Architecture '{}' is not encoder-decoder. Seq2Seq requires T5 or BART models.",
                    info.architecture.display_name()
                ),
            });
        }
    }

    // Check task - should be a seq2seq task
    match info.task {
        ModelTask::Seq2Seq | ModelTask::TextToText => {
            // Ideal - general seq2seq
        }
        ModelTask::Summarization => {
            // Good - specifically tuned for summarization
            warnings.push(format!(
                "Model '{}' is optimized for summarization. For translation, consider using flan-t5-base or flan-t5-large.",
                cli_name
            ));
        }
        ModelTask::Translation => {
            // Good - specifically tuned for translation
        }
        _ => {
            return Err(Seq2SeqError::IncompatibleModel {
                model: cli_name.to_string(),
                reason: format!(
                    "Model is designed for {:?}, not text-to-text generation.",
                    info.task
                ),
            });
        }
    }

    Ok(Seq2SeqValidation { warnings })
}

/// Get all available seq2seq models.
pub fn get_seq2seq_models() -> Vec<&'static str> {
    ModelType::all()
        .filter(|m| validate_for_seq2seq(*m).is_ok())
        .map(|m| m.cli_name())
        .collect()
}

/// Get suggested models for seq2seq tasks.
pub fn suggest_seq2seq_models() -> Vec<&'static str> {
    vec![
        "flan-t5-base",    // Good general-purpose
        "flan-t5-large",   // Higher quality
        "distilbart-cnn",  // Fast summarization
        "bart-large-cnn",  // Quality summarization
    ]
}