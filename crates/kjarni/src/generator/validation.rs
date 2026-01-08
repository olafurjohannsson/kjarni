// =============================================================================
// kjarni/src/generator/validation.rs
// =============================================================================

//! Model validation for text generation.

use kjarni_transformers::models::{ModelArchitecture, ModelTask, ModelType};
use super::types::{GeneratorError, GeneratorResult};

/// Validation result with optional warnings.
pub struct ValidationResult {
    pub warnings: Vec<String>,
}

/// Validate that a model is suitable for text generation.
pub fn validate_for_generation(model_type: ModelType) -> GeneratorResult<ValidationResult> {
    let info = model_type.info();
    let mut warnings = Vec::new();

    // Check architecture
    let valid_arch = matches!(
        info.architecture,
        ModelArchitecture::Llama
            | ModelArchitecture::Qwen2
            | ModelArchitecture::Mistral
            | ModelArchitecture::GPT
            | ModelArchitecture::Phi3
    );

    if !valid_arch {
        return Err(GeneratorError::InvalidModel(
            model_type.cli_name().to_string(),
            format!(
                "Architecture {:?} is not a decoder model. Use an encoder model for embeddings/classification.",
                info.architecture
            ),
        ));
    }

    // Warn about instruct models being used for raw generation
    if info.task == ModelTask::Chat {
        warnings.push(format!(
            "Note: '{}' is an instruction-tuned model. For raw text completion, \
             consider using a base model like 'gpt2' or 'llama3.2-1b'.",
            model_type.cli_name()
        ));
    }

    Ok(ValidationResult { warnings })
}

/// Get list of models suitable for generation.
pub fn get_generation_models() -> Vec<String> {
    ModelType::all()
        .filter(|m| {
            matches!(
                m.info().architecture,
                ModelArchitecture::Llama
                    | ModelArchitecture::Qwen2
                    | ModelArchitecture::Mistral
                    | ModelArchitecture::GPT
                    | ModelArchitecture::Phi3
            )
        })
        .map(|m| m.cli_name().to_string())
        .collect()
}