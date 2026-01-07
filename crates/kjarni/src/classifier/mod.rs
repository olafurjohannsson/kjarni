//! High-level text classification API.
//!
//! This module provides a simple interface for text classification tasks
//! like sentiment analysis, topic classification, and more.
//!
//! # Quick Start
//!
//! ```ignore
//! use kjarni::classifier;
//!
//! // One-liner
//! let result = classifier::classify("sentiment-model", "I love this!").await?;
//! println!("{}: {:.1}%", result.label, result.score * 100.0);
//!
//! // With builder
//! let classifier = Classifier::builder("sentiment-model")
//!     .gpu()
//!     .top_k(3)
//!     .build()
//!     .await?;
//!
//! let results = classifier.classify("Great product!").await?;
//! ```
//!
//! # Batch Classification
//!
//! ```ignore
//! let texts = ["I love it", "Terrible experience", "It's okay"];
//! let results = classifier.classify_batch(&texts).await?;
//! ```

mod builder;
mod model;
pub mod presets;
mod types;
mod validation;

pub use builder::ClassifierBuilder;
pub use model::Classifier;
pub use presets::{ClassifierPreset, ClassifierTier};
pub use types::{
    ClassificationOverrides, ClassificationMode, ClassificationResult, ClassifierError, ClassifierResult,
};

// ============================================================================
// Convenience Functions
// ============================================================================

/// Classify a single text using the specified model.
///
/// This is the simplest API - downloads model if needed, classifies, returns result.
///
/// # Example
///
/// ```ignore
/// let result = kjarni::classifier::classify("sentiment-model", "I love this!").await?;
/// println!("{}: {:.2}%", result.label, result.score * 100.0);
/// ```
pub async fn classify(model: &str, text: &str) -> ClassifierResult<ClassificationResult> {
    let classifier = Classifier::new(model).await?;
    classifier.classify(text).await
}

/// Classify with a preset.
pub async fn classify_preset(
    preset: &ClassifierPreset,
    text: &str,
) -> ClassifierResult<ClassificationResult> {
    let classifier = Classifier::from_preset(preset).build().await?;
    classifier.classify(text).await
}

/// Check if a model is valid for classification.
pub fn is_classifier_model(model: &str) -> ClassifierResult<()> {
    use kjarni_transformers::models::ModelType;

    let model_type = ModelType::from_cli_name(model)
        .ok_or_else(|| ClassifierError::UnknownModel(model.to_string()))?;

    validation::validate_for_classification(model_type)?;
    Ok(())
}

/// List all available classifier models.
pub fn available_models() -> Vec<&'static str> {
    validation::get_classifier_models()
}


#[cfg(test)]
mod tests;