//! Generic encoder-decoder (seq2seq) text generation.



mod builder;
mod model;
mod resolution;
mod types;
mod validation;

// Re-exports
pub use builder::Seq2SeqGeneratorBuilder;
pub use model::Seq2SeqGenerator;
pub use types::{Seq2SeqError, Seq2SeqOverrides, Seq2SeqResult, Seq2SeqToken};

/// Generate text with default settings.
pub async fn generate(model: &str, input: &str) -> Seq2SeqResult<String> {
    Seq2SeqGenerator::new(model).await?.generate(input).await
}

/// Generate with custom overrides
pub async fn generate_with_config(
    model: &str,
    input: &str,
    overrides: Seq2SeqOverrides,
) -> Seq2SeqResult<String> {
    Seq2SeqGenerator::builder(model)
        .with_overrides(overrides)
        .build()
        .await?
        .generate(input)
        .await
}

/// List all available seq2seq models.
///
/// Returns CLI names of models that can be used with Seq2SeqGenerator.
pub fn available_models() -> Vec<&'static str> {
    validation::get_seq2seq_models()
}

/// Get suggested models for seq2seq tasks.
pub fn suggested_models() -> Vec<&'static str> {
    validation::suggest_seq2seq_models()
}

/// Check if a model is valid for seq2seq.
///
/// Returns Ok(()) if valid, or an error describing why not.
pub fn is_seq2seq_model(model: &str) -> Seq2SeqResult<()> {
    use kjarni_transformers::models::ModelType;

    let model_type = ModelType::from_cli_name(model)
        .ok_or_else(|| Seq2SeqError::UnknownModel(model.to_string()))?;

    validation::validate_for_seq2seq(model_type)?;
    Ok(())
}


#[cfg(test)]
mod send_sync_tests {
    use super::*;

    const _: () = {
        const fn assert_send<T: Send>() {}
        const fn assert_sync<T: Sync>() {}

        assert_send::<Seq2SeqGenerator>();
        assert_sync::<Seq2SeqGenerator>();

        assert_send::<Seq2SeqGeneratorBuilder>();
        assert_sync::<Seq2SeqGeneratorBuilder>();

        assert_send::<Seq2SeqOverrides>();
        assert_sync::<Seq2SeqOverrides>();

        assert_send::<Seq2SeqToken>();
        assert_sync::<Seq2SeqToken>();

        assert_send::<Seq2SeqError>();
        assert_sync::<Seq2SeqError>();
    };
}


#[cfg(test)]
mod tests;