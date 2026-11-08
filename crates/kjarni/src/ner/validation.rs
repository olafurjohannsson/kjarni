//! Which models can recognize entities.

use kjarni_transformers::models::{ModelArchitecture, ModelTask, ModelType};

use super::DEFAULT_MODEL;
use super::types::{NerError, NerResult};

pub(crate) fn validate(model_type: ModelType) -> NerResult<()> {
    let info = model_type.info();
    let ok = matches!(info.task, ModelTask::TokenClassification)
        && matches!(
            info.architecture,
            ModelArchitecture::Bert | ModelArchitecture::NomicBert | ModelArchitecture::Mpnet
        );
    if ok {
        Ok(())
    } else {
        Err(NerError::IncompatibleModel {
            model: model_type.cli_name().to_string(),
            reason: format!(
                "it is a {:?} model; entity recognition needs a token classification model \
                 such as {DEFAULT_MODEL}",
                info.task
            ),
        })
    }
}
