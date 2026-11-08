//! The entity recognizer.

use std::path::PathBuf;

use kjarni_models::{Aggregation, Entity, TokenClassifier, TokenLabel};
use kjarni_transformers::models::ModelType;
use kjarni_transformers::{WgpuContext, traits::Device};

use crate::common::{default_cache_dir, ensure_model_downloaded, folder_name};

use super::builder::EntityRecognizerBuilder;
use super::types::{NerError, NerResult};
use super::validation::validate;

/// Finds named entities in text with a token classification model.
pub struct EntityRecognizer {
    inner: TokenClassifier,
    model_id: String,
    aggregation: Aggregation,
}

impl EntityRecognizer {
    /// Load a registry model, such as `distilbert-ner`, with default options.
    pub async fn new(model: &str) -> NerResult<Self> {
        EntityRecognizerBuilder::new(model).build().await
    }

    pub fn builder(model: impl Into<String>) -> EntityRecognizerBuilder {
        EntityRecognizerBuilder::new(model)
    }

    /// Load a model from a local folder: a `*ForTokenClassification` model
    /// with `config.json`, `model.safetensors` and a tokenizer.
    pub fn from_path(path: impl Into<PathBuf>) -> EntityRecognizerBuilder {
        EntityRecognizerBuilder::new("custom").model_path(path)
    }

    pub(crate) async fn from_builder(b: EntityRecognizerBuilder) -> NerResult<Self> {
        let device = b.device.to_device();
        let context = match (device, b.context) {
            (Device::Wgpu, Some(ctx)) => Some(ctx),
            (Device::Wgpu, None) => Some(
                WgpuContext::new()
                    .await
                    .map_err(|_| NerError::GpuUnavailable)?,
            ),
            _ => None,
        };

        let (inner, model_id) = match &b.model_path {
            Some(path) => {
                if !path.exists() {
                    return Err(NerError::ModelPathNotFound(path.display().to_string()));
                }
                if !b.quiet {
                    eprintln!("Loading entity recognizer from {}...", path.display());
                }
                let inner =
                    TokenClassifier::from_pretrained(path, device, context, None).map_err(|e| {
                        NerError::LoadFailed {
                            model: path.display().to_string(),
                            source: e,
                        }
                    })?;
                (inner, folder_name(path))
            }
            None => {
                let model_type = ModelType::resolve(&b.model).map_err(NerError::UnknownModel)?;
                validate(model_type)?;
                let cache_dir = b.cache_dir.clone().unwrap_or_else(default_cache_dir);
                ensure_model_downloaded(model_type, Some(&cache_dir), b.download_policy, b.quiet)
                    .await
                    .map_err(|e| match e {
                        crate::common::KjarniError::ModelNotDownloaded(m) => {
                            NerError::ModelNotDownloaded(m)
                        }
                        crate::common::KjarniError::DownloadFailed { model, source } => {
                            NerError::DownloadFailed { model, source }
                        }
                        other => NerError::Failed(other.into()),
                    })?;
                let inner = TokenClassifier::from_registry(
                    model_type,
                    Some(cache_dir),
                    device,
                    context,
                    None,
                )
                .await
                .map_err(|e| NerError::LoadFailed {
                    model: b.model.clone(),
                    source: e,
                })?;
                (inner, b.model.clone())
            }
        };

        Ok(Self {
            inner,
            model_id,
            aggregation: b.aggregation,
        })
    }

    /// The entities in a text.
    pub async fn recognize(&self, text: &str) -> NerResult<Vec<Entity>> {
        Ok(self.recognize_batch(&[text]).await?.remove(0))
    }

    /// The entities in each text.
    pub async fn recognize_batch(&self, texts: &[&str]) -> NerResult<Vec<Vec<Entity>>> {
        Ok(self.inner.entities(texts, self.aggregation).await?)
    }

    /// Every token with its own label, before tokens are grouped into
    /// entities. `O` marks a token outside any entity.
    pub async fn tokens(&self, text: &str) -> NerResult<Vec<TokenLabel>> {
        Ok(self.inner.tokens(&[text]).await?.remove(0))
    }

    /// The model's labels, such as `B-PER`, `I-PER` and `O`.
    pub fn labels(&self) -> &[String] {
        self.inner.labels()
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn aggregation(&self) -> Aggregation {
        self.aggregation
    }

    pub fn device(&self) -> Device {
        self.inner.device()
    }
}
