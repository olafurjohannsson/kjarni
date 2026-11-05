//! Named-entity recognition: people, places, organizations and the like.
//!
//! ```ignore
//! use kjarni::ner::EntityRecognizer;
//!
//! let ner = EntityRecognizer::new("distilbert-ner").await?;
//! for e in ner.recognize("Sarah Connor met Apple's CEO in San Francisco.").await? {
//!     println!("{} {} {:.2}", e.label, e.text, e.score);
//! }
//!
//! // Any *ForTokenClassification model from the Hugging Face Hub:
//! let icelandic = EntityRecognizer::from_path("IceBERT-ner").build().await?;
//! ```
//!
//! Entities come out as transformers' `pipeline("ner")` gives them for the
//! same [`Aggregation`], with the same spans and scores. The default is
//! `First`, which keeps every word whole.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use kjarni_models::TokenClassifier;
pub use kjarni_models::{Aggregation, Entity, TokenLabel};
use kjarni_transformers::models::{ModelArchitecture, ModelTask, ModelType};
use kjarni_transformers::{WgpuContext, traits::Device};

use crate::common::{DownloadPolicy, KjarniDevice, default_cache_dir, ensure_model_downloaded};

/// The model `EntityRecognizer::new` uses when given none.
pub const DEFAULT_MODEL: &str = "distilbert-ner";

/// Errors from entity recognition.
#[derive(Debug, thiserror::Error)]
pub enum NerError {
    #[error("{0}")]
    UnknownModel(String),

    #[error("Model '{model}' cannot be used for entity recognition: {reason}")]
    IncompatibleModel { model: String, reason: String },

    #[error("Model '{0}' not downloaded and download policy is set to Never")]
    ModelNotDownloaded(String),

    #[error("Model path not found: {0}")]
    ModelPathNotFound(String),

    #[error("Failed to download model '{model}': {source}")]
    DownloadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    #[error("Failed to load model '{model}': {source:#}")]
    LoadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    #[error("Entity recognition failed: {0}")]
    Failed(#[from] anyhow::Error),

    #[error("GPU requested but WebGPU context could not be created")]
    GpuUnavailable,
}

pub type NerResult<T> = Result<T, NerError>;

/// Finds named entities in text with a token classification model.
pub struct EntityRecognizer {
    inner: TokenClassifier,
    model_id: String,
    aggregation: Aggregation,
}

/// Options for [`EntityRecognizer`].
pub struct EntityRecognizerBuilder {
    model: String,
    model_path: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    device: KjarniDevice,
    context: Option<Arc<WgpuContext>>,
    download_policy: DownloadPolicy,
    aggregation: Aggregation,
    quiet: bool,
}

impl EntityRecognizerBuilder {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            model_path: None,
            cache_dir: None,
            device: KjarniDevice::Cpu,
            context: None,
            download_policy: DownloadPolicy::default(),
            aggregation: Aggregation::default(),
            quiet: false,
        }
    }

    /// Load from a local folder instead of the registry.
    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    pub fn cpu(mut self) -> Self {
        self.device = KjarniDevice::Cpu;
        self
    }

    pub fn gpu(mut self) -> Self {
        self.device = KjarniDevice::Gpu;
        self
    }

    pub fn device(mut self, device: KjarniDevice) -> Self {
        self.device = device;
        self
    }

    pub fn with_context(mut self, context: Arc<WgpuContext>) -> Self {
        self.context = Some(context);
        self
    }

    pub fn download_policy(mut self, policy: DownloadPolicy) -> Self {
        self.download_policy = policy;
        self
    }

    pub fn offline(mut self) -> Self {
        self.download_policy = DownloadPolicy::Never;
        self
    }

    /// How tokens become entities. The default, `First`, keeps words whole.
    pub fn aggregation(mut self, aggregation: Aggregation) -> Self {
        self.aggregation = aggregation;
        self
    }

    /// Say nothing on stderr while loading.
    pub fn quiet(mut self, quiet: bool) -> Self {
        self.quiet = quiet;
        self
    }

    pub async fn build(self) -> NerResult<EntityRecognizer> {
        EntityRecognizer::from_builder(self).await
    }
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

    async fn from_builder(b: EntityRecognizerBuilder) -> NerResult<Self> {
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

fn validate(model_type: ModelType) -> NerResult<()> {
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

fn folder_name(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "custom".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_model_is_a_token_classifier() {
        validate(ModelType::resolve(DEFAULT_MODEL).unwrap()).unwrap();
    }

    #[test]
    fn a_sentiment_model_is_refused_with_a_reason() {
        let err = validate(ModelType::resolve("distilbert-sentiment").unwrap())
            .unwrap_err()
            .to_string();
        assert!(err.contains("token classification"), "{err}");
    }

    #[tokio::test]
    async fn a_missing_folder_is_reported_as_missing() {
        let err = EntityRecognizer::from_path("/no/such/model/folder")
            .quiet(true)
            .build()
            .await
            .err()
            .unwrap();
        assert!(matches!(err, NerError::ModelPathNotFound(_)), "{err}");
    }

    /// Downloads distilbert-ner (261 MB) the first time. Entities are from
    /// transformers' pipeline("ner", aggregation_strategy="first") on the CPU.
    #[tokio::test]
    #[ignore = "downloads distilbert-ner"]
    async fn distilbert_ner_matches_transformers() {
        let ner = EntityRecognizer::builder(DEFAULT_MODEL)
            .quiet(true)
            .build()
            .await
            .unwrap();
        let got = ner
            .recognize("Sarah Connor met Apple's CEO in San Francisco on Tuesday.")
            .await
            .unwrap();
        let want = [
            ("PER", "Sarah Connor", 0, 12, 0.99879),
            ("ORG", "Apple", 17, 22, 0.99710),
            ("LOC", "San Francisco", 32, 45, 0.99761),
        ];
        assert_eq!(got.len(), want.len(), "{got:?}");
        for (g, (label, text, start, end, score)) in got.iter().zip(want) {
            assert_eq!(
                (g.label.as_str(), g.text.as_str(), g.start, g.end),
                (label, text, start, end)
            );
            assert!(
                (g.score - score).abs() < 1e-4,
                "{text}: {} vs {score}",
                g.score
            );
        }
    }
}
