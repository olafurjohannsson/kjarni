//! Options for loading an entity recognizer.

use std::path::PathBuf;
use std::sync::Arc;

use kjarni_models::Aggregation;
use kjarni_transformers::WgpuContext;

use crate::common::{DownloadPolicy, KjarniDevice};

use super::model::EntityRecognizer;
use super::types::NerResult;

/// Options for [`EntityRecognizer`].
pub struct EntityRecognizerBuilder {
    pub(crate) model: String,
    pub(crate) model_path: Option<PathBuf>,
    pub(crate) cache_dir: Option<PathBuf>,
    pub(crate) device: KjarniDevice,
    pub(crate) context: Option<Arc<WgpuContext>>,
    pub(crate) download_policy: DownloadPolicy,
    pub(crate) aggregation: Aggregation,
    pub(crate) quiet: bool,
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
