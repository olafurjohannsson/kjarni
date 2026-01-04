// kjarni/src/classifier.rs

use anyhow::{anyhow, Result};
use kjarni_models::sequence_classifier::SequenceClassifier;
use kjarni_transformers::{Device, ModelType, WgpuContext};
use std::sync::Arc;

/// Result of text classification.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub label: String,
    pub score: f32,
}

/// Text classifier for sentiment analysis, topic classification, etc.
///
/// # Example
///
/// ```rust
/// use kjarni::Classifier;
///
/// let classifier = Classifier::new("sentiment-distilbert").await?;
/// let result = classifier.classify("I love this product!").await?;
/// println!("{}: {:.1}%", result.label, result.score * 100.0);
/// ```
pub struct Classifier {
    inner: SequenceClassifier,
}

impl Classifier {
    /// Create classifier with default settings.
    pub async fn new(model: &str) -> Result<Self> {
        Self::builder(model).build().await
    }

    /// Create classifier builder.
    pub fn builder(model: &str) -> ClassifierBuilder {
        ClassifierBuilder::new(model)
    }

    /// Classify text, returning the top prediction.
    pub async fn classify(&self, text: &str) -> Result<ClassificationResult> {
        let predictions = self.classify_top_k(text, 1).await?;
        predictions
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No predictions returned"))
    }

    /// Classify text, returning top-k predictions.
    pub async fn classify_top_k(&self, text: &str, k: usize) -> Result<Vec<ClassificationResult>> {
        let predictions = self.inner.classify(text, k).await?;
        Ok(predictions
            .into_iter()
            .map(|(label, score)| ClassificationResult { label, score })
            .collect())
    }

    /// Classify multiple texts.
    pub async fn classify_batch(&self, texts: &[&str]) -> Result<Vec<ClassificationResult>> {
        let all_predictions = self.inner.classify_batch(texts, 1).await?;
        Ok(all_predictions
            .into_iter()
            .filter_map(|preds| preds.into_iter().next())
            .map(|(label, score)| ClassificationResult { label, score })
            .collect())
    }

    /// Get available labels.
    pub fn labels(&self) -> Option<&[String]> {
        self.inner.labels()
    }
}

pub struct ClassifierBuilder {
    model: String,
    device: Device,
    context: Option<Arc<WgpuContext>>,
}

impl ClassifierBuilder {
    fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            device: Device::Cpu,
            context: None,
        }
    }

    pub fn cpu(mut self) -> Self {
        self.device = Device::Cpu;
        self
    }
    pub fn gpu(mut self) -> Self {
        self.device = Device::Wgpu;
        self
    }

    pub async fn build(self) -> Result<Classifier> {
        let model_type = ModelType::from_cli_name(&self.model)
            .ok_or_else(|| anyhow!("Unknown model: '{}'", self.model))?;

        let context = if self.device.is_gpu() {
            Some(WgpuContext::new().await?)
        } else {
            None
        };

        let inner =
            SequenceClassifier::from_registry(model_type, None, self.device, context, None).await?;

        Ok(Classifier { inner })
    }
}
