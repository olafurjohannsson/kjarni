// kjarni/src/encoder.rs

use anyhow::{anyhow, Result};
use kjarni_models::sentence_encoder::SentenceEncoder;
use kjarni_transformers::{Device, ModelType, WgpuContext};
use std::sync::Arc;

/// Sentence/text encoder for generating embeddings.
///
/// # Example
///
/// ```ignore
/// use kjarni::Encoder;
///
/// let encoder = Encoder::new("minilm-l6-v2").await?;
/// let embedding = encoder.encode("Hello world").await?;
/// assert_eq!(embedding.len(), 384);
/// ```
pub struct Encoder {
    inner: SentenceEncoder,
}

impl Encoder {
    /// Create encoder with default settings (CPU).
    pub async fn new(model: &str) -> Result<Self> {
        Self::builder(model).build().await
    }

    /// Create encoder builder for advanced configuration.
    pub fn builder(model: &str) -> EncoderBuilder {
        EncoderBuilder::new(model)
    }

    /// Encode a single text to embedding vector.
    pub async fn encode(&self, text: &str) -> Result<Vec<f32>> {
        self.inner.encode(text).await
    }

    /// Encode a single text with options.
    pub async fn encode_with(
        &self,
        text: &str,
        pooling: Option<&str>,
        normalize: bool,
    ) -> Result<Vec<f32>> {
        self.inner.encode_with(text, pooling, normalize).await
    }

    /// Encode multiple texts.
    pub async fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.inner.encode_batch(texts).await
    }

    /// Get embedding dimension.
    pub fn dim(&self) -> usize {
        self.inner.hidden_size()
    }

    /// Get maximum sequence length.
    pub fn max_length(&self) -> usize {
        self.inner.max_seq_length()
    }
}

pub struct EncoderBuilder {
    model: String,
    device: Device,
    context: Option<Arc<WgpuContext>>,
}

impl EncoderBuilder {
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

    pub fn with_context(mut self, context: Arc<WgpuContext>) -> Self {
        self.context = Some(context);
        self.device = Device::Wgpu;
        self
    }

    pub async fn build(self) -> Result<Encoder> {
        let model_type = ModelType::from_cli_name(&self.model)
            .ok_or_else(|| anyhow!("Unknown model: '{}'", self.model))?;

        let context = if self.device.is_gpu() && self.context.is_none() {
            Some(WgpuContext::new().await?)
        } else {
            self.context
        };

        let inner =
            SentenceEncoder::from_registry(model_type, None, self.device, context, None).await?;

        Ok(Encoder { inner })
    }
}
