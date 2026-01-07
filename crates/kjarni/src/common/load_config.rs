//! Model loading configuration.

use kjarni_transformers::{models::base::ModelLoadConfig, tensor::DType};


/// High-level loading configuration with builder pattern.
///
/// This wraps `ModelLoadConfig` with a more ergonomic API.
#[derive(Debug, Clone)]
pub struct LoadConfig {
    pub(crate) inner: ModelLoadConfig,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            inner: ModelLoadConfig::default(),
        }
    }
}

impl LoadConfig {
    /// Create a new default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> LoadConfigBuilder {
        LoadConfigBuilder::new()
    }

    /// Get the inner ModelLoadConfig.
    pub fn into_inner(self) -> ModelLoadConfig {
        self.inner
    }

    /// Get a reference to the inner config.
    pub fn as_inner(&self) -> &ModelLoadConfig {
        &self.inner
    }
}

impl From<LoadConfig> for ModelLoadConfig {
    fn from(config: LoadConfig) -> Self {
        config.inner
    }
}

impl From<ModelLoadConfig> for LoadConfig {
    fn from(inner: ModelLoadConfig) -> Self {
        Self { inner }
    }
}

/// Builder for LoadConfig.
#[derive(Debug, Clone, Default)]
pub struct LoadConfigBuilder {
    offload_embeddings: bool,
    offload_lm_head: bool,
    gpu_layers: Option<usize>,
    target_dtype: Option<DType>,
    quantize_lm_head: Option<DType>,
    gpu_layer_range: Option<(usize, usize)>,
    max_batch_size: Option<usize>,
    max_sequence_length: Option<usize>,
    use_gguf: bool,
}

impl LoadConfigBuilder {
    /// Create a new builder with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Keep embedding layer on CPU to save VRAM.
    pub fn offload_embeddings(mut self, offload: bool) -> Self {
        self.offload_embeddings = offload;
        self
    }

    /// Keep language model head on CPU to save VRAM.
    pub fn offload_lm_head(mut self, offload: bool) -> Self {
        self.offload_lm_head = offload;
        self
    }

    /// Number of layers to place on GPU (None = all).
    pub fn gpu_layers(mut self, layers: Option<usize>) -> Self {
        self.gpu_layers = layers;
        self
    }

    /// Force quantization to this data type.
    pub fn target_dtype(mut self, dtype: DType) -> Self {
        self.target_dtype = Some(dtype);
        self
    }

    /// Quantize the LM head to this data type.
    pub fn quantize_lm_head(mut self, dtype: DType) -> Self {
        self.quantize_lm_head = Some(dtype);
        self
    }

    /// Only place layers in range [start, end) on GPU.
    pub fn gpu_layer_range(mut self, start: usize, end: usize) -> Self {
        self.gpu_layer_range = Some((start, end));
        self
    }

    /// Pre-allocate KV cache for this batch size.
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = Some(size);
        self
    }

    /// Pre-allocate KV cache for this sequence length.
    pub fn max_sequence_length(mut self, length: usize) -> Self {
        self.max_sequence_length = Some(length);
        self
    }

    /// Use GGUF format.
    pub fn use_gguf(mut self, use_gguf: bool) -> Self {
        self.use_gguf = use_gguf;
        self
    }

    /// Build the LoadConfig.
    pub fn build(self) -> LoadConfig {
        LoadConfig {
            inner: ModelLoadConfig {
                offload_embeddings: self.offload_embeddings,
                offload_lm_head: self.offload_lm_head,
                gpu_layers: self.gpu_layers,
                target_dtype: self.target_dtype,
                quantize_lm_head: self.quantize_lm_head,
                gpu_layer_range: self.gpu_layer_range,
                max_batch_size: self.max_batch_size,
                max_sequence_length: self.max_sequence_length,
                use_gguf: self.use_gguf,
            },
        }
    }
}