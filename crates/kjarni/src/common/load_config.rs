// =============================================================================
// kjarni/src/common/load_config.rs
// =============================================================================

//! Model loading configuration.

use kjarni_transformers::{models::base::ModelLoadConfig, tensor::DType};

/// Wrapper around ModelLoadConfig for the high-level API.
#[derive(Debug, Clone)]
pub struct LoadConfig {
    pub(crate) inner: ModelLoadConfig,
}

impl LoadConfig {
    /// Create a new default LoadConfig.
    pub fn new() -> Self {
        Self {
            inner: ModelLoadConfig::default(),
        }
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

impl Default for LoadConfig {
    fn default() -> Self {
        Self::new()
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
    inner: ModelLoadConfig,
}

impl LoadConfigBuilder {
    /// Create a new builder with defaults.
    pub fn new() -> Self {
        Self {
            inner: ModelLoadConfig::default(),
        }
    }

    /// Create a builder from an existing LoadConfig.
    pub fn from_config(config: LoadConfig) -> Self {
        Self {
            inner: config.inner,
        }
    }

    /// Create a builder from an existing ModelLoadConfig.
    pub fn from_inner(inner: ModelLoadConfig) -> Self {
        Self { inner }
    }

    /// Keep embedding layer on CPU to save VRAM.
    pub fn offload_embeddings(mut self, offload: bool) -> Self {
        self.inner.offload_embeddings = offload;
        self
    }

    /// Keep language model head on CPU to save VRAM.
    pub fn offload_lm_head(mut self, offload: bool) -> Self {
        self.inner.offload_lm_head = offload;
        self
    }

    /// Set target dtype for weights.
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.inner.target_dtype = Some(dtype);
        self
    }

    /// Use F16 precision.
    pub fn f16(self) -> Self {
        self.dtype(DType::F16)
    }

    /// Use BF16 precision.
    pub fn bf16(self) -> Self {
        self.dtype(DType::BF16)
    }

    /// Use F32 precision (default).
    pub fn f32(self) -> Self {
        self.dtype(DType::F32)
    }

    /// Quantize the LM head to specified dtype.
    pub fn quantize_lm_head(mut self, dtype: DType) -> Self {
        self.inner.quantize_lm_head = Some(dtype);
        self
    }

    /// Quantize LM head to Q8_0.
    pub fn quantize_lm_head_q8(self) -> Self {
        self.quantize_lm_head(DType::Q8_0)
    }

    /// Set maximum batch size for KV cache pre-allocation.
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.inner.max_batch_size = Some(size);
        self
    }

    /// Set maximum sequence length for KV cache pre-allocation.
    pub fn max_sequence_length(mut self, length: usize) -> Self {
        self.inner.max_sequence_length = Some(length);
        self
    }

    /// Prefer GGUF format if available.
    pub fn prefer_gguf(mut self, prefer: bool) -> Self {
        self.inner.use_gguf = prefer;
        self
    }

    /// Build the LoadConfig.
    pub fn build(self) -> LoadConfig {
        LoadConfig { inner: self.inner }
    }
}