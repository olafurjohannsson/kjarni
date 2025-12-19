//! Encoder configuration types
use crate::tensor::DType;
use std::fmt;

/// Pooling strategies for sequence outputs
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum PoolingStrategy {
    #[default]
    Mean,
    Max,
    Cls,
    LastToken,
}

impl fmt::Display for PoolingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PoolingStrategy::Mean => write!(f, "Mean"),
            PoolingStrategy::Max => write!(f, "Max"),
            PoolingStrategy::Cls => write!(f, "CLS"),
            PoolingStrategy::LastToken => write!(f, "LastToken"),
        }
    }
}

/// Configuration for encoding/embedding operations
#[derive(Clone, Debug)]
pub struct EncodingConfig {
    pub pooling_strategy: PoolingStrategy,
    pub normalize: bool,
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            pooling_strategy: PoolingStrategy::Mean,
            normalize: true,
        }
    }
}

impl fmt::Display for EncodingConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EncodingConfig {{ pooling: {}, normalize: {} }}",
            self.pooling_strategy, self.normalize
        )
    }
}

/// Configuration for CPU/GPU memory placement during encoder loading.
#[derive(Debug, Clone, Default)]
pub struct EncoderLoadConfig {
    /// If true, embedding weights stay in CPU RAM (saves ~500MB-2GB VRAM)
    pub cpu_embeddings: bool,
    /// Which layers to run on GPU: None = all, Some((start, end)) = layers [start, end)
    pub gpu_layer_range: Option<(usize, usize)>,
    /// Target dtype for quantization
    pub dtype: Option<DType>,
    /// Pre-allocation hint for batch size
    pub max_batch_size: Option<usize>,
    /// Pre-allocation hint for sequence length
    pub max_sequence_length: Option<usize>,
}

impl EncoderLoadConfig {
    /// Full GPU execution (default, maximum performance)
    pub fn full_gpu() -> Self {
        Self::default()
    }

    /// CPU embeddings, GPU layers (saves ~500MB-2GB VRAM)
    pub fn offload_embeddings() -> Self {
        Self {
            cpu_embeddings: true,
            ..Default::default()
        }
    }

    /// Quantized model loading
    pub fn quantized(dtype: DType) -> Self {
        Self {
            dtype: Some(dtype),
            ..Default::default()
        }
    }

    /// Partial GPU execution: only layers [start, end) on GPU
    pub fn partial_gpu(start: usize, end: usize) -> Self {
        Self {
            gpu_layer_range: Some((start, end)),
            ..Default::default()
        }
    }

    // Builder methods
    pub fn with_cpu_embeddings(mut self, cpu: bool) -> Self {
        self.cpu_embeddings = cpu;
        self
    }

    pub fn with_gpu_layer_range(mut self, start: usize, end: usize) -> Self {
        self.gpu_layer_range = Some((start, end));
        self
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = Some(size);
        self
    }

    pub fn with_max_sequence_length(mut self, len: usize) -> Self {
        self.max_sequence_length = Some(len);
        self
    }
}