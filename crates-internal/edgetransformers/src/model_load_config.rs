pub use crate::weights::DType;

// ============================================================================
// MODEL LOADING CONFIGURATION
// ============================================================================

/// Unified configuration for loading model components across CPU/GPU.
///
/// This configuration controls memory placement and quantization for both
/// encoder and decoder models, enabling flexible tradeoffs between
/// performance and VRAM usage.
///
/// # Examples
///
/// ```rust
/// // Full GPU (maximum performance)
/// let config = ModelLoadConfig::default();
///
/// // Offload embeddings to save ~1GB VRAM
/// let config = ModelLoadConfig::offload_embeddings();
///
/// // Maximum VRAM saving
/// let config = ModelLoadConfig::max_offload();
///
/// // Run quantized model
/// let config = ModelLoadConfig::quantized(DType::Q4_0);
///
/// // Custom configuration
/// let config = ModelLoadConfig {
///     cpu_embeddings: true,
///     gpu_layer_range: Some((0, 6)),  // Only first 6 layers on GPU
///     dtype: Some(DType::BF16),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelLoadConfig {
    /// Keep embedding weights in system RAM.
    ///
    /// When true:
    /// - Word/position embeddings stay in CPU RAM
    /// - Embedding lookup happens on CPU
    /// - Hidden states are uploaded to GPU after lookup
    ///
    /// Saves VRAM: ~500MB-2GB depending on vocab size and hidden dim.
    /// Performance impact: Minimal for long sequences, noticeable for short sequences.
    pub cpu_embeddings: bool,

    /// Keep output projection (LM head) in system RAM.
    ///
    /// When true:
    /// - Final hidden states are downloaded from GPU
    /// - Logit projection happens on CPU
    ///
    /// Only relevant for decoder and seq2seq models (encoders don't have LM heads).
    /// Saves VRAM: ~500MB-2GB depending on vocab size and hidden dim.
    pub cpu_output_head: bool,

    /// Range of layers to run on GPU: [start, end).
    ///
    /// - None = all layers on GPU (default)
    /// - Some((0, 6)) = layers 0-5 on GPU, layers 6+ on CPU
    /// - Some((6, 12)) = layers 0-5 on CPU, layers 6-11 on GPU
    ///
    /// Useful for partial offloading when model doesn't fit in VRAM.
    pub gpu_layer_range: Option<(usize, usize)>,

    /// Target dtype for weights.
    ///
    /// - None = use model's native dtype (usually F32 or BF16)
    /// - Some(DType::BF16) = convert to BF16 (halves memory, slight quality loss)
    /// - Some(DType::Q4_0) = 4-bit quantization (~4x memory reduction)
    ///
    /// Supported dtypes for V1: F32, BF16, Q4_0
    pub dtype: Option<DType>,

    /// Pre-allocate buffers for this batch size.
    ///
    /// Setting this can reduce memory fragmentation and allocation overhead
    /// during inference. If None, buffers are allocated on-demand.
    pub max_batch_size: Option<usize>,

    /// Pre-allocate buffers for this sequence length.
    ///
    /// Combined with max_batch_size, this pre-allocates the tensor pool.
    /// If None, buffers are allocated on-demand.
    pub max_sequence_length: Option<usize>,
}

impl Default for ModelLoadConfig {
    fn default() -> Self {
        Self {
            cpu_embeddings: false,
            cpu_output_head: false,
            gpu_layer_range: None,
            dtype: None,
            max_batch_size: None,
            max_sequence_length: None,
        }
    }
}

impl ModelLoadConfig {
    /// Full GPU execution, maximum performance.
    ///
    /// All model components loaded to GPU. Best for systems with sufficient VRAM.
    pub fn full_gpu() -> Self {
        Self::default()
    }

    /// Offload embeddings to CPU to save VRAM.
    ///
    /// Saves ~500MB-2GB depending on model. Minimal performance impact.
    pub fn offload_embeddings() -> Self {
        Self {
            cpu_embeddings: true,
            ..Default::default()
        }
    }

    /// Maximum VRAM saving: embeddings and output head on CPU.
    ///
    /// Saves ~1-4GB depending on model. Some performance impact on short sequences.
    pub fn max_offload() -> Self {
        Self {
            cpu_embeddings: true,
            cpu_output_head: true,
            ..Default::default()
        }
    }

    /// Run with specified quantization dtype.
    ///
    /// # Arguments
    /// * `dtype` - Target dtype (Q4_0, Q8_0, BF16, etc.)
    pub fn quantized(dtype: DType) -> Self {
        Self {
            dtype: Some(dtype),
            ..Default::default()
        }
    }

    /// Run only specified layer range on GPU.
    ///
    /// # Arguments
    /// * `start` - First layer on GPU (inclusive)
    /// * `end` - Last layer on GPU (exclusive)
    ///
    /// # Example
    /// ```rust
    /// // First 6 layers on GPU, rest on CPU
    /// let config = ModelLoadConfig::partial_gpu(0, 6);
    /// ```
    pub fn partial_gpu(start: usize, end: usize) -> Self {
        Self {
            gpu_layer_range: Some((start, end)),
            ..Default::default()
        }
    }

    /// Builder: set CPU embeddings
    pub fn with_cpu_embeddings(mut self, value: bool) -> Self {
        self.cpu_embeddings = value;
        self
    }

    /// Builder: set CPU output head
    pub fn with_cpu_output_head(mut self, value: bool) -> Self {
        self.cpu_output_head = value;
        self
    }

    /// Builder: set GPU layer range
    pub fn with_gpu_layer_range(mut self, start: usize, end: usize) -> Self {
        self.gpu_layer_range = Some((start, end));
        self
    }

    /// Builder: set dtype
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Builder: set max batch size for pre-allocation
    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = Some(size);
        self
    }

    /// Builder: set max sequence length for pre-allocation
    pub fn with_max_sequence_length(mut self, length: usize) -> Self {
        self.max_sequence_length = Some(length);
        self
    }

    /// Check if a layer index should run on GPU given this config.
    pub fn layer_on_gpu(&self, layer_idx: usize, total_layers: usize) -> bool {
        match self.gpu_layer_range {
            None => true, // All layers on GPU
            Some((start, end)) => layer_idx >= start && layer_idx < end,
        }
    }

    /// Get the GPU layer range, defaulting to all layers if not specified.
    pub fn get_gpu_range(&self, total_layers: usize) -> (usize, usize) {
        self.gpu_layer_range.unwrap_or((0, total_layers))
    }
}