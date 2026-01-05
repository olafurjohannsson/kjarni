//! Base traits and types for language model inference.
//!
//! This module provides high-level, user-facing traits and configuration types
//! that abstract over the low-level architecture-specific implementations. It
//! defines the [`LanguageModel`] trait for tokenization and text generation,
//! along with supporting types for model configuration and input handling.
//!
//! # Overview
//!
//! The main components are:
//!
//! - [`LanguageModel`] — Core trait implemented by all language models
//! - [`ModelInput`] — Flexible input enum supporting CPU/GPU tensors or tokens
//! - [`ModelLoadConfig`] — Configuration for model loading and device placement
//! - [`RopeScalingConfig`] — RoPE (Rotary Position Embedding) scaling parameters
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::models::base::{LanguageModel, ModelInput};
//! use ndarray::Array2;
//!
//! // Tokenize text
//! let tokens = model.tokenize("Hello, world!")?;
//!
//! // Create input from tokens
//! let input = ModelInput::from_array(tokens.view());
//!
//! // Decode output tokens back to text
//! let output_text = model.decode(&[123, 456, 789])?;
//! ```
//!
//! # See Also
//!
//! - [`crate::traits::InferenceModel`] — Low-level inference trait
//! - [`crate::models::registry`] — Pretrained model metadata

pub use crate::tensor::DType;
use crate::Cache;
use crate::{gpu_ops::GpuTensor, traits::InferenceModel};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ndarray::{Array2, ArrayView2, ArrayView3};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

/// Configuration for Rotary Position Embedding (RoPE) scaling.
///
/// RoPE scaling allows models to handle sequence lengths longer than their
/// original training context by adjusting the frequency bands used in position
/// embeddings. This is particularly important for long-context models like
/// Llama 3.1 and Phi-3.
///
/// # Fields
///
/// * `factor` — Global scaling factor applied to all frequencies.
/// * `high_freq_factor` — Scaling factor for high-frequency components.
/// * `low_freq_factor` — Scaling factor for low-frequency components.
/// * `original_max_position_embeddings` — Maximum sequence length from base model training.
/// * `rope_type` — Scaling strategy (e.g., "linear", "dynamic", "yarn").
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::models::base::RopeScalingConfig;
///
/// let config = RopeScalingConfig {
///     factor: 8.0,
///     high_freq_factor: 4.0,
///     low_freq_factor: 1.0,
///     original_max_position_embeddings: 8192,
///     rope_type: "llama3".to_string(),
/// };
/// ```
///
/// # See Also
///
/// - [RoPE paper](https://arxiv.org/abs/2104.09864) — Original Rotary Position Embedding
/// - [YaRN](https://arxiv.org/abs/2309.00071) — Yet another RoPE extensioN method
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RopeScalingConfig {
    /// Global scaling factor applied to all frequencies.
    pub factor: f32,
    /// Scaling factor for high-frequency components.
    pub high_freq_factor: f32,
    /// Scaling factor for low-frequency components.
    pub low_freq_factor: f32,
    /// Maximum sequence length from base model training.
    pub original_max_position_embeddings: usize,
    /// Scaling strategy (e.g., "linear", "dynamic", "yarn").
    pub rope_type: String,
}

/// Defines the autoregressive generation loop strategy for decoder models.
///
/// Different model implementations use different strategies for transitioning
/// from the prefill phase (processing the prompt) to the decode phase (generating
/// tokens one at a time). This enum allows matching the exact behavior of
/// reference implementations.
///
/// # Variants
///
/// * [`Pipelined`](AutoregressiveLoop::Pipelined) — Efficient approach that uses prefill output directly.
/// * [`Legacy`](AutoregressiveLoop::Legacy) — Compatibility mode for older implementations.
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::models::base::AutoregressiveLoop;
///
/// // Most modern models use pipelined generation
/// let loop_type = AutoregressiveLoop::Pipelined;
///
/// // GPT-2 requires legacy mode for Hugging Face parity
/// let gpt2_loop = AutoregressiveLoop::Legacy;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoregressiveLoop {
    /// Efficient, pipelined logic that uses the prefill output directly.
    ///
    /// This is the theoretically correct approach used by modern models like
    /// Llama. The final hidden state from the prefill pass is immediately used
    /// to generate the first new token without recomputation.
    Pipelined,

    /// Inefficient, two-pass logic for compatibility with legacy implementations.
    ///
    /// Discards the prefill output and re-processes the last prompt token to
    /// get the first logits. Required for exact parity with Hugging Face's
    /// GPT-2 implementation.
    Legacy,
}

/// Flexible input type for model inference supporting multiple formats and devices.
///
/// `ModelInput` allows passing either token IDs or pre-computed hidden states,
/// on either CPU or GPU. This flexibility enables efficient multi-stage pipelines
/// where embeddings are computed separately or cached across requests.
///
/// # Variants
///
/// * [`TokensGpu`](ModelInput::TokensGpu) — Token IDs stored in GPU memory.
/// * [`TokensCpu`](ModelInput::TokensCpu) — Token IDs stored in CPU memory.
/// * [`HiddenGpu`](ModelInput::HiddenGpu) — Pre-computed embeddings on GPU.
/// * [`HiddenCpu`](ModelInput::HiddenCpu) — Pre-computed embeddings on CPU.
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::models::base::ModelInput;
///
/// // From a slice of tokens (single sequence)
/// let tokens = vec![1, 2, 3, 4, 5];
/// let input = ModelInput::from_tokens(&tokens);
///
/// // From a 2D array (batched)
/// let batch = Array2::from_shape_vec((2, 5), tokens)?;
/// let input = ModelInput::from_array(batch.view());
///
/// // From GPU tensor
/// let gpu_input = ModelInput::from_gpu_tokens(&gpu_tensor);
/// ```
///
/// # See Also
///
/// - [`GpuTensor`] — GPU tensor storage
#[derive(Debug)]
pub enum ModelInput<'a> {
    /// Token IDs stored in GPU memory.
    ///
    /// Shape: `[batch, seq]` where `batch` is the number of sequences and
    /// `seq` is the sequence length. All sequences must have the same length.
    TokensGpu(&'a GpuTensor),

    /// Token IDs stored in CPU memory.
    ///
    /// Shape: `[batch, seq]`. Uses `ArrayView2` to handle both flat slices
    /// and 2D arrays efficiently without copying data.
    TokensCpu(ndarray::ArrayView2<'a, u32>),

    /// Pre-computed hidden states stored in GPU memory.
    ///
    /// Shape: `[batch, seq, hidden]` where `hidden` is the model's hidden size.
    /// Use this variant to skip the embedding layer when passing cached embeddings.
    HiddenGpu(&'a GpuTensor),

    /// Pre-computed hidden states stored in CPU memory.
    ///
    /// Shape: `[batch, seq, hidden]`. Useful for embedding models where you
    /// want to separate embedding computation from the main model.
    HiddenCpu(ndarray::ArrayView3<'a, f32>),
}
impl<'a> ModelInput<'a> {
    /// Creates a `ModelInput` from a slice of token IDs.
    ///
    /// Assumes batch size of 1. This is the most common case for inference.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tokens = vec![1, 2, 3, 4, 5];
    /// let input = ModelInput::from_tokens(&tokens);
    /// ```
    pub fn from_tokens(tokens: &'a [u32]) -> Self {
        let view = ArrayView2::from_shape((1, tokens.len()), tokens)
            .expect("Failed to create token view from slice");
        ModelInput::TokensCpu(view)
    }

    /// Creates a `ModelInput` from a 2D array of token IDs.
    ///
    /// Use this for batched inputs.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let batch = Array2::from_shape_vec((2, 5), tokens)?;
    /// let input = ModelInput::from_array(batch.view());
    /// ```
    pub fn from_array(array: ArrayView2<'a, u32>) -> Self {
        ModelInput::TokensCpu(array)
    }

    /// Creates a `ModelInput` from a GPU tensor of token IDs.
    pub fn from_gpu_tokens(tensor: &'a GpuTensor) -> Self {
        ModelInput::TokensGpu(tensor)
    }

    /// Creates a `ModelInput` from pre-computed CPU hidden states.
    pub fn from_hidden(hidden: ArrayView3<'a, f32>) -> Self {
        ModelInput::HiddenCpu(hidden)
    }

    /// Creates a `ModelInput` from pre-computed GPU hidden states.
    pub fn from_gpu_hidden(tensor: &'a GpuTensor) -> Self {
        ModelInput::HiddenGpu(tensor)
    }

    /// Returns the batch size.
    pub fn batch_size(&self) -> usize {
        match self {
            ModelInput::TokensGpu(t) => t.shape()[0],
            ModelInput::TokensCpu(a) => a.shape()[0],
            ModelInput::HiddenGpu(t) => t.shape()[0],
            ModelInput::HiddenCpu(a) => a.shape()[0],
        }
    }

    /// Returns the sequence length.
    pub fn seq_len(&self) -> usize {
        match self {
            ModelInput::TokensGpu(t) => t.shape()[1],
            ModelInput::TokensCpu(a) => a.shape()[1],
            ModelInput::HiddenGpu(t) => t.shape()[1],
            ModelInput::HiddenCpu(a) => a.shape()[1],
        }
    }

    /// Returns true if this is a token input (vs hidden states).
    pub fn is_tokens(&self) -> bool {
        matches!(self, ModelInput::TokensGpu(_) | ModelInput::TokensCpu(_))
    }

    /// Returns true if this input is on the GPU.
    pub fn is_gpu(&self) -> bool {
        matches!(self, ModelInput::TokensGpu(_) | ModelInput::HiddenGpu(_))
    }
}
/// Configuration for model loading and device placement.
///
/// `ModelLoadConfig` controls how a model is loaded into memory, including
/// which layers run on GPU vs CPU, quantization settings, and memory limits.
/// This allows fine-tuning the memory/performance trade-off based on available
/// hardware and use case requirements.
///
/// # Fields
///
/// * `offload_embeddings` — Keep embedding layer on CPU to save VRAM (500MB-2GB).
/// * `offload_lm_head` — Keep language model head on CPU to save VRAM.
/// * `gpu_layers` — Number of decoder layers to place on GPU (None = all).
/// * `target_dtype` — Force quantization to this data type (overrides file format).
/// * `quantize_lm_head` — Quantize the language model head to this data type.
/// * `gpu_layer_range` — Only place layers `[start, end)` on GPU.
/// * `max_batch_size` — Pre-allocate KV cache for this batch size.
/// * `max_sequence_length` — Pre-allocate KV cache for this sequence length.
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::models::base::ModelLoadConfig;
/// use kjarni_transformers::tensor::DType;
///
/// // Full GPU execution (default)
/// let config = ModelLoadConfig::full_gpu();
///
/// // Save VRAM by keeping embeddings on CPU
/// let config = ModelLoadConfig::default()
///     .with_offload_embeddings(true);
///
/// // Quantized model with 4-bit weights
/// let config = ModelLoadConfig::quantized(DType::Q4_K);
///
/// // Partial GPU: only middle layers on GPU
/// let config = ModelLoadConfig::partial_gpu(4, 20);
/// ```
///
/// # See Also
///
/// - [`DType`] — Supported quantization data types
#[derive(Debug, Clone, Copy)]
pub struct ModelLoadConfig {
    /// Keep embedding layer on CPU to save VRAM (500MB-2GB).
    pub offload_embeddings: bool,
    /// Keep language model head on CPU to save VRAM.
    pub offload_lm_head: bool,
    /// Number of decoder layers to place on GPU (None = all layers).
    pub gpu_layers: Option<usize>,
    /// Force quantization to this data type (overrides file format).
    pub target_dtype: Option<DType>,
    /// Quantize the language model head to this data type.
    pub quantize_lm_head: Option<DType>,
    /// Only place layers `[start, end)` on GPU.
    pub gpu_layer_range: Option<(usize, usize)>,
    /// Pre-allocate KV cache for this batch size.
    pub max_batch_size: Option<usize>,
    /// Pre-allocate KV cache for this sequence length.
    pub max_sequence_length: Option<usize>,
    /// Use gguf
    pub use_gguf: bool,
}

impl Default for ModelLoadConfig {
    fn default() -> Self {
        Self {
            offload_embeddings: false,
            offload_lm_head: false,
            gpu_layers: None,
            target_dtype: None, // Default to "detect from file"
            quantize_lm_head: None,
            gpu_layer_range: None,
            max_batch_size: None,
            max_sequence_length: None,
            use_gguf: false,
        }
    }
}

impl ModelLoadConfig {
    /// Creates a configuration for full GPU execution.
    ///
    /// All model layers run on GPU for maximum performance. This is the default
    /// configuration and provides the fastest inference speed at the cost of
    /// maximum VRAM usage.
    ///
    /// # Returns
    ///
    /// A configuration with all layers on GPU.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = ModelLoadConfig::full_gpu();
    /// let model = LlamaModel::load(weights, config)?;
    /// ```
    pub fn full_gpu() -> Self {
        Self::default()
    }

    /// Creates a configuration with embeddings offloaded to CPU.
    ///
    /// Keeps the embedding layer on CPU while running transformer layers on GPU.
    /// This saves approximately 500MB-2GB of VRAM depending on vocabulary size,
    /// with minimal impact on inference speed.
    ///
    /// # Returns
    ///
    /// A configuration with CPU embeddings and GPU layers.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = ModelLoadConfig::set_offload_embeddings();
    /// let model = LlamaModel::load(weights, config)?;
    /// ```
    pub fn set_offload_embeddings() -> Self {
        Self {
            offload_embeddings: true,
            ..Default::default()
        }
    }

    /// Creates a configuration for quantized model loading.
    ///
    /// Forces all weights to be loaded or converted to the specified data type,
    /// regardless of the format in the model file. Use this to reduce memory
    /// usage at the cost of some precision.
    ///
    /// # Arguments
    ///
    /// * `dtype` - Target quantization data type (e.g., Q4_K, Q8_0, BF16).
    ///
    /// # Returns
    ///
    /// A configuration that quantizes weights to the specified dtype.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use kjarni_transformers::tensor::DType;
    ///
    /// // Load model with 4-bit quantization
    /// let config = ModelLoadConfig::quantized(DType::Q4_K);
    /// let model = LlamaModel::load(weights, config)?;
    /// ```
    pub fn quantized(dtype: DType) -> Self {
        Self {
            target_dtype: Some(dtype),
            ..Default::default()
        }
    }

    /// Creates a configuration for partial GPU execution.
    ///
    /// Only places layers in the range `[start, end)` on GPU, keeping all other
    /// layers on CPU. Useful for large models that don't fit entirely in VRAM,
    /// or for balancing computation across devices.
    ///
    /// # Arguments
    ///
    /// * `start` - Index of first layer to place on GPU (inclusive).
    /// * `end` - Index of last layer to place on GPU (exclusive).
    ///
    /// # Returns
    ///
    /// A configuration with the specified layer range on GPU.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Place layers 4-20 on GPU, rest on CPU
    /// let config = ModelLoadConfig::partial_gpu(4, 20);
    /// let model = LlamaModel::load(weights, config)?;
    /// ```
    pub fn partial_gpu(start: usize, end: usize) -> Self {
        Self {
            gpu_layer_range: Some((start, end)),
            ..Default::default()
        }
    }

    /// Sets quantization for the language model head.
    ///
    /// The language model head (vocabulary projection) is often the largest single
    /// weight matrix in the model. Quantizing it can save significant VRAM.
    ///
    /// # Arguments
    ///
    /// * `dtype` - Quantization data type for the LM head.
    ///
    /// # Returns
    ///
    /// The modified configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = ModelLoadConfig::full_gpu()
    ///     .with_quantized_lm_head(DType::Q8_0);
    /// ```
    pub fn with_quantized_lm_head(mut self, dtype: DType) -> Self {
        self.quantize_lm_head = Some(dtype);
        self
    }

    /// Sets whether to offload embeddings to CPU.
    ///
    /// # Arguments
    ///
    /// * `offload` - If `true`, keep embeddings on CPU. If `false`, keep on GPU.
    ///
    /// # Returns
    ///
    /// The modified configuration.
    pub fn with_offload_embeddings(mut self, offload: bool) -> Self {
        self.offload_embeddings = offload;
        self
    }

    /// Sets the range of layers to place on GPU.
    ///
    /// # Arguments
    ///
    /// * `start` - Index of first layer to place on GPU (inclusive).
    /// * `end` - Index of last layer to place on GPU (exclusive).
    ///
    /// # Returns
    ///
    /// The modified configuration.
    pub fn with_gpu_layer_range(mut self, start: usize, end: usize) -> Self {
        self.gpu_layer_range = Some((start, end));
        self
    }

    /// Sets the target quantization data type.
    ///
    /// # Arguments
    ///
    /// * `dtype` - Target quantization data type.
    ///
    /// # Returns
    ///
    /// The modified configuration.
    pub fn with_target_dtype(mut self, dtype: DType) -> Self {
        self.target_dtype = Some(dtype);
        self
    }

    /// Sets the maximum batch size for KV cache pre-allocation.
    ///
    /// # Arguments
    ///
    /// * `size` - Maximum batch size.
    ///
    /// # Returns
    ///
    /// The modified configuration.
    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = Some(size);
        self
    }

    /// Sets the maximum sequence length for KV cache pre-allocation.
    ///
    /// # Arguments
    ///
    /// * `len` - Maximum sequence length.
    ///
    /// # Returns
    ///
    /// The modified configuration.
    pub fn with_max_sequence_length(mut self, len: usize) -> Self {
        self.max_sequence_length = Some(len);
        self
    }
}


/// Core trait for all language models providing tokenization and metadata.
///
/// `LanguageModel` is the high-level user-facing trait implemented by all model
/// architectures in Kjarni. It provides tokenization, model configuration access,
/// and cache management. Models implementing this trait can be used interchangeably
/// for inference and text generation.
///
/// This trait is implemented by:
/// - Encoder-only models (BERT, RoBERTa)
/// - Decoder-only models (Llama, GPT, Mistral)
/// - Encoder-decoder models (T5, BART)
///
/// # Required Methods
///
/// Implementors must provide model metadata and tokenization functionality.
/// Most methods have default implementations that cover common patterns.
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::models::base::LanguageModel;
///
/// fn print_model_info(model: &impl LanguageModel) {
///     println!("Vocabulary size: {}", model.vocab_size());
///     println!("Hidden size: {}", model.hidden_size());
///     println!("Context size: {}", model.context_size());
/// }
///
/// // Tokenize text
/// let tokens = model.tokenize("Hello, world!")?;
///
/// // Decode tokens back to text
/// let text = model.decode(&[1, 2, 3, 4, 5])?;
/// ```
///
/// # See Also
///
/// - [`InferenceModel`] — Low-level trait for forward pass implementation
/// - [`Cache`] — KV cache for autoregressive generation
#[async_trait]
pub trait LanguageModel: InferenceModel {
    /// Returns the vocabulary size of the model.
    fn vocab_size(&self) -> usize;

    /// Returns the hidden dimension size of the model.
    fn hidden_size(&self) -> usize;

    /// Returns the number of transformer layers in the model.
    fn num_layers(&self) -> usize;

    /// Returns the number of attention heads per layer.
    fn num_heads(&self) -> usize;

    /// Returns the maximum context size (sequence length) supported by the model.
    fn context_size(&self) -> usize;

    /// Returns a reference to the tokenizer used by this model.
    fn tokenizer(&self) -> &Tokenizer;

    /// Returns the end-of-sequence token ID if defined.
    fn eos_token_id(&self) -> Option<u32>;

    /// Returns multiple end-of-sequence token IDs if the model supports them.
    ///
    /// Some models define multiple EOS tokens (e.g., Llama 3 with `<|eot_id|>`).
    /// Default implementation returns `None`.
    fn eos_token_ids(&self) -> Option<Vec<u32>> {
        None
    }

    /// Returns the beginning-of-sequence token ID if defined.
    fn bos_token_id(&self) -> Option<u32>;

    /// Returns the forced beginning-of-sequence token ID for seq2seq models.
    ///
    /// Used by encoder-decoder models to start generation with a specific token.
    fn forced_bos_token_id(&self) -> Option<u32>;

    /// Returns the forced end-of-sequence token ID for seq2seq models.
    fn forced_eos_token_id(&self) -> Option<u32>;

    /// Returns the padding token ID used for batched inference.
    fn pad_token_id(&self) -> Option<u32>;

    /// Creates a new KV cache for autoregressive generation.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of sequences to cache simultaneously.
    /// * `max_len` - Maximum sequence length to allocate cache for.
    /// * `num_beams` - Number of beams for beam search (1 for greedy decoding).
    ///
    /// # Returns
    ///
    /// A boxed cache implementation appropriate for this model architecture.
    ///
    /// # Errors
    ///
    /// Returns an error if cache allocation fails (e.g., out of memory).
    fn new_cache(
        &self,
        batch_size: usize,
        max_len: usize,
        num_beams: usize,
    ) -> Result<Box<dyn Cache>>;

    // --- Default Implementations ---

    /// Returns the set of token IDs that should stop generation.
    ///
    /// By default, includes the model's EOS token and Llama 3's `<|eot_id|>` if present.
    /// Override this method to customize generation stopping behavior.
    ///
    /// # Returns
    ///
    /// A set of token IDs that indicate generation should stop.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let stop_ids = model.stop_token_ids();
    /// for token in generated_tokens {
    ///     if stop_ids.contains(&token) {
    ///         break;
    ///     }
    /// }
    /// ```
    fn stop_token_ids(&self) -> std::collections::HashSet<u32> {
        let mut set = std::collections::HashSet::new();
        if let Some(id) = self.eos_token_id() {
            set.insert(id);
        }
        // Llama 3 specific EOT ID
        if let Some(eot) = self.tokenizer().token_to_id("<|eot_id|>") {
            set.insert(eot);
        }
        set
    }

    /// Returns the maximum generation length.
    ///
    /// By default, returns the model's context size. Override to enforce
    /// different limits for generation.
    ///
    /// # Returns
    ///
    /// Maximum number of tokens that can be generated.
    fn max_length(&self) -> usize {
        self.context_size()
    }

    /// Tokenizes a single text string into token IDs.
    ///
    /// Converts the input text into a 2D array of token IDs with shape `[1, seq_len]`.
    /// This is the most common case for single-sequence inference.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to tokenize.
    ///
    /// # Returns
    ///
    /// A 2D array of token IDs with shape `[1, seq_len]`.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tokens = model.tokenize("Hello, world!")?;
    /// assert_eq!(tokens.shape()[0], 1); // batch size
    /// ```
    fn tokenize(&self, text: &str) -> Result<Array2<u32>> {
        let encoding = self.tokenizer()
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let ids = encoding.get_ids().to_vec();
        let seq_len = ids.len();
        Ok(Array2::from_shape_vec((1, seq_len), ids)?)
    }

    /// Tokenizes a batch of texts with padding to the maximum length.
    ///
    /// Each text is tokenized independently, then all sequences are padded
    /// to match the length of the longest sequence in the batch.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of text strings to tokenize.
    ///
    /// # Returns
    ///
    /// A 2D array of token IDs with shape `[batch_size, max_seq_len]` where
    /// `max_seq_len` is the length of the longest tokenized sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The input batch is empty.
    /// - Tokenization fails for any text.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let texts = vec!["Hello", "Hello, world!"];
    /// let tokens = model.tokenize_batch(&texts)?;
    /// assert_eq!(tokens.shape()[0], 2); // batch size
    /// ```
    fn tokenize_batch(&self, texts: &[&str]) -> Result<Array2<u32>> {
        if texts.is_empty() {
            return Err(anyhow!("Cannot tokenize empty batch"));
        }

        let mut encodings = Vec::new();
        let mut max_len = 0;

        for text in texts {
            let encoding = self
                .tokenizer()
                .encode(*text, true)
                .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

            max_len = max_len.max(encoding.len());
            encodings.push(encoding);
        }
        // Rust is a multi-paradigm programming language that emphasizes performance, type safety, and concurrency . It enforces memory safety without using a garbage collector . To simultaneously enforce memory safety and prevent data races, its 'borrow checker' tracks the object lifetime of all references in a program during compilation .
        // Rust is a multi-paradigm programming language that emphasizes performance, type safety, and concurrency . It enforces memory safety—meaning that all references point to valid memory—without using a garbage collector . To simultaneously enforce memory safety and prevent data races, its 'borrow checker' tracks the object lifetime
        let pad_id = self.pad_token_id().unwrap_or(0);
        let batch_size = texts.len();

        let mut batch = Array2::from_elem((batch_size, max_len), pad_id);

        for (i, encoding) in encodings.iter().enumerate() {
            for (j, &token_id) in encoding.get_ids().iter().enumerate() {
                batch[[i, j]] = token_id;
            }
        }

        Ok(batch)
    }

    /// Decodes token IDs back to text.
    ///
    /// Converts a sequence of token IDs into a human-readable string using
    /// the model's tokenizer. Special tokens are typically removed during decoding.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Slice of token IDs to decode.
    ///
    /// # Returns
    ///
    /// The decoded text string.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails (e.g., invalid token IDs).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let generated = vec![128000, 9906, 11, 1917, 0]; // "Hello, world!"
    /// let text = model.decode(&generated)?;
    /// println!("{}", text);
    /// ```
    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.tokenizer()
            .decode(token_ids, true)
            .map_err(|e| anyhow!("Decoding failed: {}", e))
    }

    /// Decodes a batch of token ID sequences.
    ///
    /// Converts multiple sequences of token IDs into text strings. Each sequence
    /// is decoded independently.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Slice of token ID vectors to decode.
    ///
    /// # Returns
    ///
    /// A vector of decoded text strings.
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails for any sequence.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let batch = vec![
    ///     vec![128000, 9906],      // "Hello"
    ///     vec![128000, 9906, 11, 1917, 0],  // "Hello, world!"
    /// ];
    /// let texts = model.decode_batch(&batch)?;
    /// assert_eq!(texts.len(), 2);
    /// ```
    fn decode_batch(&self, token_ids: &[Vec<u32>]) -> Result<Vec<String>> {
        token_ids.iter().map(|ids| self.decode(ids)).collect()
    }
}
