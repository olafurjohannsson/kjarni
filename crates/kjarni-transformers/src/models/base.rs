//! Base traits and types for language model inference.

pub use crate::tensor::DType;
use crate::Cache;
use crate::{gpu::GpuTensor, traits::InferenceModel};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ndarray::{Array2, ArrayView2, ArrayView3};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

/// Configuration for Rotary Position Embedding (RoPE) scaling.
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

pub enum PaddingSide {
    Left,
    Right,
}


/// Defines the autoregressive generation loop strategy for decoder models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoregressiveLoop {
    /// uses the prefill output directly
    Pipelined,

    ///  two-pass logic for compatibility with legacy implementations.
    Legacy,
}

/// Flexible input type for model inference supporting multiple format
#[derive(Debug)]
pub enum ModelInput<'a> {
    /// Token IDs stored in GPU memory.
    TokensGpu(&'a GpuTensor),

    /// Token IDs stored in CPU memory.
    TokensCpu(ndarray::ArrayView2<'a, u32>),

    /// Pre-computed hidden states stored in GPU memory.
    HiddenGpu(&'a GpuTensor),

    /// Pre-computed hidden states stored in CPU memory.
    HiddenCpu(ndarray::ArrayView3<'a, f32>),
}
impl<'a> ModelInput<'a> {
    /// Creates a `ModelInput` from a slice of token IDs.
    pub fn from_tokens(tokens: &'a [u32]) -> Self {
        let view = ArrayView2::from_shape((1, tokens.len()), tokens)
            .expect("Failed to create token view from slice");
        ModelInput::TokensCpu(view)
    }

    /// Creates a `ModelInput` from a 2D array of token IDs.
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
#[derive(Debug, Clone, Copy)]
pub struct ModelLoadConfig {
    /// Keep embedding layer on CPU to save VRAM (500MB-2GB).
    pub offload_embeddings: bool,
    /// Keep language model head on CPU to save VRAM.
    pub offload_lm_head: bool,
    /// Force quantization to this data type (overrides file format).
    pub target_dtype: Option<DType>,
    /// Quantize the language model head to this data type.
    pub quantize_lm_head: Option<DType>,
    /// Quantize the embeddings to this data type.
    pub quantize_embeddings: Option<DType>,
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
            target_dtype: None, // Default to "detect from file"
            quantize_lm_head: None,
            quantize_embeddings: None,
            max_batch_size: None,
            max_sequence_length: None,
            use_gguf: false,
        }
    }
}

impl ModelLoadConfig {
    /// Creates a configuration for full GPU execution.
    pub fn full_gpu() -> Self {
        Self::default()
    }

    /// Creates a configuration with embeddings offloaded to CPU.
    pub fn set_offload_embeddings() -> Self {
        Self {
            offload_embeddings: true,
            ..Default::default()
        }
    }

    /// Creates a configuration for quantized model loading
    pub fn quantized(dtype: DType) -> Self {
        Self {
            target_dtype: Some(dtype),
            ..Default::default()
        }
    }

    /// Sets quantization for the language model head
    pub fn with_quantized_lm_head(mut self, dtype: DType) -> Self {
        self.quantize_lm_head = Some(dtype);
        self
    }

    /// Sets whether to offload embeddings to CPU.
    pub fn with_offload_embeddings(mut self, offload: bool) -> Self {
        self.offload_embeddings = offload;
        self
    }

    /// Sets the target quantization data type
    pub fn with_target_dtype(mut self, dtype: DType) -> Self {
        self.target_dtype = Some(dtype);
        self
    }

    /// Sets the maximum batch size for KV cache pre-allocation
    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = Some(size);
        self
    }

    /// Sets the maximum sequence length for KV cache pre-allocation
    pub fn with_max_sequence_length(mut self, len: usize) -> Self {
        self.max_sequence_length = Some(len);
        self
    }
}


/// Core trait for all language models providing tokenization and metadata.
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
    fn new_cache(
        &self,
        batch_size: usize,
        max_len: usize,
        num_beams: usize,
    ) -> Result<Box<dyn Cache>>;

    /// Returns the set of token IDs that should stop generation.
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
    fn max_length(&self) -> usize {
        self.context_size()
    }

    /// Tokenizes a single text string into token IDs.
    fn tokenize(&self, text: &str) -> Result<Array2<u32>> {
        let encoding = self.tokenizer()
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let ids = encoding.get_ids().to_vec();
        let seq_len = ids.len();
        Ok(Array2::from_shape_vec((1, seq_len), ids)?)
    }

    /// Tokenizes a batch of texts with padding to the maximum length.
    fn tokenize_batch(&self, texts: &[&str], side: PaddingSide) -> Result<Array2<u32>> {
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

        let pad_id = self.pad_token_id().unwrap_or(0);
        let batch_size = texts.len();
        let mut batch = Array2::from_elem((batch_size, max_len), pad_id);

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let len = ids.len();

            match side {
                PaddingSide::Right => {
                    // Standard for Encoders: [Token, Token, Pad, Pad]
                    for (j, &token_id) in ids.iter().enumerate() {
                        batch[[i, j]] = token_id;
                    }
                }
                PaddingSide::Left => {
                    // Standard for Decoders: [Pad, Pad, Token, Token]
                    let start_col = max_len - len;
                    for (j, &token_id) in ids.iter().enumerate() {
                        batch[[i, start_col + j]] = token_id;
                    }
                }
            }
        }

        Ok(batch)
    }

    /// Decodes token IDs back to text.
    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.tokenizer()
            .decode(token_ids, true)
            .map_err(|e| anyhow!("Decoding failed: {}", e))
    }

    /// Decodes a batch of token ID sequences.
    fn decode_batch(&self, token_ids: &[Vec<u32>]) -> Result<Vec<String>> {
        token_ids.iter().map(|ids| self.decode(ids)).collect()
    }
}
