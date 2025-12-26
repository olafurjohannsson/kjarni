//! Base traits for language models
//!
//! This module provides high-level, user-facing traits that abstract over
//! the low-level architecture traits in `traits.rs`.

pub use crate::tensor::DType;
use crate::{gpu_ops::GpuTensor, traits::InferenceModel};
use crate::Cache;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ndarray::{Array2, ArrayView2, ArrayView3};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScalingConfig {
    pub factor: f32,
    pub high_freq_factor: f32,
    pub low_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoregressiveLoop {
    /// Efficient, pipelined logic. Uses the prefill output directly.
    /// Correct for Llama and the "theoretically correct" approach.
    Pipelined,
    /// Inefficient, two-pass logic. Discards prefill output and re-processes
    /// the last prompt token to get the first logits.
    /// Required for parity with Hugging Face's GPT-2 implementation.
    Legacy,
}

#[derive(Debug)]
pub enum ModelInput<'a> {
    /// Token IDs on GPU. Shape: [batch, seq]
    TokensGpu(&'a GpuTensor),

    /// Token IDs on CPU. Shape: [batch, seq]
    /// We use ArrayView2 to handle both flat slices and 2D arrays.
    TokensCpu(ndarray::ArrayView2<'a, u32>),

    /// Pre-computed hidden states on GPU. Shape: [batch, seq, hidden]
    HiddenGpu(&'a GpuTensor),

    /// Pre-computed hidden states on CPU. Shape: [batch, seq, hidden]
    HiddenCpu(ndarray::ArrayView3<'a, f32>),
}
impl<'a> ModelInput<'a> {
    /// Creates a `ModelInput` from a slice of token IDs.
    ///
    /// Assumes batch size of 1. This is the most common case for inference.
    ///
    /// # Example
    ///
    /// ```rust
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
    /// ```rust
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
#[derive(Debug, Clone, Copy)]
pub struct ModelLoadConfig {
    pub offload_embeddings: bool,
    pub offload_lm_head: bool,
    pub gpu_layers: Option<usize>,
    pub target_dtype: Option<DType>, // User override
    pub gpu_layer_range: Option<(usize, usize)>,
    pub max_batch_size: Option<usize>,
    pub max_sequence_length: Option<usize>,
}

impl Default for ModelLoadConfig {
    fn default() -> Self {
        Self {
            offload_embeddings: false,
            offload_lm_head: false,
            gpu_layers: None,
            target_dtype: None, // Default to "detect from file"
            gpu_layer_range: None,
            max_batch_size: None,
            max_sequence_length: None,
        }
    }
}

impl ModelLoadConfig {
    /// Full GPU execution (default, maximum performance)
    pub fn full_gpu() -> Self {
        Self::default()
    }

    /// CPU embeddings, GPU layers (saves ~500MB-2GB VRAM)
    pub fn set_offload_embeddings() -> Self {
        Self {
            offload_embeddings: true,
            ..Default::default()
        }
    }

    /// Quantized model loading
    pub fn quantized(dtype: DType) -> Self {
        Self {
            target_dtype: Some(dtype),
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
    pub fn with_offload_embeddings(mut self, offload: bool) -> Self {
        self.offload_embeddings = offload;
        self
    }

    pub fn with_gpu_layer_range(mut self, start: usize, end: usize) -> Self {
        self.gpu_layer_range = Some((start, end));
        self
    }

    pub fn with_target_dtype(mut self, dtype: DType) -> Self {
        self.target_dtype = Some(dtype);
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


/// Base trait for all language models - provides tokenization
///
/// This is implemented by encoder-only (BERT), decoder-only (GPT),
/// and encoder-decoder (BART) models.
#[async_trait]
pub trait LanguageModel: InferenceModel {
    fn vocab_size(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn num_heads(&self) -> usize;
    fn context_size(&self) -> usize;
    fn tokenizer(&self) -> &Tokenizer;

    fn eos_token_id(&self) -> Option<u32>;
    fn bos_token_id(&self) -> Option<u32>;
    fn forced_bos_token_id(&self) -> Option<u32>;
    fn forced_eos_token_id(&self) -> Option<u32>;
    fn pad_token_id(&self) -> Option<u32>;

    fn new_cache(
        &self,
        batch_size: usize,
        max_len: usize,
        num_beams: usize,
    ) -> Result<Box<dyn Cache>>;

    // --- Logic Methods (Default implementations are now safe) ---

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

    fn max_length(&self) -> usize {
        self.context_size()
    }

    fn tokenize(&self, text: &str) -> Result<Array2<u32>> {
        let encoding = self.tokenizer()
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let ids = encoding.get_ids().to_vec();
        let seq_len = ids.len();
        Ok(Array2::from_shape_vec((1, seq_len), ids)?)
    }

    /// Tokenize batch of texts (with padding to max length)
    ///
    /// Returns Array2<f32> with shape [batch_size, max_seq_len]
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

    /// Decode token IDs back to text
    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.tokenizer()
            .decode(token_ids, true)
            .map_err(|e| anyhow!("Decoding failed: {}", e))
    }

    /// Decode batch of token IDs
    fn decode_batch(&self, token_ids: &[Vec<u32>]) -> Result<Vec<String>> {
        token_ids.iter().map(|ids| self.decode(ids)).collect()
    }
}
