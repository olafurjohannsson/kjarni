//! Base traits for language models
//!
//! This module provides high-level, user-facing traits that abstract over
//! the low-level architecture traits in `traits.rs`.

use crate::gpu_ops::GpuTensor;
use crate::gpu_ops::GpuTensorPool;
use crate::pooling::mean_pool;
use crate::traits::{
    Decoder, DecoderOutput, Encoder, EncoderOutput, LanguageModelConfig, TransformerModel,
};
use crate::utils::create_full_attention_mask;
pub use crate::weights::DType;
use crate::Cache;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};
use std::fmt;
use tokenizers::Tokenizer;

#[derive(Clone, Debug)]
pub struct EncodingConfig {
    pub pooling_strategy: PoolingStrategy,
    pub normalize: bool,
}
impl fmt::Display for EncodingConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EncodingConfig {{ pooling_strategy: {}, normalize: {} }}",
            self.pooling_strategy, self.normalize
        )
    }
}
/// Pooling strategies for sequence outputs
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PoolingStrategy {
    Mean,
    Max,
    Cls,
    LastToken,
}
impl fmt::Display for PoolingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PoolingStrategy::Mean => "Mean",
            PoolingStrategy::Max => "Max",
            PoolingStrategy::Cls => "CLS",
            PoolingStrategy::LastToken => "LastToken",
        };
        write!(f, "{}", s)
    }
}

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

// impl Default for GenerationConfig {
//     fn default() -> Self {
//         Self {
//             // General sampling params
//             max_new_tokens: None,
//             temperature: 0.7,
//             top_k: Some(50),
//             top_p: Some(0.9),
//             repetition_penalty: 1.1,
//             sampling_strategy: SamplingStrategy::TopKTopP,
//             eos_token_id: Some(50256), // default GPT-2 EOS
//             pad_token_id: Some(50256),
//             // Beam search specific params (from BART config)
//             num_beams: 4,
//             min_length: 56,
//             max_length: 142,
//             no_repeat_ngram_size: 3,
//             length_penalty: 2.0,
//             early_stopping: true,
//             add_bos_token: true,
//         }
//     }
// }
/// Parameters for sampling-based decoding (Top-K, Top-P, Temperature).
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

/// Parameters for beam search decoding.
#[derive(Clone, Debug)]
pub struct BeamSearchParams {
    pub num_beams: usize,
    pub length_penalty: f32,
    pub early_stopping: bool,
}

/// The user-facing decoding algorithm and its specific parameters.
#[derive(Clone, Debug)]
pub enum DecodingStrategy {
    /// Select the most likely token (argmax).
    Greedy,
    /// Sample from the distribution using various techniques.
    Sample(SamplingParams),
    /// Explore multiple hypotheses to find the most likely sequence.
    BeamSearch(BeamSearchParams),
}

/// The main, unified configuration struct for text generation.
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    // --- Common Parameters for all strategies ---
    pub max_new_tokens: Option<usize>,
    pub max_length: usize,
    pub min_length: usize,
    pub repetition_penalty: f32,
    pub no_repeat_ngram_size: usize,
    pub add_bos_token: bool,

    // --- The specific decoding strategy and its parameters ---
    pub strategy: DecodingStrategy,
}
/// A sensible default for decoder-only models (like GPT-2 or Llama).
impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: Some(50),
            max_length: 100,
            min_length: 0,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: 0,
            add_bos_token: true,
            strategy: DecodingStrategy::Sample(SamplingParams {
                temperature: 0.7,
                top_k: Some(50),
                top_p: Some(0.9),
            }),
        }
    }
}

// A struct to hold all the inputs for a single generation step.
// This keeps the `forward` signature clean.
pub struct StepInput<'a, T> {
    pub tokens: &'a T,
    pub encoder_state: Option<&'a T>, // Optional for decoder-only models
    pub attention_mask: &'a T,
}

/// Base trait for all language models - provides tokenization
///
/// This is implemented by encoder-only (BERT), decoder-only (GPT),
/// and encoder-decoder (BART) models.
#[async_trait]
pub trait LanguageModel: TransformerModel {
    /// Get the model's configuration
    ///
    /// This provides access to all architectural parameters like
    /// vocab_size, max_position_embeddings, hidden_size, etc.
    fn config(&self) -> &dyn LanguageModelConfig;

    /// Creates a new, empty Key-Value cache appropriately configured for this model.
    ///
    /// The model implementation is responsible for choosing the correct cache type
    /// (CPU/GPU) and initializing it with the correct dimensions.
    fn new_cache(
        &self,
        batch_size: usize,
        max_len: usize,
        num_beams: usize,
    ) -> Result<Box<dyn Cache>>;

    /// Maximum sequence length the model can handle
    fn max_length(&self) -> usize {
        self.config().max_position_embeddings()
    }

    /// Size of the vocabulary
    fn vocab_size(&self) -> usize {
        self.config().vocab_size()
    }

    /// Hidden state dimensionality
    fn hidden_size(&self) -> usize {
        self.config().hidden_size()
    }

    /// Number of transformer layers
    fn num_layers(&self) -> usize {
        self.config().num_hidden_layers()
    }

    /// Number of attention heads
    fn num_heads(&self) -> usize {
        self.config().num_attention_heads()
    }

    /// Get the tokenizer
    fn tokenizer(&self) -> &Tokenizer;

    /// Get the end-of-sequence token ID (if applicable)
    fn eos_token_id(&self) -> Option<u32> {
        self.config()
            .eos_token_id()
            .or_else(|| self.tokenizer().token_to_id("</s>"))
            .or_else(|| self.tokenizer().token_to_id("<|endoftext|>"))
    }

    /// Get the padding token ID (if applicable)
    fn pad_token_id(&self) -> Option<u32> {
        self.config()
            .pad_token_id()
            .or_else(|| self.tokenizer().token_to_id("<pad>"))
            .or_else(|| self.tokenizer().token_to_id("[PAD]"))
    }

    /// Get the beginning-of-sequence token ID (if applicable)
    fn bos_token_id(&self) -> Option<u32> {
        self.config()
            .bos_token_id()
            .or_else(|| self.tokenizer().token_to_id("<s>"))
            .or_else(|| self.tokenizer().token_to_id("[CLS]"))
    }

    /// Tokenize text into token IDs
    ///
    /// Returns Array2<f32> with shape [1, seq_len]
    fn tokenize(&self, text: &str) -> Result<Array2<u32>> {
        let encoding = self
            .tokenizer()
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

/// Trait for encoder-only language models (BERT, RoBERTa, etc.)
///
/// These models encode text into fixed-size embeddings.
#[async_trait]
pub trait EncoderLanguageModel: LanguageModel {
    /// Get the encoder backend
    fn encoder(&self) -> &dyn Encoder<Input=Array2<u32>, Output=EncoderOutput>;

    /// Get hidden states for input text
    async fn get_hidden_states(&self, text: &str) -> Result<Array3<f32>> {
        let input_ids = self.tokenize(text)?;
        let seq_len = input_ids.ncols();
        let attention_mask = create_full_attention_mask(1, seq_len);

        self.encoder()
            .get_hidden_states(&input_ids, &attention_mask, None) // TODO: token_type_ids ???
            .await
    }

    /// Get hidden states for a batch of texts
    async fn get_hidden_states_batch(&self, texts: &[&str]) -> Result<(Array3<f32>, Array2<f32>)> {
        if texts.is_empty() {
            return Err(anyhow!("Cannot get hidden states for an empty batch"));
        }
        let input_ids = self.tokenize_batch(texts)?;

        // Create an attention mask that ignores padding tokens
        let pad_id = self.pad_token_id().unwrap_or(0);
        let attention_mask = input_ids.mapv(|id| if id == pad_id { 0.0 } else { 1.0 });

        let hidden_states = self
            .encoder()
            .get_hidden_states(&input_ids, &attention_mask, None) // TODO: token_type_ids ???
            .await?;

        Ok((hidden_states, attention_mask))
    }

    /// Encode text into a single, L2-normalized embedding vector.
    ///
    /// # Arguments
    /// * `text` - Input text
    /// * `pooling` - Pooling strategy: "cls", or "mean"
    /// * `normalize` -
    async fn encode(&self, text: &str, config: &EncodingConfig) -> Result<Vec<f32>> {
        let (hidden, attention_mask) = self.get_hidden_states_batch(&[text]).await?;

        // Mean pooling requires the attention mask to work correctly
        let embedding = match config.pooling_strategy {
            PoolingStrategy::Cls => {
                // Use [CLS] token (first token)
                hidden.slice(ndarray::s![0, 0, ..]).to_owned()
            }
            PoolingStrategy::Mean => {
                // Mean pool using the attention mask
                mean_pool(&hidden, &attention_mask)?.row(0).to_owned()
            }
            PoolingStrategy::Max => {
                unimplemented!()
            }
            PoolingStrategy::LastToken => {
                unimplemented!()
            }
            _ => {
                return Err(anyhow!(
                    "Unknown pooling strategy: {}",
                    config.pooling_strategy
                ));
            }
        };

        // L2 Normalize the final embedding
        if config.normalize {
            let norm = (embedding.dot(&embedding)).sqrt();
            let normalized_embedding = if norm > 0.0 {
                embedding / norm
            } else {
                embedding
            };

            return Ok(normalized_embedding.to_vec());
        }

        Ok(embedding.to_vec())
    }

    /// Encode a batch of texts into embedding vectors.
    ///
    /// # Arguments
    /// * `texts` - Input texts
    /// * `pooling` - Pooling strategy: "cls", or "mean"
    /// * `normalize` - Whether to L2-normalize the outputs
    async fn encode_batch(&self, texts: &[&str], config: &EncodingConfig) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let (hidden_states, attention_mask) = self.get_hidden_states_batch(texts).await?;

        // Apply pooling to the whole batch
        let mut pooled = match config.pooling_strategy {
            PoolingStrategy::Cls => {
                // Use [CLS] token (first token of each item in the batch)
                hidden_states.slice(ndarray::s![.., 0, ..]).to_owned()
            }
            PoolingStrategy::Mean => mean_pool(&hidden_states, &attention_mask)?,
            PoolingStrategy::LastToken => {
                unimplemented!()
            }
            PoolingStrategy::Max => {
                unimplemented!()
            }
            _ => {
                return Err(anyhow!(
                    "Unknown pooling strategy: {}",
                    config.pooling_strategy
                ));
            }
        };

        // Conditionally L2 normalize the entire batch of embeddings
        if config.normalize {
            l2_normalize_inplace(&mut pooled);
        }

        Ok(pooled.outer_iter().map(|row| row.to_vec()).collect())
    }
}

pub trait GpuClassificationHead {
    fn project(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor>;
}
//
// // ============================================================================
// // GPU ENCODER
// // ============================================================================
//
// /// Flexible input for GPU encoder - supports hybrid execution.
// ///
// /// This enum allows the encoder to accept input in various forms,
// /// enabling efficient hybrid CPU/GPU workflows.
// pub enum GpuEncoderInput<'a> {
//     /// Token IDs already on GPU.
//     ///
//     /// Use when: Full GPU path, token IDs already uploaded.
//     /// Shape: `[batch_size, sequence_length]` u32
//     TokensGpu(&'a GpuTensor),
//
//     /// Token IDs on CPU - will do CPU embedding lookup, upload hidden states.
//     ///
//     /// Use when: VRAM saving mode with cpu_embeddings=true.
//     /// The encoder will:
//     /// 1. Perform embedding lookup on CPU
//     /// 2. Upload resulting hidden states to GPU
//     /// 3. Continue with GPU layers
//     TokensCpu(&'a Array2<u32>),
//
//     /// Pre-computed hidden states on GPU (skip embedding).
//     ///
//     /// Use when: Continuing from partial GPU execution or external embedding.
//     /// Shape: `[batch_size, sequence_length, hidden_size]` f32
//     HiddenGpu(&'a GpuTensor),
//
//     /// Pre-computed hidden states on CPU - will upload.
//     ///
//     /// Use when: Continuing from CPU encoder/layers to GPU.
//     /// The encoder will upload the hidden states and continue with GPU layers.
//     HiddenCpu(&'a Array3<f32>),
// }
//
// /// Output from GPU encoder.
// #[derive(Debug)]
// pub struct GpuEncoderOutput {
//     /// Final hidden states on GPU: `[batch_size, sequence_length, hidden_size]`
//     pub last_hidden_state: GpuTensor,
// }
//
// /// GPU-based transformer encoder trait.
// ///
// /// Provides methods for embedding lookup, normalization, and layer execution
// /// on GPU with support for hybrid CPU/GPU workflows through `GpuEncoderInput`.
// ///
// pub trait GpuEncoder: Send + Sync {
//     /// Compute embeddings only (handles CPU/GPU input).
//     ///
//     /// Does NOT apply the initial layer normalization.
//     ///
//     /// # Arguments
//     /// * `cmd_encoder` - WGPU command encoder for recording GPU commands
//     /// * `pool` - Tensor pool for intermediate allocations
//     /// * `input` - Token IDs or hidden states (see `GpuEncoderInput`)
//     /// * `token_type_ids` - Optional token type IDs on GPU
//     ///
//     /// # Returns
//     /// Hidden states on GPU `[batch_size, sequence_length, hidden_size]`
//     fn embed(
//         &self,
//         cmd_encoder: &mut wgpu::CommandEncoder,
//         pool: &mut GpuTensorPool,
//         input: GpuEncoderInput,
//         token_type_ids: Option<&GpuTensor>,
//     ) -> Result<GpuTensor>;
//
//     /// Compute embeddings + initial normalization.
//     ///
//     /// This produces hidden states ready to be processed by encoder layers.
//     ///
//     /// # Arguments
//     /// * `cmd_encoder` - WGPU command encoder
//     /// * `pool` - Tensor pool
//     /// * `input` - Token IDs or hidden states
//     /// * `token_type_ids` - Optional token type IDs
//     ///
//     /// # Returns
//     /// Normalized hidden states on GPU
//     fn embed_and_normalize(
//         &self,
//         cmd_encoder: &mut wgpu::CommandEncoder,
//         pool: &mut GpuTensorPool,
//         input: GpuEncoderInput,
//         token_type_ids: Option<&GpuTensor>,
//     ) -> Result<GpuTensor>;
//
//     /// Run layers `[start_layer, end_layer)` on hidden states.
//     ///
//     /// # Arguments
//     /// * `cmd_encoder` - WGPU command encoder
//     /// * `pool` - Tensor pool
//     /// * `hidden_states` - Input hidden states on GPU
//     /// * `attention_mask` - Attention mask on GPU
//     /// * `start_layer` - First layer to execute (inclusive)
//     /// * `end_layer` - Last layer to execute (exclusive)
//     ///
//     /// # Returns
//     /// Hidden states after processing through the specified layers
//     fn forward_layers(
//         &self,
//         cmd_encoder: &mut wgpu::CommandEncoder,
//         pool: &mut GpuTensorPool,
//         hidden_states: &GpuTensor,
//         attention_mask: &GpuTensor,
//         start_layer: usize,
//         end_layer: usize,
//     ) -> Result<GpuTensor>;
//
//     /// Number of encoder layers in this model.
//     fn num_layers(&self) -> usize;
//
//     /// Hidden dimension of the model.
//     fn hidden_size(&self) -> usize;
//
//     /// Full forward pass through the encoder.
//     ///
//     /// Default implementation calls embed_and_normalize + forward_layers(0, num_layers).
//     fn forward(
//         &self,
//         cmd_encoder: &mut wgpu::CommandEncoder,
//         pool: &mut GpuTensorPool,
//         input: GpuEncoderInput,
//         attention_mask: &GpuTensor,
//         token_type_ids: Option<&GpuTensor>,
//     ) -> Result<GpuEncoderOutput> {
//         let hidden = self.embed_and_normalize(cmd_encoder, pool, input, token_type_ids)?;
//         let output = self.forward_layers(
//             cmd_encoder,
//             pool,
//             &hidden,
//             attention_mask,
//             0,
//             self.num_layers(),
//         )?;
//         Ok(GpuEncoderOutput {
//             last_hidden_state: output,
//         })
//     }
// }

// #[async_trait(?Send)]
// pub trait GpuEncoder: Send + Sync {
//     /// Encodes input IDs into hidden states, keeping the result on the GPU.
//     fn forward(
//         &self,
//         encoder: &mut wgpu::CommandEncoder,
//         pool: &mut GpuTensorPool,
//         input_ids: &GpuTensor,
//         attention_mask: &GpuTensor,
//     ) -> Result<GpuTensor>;
// }

/// Flexible input for the decoder.
/// - Use `Gpu` when you have VRAM to spare (fastest).
/// - Use `Cpu` when VRAM is tight (saves ~1GB for Llama 3.2 1B).
pub enum DecoderInput<'a> {
    Gpu(&'a GpuTensor),
    Cpu(&'a [u32]),
}
// Phase 2: Implement Backends and GPU Components
// TODO #4: Implement BartGpuDecoder.
// Action: Create the BartGpuDecoder struct and implement GpuCrossAttentionDecoder for it. This will be a new file.
// TODO #5: Update BartModel for GPU.
// Action:
// In from_pretrained, instantiate BartGpuEncoder and BartGpuDecoder.
// Load the LM head into a GpuTensor.
// Implement the new EncoderDecoderLanguageModel accessor trait.
// TODO #6: Implement GpuBackend against the new trait.
// Action:
// Implement async fn encode. This will call model.gpu_encoder().forward(...).
// Implement async fn decode_step. This will call model.gpu_decoder().forward(...) followed by a GPU-based matrix multiplication with model.gpu_lm_head_weights(), and finally download the resulting logits.
// TODO #7: Refactor CpuBackend against the new trait.
// Action:
// Implement async fn encode. This will call model.cpu_encoder().forward(...) and then pre-compute the cross-attention K/V cache, bundling it all into the CpuTensor::EncoderState.
// Implement async fn decode_step. This will unpack the CpuTensor, call model.cpu_decoder().forward(...), and then project to logits using model.lm_head_layer().
// Phase 3: Final Generator Integration
// TODO #8: Update run_beam_search.
// Status: ALMOST DONE. This function is already generic, which is great. It just needs minor updates to call the new backend methods.
// Action:
// Remove the call to backend.prepare_encoder_state. The initial encoder_state will now come from the new backend.encode(...) method, which you'll call once before the loop.
// Inside the loop, replace the call to backend.forward(...) and the manual logit projection with a single call to backend.decode_step(...).
#[derive(Debug, Clone, Copy)]
pub struct DecoderLoadConfig {
    /// If true, embedding weights are kept in system RAM and lookup happens on CPU.
    /// Saves VRAM (~1GB for Llama 1B).
    pub offload_embeddings: bool,

    /// If true, the LM Head (final projection) is kept in system RAM.
    /// Saves VRAM (~1GB for Llama 1B).
    pub offload_lm_head: bool,

    /// Number of layers to run on GPU. If None, all layers are on GPU.
    /// Useful for partial offloading.
    pub gpu_layers: Option<usize>,

    pub target_dtype: Option<DType>,
}

impl Default for DecoderLoadConfig {
    fn default() -> Self {
        Self {
            offload_embeddings: false, // Default to Performance (Pure GPU)
            offload_lm_head: false,
            gpu_layers: None,
            target_dtype: None,
        }
    }
}

#[async_trait(?Send)]
pub trait GpuDecoder: Send + Sync {
    /// Performs a forward pass on the GPU.
    async fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder, // <-- Pass in the encoder
        pool: &mut crate::gpu_ops::GpuTensorPool, // <-- Pass in the pool
        // input_ids: &GpuTensor,
        input: DecoderInput<'_>,
        attention_mask: &GpuTensor,
        position_offset: usize,
        cache: Option<&mut crate::cache::GpuKVCache>,
        encoder_hidden_states: Option<&GpuTensor>,
    ) -> Result<GpuTensor>; // Returns the final hidden states as a GpuTensor
}

/// Trait for decoder-only language models (GPT-2, GPT-3, Llama, etc.)
///
/// These models generate text autoregressively.
#[async_trait(?Send)]
pub trait DecoderLanguageModel: LanguageModel {
    /// Get the decoder backend
    fn decoder(&self) -> &dyn Decoder<Input=Array2<u32>, Output=DecoderOutput>;

    /// Get the GPU-native decoder.
    /// Returns an error if the model was not loaded with GPU support.
    fn gpu_decoder(&self) -> Result<&(dyn GpuDecoder + Send + Sync)> {
        Err(anyhow::anyhow!(
            "This model does not have a GPU decoder implementation."
        ))
    }

    /// Get the GPU tensor for the LM head weights.
    /// Returns an error if the model was not loaded with GPU support.
    fn gpu_lm_head(&self) -> Result<&GpuTensor> {
        Err(anyhow!("This model does not have GPU LM head weights."))
    }
    /// Get the PRE-TRANSPOSED GPU tensor for the LM head weights.
    fn gpu_lm_head_transposed(&self) -> Result<&GpuTensor> {
        Err(anyhow!(
            "This model does not have transposed GPU LM head weights."
        ))
    }

    /// Get the LM head (projection to vocabulary)
    fn lm_head(&self) -> &Array2<f32>;

    /// Specifies the generation loop strategy required for this model
    /// to maintain parity with its reference implementation.
    fn autoregressive_loop(&self) -> AutoregressiveLoop;

    fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>>;

    /// Get hidden states for input text
    async fn get_hidden_states(&self, text: &str) -> Result<Array3<f32>> {
        let input_ids = self.tokenize(text)?;
        let seq_len = input_ids.ncols();
        let attention_mask = create_full_attention_mask(1, seq_len);

        self.decoder()
            .get_hidden_states(&input_ids, &attention_mask)
            .await
    }
    fn get_default_generation_config(&self) -> GenerationConfig {
        GenerationConfig::default() // Default implementation
    }
    /// Get raw logits for input text
    ///
    /// Useful for:
    /// - Perplexity calculation
    /// - Comparing model outputs
    /// - Custom sampling strategies
    async fn get_logits(&self, text: &str) -> Result<Array3<f32>> {
        let input_ids = self.tokenize(text)?;
        let seq_len = input_ids.ncols();
        let attention_mask = create_full_attention_mask(1, seq_len);

        let decoder_output = self
            .decoder()
            .forward(&input_ids, &attention_mask, None)
            .await?;

        self.project_to_logits(&decoder_output.last_hidden_state)
    }
}

/// Trait for encoder-decoder models (BART, T5, etc.)
///
/// These models encode input and decode output (seq2seq).
#[async_trait]
pub trait Seq2SeqLanguageModel: LanguageModel {
    /// Encode input text to hidden states
    async fn encode_input(&self, text: &str) -> Result<EncoderOutput>;

    async fn generate(&self, input_text: &str, config: &GenerationConfig) -> Result<String>;

    /// Generate output from encoder hidden states.
    async fn generate_from_encoding(
        &self,
        encoder_output: &EncoderOutput,
        encoder_attention_mask: &Array2<f32>, // The mask is needed for cross-attention
        config: &GenerationConfig,
    ) -> Result<String>;
}

/// Helper: Project hidden states to vocabulary logits
///
/// Performs matrix multiplication: [batch, seq, hidden] @ [hidden, vocab] → [batch, seq, vocab]
pub fn project_to_vocab(hidden_states: &Array3<f32>, lm_head: &Array2<f32>) -> Result<Array3<f32>> {
    let (batch_size, seq_len, hidden_size) = hidden_states.dim();
    let vocab_size = lm_head.ncols();

    if lm_head.nrows() != hidden_size {
        return Err(anyhow!(
            "LM head shape mismatch: expected [{}x{}], got [{}x{}]",
            hidden_size,
            vocab_size,
            lm_head.nrows(),
            lm_head.ncols()
        ));
    }

    // Reshape to 2D for efficient matmul
    let hidden_2d = hidden_states
        .view()
        .into_shape_with_order((batch_size * seq_len, hidden_size))?;

    // Matrix multiplication: [batch*seq, hidden] @ [hidden, vocab]
    let logits_2d = hidden_2d.dot(lm_head);

    // Reshape back to 3D
    let logits = logits_2d.into_shape_with_order((batch_size, seq_len, vocab_size))?;

    Ok(logits)
}

/// Helper: Apply L2 normalization to embeddings
pub fn l2_normalize(embeddings: &Array2<f32>) -> Array2<f32> {
    let mut normalized = embeddings.clone();

    for mut row in normalized.rows_mut() {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            row /= norm;
        }
    }

    normalized
}

pub fn l2_normalize_inplace(embeddings: &mut Array2<f32>) {
    for mut row in embeddings.rows_mut() {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            row /= norm;
        }
    }
}
