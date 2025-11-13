//! Base traits for language models
//!
//! This module provides high-level, user-facing traits that abstract over
//! the low-level architecture traits in `traits.rs`.

use crate::Cache;
use crate::cache::CpuKVCache;
use crate::pooling::mean_pool;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array1, Array2, Array3};
use tokenizers::Tokenizer;
use std::sync::Arc;
use crate::traits::{
    Decoder, DecoderOutput, Encoder, EncoderOutput, LanguageModelConfig, TransformerModel, CrossAttentionDecoder
};
use crate::utils::create_full_attention_mask;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScalingConfig {
    pub factor: f32,
    pub high_freq_factor: f32,
    pub low_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: String,
}

/// Configuration for text generation
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    // Let's make this optional. If it's `Some`, it takes precedence.
    pub max_new_tokens: Option<usize>,

    // This will be our ultimate authority for loop termination.
    pub max_length: usize,

    // pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub sampling_strategy: SamplingStrategy,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub num_beams: usize,
    pub min_length: usize,

    pub length_penalty: f32,
    pub early_stopping: bool,
    pub no_repeat_ngram_size: usize,
    
    /// Whether to prepend the BOS token to the prompt
    /// Defaults to true
    pub add_bos_token: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationStrategy {
    /// Efficient, pipelined logic. Uses the prefill output directly.
    /// Correct for Llama and the "theoretically correct" approach.
    Pipelined,
    /// Inefficient, two-pass logic. Discards prefill output and re-processes
    /// the last prompt token to get the first logits.
    /// Required for parity with Hugging Face's GPT-2 implementation.
    Legacy,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            // General sampling params
            max_new_tokens: None,
            temperature: 0.7,
            top_k: Some(50),
            top_p: Some(0.9),
            repetition_penalty: 1.1,
            sampling_strategy: SamplingStrategy::TopKTopP,
            eos_token_id: Some(50256), // default GPT-2 EOS
            pad_token_id: Some(50256),
            // Beam search specific params (from BART config)
            num_beams: 4,
            min_length: 56,
            max_length: 142,
            no_repeat_ngram_size: 3,
            length_penalty: 2.0,
            early_stopping: true,
            add_bos_token: true,
        }
    }
}

pub struct BeamHypothesis {
    pub tokens: Vec<u32>,
    pub score: f32,
    pub cache: Box<dyn Cache>,
}

// No changes needed here
#[derive(Clone, Debug)]
pub enum SamplingStrategy {
    Greedy,
    TopK,
    TopP,
    TopKTopP,
    Temperature,
    BeamSearch, // We'll use this strategy implicitly for Seq2Seq
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
    fn new_cache(&self, batch_size: usize, max_len: usize) -> Result<Box<dyn Cache>>;
    
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
        self.tokenizer()
            .token_to_id("<|endoftext|>")
            .or_else(|| self.tokenizer().token_to_id("</s>"))
    }

    /// Get the padding token ID (if applicable)
    fn pad_token_id(&self) -> Option<u32> {
        self.tokenizer()
            .token_to_id("<pad>")
            .or_else(|| self.tokenizer().token_to_id("[PAD]"))
    }

    /// Get the beginning-of-sequence token ID (if applicable)
    fn bos_token_id(&self) -> Option<u32> {
        self.tokenizer()
            .token_to_id("<s>")
            .or_else(|| self.tokenizer().token_to_id("[CLS]"))
    }

    /// Tokenize text into token IDs
    ///
    /// Returns Array2<f32> with shape [1, seq_len]
    fn tokenize(&self, text: &str) -> Result<Array2<f32>> {
        let encoding = self
            .tokenizer()
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let ids: Vec<f32> = encoding.get_ids().iter().map(|&id| id as f32).collect();

        let seq_len = ids.len();
        Ok(Array2::from_shape_vec((1, seq_len), ids)?)
    }

    /// Tokenize batch of texts (with padding to max length)
    ///
    /// Returns Array2<f32> with shape [batch_size, max_seq_len]
    fn tokenize_batch(&self, texts: &[&str]) -> Result<Array2<f32>> {
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
        let pad_id = self.pad_token_id().unwrap_or(0) as f32;
        let batch_size = texts.len();

        let mut batch = Array2::from_elem((batch_size, max_len), pad_id);

        for (i, encoding) in encodings.iter().enumerate() {
            for (j, &token_id) in encoding.get_ids().iter().enumerate() {
                batch[[i, j]] = token_id as f32;
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
    fn encoder(&self) -> &dyn Encoder<Input = Array2<f32>, Output = EncoderOutput>;

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
        let pad_id = self.pad_token_id().unwrap_or(0) as f32;
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
    async fn encode(&self, text: &str, pooling: &str, normalize: bool) -> Result<Vec<f32>> {
        let (hidden, attention_mask) = self.get_hidden_states_batch(&[text]).await?;

        // Mean pooling requires the attention mask to work correctly
        let embedding = match pooling {
            "cls" => {
                // Use [CLS] token (first token)
                hidden.slice(ndarray::s![0, 0, ..]).to_owned()
            }
            "mean" => {
                // Mean pool using the attention mask
                mean_pool(&hidden, &attention_mask)?.row(0).to_owned()
            }
            _ => return Err(anyhow!("Unknown pooling strategy: {}", pooling)),
        };

        // L2 Normalize the final embedding
        if normalize {
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
    async fn encode_batch(
        &self,
        texts: &[&str],
        pooling: &str,
        normalize: bool,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let (hidden_states, attention_mask) = self.get_hidden_states_batch(texts).await?;

        // Apply pooling to the whole batch
        let mut pooled = match pooling {
            "cls" => {
                // Use [CLS] token (first token of each item in the batch)
                hidden_states.slice(ndarray::s![.., 0, ..]).to_owned()
            }
            "mean" => mean_pool(&hidden_states, &attention_mask)?,
            _ => return Err(anyhow!("Unknown pooling strategy: {}", pooling)),
        };

        // Conditionally L2 normalize the entire batch of embeddings
        if normalize {
            l2_normalize_inplace(&mut pooled);
        }

        Ok(pooled.outer_iter().map(|row| row.to_vec()).collect())
    }
}

#[async_trait]
pub trait EncoderDecoderLanguageModel: LanguageModel {
    /// Returns a reference to the model's encoder component.
    fn encoder(&self) -> &dyn Encoder<Input = Array2<f32>, Output = EncoderOutput>;

    /// Returns a reference to the model's decoder component.
    fn decoder(&self) -> &dyn CrossAttentionDecoder<Input = Array2<f32>, Output = DecoderOutput>;

    /// Projects the final hidden states from the decoder to vocabulary logits.
    fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>>;
    
    fn generation_config_from_preset(&self) -> GenerationConfig;

    /// The token ID that should be used to start the decoding process.
    fn decoder_start_token_id(&self) -> u32;
}

/// Trait for decoder-only language models (GPT-2, GPT-3, Llama, etc.)
///
/// These models generate text autoregressively.
#[async_trait]
pub trait DecoderLanguageModel: LanguageModel {
    /// Get the decoder backend
    fn decoder(&self) -> &dyn Decoder<Input = Array2<f32>, Output = DecoderOutput>;

    /// Get the LM head (projection to vocabulary)
    fn lm_head(&self) -> &Array2<f32>;

    /// Specifies the generation loop strategy required for this model
    /// to maintain parity with its reference implementation.
    fn generation_strategy(&self) -> GenerationStrategy;

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
