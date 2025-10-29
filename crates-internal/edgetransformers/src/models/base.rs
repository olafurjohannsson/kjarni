//! Base traits for language models
//!
//! This module provides high-level, user-facing traits that abstract over
//! the low-level architecture traits in `traits.rs`.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array1, Array2, Array3};
use tokenizers::Tokenizer;

use crate::traits::{TransformerModel, Encoder, Decoder, EncoderOutput, DecoderOutput, LanguageModelConfig};
use crate::utils::create_full_attention_mask;

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
        let encoding = self.tokenizer()
            .encode(text, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        
        let ids: Vec<f32> = encoding.get_ids()
            .iter()
            .map(|&id| id as f32)
            .collect();
        
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
            let encoding = self.tokenizer()
                .encode(*text, true)
                .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
            
            max_len = max_len.max(encoding.len());
            encodings.push(encoding);
        }
        
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
        token_ids.iter()
            .map(|ids| self.decode(ids))
            .collect()
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
            .get_hidden_states(&input_ids, &attention_mask)
            .await
    }
    
    /// Encode text into a single embedding vector with pooling
    ///
    /// # Arguments
    /// * `text` - Input text
    /// * `pooling` - Pooling strategy: "cls", "mean", or "max"
    async fn encode(&self, text: &str, pooling: &str) -> Result<Vec<f32>> {
        let hidden = self.get_hidden_states(text).await?;
        
        // Apply pooling
        let embedding = match pooling {
            "cls" => {
                // Use [CLS] token (first token)
                hidden.slice(ndarray::s![0, 0, ..]).to_owned()
            }
            "mean" => {
                // Mean pooling over sequence
                hidden.mean_axis(ndarray::Axis(1))
                    .ok_or_else(|| anyhow!("Mean pooling failed"))?
                    .row(0)
                    .to_owned()
            }
            "max" => {
                // Max pooling over sequence
                let (_, seq_len, hidden_size) = hidden.dim();
                let mut result = Array1::<f32>::from_elem(hidden_size, f32::NEG_INFINITY);
                
                for s in 0..seq_len {
                    for h in 0..hidden_size {
                        result[h] = result[h].max(hidden[[0, s, h]]);
                    }
                }
                result
            }
            _ => return Err(anyhow!("Unknown pooling strategy: {}", pooling)),
        };
        
        Ok(embedding.to_vec())
    }
    
    /// Encode batch of texts
    async fn encode_batch(&self, texts: &[&str], pooling: &str) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::new();
        for text in texts {
            results.push(self.encode(text, pooling).await?);
        }
        Ok(results)
    }
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
        
        let decoder_output = self.decoder()
            .forward(&input_ids, &attention_mask, None)
            .await?;
        
        // Project to vocabulary
        let logits = project_to_vocab(
            &decoder_output.last_hidden_state,
            self.lm_head(),
        )?;
        
        Ok(logits)
    }
}

/// Trait for encoder-decoder models (BART, T5, etc.)
///
/// These models encode input and decode output (seq2seq).
#[async_trait]
pub trait Seq2SeqLanguageModel: LanguageModel {
    /// Encode input text to hidden states
    async fn encode_input(&self, text: &str) -> Result<EncoderOutput>;
    
    async fn generate(&self, input_text: &str, max_length: usize) -> Result<String>;

    
    /// Generate output from encoder hidden states.
    async fn generate_from_encoding(
        &self,
        encoder_output: &EncoderOutput,
        encoder_attention_mask: &Array2<f32>, // The mask is needed for cross-attention
        max_length: usize,
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
            hidden_size, vocab_size, lm_head.nrows(), lm_head.ncols()
        ));
    }
    
    // Reshape to 2D for efficient matmul
    let hidden_2d = hidden_states
        .view()
        .into_shape((batch_size * seq_len, hidden_size))?;
    
    // Matrix multiplication: [batch*seq, hidden] @ [hidden, vocab]
    let logits_2d = hidden_2d.dot(lm_head);
    
    // Reshape back to 3D
    let logits = logits_2d.into_shape((batch_size, seq_len, vocab_size))?;
    
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