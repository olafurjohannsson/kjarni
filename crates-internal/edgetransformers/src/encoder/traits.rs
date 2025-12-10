//! Base traits for language models
//!
//! This module provides high-level, user-facing traits that abstract over
//! the low-level architecture traits in `traits.rs`.

use crate::gpu_ops::GpuTensor;
use crate::gpu_ops::GpuTensorPool;
use crate::models::base::LanguageModel;
use crate::pooling::mean_pool;
use crate::traits::{Encoder, EncoderOutput, LanguageModelConfig, LayerAttentionNames, LayerFeedForwardNames};
use crate::utils::create_full_attention_mask;
pub use crate::weights::DType;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ndarray::{Array2, Array3};
use std::fmt;

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

/// Describes the specific architectural details of an Encoder-only model (e.g., BERT, RoBERTa).
///
/// This trait acts as a "blueprint" that a generic `TransformerEncoder` can use to
/// construct itself. It provides a mapping from abstract component concepts (e.g., "the query
/// projection of the first layer's attention") to the concrete tensor names found in a
/// `safetensors` weight file.
pub trait EncoderArchitecture: LanguageModelConfig {
    /// Returns the tensor names for the word, position, and token type embeddings.
    //fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>); // RoBERTa has no token_type_embeddings

    /// Returns the tensor names for the LayerNorm applied after the embedding layer.
    fn get_embedding_layer_norm_names(&self) -> (&str, &str);

    /// Returns the names of all weights and biases for the attention component of a specific encoder layer.
    fn get_attention_names(&self, layer_index: usize) -> LayerAttentionNames;

    /// Returns the names of all weights and biases for the feed-forward component of a specific encoder layer.
    fn get_feed_forward_names(&self, layer_index: usize) -> LayerFeedForwardNames;
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

// ============================================================================
// GPU ENCODER
// ============================================================================

/// Flexible input for GPU encoder - supports hybrid execution.
///
/// This enum allows the encoder to accept input in various forms,
/// enabling efficient hybrid CPU/GPU workflows.
pub enum GpuEncoderInput<'a> {
    /// Token IDs already on GPU.
    ///
    /// Use when: Full GPU path, token IDs already uploaded.
    /// Shape: `[batch_size, sequence_length]` u32
    TokensGpu(&'a GpuTensor),

    /// Token IDs on CPU - will do CPU embedding lookup, upload hidden states.
    ///
    /// Use when: VRAM saving mode with cpu_embeddings=true.
    /// The encoder will:
    /// 1. Perform embedding lookup on CPU
    /// 2. Upload resulting hidden states to GPU
    /// 3. Continue with GPU layers
    TokensCpu(&'a Array2<u32>),

    /// Pre-computed hidden states on GPU (skip embedding).
    ///
    /// Use when: Continuing from partial GPU execution or external embedding.
    /// Shape: `[batch_size, sequence_length, hidden_size]` f32
    HiddenGpu(&'a GpuTensor),

    /// Pre-computed hidden states on CPU - will upload.
    ///
    /// Use when: Continuing from CPU encoder/layers to GPU.
    /// The encoder will upload the hidden states and continue with GPU layers.
    HiddenCpu(&'a Array3<f32>),
}

/// Output from GPU encoder.
#[derive(Debug)]
pub struct GpuEncoderOutput {
    /// Final hidden states on GPU: `[batch_size, sequence_length, hidden_size]`
    pub last_hidden_state: GpuTensor,
}

/// GPU-based transformer encoder trait.
///
/// Provides methods for embedding lookup, normalization, and layer execution
/// on GPU with support for hybrid CPU/GPU workflows through `GpuEncoderInput`.
///
/// # Example
/// ```rust
/// // Full GPU path
/// let output = encoder.forward(cmd, pool, GpuEncoderInput::TokensGpu(&ids), &mask, None)?;
///
/// // VRAM saving: CPU embeddings → GPU layers
/// let output = encoder.forward(cmd, pool, GpuEncoderInput::TokensCpu(&ids_cpu), &mask, None)?;
///
/// // Hybrid: continue from CPU hidden states
/// let cpu_hidden = cpu_encoder.forward_layers(&hidden, &mask, 0, 6)?;
/// let output = encoder.forward_layers(cmd, pool, &uploaded_hidden, &mask, 6, 12)?;
/// ```
pub trait GpuEncoder: Send + Sync {
    /// Compute embeddings only (handles CPU/GPU input).
    ///
    /// Does NOT apply the initial layer normalization.
    ///
    /// # Arguments
    /// * `cmd_encoder` - WGPU command encoder for recording GPU commands
    /// * `pool` - Tensor pool for intermediate allocations
    /// * `input` - Token IDs or hidden states (see `GpuEncoderInput`)
    /// * `token_type_ids` - Optional token type IDs on GPU
    ///
    /// # Returns
    /// Hidden states on GPU `[batch_size, sequence_length, hidden_size]`
    fn embed(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: GpuEncoderInput,
        token_type_ids: Option<&GpuTensor>,
    ) -> Result<GpuTensor>;

    /// Compute embeddings + initial normalization.
    ///
    /// This produces hidden states ready to be processed by encoder layers.
    ///
    /// # Arguments
    /// * `cmd_encoder` - WGPU command encoder
    /// * `pool` - Tensor pool
    /// * `input` - Token IDs or hidden states
    /// * `token_type_ids` - Optional token type IDs
    ///
    /// # Returns
    /// Normalized hidden states on GPU
    fn embed_and_normalize(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: GpuEncoderInput,
        token_type_ids: Option<&GpuTensor>,
    ) -> Result<GpuTensor>;

    /// Run layers `[start_layer, end_layer)` on hidden states.
    ///
    /// # Arguments
    /// * `cmd_encoder` - WGPU command encoder
    /// * `pool` - Tensor pool
    /// * `hidden_states` - Input hidden states on GPU
    /// * `attention_mask` - Attention mask on GPU
    /// * `start_layer` - First layer to execute (inclusive)
    /// * `end_layer` - Last layer to execute (exclusive)
    ///
    /// # Returns
    /// Hidden states after processing through the specified layers
    fn forward_layers(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<GpuTensor>;

    /// Number of encoder layers in this model.
    fn num_layers(&self) -> usize;

    /// Hidden dimension of the model.
    fn hidden_size(&self) -> usize;

    /// Full forward pass through the encoder.
    ///
    /// Default implementation calls embed_and_normalize + forward_layers(0, num_layers).
    fn forward(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: GpuEncoderInput,
        attention_mask: &GpuTensor,
        token_type_ids: Option<&GpuTensor>,
    ) -> Result<GpuEncoderOutput> {
        let hidden = self.embed_and_normalize(cmd_encoder, pool, input, token_type_ids)?;
        let output = self.forward_layers(
            cmd_encoder,
            pool,
            &hidden,
            attention_mask,
            0,
            self.num_layers(),
        )?;
        Ok(GpuEncoderOutput {
            last_hidden_state: output,
        })
    }
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
