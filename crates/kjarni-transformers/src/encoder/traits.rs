//! Base traits for language models
//!
//! This module provides high-level, user-facing traits that abstract over
//! the low-level architecture traits in `traits.rs`.

use crate::encoder::config::{EncodingConfig, PoolingStrategy};
use crate::gpu_ops::{GpuFrameContext, GpuTensor, GpuTensorPool};
use crate::{last_token_pool, max_pool};
use crate::models::base::{LanguageModel, ModelInput};
use crate::pooling::mean_pool;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ndarray::{Array2, Array3};

/// Trait for encoder-only language models (BERT, RoBERTa, etc.)
///
/// These models encode text into fixed-size embeddings.
#[async_trait(?Send)]
pub trait EncoderLanguageModel: LanguageModel {
    fn encoder_cpu_ops(&self) -> Option<&dyn CpuEncoderOps>;

    fn encoder_gpu_ops(&self) -> Option<&dyn GpuEncoderOps>;

    fn dimension(&self) -> usize {
        self.hidden_size()
    }
    fn encoder_dimensions(&self) -> usize {
        // either GPU or CPU
        match self.device() {
            crate::traits::Device::Cpu => {
                let ops = self
                    .encoder_cpu_ops()
                    .expect("CPU ops not implemented for this model");
                ops.encoder().hidden_size()
            }
            crate::traits::Device::Wgpu => {
                let ops = self
                    .encoder_gpu_ops()
                    .expect("GPU ops not implemented for this model");
                ops.encoder().hidden_size()
            }
        }
    }
    /// Get the encoder backend
    // fn encoder(&self) -> &dyn Encoder<Input = Array2<u32>, Output = EncoderOutput>;

    /// Get hidden states for input text
    async fn get_hidden_states(&self, text: &str) -> Result<Array3<f32>> {
        let (batch_hidden_states, _) = self.get_hidden_states_batch(&[text]).await?;
        Ok(batch_hidden_states)
    }

    /// Get hidden states for a batch of texts
    async fn get_hidden_states_batch(&self, texts: &[&str]) -> Result<(Array3<f32>, Array2<f32>)> {
        if texts.is_empty() {
            return Ok((Array3::zeros((0, 0, 0)), Array2::zeros((0, 0))));
        }

        // 1. Tokenize and create attention mask (this logic is good)
        let input_ids = self.tokenize_batch(texts)?;
        let pad_id = self.pad_token_id().unwrap_or(0);
        let attention_mask = input_ids.mapv(|id| if id == pad_id { 0.0 } else { 1.0 });

        // 2. Dispatch to the correct backend using the ops traits
        let hidden_states = if let Some(ops) = self.encoder_cpu_ops() {
            // --- CPU PATH ---
            // The `token_type_ids` can be added here later if a model needs them.
            ops.encoder()
                .forward(&input_ids, &attention_mask, None)?
                .last_hidden_state
        } else if let Some(ops) = self.encoder_gpu_ops() {
            // --- GPU PATH ---
            let context = self
                .context()
                .ok_or_else(|| anyhow!("GPU model missing context"))?;
            let pool = context.get_inference_pool();
            let mut pool_guard = pool.lock().await;
            // Use a GpuFrameContext to manage resources
            let mut frame = GpuFrameContext::new(&context, pool_guard);
            let (encoder_cmd, pool_ref) = frame.resources();

            // Upload data to GPU
            let input_ids_gpu = GpuTensor::from_ndarray(&context, &input_ids)?;
            let attention_mask_gpu = GpuTensor::from_ndarray(&context, &attention_mask)?;

            // Run the forward pass
            let gpu_output = ops.encoder().forward(
                encoder_cmd,
                pool_ref,
                ModelInput::TokensGpu(&input_ids_gpu),
                &attention_mask_gpu,
                None, // token_type_ids can be added here
            )?;

            frame.finish();

            // Download the result back to CPU
            gpu_output.last_hidden_state.to_ndarray_3d().await?
        } else {
            return Err(anyhow!(
                "No available CPU or GPU encoder implementation for this model."
            ));
        };

        Ok((hidden_states, attention_mask))
    }
}

/// Defines the application of an encoder model for sentence similarity tasks.
///
/// This trait provides methods to convert raw hidden states into final, pooled,
/// and normalized sentence embeddings.
#[async_trait(?Send)]
pub trait SentenceEncoderModel: EncoderLanguageModel {
    /// Encode a batch of texts into embedding vectors.
    async fn encode_batch(&self, texts: &[&str], config: &EncodingConfig) -> Result<Vec<Vec<f32>>>;

    /// Encode a single text into an embedding vector.
    async fn encode(&self, text: &str, config: &EncodingConfig) -> Result<Vec<f32>> {
        // Default implementation for single text encoding.
        let batch_result = self.encode_batch(&[text], config).await?;
        batch_result
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("Batch encoding returned no results for a single item."))
    }
}

// Now, we provide a GENERIC implementation that works for ANY EncoderLanguageModel.
// This is the key to reusability.
#[async_trait(?Send)]
impl<T: EncoderLanguageModel + Sync> SentenceEncoderModel for T {
    async fn encode_batch(&self, texts: &[&str], config: &EncodingConfig) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Get the base hidden states from the core trait.
        let (hidden_states, attention_mask) = self.get_hidden_states_batch(texts).await?;

        // 2. Apply the application-specific pooling logic.
        // Apply pooling to the whole batch
        let mut pooled = match config.pooling_strategy {
            PoolingStrategy::Cls => {
                // Use [CLS] token (first token of each item in the batch)
                hidden_states.slice(ndarray::s![.., 0, ..]).to_owned()
            }
            PoolingStrategy::Mean => mean_pool(&hidden_states, &attention_mask)?,
            PoolingStrategy::LastToken => last_token_pool(&hidden_states, &attention_mask)?,
            PoolingStrategy::Max => max_pool(&hidden_states, &attention_mask)?,
            _ => {
                return Err(anyhow!(
                    "Unknown pooling strategy: {}",
                    config.pooling_strategy
                ));
            }
        };

        // 3. Apply the application-specific normalization logic.
        if config.normalize {
            l2_normalize_inplace(&mut pooled);
        }

        Ok(pooled.outer_iter().map(|row| row.to_vec()).collect())
    }
}

// ============================================================================
// CPU ENCODER
// ============================================================================

/// Output from a CPU encoder.
#[derive(Clone, Debug)]
pub struct CpuEncoderOutput {
    /// Final hidden states: `[batch_size, sequence_length, hidden_size]`
    pub last_hidden_state: Array3<f32>,
}

/// CPU-based transformer encoder trait.
///
/// Provides methods for embedding lookup, normalization, and layer execution
/// with support for partial layer execution (for hybrid CPU/GPU workflows).
///

/// ```
pub trait CpuEncoder: Send + Sync {
    /// Compute embeddings only (word + position + token_type).
    ///
    /// Does NOT apply the initial layer normalization.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs `[batch_size, sequence_length]`
    /// * `token_type_ids` - Optional token type IDs for models like BERT
    ///
    /// # Returns
    /// Hidden states `[batch_size, sequence_length, hidden_size]`
    fn embed(&self, input_ids: &Array2<u32>, token_type_ids: Option<&Array2<u32>>) -> Array3<f32>;

    /// Compute embeddings + initial normalization.
    ///
    /// This produces hidden states ready to be processed by encoder layers.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs `[batch_size, sequence_length]`
    /// * `token_type_ids` - Optional token type IDs
    ///
    /// # Returns
    /// Normalized hidden states `[batch_size, sequence_length, hidden_size]`
    fn embed_and_normalize(
        &self,
        input_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
    ) -> Array3<f32>;

    /// Run layers `[start_layer, end_layer)` on pre-computed hidden states.
    ///
    /// Useful for:
    /// - Hybrid CPU/GPU execution
    /// - Layer-by-layer debugging
    /// - Partial model execution
    ///
    /// # Arguments
    /// * `hidden_states` - Input hidden states `[batch_size, seq_len, hidden_size]`
    /// * `attention_mask` - Attention mask `[batch_size, seq_len]`
    /// * `start_layer` - First layer to execute (inclusive)
    /// * `end_layer` - Last layer to execute (exclusive)
    ///
    /// # Returns
    /// Hidden states after processing through the specified layers
    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>>;

    /// Number of encoder layers in this model.
    fn num_layers(&self) -> usize;

    /// Hidden dimension of the model.
    fn hidden_size(&self) -> usize;

    /// Full forward pass through the encoder.
    ///
    /// Default implementation calls embed_and_normalize + forward_layers(0, num_layers).
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs `[batch_size, sequence_length]`
    /// * `attention_mask` - Attention mask `[batch_size, sequence_length]`
    /// * `token_type_ids` - Optional token type IDs
    ///
    /// # Returns
    /// Encoder output containing the final hidden states
    fn forward(
        &self,
        input_ids: &Array2<u32>,
        attention_mask: &Array2<f32>,
        token_type_ids: Option<&Array2<u32>>,
    ) -> Result<CpuEncoderOutput> {
        let hidden = self.embed_and_normalize(input_ids, token_type_ids);
        let output = self.forward_layers(&hidden, attention_mask, 0, self.num_layers())?;
        Ok(CpuEncoderOutput {
            last_hidden_state: output,
        })
    }
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
        input: ModelInput<'_>,
        token_type_ids: Option<ModelInput<'_>>,
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
        input: ModelInput<'_>,
        token_type_ids: Option<ModelInput<'_>>,
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
        input: ModelInput<'_>,
        attention_mask: &GpuTensor,
        token_type_ids: Option<ModelInput<'_>>,
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

pub trait CpuEncoderOps: Send + Sync {
    fn encoder(&self) -> &dyn CpuEncoder;
}

pub trait GpuEncoderOps: Send + Sync {
    fn encoder(&self) -> &dyn GpuEncoder;
}

pub fn l2_normalize_inplace(embeddings: &mut Array2<f32>) {
    for mut row in embeddings.rows_mut() {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            row /= norm;
        }
    }
}
