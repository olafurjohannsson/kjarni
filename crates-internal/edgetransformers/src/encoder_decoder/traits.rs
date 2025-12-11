use crate::cache::Cache;
use crate::gpu_ops::GpuTensor;
// use crate::models::base::EncoderDecoderLanguageModel;

use crate::encoder::prelude::*;
use crate::gpu_ops::GpuTensorPool;
use crate::linear_layer::LinearLayer;
use crate::models::base::{GenerationConfig, LanguageModel};
use crate::prelude::*;
// use crate::models::base::De
use crate::traits::DecoderOutput;
use crate::traits::EncoderOutput;
use anyhow::Result;
use async_trait::async_trait;
use bytemuck;
use ndarray::{Array1, Array2, Array3};

pub struct StepInput<'a, T> {
    pub tokens: &'a T,
    pub encoder_state: Option<&'a T>,
    pub attention_mask: &'a T,
}
pub trait HasShape {
    fn shape(&self) -> &[usize];
}
impl HasShape for GpuTensor {
    fn shape(&self) -> &[usize] {
        self.shape()
    }
}
impl<S: ndarray::Data, D: ndarray::Dimension> HasShape for ndarray::ArrayBase<S, D> {
    fn shape(&self) -> &[usize] {
        self.shape()
    }
}
// #[async_trait(?Send)]
// pub trait GenerationBackend: Send + Sync {
//     type Cache: Cache;
//     type Tensor: Send + Sync + HasShape;

//     // async fn encode(&self, model: &dyn LanguageModel, tokens: &[u32]) -> Result<Self::Tensor> ;

//     async fn forward<'a>(
//         &'a self,
//         model: &'a dyn EncoderDecoderLanguageModel,
//         inputs: StepInput<'a, Self::Tensor>,
//         cache: &'a mut dyn Cache,
//     ) -> Result<Array3<f32>>;

//     fn create_token_tensor(&self, tokens: &[u32], num_beams: usize) -> Result<Self::Tensor>;
//     fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_tokens: &[u32]) -> Result<()>;
//     fn prepare_encoder_state(
//         &self,
//         model: &dyn EncoderDecoderLanguageModel,
//         encoder_output: &EncoderOutput,
//     ) -> Result<Self::Tensor>;
//     fn prepare_attention_mask(&self, seq_len: usize, num_beams: usize) -> Result<Self::Tensor>;
//     fn reorder_cache(&self, cache: &mut dyn Cache, indices: &[usize]) -> Result<()>;
// }

/// Trait for encoder-decoder models (BART, T5, etc.), acting as a pure data accessor.
///
/// This trait provides a unified interface for a backend to access the necessary
/// components of a model (CPU or GPU versions), without knowing the model's
/// internal implementation.
#[async_trait]
pub trait EncoderDecoderLanguageModel: LanguageModel {
    // --- CPU Component Accessors ---

    /// Returns a reference to the model's CPU encoder, if available.
    fn cpu_encoder(&self) -> Result<&dyn CpuEncoder>;

    fn encoder(&self) -> Result<&dyn Encoder<Input=Array2<u32>, Output=EncoderOutput>>;

    /// Returns a reference to the model's CPU decoder, if available.
    fn cpu_decoder(&self) -> Result<&dyn CrossAttentionDecoder<
        TokenInput=Array2<u32>,
        EncoderStateInput=Array3<f32>,
        MaskInput=Array2<f32>,
        Output=DecoderOutput,
    >>;

    // --- GPU Component Accessors ---

    /// Returns a reference to the model's GPU encoder, if available.
    fn gpu_encoder(&self) -> Result<&dyn GpuEncoder>;

    /// Returns a reference to the model's GPU cross-attention decoder, if available.
    fn gpu_decoder(&self) -> Result<&dyn GpuCrossAttentionDecoder>;

    // --- LM Head Accessors (Backend Agnostic) ---

    /// Returns a reference to the CPU `LinearLayer` for the LM head.
    /// Used by the `CpuBackend`.
    fn lm_head_layer(&self) -> &LinearLayer;

    /// Returns a reference to the GPU tensor for the LM head weights.
    /// The tensor should have the shape `[vocab_size, hidden_size]`.
    /// Used by the `GpuBackend`.
    fn gpu_lm_head_weights(&self) -> Result<&GpuTensor>;

    /// Returns a reference to the optional CPU final logits bias.
    /// Used by the `CpuBackend`.
    fn final_logits_bias(&self) -> Option<&Array1<f32>>;

    /// Returns a reference to the optional GPU tensor for the final logits bias.
    /// Used by the `GpuBackend`.
    fn gpu_final_logits_bias(&self) -> Result<Option<&GpuTensor>>;

    // --- Configuration Accessors ---

    /// The token ID that should be used to start the decoding process.
    fn decoder_start_token_id(&self) -> u32;

    /// Returns the default generation configuration for this model.
    fn get_default_generation_config(&self) -> GenerationConfig;
}

/// Defines the synchronous interface for a GPU-native cross-attention decoder.
///
/// This component's `forward` method is responsible for recording the GPU commands
/// for a single decoder pass into a command encoder provided by a backend.
#[async_trait(?Send)]
pub trait GpuCrossAttentionDecoder: Send + Sync {
    /// Pre-computes the Key and Value matrices for cross-attention from the encoder's output.
    ///
    /// This is a critical optimization that avoids redundant projections in the generation loop.
    ///
    /// # Returns
    /// A vector of (Key, Value) `GpuTensor` tuples, one for each decoder layer.
    fn precompute_cross_attention_kv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        encoder_hidden_states: &GpuTensor,
    ) -> Result<Vec<(GpuTensor, GpuTensor)>>;

    /// Records the GPU commands for a forward pass.
    ///
    /// # Arguments
    /// * `cross_attention_kv_cache` - The pre-computed K/V matrices from the encoder.
    ///
    /// # Returns
    /// A `GpuTensor` representing the final hidden states of the decoder, resident on the GPU.
    fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        decoder_input_ids: &GpuTensor,
        // This is now optional; if the cache is provided, this can be None.
        encoder_hidden_states: Option<&GpuTensor>,
        cross_attention_kv_cache: Option<&Vec<(GpuTensor, GpuTensor)>>,
        decoder_attention_mask: &GpuTensor,
        cache: &mut dyn Cache,
    ) -> Result<GpuTensor>;
}

/// Defines the asynchronous interface for a decoder that uses cross-attention (e.g., BART Decoder).
///
/// This type of decoder attends to two sources: its own previously generated tokens
/// (self-attention) and the output of an encoder (cross-attention).
#[async_trait(?Send)]
pub trait CrossAttentionDecoder: Send + Sync + TransformerModel {
    type TokenInput;
    type EncoderStateInput;
    type MaskInput;
    type Output;

    /// Asynchronously performs a forward pass through the full encoder-decoder stack.
    async fn forward<'a>(
        &self,
        decoder_input_ids: &Self::TokenInput,
        encoder_hidden_states: &'a Self::EncoderStateInput,
        encoder_attention_mask: Option<&'a Self::MaskInput>,
        decoder_attention_mask: Option<&'a Self::MaskInput>,
        cache: Option<&mut dyn Cache>,
        // NEW: Optional pre-computed Cross KV
        // Vector of tuples (K, V) matching the layers
        cross_kv_caches: Option<&Vec<(ndarray::Array4<f32>, ndarray::Array4<f32>)>>,
    ) -> Result<Self::Output>;

    fn precompute_cross_attention_kv(
        &self,
        encoder_state: &Self::EncoderStateInput,
    ) -> Result<Vec<(ndarray::Array4<f32>, ndarray::Array4<f32>)>> {
        // Default implementation can return empty if not optimized,
        // but for CPU backend we need this implemented.
        Err(anyhow::anyhow!("Pre-computation not implemented for this decoder"))
    }
    // fn as_any(&self) -> &dyn Any;
}

#[async_trait(?Send)]
pub trait EncoderDecoderGenerationBackend: Send + Sync {
    // REMOVED: `type Cache` is no longer needed here.
    type Tensor: Send + Sync;

    async fn encode(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        tokens: &[u32],
        num_beams: usize,
    ) -> Result<Self::Tensor>;

    // UPDATED: Now generic over the Cache type `C`.
    async fn decode_step(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        decoder_tokens: &Self::Tensor,
        encoder_state: &Self::Tensor,
        cache: &mut dyn Cache,
    ) -> Result<Array3<f32>>;

    fn create_token_tensor(&self, tokens: &[u32], num_beams: usize) -> Result<Self::Tensor>;
    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_tokens: &[u32]) -> Result<()>;

    fn reorder_cache(&self, cache: &mut dyn Cache, indices: &[usize]) -> Result<()>;
}