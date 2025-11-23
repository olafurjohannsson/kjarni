use anyhow::Result;
use async_trait::async_trait;
use edgetransformers::WgpuContext;
use edgetransformers::cache::{Cache, CpuKVCache, GpuKVCache};
use edgetransformers::gpu_ops::Kernel;
use edgetransformers::gpu_ops::primitives::bmm::GpuBatchedMatMul;
use edgetransformers::gpu_ops::primitives::layout::slice::GpuSlice;
use edgetransformers::gpu_ops::primitives::matmul::GpuMatMul;
use edgetransformers::gpu_ops::{GpuFrameContext, GpuTensor, GpuTensorPool};
use edgetransformers::models::DecoderLanguageModel;
pub use edgetransformers::models::base::{AutoregressiveLoop, DecodingStrategy, GenerationConfig};
use ndarray::{Array1, Array2, s};
use std::sync::Arc;
use tokio::sync::Mutex;
use wgpu::CommandEncoder;

pub trait IsTensor: Send + Sync {}

#[async_trait(?Send)]
pub trait DecoderGenerationBackend {
    type Tensor: IsTensor;

    /// Creates a tensor for the initial prompt tokens.
    fn prime_tokens(&self, tokens: &[u32]) -> Result<Self::Tensor>;

    /// Creates a tensor to hold the single token for iterative generation.
    fn new_token_tensor(&self) -> Result<Self::Tensor>;

    /// Efficiently updates the token tensor with the next token's ID.
    /// This is a key optimization that avoids new allocations.
    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_token_id: u32) -> Result<()>;

    /// Prepares the attention mask for a given sequence length.
    fn prepare_attention_mask(&self, seq_len: usize, max_len: usize) -> Result<Self::Tensor>;

    async fn prefill<'a>(
        &'a self,
        model: &'a dyn DecoderLanguageModel,
        initial_tokens: &[u32],
        cache: &'a mut dyn Cache,
    ) -> Result<Array1<f32>>;

    /// Decodes a single new token and returns the next set of logits.
    async fn decode_one<'a>(
        &'a self,
        model: &'a dyn DecoderLanguageModel,
        token_id: u32,  // Pass the single token ID
        seq_len: usize, // The new total sequence length
        cache: &'a mut dyn Cache,
    ) -> Result<Array1<f32>>;
}

// Make GpuTensor compatible with our marker trait
impl IsTensor for GpuTensor {}
impl IsTensor for Array2<u32> {}
impl IsTensor for Array2<f32> {}