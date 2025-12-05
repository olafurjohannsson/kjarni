use anyhow::{anyhow, Result};
use async_trait::async_trait;
use bytemuck;
use edgetransformers::cache::{Cache, GpuBeamKVCache};
use edgetransformers::gpu_context::WgpuContext;
use edgetransformers::gpu_ops::{GpuTensor, GpuTensorPool};
use edgetransformers::models::base::{
    DecodingStrategy, EncoderDecoderLanguageModel, GenerationConfig, LanguageModel,
};
// use crate::generation::
use edgetransformers::prelude::*;
use edgetransformers::traits::EncoderOutput;
use ndarray::{s, Array1, Array2, Array3};
use std::sync::Arc;
use tokio::sync::Mutex;

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
#[async_trait(?Send)]
pub trait GenerationBackend: Send + Sync {
    type Cache: Cache;
    type Tensor: Send + Sync + HasShape;

    async fn forward<'a>(
        &'a self,
        model: &'a dyn EncoderDecoderLanguageModel,
        inputs: StepInput<'a, Self::Tensor>,
        cache: &'a mut dyn Cache,
    ) -> Result<Array3<f32>>;

    fn create_token_tensor(&self, tokens: &[u32], num_beams: usize) -> Result<Self::Tensor>;
    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_tokens: &[u32]) -> Result<()>;
    fn prepare_encoder_state(&self, model: &dyn EncoderDecoderLanguageModel, encoder_output: &EncoderOutput) -> Result<Self::Tensor>;
    fn prepare_attention_mask(&self, seq_len: usize, num_beams: usize) -> Result<Self::Tensor>;
    fn reorder_cache(&self, cache: &mut dyn Cache, indices: &[usize]) -> Result<()>;
}
