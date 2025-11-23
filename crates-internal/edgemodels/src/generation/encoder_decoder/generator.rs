use anyhow::{anyhow, Result};
use async_trait::async_trait;
use bytemuck;
use edgetransformers::cache::{Cache, GpuBeamKVCache, CpuBeamKVCache};
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
use crate::generation::encoder_decoder::{
    GenerationBackend,
    StepInput,
    GpuBackend,
    HasShape,
    run_beam_search,
    find_best_beams_and_get_indices
};

/// An enum to wrap different ndarray tensor types to satisfy the GenerationBackend trait,
/// which requires a single associated `Tensor` type.
#[derive(Debug)]
pub enum CpuTensor {
    U32(Array2<u32>),
    F32_2D(Array2<f32>),
    F32_3D(Array3<f32>),
}

impl HasShape for CpuTensor {
    fn shape(&self) -> &[usize] {
        match self {
            CpuTensor::U32(a) => a.shape(),
            CpuTensor::F32_2D(a) => a.shape(),
            CpuTensor::F32_3D(a) => a.shape(),
        }
    }
}

struct CpuBackend;

#[async_trait(?Send)]
impl GenerationBackend for CpuBackend {
    type Cache = CpuBeamKVCache;
    type Tensor = CpuTensor;

    async fn forward<'a>(
        &'a self,
        model: &'a dyn EncoderDecoderLanguageModel,
        inputs: StepInput<'a, Self::Tensor>,
        cache: &'a mut dyn Cache,
    ) -> Result<Array3<f32>> {
        // 1. Downcast the generic inputs to concrete ndarray types
        let tokens = match inputs.tokens {
            CpuTensor::U32(t) => t,
            _ => return Err(anyhow!("Invalid tensor type for tokens, expected U32")),
        };
        let encoder_state = match inputs.encoder_state.unwrap() {
            CpuTensor::F32_3D(s) => s,
            _ => return Err(anyhow!("Invalid tensor type for encoder_state, expected F32_3D")),
        };
        let attention_mask = match inputs.attention_mask {
            CpuTensor::F32_2D(m) => m,
            _ => return Err(anyhow!("Invalid tensor type for attention_mask, expected F32_2D")),
        };

        // 2. Call the model's CPU decoder
        let decoder_output = model
            .decoder()
            .forward(tokens, encoder_state, None, Some(attention_mask), Some(cache))
            .await?;

        Ok(decoder_output.last_hidden_state)
    }

    fn create_token_tensor(&self, tokens: &[u32], num_beams: usize) -> Result<Self::Tensor> {
        let seq_len = if num_beams > 0 { tokens.len() / num_beams } else { 0 };
        let tokens_ndarray = Array2::from_shape_vec((num_beams, seq_len), tokens.to_vec())?;
        Ok(CpuTensor::U32(tokens_ndarray))
    }

    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_tokens: &[u32]) -> Result<()> {
        let current_tensor = match tensor {
            CpuTensor::U32(t) => t,
            _ => return Err(anyhow!("Invalid tensor type for update_token_tensor, expected U32")),
        };
        // The new tokens represent the next single token for each beam.
        let new_tokens_ndarray =
            Array2::from_shape_vec((new_tokens.len(), 1), new_tokens.to_vec())?;
        *current_tensor = new_tokens_ndarray;
        Ok(())
    }

    fn prepare_encoder_state(&self, encoder_output: &EncoderOutput) -> Result<Self::Tensor> {
        // The encoder output is already on the CPU, just clone and wrap it.
        Ok(CpuTensor::F32_3D(encoder_output.last_hidden_state.clone()))
    }

    fn prepare_attention_mask(&self, seq_len: usize, num_beams: usize) -> Result<Self::Tensor> {
        let mask: Array2<f32> = Array2::ones((num_beams, seq_len));
        Ok(CpuTensor::F32_2D(mask))
    }

    fn reorder_cache(&self, cache: &mut dyn Cache, indices: &[usize]) -> Result<()> {
        let cpu_cache = cache
            .as_any_mut()
            .downcast_mut::<CpuBeamKVCache>() // <-- Change this to CpuBeamKVCache
            .ok_or_else(|| anyhow!("Failed to downcast to CpuBeamKVCache for reordering"))?;

        // This is now a simple, efficient call.
        cpu_cache.reorder(indices);

        Ok(())
    }
}

pub struct Seq2SeqGenerator {
    pub model: Box<dyn EncoderDecoderLanguageModel>,
}

impl Seq2SeqGenerator {
    pub fn new(model: Box<dyn EncoderDecoderLanguageModel>) -> Self {
        Self { model }
    }

    pub async fn generate(&self, input_text: &str, config: &GenerationConfig) -> Result<String> {
        let encoder_output = self.encode_input(input_text).await?;
        self.generate_from_encoding(&encoder_output, config).await
    }

    async fn encode_input(&self, text: &str) -> Result<EncoderOutput> {
        let encoding = self
            .model
            .tokenizer()
            .encode(text, true)
            .map_err(|e| anyhow!(e))?;
        let input_ids = Array2::from_shape_vec((1, encoding.len()), encoding.get_ids().to_vec())?;
        let attention_mask = Array2::ones(input_ids.dim());
        self.model
            .encoder()
            .forward(&input_ids, &attention_mask, None)
            .await
    }

    pub async fn generate_from_encoding(
        &self,
        encoder_output: &EncoderOutput,
        config: &GenerationConfig,
    ) -> Result<String> {
        match self.model.device() {
            Device::Cpu => {
                let backend = CpuBackend;

                // 1. Determine Beam Count
                let num_beams = match &config.strategy {
                    DecodingStrategy::BeamSearch(params) => params.num_beams,
                    DecodingStrategy::Greedy => 1,
                    _ => return Err(anyhow!("Only BeamSearch and Greedy are supported.")),
                };

                let original_encoder_state_cpu = &encoder_output.last_hidden_state;

                // 2. Expand encoder state to match num_beams
                let expanded_encoder_state_cpu = original_encoder_state_cpu
                    .broadcast((
                        num_beams,
                        original_encoder_state_cpu.shape()[1],
                        original_encoder_state_cpu.shape()[2],
                    ))
                    .unwrap()
                    .to_owned();

                let expanded_encoder_output = EncoderOutput {
                    last_hidden_state: expanded_encoder_state_cpu,
                    
                };

                // 3. Run the generation loop
                run_beam_search(
                    self.model.as_ref(),
                    backend,
                    &expanded_encoder_output,
                    config,
                )
                    .await
            }
            Device::Wgpu => {
                let context = self.model.context().unwrap();
                let backend = GpuBackend {
                    context: context.clone(),
                    pool: Arc::new(Mutex::new(GpuTensorPool::new(context))),
                };

                // 1. Determine Beam Count
                let num_beams = match &config.strategy {
                    DecodingStrategy::BeamSearch(params) => params.num_beams,
                    DecodingStrategy::Greedy => 1,
                    _ => return Err(anyhow!("Only BeamSearch and Greedy are supported.")),
                };

                let original_encoder_state_cpu = &encoder_output.last_hidden_state;

                // 2. Expand encoder state to match num_beams
                // (Even for greedy/1 beam, this ensures dimensions are consistent)
                let expanded_encoder_state_cpu = original_encoder_state_cpu
                    .broadcast((
                        num_beams,
                        original_encoder_state_cpu.shape()[1],
                        original_encoder_state_cpu.shape()[2],
                    ))
                    .unwrap()
                    .to_owned();

                let expanded_encoder_output = EncoderOutput {
                    last_hidden_state: expanded_encoder_state_cpu,
                };

                // 3. Run the generation loop
                run_beam_search(
                    self.model.as_ref(),
                    backend,
                    &expanded_encoder_output,
                    config,
                )
                    .await
            }
        }
    }
}
