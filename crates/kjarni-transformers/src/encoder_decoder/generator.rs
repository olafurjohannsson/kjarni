use crate::cache::Cache;
use crate::common::StreamedToken;
use crate::common::{DecodingStrategy, GenerationConfig};
use crate::encoder_decoder::cpu_backend::{self, CpuBackend};
use crate::encoder_decoder::gpu_backend::{self, GpuBackend};
use crate::encoder_decoder::traits::{
    EncoderDecoderGenerationBackend, EncoderDecoderLanguageModel,
};
use crate::encoder_decoder::{run_beam_search, run_beam_search_stream};
use crate::models::base::LanguageModel;
use crate::prelude::*;
use anyhow::{anyhow, Result};
use async_stream::try_stream;
use async_trait::async_trait;
use futures_core::stream::Stream;
use ndarray::Array3;
use std::any::Any;

pub enum AnyEncoderDecoderBackend {
    Cpu(CpuBackend),
    Gpu(GpuBackend),
}

#[async_trait(?Send)]
impl EncoderDecoderGenerationBackend for AnyEncoderDecoderBackend {
    // The associated Tensor type remains the same, using Box<dyn Any> for type erasure.
    type Tensor = Box<dyn Any + Send + Sync>;

    async fn encode(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        tokens: &[u32],
        num_beams: usize,
    ) -> Result<Self::Tensor> {
        match self {
            AnyEncoderDecoderBackend::Cpu(b) => {
                let tensor = b.encode(model, tokens, num_beams).await?;
                Ok(Box::new(tensor))
            }
            AnyEncoderDecoderBackend::Gpu(b) => {
                let tensor = b.encode(model, tokens, num_beams).await?;
                Ok(Box::new(tensor))
            }
        }
    }

    // The decode_step and reorder_cache methods are now generic over the Cache type.
    async fn decode_step(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        decoder_tokens: &Self::Tensor,
        encoder_state: &Self::Tensor,
        cache: &mut dyn Cache,
    ) -> Result<Array3<f32>> {
        match self {
            AnyEncoderDecoderBackend::Cpu(b) => {
                let tokens = decoder_tokens
                    .downcast_ref::<cpu_backend::CpuSeq2SeqState>()
                    .ok_or_else(|| anyhow!("Mismatched Tensor type for CpuBackend"))?;
                let state = encoder_state
                    .downcast_ref::<cpu_backend::CpuSeq2SeqState>()
                    .ok_or_else(|| anyhow!("Mismatched Tensor type for CpuBackend"))?;
                b.decode_step(model, tokens, state, cache).await
            }
            AnyEncoderDecoderBackend::Gpu(b) => {
                let tokens = decoder_tokens
                    .downcast_ref::<gpu_backend::GpuSeq2SeqState>()
                    .ok_or_else(|| anyhow!("Mismatched Tensor type for GpuBackend"))?;
                let state = encoder_state
                    .downcast_ref::<gpu_backend::GpuSeq2SeqState>()
                    .ok_or_else(|| anyhow!("Mismatched Tensor type for GpuBackend"))?;
                b.decode_step(model, tokens, state, cache).await
            }
        }
    }

    fn create_token_tensor(&self, tokens: &[u32], num_beams: usize) -> Result<Self::Tensor> {
        match self {
            AnyEncoderDecoderBackend::Cpu(b) => {
                let tensor = b.create_token_tensor(tokens, num_beams)?;
                Ok(Box::new(tensor))
            }
            AnyEncoderDecoderBackend::Gpu(b) => {
                let tensor = b.create_token_tensor(tokens, num_beams)?;
                Ok(Box::new(tensor))
            }
        }
    }

    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_tokens: &[u32]) -> Result<()> {
        match self {
            AnyEncoderDecoderBackend::Cpu(b) => {
                let concrete_tensor = tensor
                    .downcast_mut::<cpu_backend::CpuSeq2SeqState>()
                    .ok_or_else(|| anyhow!("Mismatched Tensor type for CpuBackend"))?;
                b.update_token_tensor(concrete_tensor, new_tokens)
            }
            AnyEncoderDecoderBackend::Gpu(b) => {
                let concrete_tensor = tensor
                    .downcast_mut::<gpu_backend::GpuSeq2SeqState>()
                    .ok_or_else(|| anyhow!("Mismatched Tensor type for GpuBackend"))?;
                b.update_token_tensor(concrete_tensor, new_tokens)
            }
        }
    }

    fn reorder_cache(&self, cache: &mut dyn Cache, indices: &[usize]) -> Result<()> {
        match self {
            AnyEncoderDecoderBackend::Cpu(b) => b.reorder_cache(cache, indices),
            AnyEncoderDecoderBackend::Gpu(b) => b.reorder_cache(cache, indices),
        }
    }
}

pub struct EncoderDecoderGenerator {
    pub model: Box<dyn EncoderDecoderLanguageModel>,
    backend: AnyEncoderDecoderBackend,
}

impl EncoderDecoderGenerator {
    pub fn new(model: Box<dyn EncoderDecoderLanguageModel>) -> Result<Self> {
        let backend = match model.device() {
            Device::Cpu => AnyEncoderDecoderBackend::Cpu(CpuBackend),
            Device::Wgpu => {
                let context = model
                    .context()
                    .ok_or_else(|| anyhow!("GPU model missing WgpuContext"))?;
                AnyEncoderDecoderBackend::Gpu(GpuBackend::new(context)?)
            }
        };
        Ok(Self { model, backend })
    }

    pub async fn generate(
        &self,
        input_text: &str,
        config: Option<&GenerationConfig>,
    ) -> Result<String> {
        let t_start = std::time::Instant::now();
        let generation_config =
            config.map_or_else(|| self.model.get_default_generation_config(), |c| c.clone());

        let result = run_beam_search(
            self.model.as_ref(),
            &self.backend,
            input_text,
            &generation_config,
        )
            .await;

        let elapsed = t_start.elapsed();
        if let Ok(ref text) = result {
            let num_tokens = self
                .model
                .tokenizer()
                .encode(text.as_str(), false)
                .map_or(0, |e| e.len());
            if num_tokens > 0 && elapsed.as_secs_f32() > 0.0 {
                let tps = num_tokens as f32 / elapsed.as_secs_f32();
                log::info!(
                    "[Seq2Seq] Generated {} tokens in {:?}. Speed: {:.2} t/s",
                    num_tokens,
                    elapsed,
                    tps
                );
            } else {
                log::info!("[Seq2Seq] Total Generation Time: {:?}", elapsed);
            }
        } else {
            log::info!("[Seq2Seq] Total Generation Time (failed): {:?}", elapsed);
        }

        result
    }

    // The method signature is now slightly different to accommodate the owned config
    pub fn generate_stream<'a>(
        &'a self,
        input_text: &'a str,
        config: Option<&GenerationConfig>,
    ) -> impl Stream<Item=Result<StreamedToken>> + 'a {
        let owned_config =
            config.map_or_else(|| self.model.get_default_generation_config(), |c| c.clone());

        // We use an async_stream block to build the stream. `async move` allows it
        // to take ownership of `owned_config`.
        try_stream! {
            if let DecodingStrategy::BeamSearch(params) = &owned_config.strategy {
                if params.num_beams > 1 {
                    log::warn!(
                        "Streaming with beam search is enabled. The optimal sequence may change \
                         during generation, potentially causing the streamed output to differ from \
                         the final result of the non-streaming `generate` method."
                    );
                }
            }

            // `run_beam_search_stream` is now called inside the stream block.
            let mut stream = run_beam_search_stream(
                self.model.as_ref(),
                &self.backend,
                input_text,
                &owned_config, // Move the owned config into the function
            );

            // We yield each item from the inner stream.
            // futures_util::pin_mut!(stream);
            // while let Some(token) = futures_util::StreamExt::next(&mut stream).await {
            //     yield token?;
            // }
            let t_start = std::time::Instant::now();
            let mut token_count = 0;

            futures_util::pin_mut!(stream);
            while let Some(token) = futures_util::StreamExt::next(&mut stream).await {
                token_count += 1;
                let elapsed = t_start.elapsed();
                if elapsed.as_secs_f32() > 0.0 {
                    let tps = token_count as f32 / elapsed.as_secs_f32();
                    log::info!("[Stream] Token #{}, Speed: {:.2} t/s", token_count, tps);
                }
                yield token?;
            }
        }
    }
}
