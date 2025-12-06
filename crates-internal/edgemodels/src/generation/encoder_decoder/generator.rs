use crate::generation::encoder_decoder::{CpuBackend, GpuBackend};
use anyhow::{anyhow, Result};
use async_stream::try_stream;
use bytemuck;
use edgetransformers::common::{StreamedToken, TokenType};
use edgetransformers::encoder_decoder::{run_beam_search, run_beam_search_stream, HasShape};

use edgetransformers::models::base::{
    DecodingStrategy, EncoderDecoderLanguageModel, GenerationConfig, LanguageModel,
};
use edgetransformers::prelude::*;
use edgetransformers::traits::EncoderOutput;
use futures_core::stream::Stream;
use ndarray::Array2;

pub enum AnySeq2SeqBackend {
    Cpu(CpuBackend),
    Gpu(GpuBackend),
}

pub struct Seq2SeqGenerator {
    pub model: Box<dyn EncoderDecoderLanguageModel>,
    backend: AnySeq2SeqBackend,
}

impl Seq2SeqGenerator {
    pub fn new(model: Box<dyn EncoderDecoderLanguageModel>) -> Result<Self> {
        let backend = match model.device() {
            Device::Cpu => AnySeq2SeqBackend::Cpu(CpuBackend),
            Device::Wgpu => {
                let context = model
                    .context()
                    .ok_or_else(|| anyhow!("GPU model missing WgpuContext"))?;
                AnySeq2SeqBackend::Gpu(GpuBackend::new(context)?)
            }
        };

        Ok(Self { model, backend })
    }

    pub async fn generate(&self, input_text: &str, config: &GenerationConfig) -> Result<String> {
        let encoder_output = self.encode_input(input_text).await?;
        self.generate_from_encoding(&encoder_output, config).await
    }

    pub async fn encode_input(&self, text: &str) -> Result<EncoderOutput> {
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
    fn prepare_beam_state(
        &self,
        encoder_output: &EncoderOutput,
        num_beams: usize,
    ) -> Result<EncoderOutput> {
        if num_beams == 1 {
            // No copy needed for greedy search
            return Ok(encoder_output.clone());
        }

        let original = &encoder_output.last_hidden_state;
        let expanded = original
            .broadcast((num_beams, original.shape()[1], original.shape()[2]))
            .ok_or_else(|| anyhow!("Failed to broadcast encoder state"))?
            .to_owned();

        Ok(EncoderOutput {
            last_hidden_state: expanded,
        })
    }
    pub async fn generate_from_encoding(
        &self,
        encoder_output: &EncoderOutput,
        config: &GenerationConfig,
    ) -> Result<String> {
        let t_start = std::time::Instant::now();

        // 1. Get Params
        let num_beams = match &config.strategy {
            DecodingStrategy::BeamSearch(p) => p.num_beams,
            DecodingStrategy::Greedy => 1,
            _ => return Err(anyhow!("Unsupported strategy")),
        };

        // 2. Prepare Data (Device Agnostic)
        let beam_encoder_output = self.prepare_beam_state(encoder_output, num_beams)?;

        // 3. Dispatch (Code is now much smaller)
        let result = match &self.backend {
            AnySeq2SeqBackend::Cpu(backend) => {
                run_beam_search(self.model.as_ref(), backend, &beam_encoder_output, config).await
            }
            AnySeq2SeqBackend::Gpu(backend) => {
                run_beam_search(self.model.as_ref(), backend, &beam_encoder_output, config).await
            }
        };

        log::info!("[Seq2Seq] Total Time: {:?}", t_start.elapsed());
        result
    }
    pub async fn generate_stream_from_encoding(
        &self,
        encoder_output: &EncoderOutput,
        config: &GenerationConfig,
    ) -> Result<impl Stream<Item=Result<StreamedToken>>> {
        let num_beams = match &config.strategy {
            DecodingStrategy::BeamSearch(params) => params.num_beams,
            DecodingStrategy::Greedy => 1,
            _ => return Err(anyhow!("Only BeamSearch and Greedy are supported.")),
        };

        let original_encoder_state = &encoder_output.last_hidden_state;

        let expanded_encoder_state = original_encoder_state
            .broadcast((
                num_beams,
                original_encoder_state.shape()[1],
                original_encoder_state.shape()[2],
            ))
            .unwrap()
            .to_owned();

        let expanded_encoder_output = EncoderOutput {
            last_hidden_state: expanded_encoder_state,
        };

        // Match on backend and call run_beam_search with concrete type
        let final_text = match &self.backend {
            AnySeq2SeqBackend::Cpu(backend) => {
                run_beam_search(
                    self.model.as_ref(),
                    backend, // Concrete CpuBackend
                    &expanded_encoder_output,
                    config,
                )
                    .await?
            }
            AnySeq2SeqBackend::Gpu(backend) => {
                run_beam_search(
                    self.model.as_ref(),
                    backend, // Concrete GpuBackend
                    &expanded_encoder_output,
                    config,
                )
                    .await?
            }
        };

        // Tokenize and stream
        let tokenizer = self.model.tokenizer();
        let encoding = tokenizer
            .encode(final_text.as_str(), false)
            .map_err(|e| anyhow!(e))?;
        let token_ids = encoding.get_ids().to_vec();

        Ok(try_stream! {
            for token_id in token_ids {
                let decoded = tokenizer.decode(&[token_id], false).map_err(|e| anyhow!(e))?;
                yield StreamedToken {
                    text: decoded,
                    id: token_id,
                    token_type: TokenType::Generated,
                };
            }
        })
    }

    pub async fn generate_stream<'a>(
        &'a self,
        input_text: &'a str,
        config: &'a GenerationConfig,
    ) -> impl Stream<Item=Result<StreamedToken>> + 'a {
        try_stream! {
            let encoder_output = self.encode_input(input_text).await?;

            let num_beams = match &config.strategy {
                DecodingStrategy::BeamSearch(params) => params.num_beams,
                DecodingStrategy::Greedy => 1,
                _ => Err(anyhow!("Only BeamSearch and Greedy supported"))?,
            };

            let expanded = encoder_output.last_hidden_state
                .broadcast((num_beams, encoder_output.last_hidden_state.shape()[1], encoder_output.last_hidden_state.shape()[2]))
                .ok_or_else(|| anyhow!("Broadcast failed"))?
                .to_owned();

            let expanded_output = EncoderOutput { last_hidden_state: expanded };

            match &self.backend {
                AnySeq2SeqBackend::Cpu(backend) => {
                    let stream = run_beam_search_stream(self.model.as_ref(), backend, &expanded_output, config);
                    for await token in stream {
                        yield token?;
                    }
                }
                AnySeq2SeqBackend::Gpu(backend) => {
                    let stream = run_beam_search_stream(self.model.as_ref(), backend, &expanded_output, config);
                    for await token in stream {
                        yield token?;
                    }
                }
            }
        }
    }
}
