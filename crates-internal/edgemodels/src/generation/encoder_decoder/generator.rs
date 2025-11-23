use crate::generation::common::{StreamedToken, TokenType};
use crate::generation::encoder_decoder::{run_beam_search, CpuBackend, GpuBackend, HasShape};
use anyhow::{anyhow, Result};
use async_stream::try_stream;
use bytemuck;
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
            Device::Cpu => {
                AnySeq2SeqBackend::Cpu(CpuBackend)
            }
            Device::Wgpu => {
                let context = model.context()
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
            )).unwrap()
            .to_owned();

        let expanded_encoder_output = EncoderOutput {
            last_hidden_state: expanded_encoder_state,
        };

        // Match on backend and call run_beam_search with concrete type
        let final_text = match &self.backend {
            AnySeq2SeqBackend::Cpu(backend) => {
                run_beam_search(
                    self.model.as_ref(),
                    &self.backend,  // Concrete CpuBackend
                    &expanded_encoder_output,
                    config,
                ).await?
            }
            AnySeq2SeqBackend::Gpu(backend) => {
                run_beam_search(
                    self.model.as_ref(),
                    &self.backend,  // Concrete GpuBackend
                    &expanded_encoder_output,
                    config,
                ).await?
            }
        };

        // Tokenize and stream
        let tokenizer = self.model.tokenizer();
        let encoding = tokenizer.encode(&final_text, false).map_err(|e| anyhow!(e))?;
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
}
