//! A generic, backend-agnostic transformer encoder-decoder implementation

mod beams;
mod cpu_backend;
mod generator;
pub mod traits;

pub use crate::{Cache, CpuKVCache};
pub use cpu_backend::CpuBackend;

pub mod config;

pub mod decoder_self_attn;



pub use crate::encoder_decoder::config::{SummarizationParams, TaskSpecificParams};
pub use crate::encoder_decoder::traits::{
    EncoderDecoderGenerationBackend, EncoderDecoderLanguageModel,
};
pub use beams::{run_beam_search, run_beam_search_stream, BeamHypothesis};
pub use cpu_backend::CpuSeq2SeqState;

pub use decoder_self_attn::DecoderSelfAttention;
pub use generator::{AnyEncoderDecoderBackend, EncoderDecoderGenerator};

