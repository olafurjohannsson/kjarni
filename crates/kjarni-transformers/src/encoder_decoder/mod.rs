//! A generic, backend-agnostic transformer encoder-decoder implementation.
//!
//! This module provides `TransformerEncoderDecoder`, a reusable component that can represent
//! various seq2seq models like BART, T5, etc. It is designed to be
//! backend-aware, containing either a CPU or a GPU implementation.
//!
//! The model is constructed generically by relying on the `EncoderDecoderArchitecture`
//! trait, which provides the specific weight names and hyperparameters.

mod cpu;
mod gpu;
mod beams;
mod generator;
mod gpu_backend;
mod cpu_backend;
pub mod traits;

pub use cpu_backend::CpuBackend;
pub use gpu_backend::GpuBackend;
pub use crate::{Cache, CpuKVCache, GpuKVCache};
pub mod decoder_cross_attn;
pub mod decoder_self_attn;
pub mod decoder_cross_attn_layer;
pub use decoder_cross_attn::DecoderCrossAttention;
pub use decoder_self_attn::DecoderSelfAttention;
pub use generator::{Seq2SeqGenerator, AnyEncoderDecoderBackend};
pub use beams::{run_beam_search, run_beam_search_stream, BeamHypothesis};
pub use gpu_backend::GpuSeq2SeqState;
pub use cpu_backend::CpuSeq2SeqState;
pub use crate::encoder_decoder::traits::{EncoderDecoderGenerationBackend,
    EncoderDecoderLanguageModel
};
use serde::Deserialize;
#[derive(Debug, Clone, Deserialize, Copy)]
#[allow(non_snake_case)] // To allow serde to match the camelCase keys
pub struct SummarizationParams {
    pub early_stopping: bool,
    pub length_penalty: f32,
    pub max_length: usize,
    pub min_length: usize,
    pub no_repeat_ngram_size: usize,
    pub num_beams: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(non_snake_case)]
pub struct TaskSpecificParams {
    pub summarization: SummarizationParams,
}
