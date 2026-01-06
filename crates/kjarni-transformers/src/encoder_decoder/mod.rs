//! A generic, backend-agnostic transformer encoder-decoder implementation.
//!
//! This module provides `TransformerEncoderDecoder`, a reusable component that can represent
//! various seq2seq models like BART, T5, etc. It is designed to be
//! backend-aware, containing either a CPU or a GPU implementation.
//!
//! The model is constructed generically by relying on the `EncoderDecoderArchitecture`
//! trait, which provides the specific weight names and hyperparameters.

mod beams;
mod cpu_backend;
mod generator;
mod gpu_backend;
pub mod traits;

pub use crate::{Cache, CpuKVCache, GpuKVCache};
pub use cpu_backend::CpuBackend;
pub use gpu_backend::GpuBackend;
mod config;
pub mod decoder_cross_attn;
pub mod decoder_cross_attn_layer;
pub mod decoder_self_attn;
pub mod cpu_encoder;

pub mod relative_position_bias;
pub use crate::encoder_decoder::config::{SummarizationParams, TaskSpecificParams};
pub use crate::encoder_decoder::traits::{
    EncoderDecoderGenerationBackend, EncoderDecoderLanguageModel,
};
pub use beams::{run_beam_search, run_beam_search_stream, BeamHypothesis};
pub use cpu_backend::CpuSeq2SeqState;
pub use decoder_cross_attn::DecoderCrossAttention;
pub use decoder_self_attn::DecoderSelfAttention;
pub use generator::{AnyEncoderDecoderBackend, EncoderDecoderGenerator};
pub use gpu_backend::GpuSeq2SeqState;


#[cfg(test)]
mod generator_test;
pub mod cpu_decoder;