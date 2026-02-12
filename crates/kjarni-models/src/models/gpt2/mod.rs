//! GPT-2 style decoder-only language model.
//!
//! This module provides the `Gpt2Model`, a model container responsible for loading
//! weights and configuration for models like GPT-2, DistilGPT2, etc.
//!
//! The actual text generation is handled by the generic `Generator` struct,
//! which can operate on any model that implements the `DecoderLanguageModel` trait.

mod config;
mod model;
mod gpu_decoder;
mod cpu_decoder;

#[cfg(test)]
mod tests;


pub use model::Gpt2Model;
pub use config::Gpt2Config;