//! LLaMA-style decoder-only language model.
//!
//! This module provides the `LlamaModel`, a model container responsible for loading
//! weights and configuration for Llama and its variants.
//!
//! The actual text generation is handled by the generic `Generator` struct.



pub mod config;
pub mod model;

pub use model::MistralModel;
pub use config::MistralConfig;
