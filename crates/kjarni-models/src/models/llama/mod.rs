//! LLaMA-style decoder-only language model.
//!
//! This module provides the `LlamaModel`, a model container responsible for loading
//! weights and configuration for Llama and its variants.
//!
//! The actual text generation is handled by the generic `Generator` struct.



pub mod chat_template;
pub mod config;
pub mod gpu_decoder;
pub mod cpu_decoder;
pub mod model;

pub use model::LlamaModel;
pub use config::LlamaConfig;
pub use chat_template::{Llama3ChatTemplate, Llama2ChatTemplate};

#[cfg(test)]
mod tests;