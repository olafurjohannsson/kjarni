//! LLaMA-style decoder-only language model.
//!
//! This module provides the `LlamaModel`, a model container responsible for loading
//! weights and configuration for Llama and its variants.
//!
//! The actual text generation is handled by the generic `Generator` struct.


use anyhow::{Result, anyhow};
use async_trait::async_trait;
use edgetransformers::decoder::TransformerDecoder;
use edgetransformers::gpu_ops::GpuTensor;
use edgetransformers::gpu_ops::blocks::rope::GpuRoPE;
use edgetransformers::models::base::GpuDecoder;
use edgetransformers::models::base::{AutoregressiveLoop, DecodingStrategy};
use edgetransformers::models::download_model_files;
use edgetransformers::models::{DecoderLanguageModel, LanguageModel, ModelArchitecture, ModelType};
use edgetransformers::prelude::*;
use edgetransformers::rope::RoPE;
use edgetransformers::traits::{Decoder, DecoderArchitecture, DecoderOutput, LanguageModelConfig};
use edgetransformers::weights::ModelWeights;
use ndarray::{Array2, Array3, s};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;


pub mod config;
pub mod gpu_decoder;
pub mod model;
pub mod gguf_loader;

#[cfg(test)]
mod tests;