//! GPT-2 style decoder-only language model.
//!
//! This module provides the `Gpt2Model`, a model container responsible for loading
//! weights and configuration for models like GPT-2, DistilGPT2, etc.
//!
//! The actual text generation is handled by the generic `Generator` struct,
//! which can operate on any model that implements the `DecoderLanguageModel` trait.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use edgetransformers::decoder::TransformerDecoder;
use edgetransformers::models::base::{AutoregressiveLoop, DecodingStrategy};
use edgetransformers::models::download_model_files;
use edgetransformers::models::{DecoderLanguageModel, LanguageModel, ModelArchitecture, ModelType};
use edgetransformers::prelude::*;
use edgetransformers::traits::{Decoder, DecoderArchitecture, DecoderOutput, LanguageModelConfig};
use edgetransformers::weights::ModelWeights;
use ndarray::{Array2, Array3};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

mod config;
mod model;
mod gpu_decoder;

#[cfg(test)]
mod tests;


pub use model::Gpt2Model;
pub use config::Gpt2Config;