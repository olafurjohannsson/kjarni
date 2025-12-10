use anyhow::Result;
use edgetransformers::models::{
    base::DecodingStrategy,
    ModelType,
};
use edgetransformers::prelude::*;
use edgetransformers::traits::{
    Encoder, EncoderDecoderArchitecture,
    LanguageModelConfig, TransformerModel,
};

use edgetransformers::TransformerConfig;
use ndarray::{s, Array1, Array2, Array3};
use serde::Deserialize;
use std::ops::AddAssign;
use std::sync::Arc;

// pub mod seq2seq_model;


pub use crate::models::bart::config::BartConfig;
// pub use seq2seq_model::{AnySeq2SeqModel, Seq2SeqModel};


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

// #[cfg(test)]
// mod tests;

