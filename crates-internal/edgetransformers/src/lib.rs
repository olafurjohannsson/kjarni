//! Core transformer components for building transformer-based models
//!
//! This crate provides the fundamental building blocks for transformer architectures
//! without model-specific implementations.

//! EdgeTransformers: Fast transformer models for Rust

pub mod activations;
pub mod attention;
pub mod cache;
pub mod common;
pub mod decoder;
pub mod decoder_attention;
pub mod decoder_cross_attn_layer;
pub mod decoder_layer;
pub mod embeddings;
pub mod encoder;
pub mod encoder_decoder;
pub mod encoder_layer;
pub mod feedforward;
pub mod gpu_context;
pub mod gpu_ops;
pub mod linear_layer;
pub mod models;
pub mod normalization;
pub mod pooling;
pub mod rope;
pub mod traits;
pub mod utils;
pub mod weights;

// Re-export commonly used items
pub use crate::{
    attention::MultiHeadAttention,
    embeddings::Embeddings,
    feedforward::FeedForward,
    normalization::Normalization,
    pooling::{cls_pool, last_token_pool, max_pool, mean_pool, PoolingStrategy},
    weights::ModelWeights,
};
pub use cache::{Cache, CpuKVCache, GpuKVCache};
pub use gpu_context::WgpuContext;
pub use traits::{
    Decoder, Device, Encoder, TransformerConfig, TransformerModel,
};

// Re-export model traits and registry
pub use models::{
    DecoderLanguageModel, EncoderLanguageModel, LanguageModel, ModelArchitecture, ModelType,
    Seq2SeqLanguageModel,
};

// Prelude for easy imports
pub mod prelude {
    pub use crate::cache::{Cache, CpuKVCache, GpuKVCache};
    pub use crate::gpu_context::WgpuContext;
    pub use crate::models::{DecoderLanguageModel, EncoderLanguageModel, LanguageModel};
    pub use crate::traits::{Decoder, Device, Encoder, TransformerModel};
}

#[cfg(test)]
pub mod tests;

