
//! Core transformer components for building transformer-based models
//!
//! This crate provides the fundamental building blocks for transformer architectures
//! without model-specific implementations.

//! EdgeTransformers: Fast transformer models for Rust


pub mod activations;
pub mod attention;
pub mod cache;
pub mod decoder;
pub mod embeddings;
pub mod adaptive_embeddings;
pub mod encoder;
pub mod encoder_decoder;
pub mod feedforward;
pub mod decoder_attention;
pub mod linear_layer;
pub mod normalization;
pub mod gpu_context;
pub mod gpu_ops;
pub mod models;
pub mod pooling;
pub mod traits;
pub mod utils;
pub mod weights;
pub mod decoder_cross_attn_layer;
pub mod decoder_layer;
pub mod encoder_layer;
pub mod rope;
pub mod decoder_cross_attn;


// Re-export commonly used items
pub use crate::{
    attention::MultiHeadAttention,
    embeddings::Embeddings,
    feedforward::FeedForward,
    normalization::Normalization,
    pooling::{PoolingStrategy, cls_pool, last_token_pool, max_pool, mean_pool},
    weights::ModelWeights,
};
pub use cache::{Cache, CpuKVCache, GpuKVCache};
pub use gpu_context::WgpuContext;
pub use traits::{
    CrossAttentionDecoder, Decoder, Device, Encoder, TransformerConfig, TransformerModel,
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