//! Core transformer components for building transformer-based models
//!
//! This crate provides the fundamental building blocks for transformer architectures
//! without model-specific implementations.

//! Kjarni Transformers: Fast transformer models for Rust

pub mod activations;
pub mod attention;
pub mod cache;
pub mod common;
pub mod decoder;
pub mod embeddings;
pub mod encoder;
pub mod encoder_decoder;
pub mod feedforward;
pub mod gpu_context;
pub mod gpu_ops;
pub mod linear_layer;
pub mod linear_layer_old;
pub mod models;
pub mod normalization;
pub mod pooling;
pub mod rope;
pub mod traits;
pub mod utils;
pub mod weights;
pub mod weights_old;
pub mod tensor;
pub mod chat;

// Re-export commonly used items
pub use crate::{
    chat::templates::{ChatTemplate, Conversation, Message, Role},
    attention::MultiHeadAttention,
    embeddings::Embeddings,
    feedforward::FeedForward,
    normalization::Normalization,
    pooling::{cls_pool, last_token_pool, max_pool, mean_pool, PoolingStrategy},
    weights_old::ModelWeights,
};
pub use cache::{Cache, CpuKVCache, GpuKVCache};
pub use gpu_context::WgpuContext;
pub use traits::{
    Decoder, Device, Encoder, TransformerConfig, TransformerModel,
};

// Re-export model traits and registry
pub use models::{
    LanguageModel, ModelArchitecture, ModelType,
};

// Prelude for easy imports
pub mod prelude {
    pub use crate::cache::{Cache, CpuKVCache, GpuKVCache};
    pub use crate::gpu_context::WgpuContext;
    pub use crate::models::{LanguageModel};
    pub use crate::traits::{Decoder, Device, Encoder, TransformerModel};
}

#[cfg(test)]
pub mod tests;

