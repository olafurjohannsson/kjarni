//! Core transformer components for building transformer-based models
//!
//! This crate provides the fundamental building blocks for transformer architectures
//! without model-specific implementations.

//! Kjarni Transformers: Fast transformer models for Rust

pub mod activations;
pub mod attention;
pub mod cache;
pub mod chat;
pub mod common;
pub mod decoder;
pub mod embeddings;
pub mod encoder;
pub mod encoder_decoder;
pub mod feedforward;
pub mod gpu_ops;
pub mod kernels;
pub mod linear_layer;
pub mod models;
pub mod normalization;
pub mod ops;
pub mod pooling;
pub mod rope;
pub mod tensor;
pub mod traits;
pub mod utils;
pub mod weights;

// Re-export commonly used items
pub use crate::{
    attention::MultiHeadAttention,
    chat::templates::{ChatTemplate, Conversation, Message, Role},
    embeddings::Embeddings,
    feedforward::FeedForward,
    normalization::Normalization,
    pooling::{cls_pool, last_token_pool, max_pool, mean_pool, PoolingStrategy},
};
pub use cache::{Cache, CpuKVCache, GpuKVCache};
pub use gpu_ops::context::WgpuContext;
pub use traits::Device;

// Re-export model traits and registry
pub use models::{LanguageModel, ModelArchitecture, ModelType};

// Prelude for easy imports
pub mod prelude {
    pub use crate::cache::{Cache, CpuKVCache, GpuKVCache};
    pub use crate::gpu_ops::context::WgpuContext;
    pub use crate::models::LanguageModel;
    pub use crate::traits::Device;
}

#[cfg(test)]
pub mod tests;
