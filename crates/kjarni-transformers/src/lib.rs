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
pub mod cpu;
pub mod decoder;

pub mod encoder_decoder;
pub mod feedforward;
pub mod gpu_ops;

pub mod linear_layer;
pub mod models;
pub mod normalization;

pub mod audio;
pub mod execution;
pub mod lm_head;
pub mod pipeline;
pub mod pooling;
pub mod rope;
pub mod stats;
pub mod tensor;
pub mod tokenizer;
pub mod traits;
pub mod utils;
pub mod weights;

pub use audio::{
    AudioConvFrontend, AudioData, AudioLoaderConfig, AudioPipeline, MelConfig, create_silence,
    create_sine_wave, load_audio, load_audio_bytes, load_audio_for_whisper,
};

// Re-export commonly used items
pub use crate::{
    attention::MultiHeadAttention,
    chat::templates::{ChatTemplate, Conversation, Message, Role},
    cpu::embeddings::{
        EmbeddingConfig, EmbeddingConfigBuilder, EmbeddingData, EmbeddingInput, Embeddings,
        LoadedEmbeddings,
    },
    execution::ExecutionPlan,
    feedforward::FeedForward,
    lm_head::{LMHeadConfig, LoadedLMHead},
    normalization::Normalization,
    pipeline::{DecoderPipeline, DecoderPipelineConfig},
    pooling::{PoolingStrategy, cls_pool, last_token_pool, max_pool, mean_pool},
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

#[global_allocator]
static GLOBAL: crate::utils::alloc_stats::TracingAllocator =
    crate::utils::alloc_stats::TracingAllocator;
