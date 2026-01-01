//! An engine for autoregressive decoding with transformer-based language models.
//!
//! This module provides a flexible, multi-layered architecture for running
//! decoder-only models on both CPU and GPU backends.
//!
//! The primary entry point for users is the [`prelude`], which exports all necessary
//! traits and types.

// Public modules for the library's public API
pub mod config;
pub mod traits;
pub mod generator;
pub mod backend;


// Internal components. Not part of the public API.
mod cpu;
mod gpu;


pub mod prelude {
    pub use crate::decoder::{
        config::DecoderLoadConfig,
        cpu::{
            DecoderAttention,
            DecoderLayer,
            CpuRoPEDecoderLayer,
            CpuDecoderBackend,
        },
        gpu::{
            GpuDecoderBackend,
            GpuPreNormDecoderLayer,
            GpuRoPEDecoderLayer,
        },
        backend::AnyDecoderBackend,
        generator::{
            DecoderGenerator
        },
        traits::{
            CpuDecoder,
            GpuDecoder,
            DecoderGenerationBackend,
            DecoderLanguageModel,
            CpuDecoderOps,
            GpuDecoderOps,
        },
    };
}