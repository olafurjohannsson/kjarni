//! An engine for autoregressive decoding with transformer-based language models.
//!
//! This module provides a flexible, multi-layered architecture for running
//! decoder-only models on both CPU and GPU backends.
//!
//! The primary entry point for users is the [`prelude`], which exports all necessary
//! traits and types.

// Public modules for the library's public API
pub mod traits;
pub mod generator;
pub mod backend;


// Internal components. Not part of the public API.
mod gpu;


pub mod prelude {
    pub use crate::cpu::decoder::{
        CpuDecoderBackend,
        CpuRoPEDecoderLayer,
        DecoderAttention,
        DecoderLayer,
    };
    pub use crate::gpu::{
        decoder::backend::GpuDecoderBackend,
    };
    pub use crate::decoder::{
        backend::AnyDecoderBackend,
        generator::DecoderGenerator,
        gpu::{
            GpuPreNormDecoderLayer,
            GpuRoPEDecoderLayer,
        },
        traits::{
            CpuDecoder,
            CpuDecoderOps,
            DecoderGenerationBackend,
            DecoderLanguageModel,
            GpuDecoder,
            GpuDecoderOps,
        },
    };
}

#[cfg(test)]
mod test_generator;