//! A generic, backend-agnostic transformer encoder implementation.
//!
//! This module provides `TransformerEncoder`, a reusable component that can represent
//! various encoder-only models like BERT, RoBERTa, etc. It is designed to be
//! backend-aware, containing either a CPU or a (future) GPU implementation.
//!
//! The encoder is constructed generically by relying on the `EncoderArchitecture`
//! trait, which provides the specific weight names and hyperparameters for a

//! given model, allowing for maximum code reuse.

mod cpu;
mod gpu;
pub mod encoder_self_attention;
pub mod traits;
pub mod classifier;
pub mod pooler;
pub mod config;
pub mod encoder_layer;

use crate::encoder::traits::EncoderArchitecture;
use crate::traits::TransformerModel;
pub use cpu::CpuTransformerEncoder;
pub use gpu::GpuTransformerEncoder;
pub use traits::{CpuEncoder, CpuEncoderOps, GpuEncoder, GpuEncoderOps, SentenceEncoderModel};


pub mod prelude {
    pub use crate::encoder::{
        classifier::{
            CpuSequenceClassificationHead,
            GpuSequenceClassificationHead,
        },
        config::{
            EncoderLoadConfig,
            EncodingConfig,
            PoolingStrategy,
        },
        encoder_self_attention::EncoderSelfAttention,
        pooler::{
            CpuPooler,
            GpuPooler,
            StandardCpuPooler,
        },
        traits::{
            CpuEncoder,
            CpuEncoderOutput,
            EncoderLanguageModel,
            GpuEncoder,
            GpuEncoderInput,
            GpuEncoderOutput,
        },
    };
}

