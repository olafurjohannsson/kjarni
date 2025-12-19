//! GPU-specific building blocks for constructing decoder compute components.

pub mod blocks;
pub mod backend;

pub use backend::GpuDecoderBackend;
pub use blocks::{
    // attention::DecoderAttention,
    layer::GpuPreNormDecoderLayer,
};