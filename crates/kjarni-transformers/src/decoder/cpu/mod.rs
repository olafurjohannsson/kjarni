//! CPU-specific building blocks for constructing decoder compute components.

pub mod blocks;
pub mod backend;

pub use backend::CpuDecoderBackend;
pub use blocks::{
    attention::DecoderAttention,
    layer::{CpuRoPEDecoderLayer, DecoderLayer},
};