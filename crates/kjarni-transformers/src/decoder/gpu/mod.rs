//! GPU-specific building blocks for constructing decoder compute components.

pub mod backend;

pub use backend::GpuDecoderBackend;

pub use crate::gpu_ops::blocks::layers::{
    GpuPreNormDecoderLayer,
    GpuRoPEDecoderLayer,
};