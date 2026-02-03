//! Cache implementations for transformer models

mod gpu;
mod gpu_beam;
pub use gpu_beam::GpuBeamKVCache;
pub use gpu::GpuKVCache;

use std::any::Any;
