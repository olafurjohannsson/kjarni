//! Weight loading infrastructure for model files.
//!
//! Supports SafeTensors and GGUF formats with memory-mapped loading.

mod gguf_conversion;
mod gguf_loader;
mod model_weights;
mod mmap_cache;
mod safetensors_loader;

use std::any::Any;

use anyhow::Result;

use crate::tensor::raw_tensor::TensorView;

pub use gguf_conversion::{cast_or_copy, raw_to_typed_gguf};
pub use gguf_loader::{GgufHfMapper, GgufLoader};
pub use model_weights::{AttentionLayout, ModelWeights, raw_to_typed};
pub use mmap_cache::{clear_mmap_cache, mmap_cache_stats};
pub use safetensors_loader::SafeTensorsLoader;

/// Trait for loading model weights from various file formats.
///
/// Implementations handle format-specific details while providing a unified
/// interface for weight access. Object-safe for use as `Box<dyn WeightLoader>`.
pub trait WeightLoader: Send + Sync {
    /// Returns a raw tensor view by name.
    ///
    /// The view borrows from mmap'd memory and should be consumed immediately.
    fn get_raw(&self, name: &str) -> Result<TensorView<'_>>;

    /// Returns `true` if a tensor with the given name exists.
    fn contains(&self, name: &str) -> bool;

    /// Returns a string from metadata (GGUF only).
    fn get_string(&self, _key: &str) -> Option<&str> {
        None
    }

    /// Returns a u32 from metadata (GGUF only).
    fn get_u32(&self, _key: &str) -> Option<u32> {
        None
    }

    /// Returns a f32 from metadata (GGUF only).
    fn get_f32(&self, _key: &str) -> Option<f32> {
        None
    }

    /// Returns `true` if this loader has embedded metadata.
    fn has_metadata(&self) -> bool {
        false
    }

    /// Downcasts to a concrete type.
    fn as_any(&self) -> &dyn Any;
}