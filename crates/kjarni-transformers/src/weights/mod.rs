//! Weight loading infrastructure for model files.

mod gguf_conversion;
mod gguf_loader;
mod model_weights;
mod mmap_cache;
mod safetensors_loader;

pub use gguf_conversion::{cast_or_copy, raw_to_typed_gguf};
pub use gguf_loader::{GgufHfMapper, GgufLoader};
pub use model_weights::{AttentionLayout, ModelWeights, raw_to_typed};
pub use mmap_cache::{clear_mmap_cache, mmap_cache_stats};
pub use safetensors_loader::SafeTensorsLoader;

use crate::tensor::raw_tensor::TensorView;
use anyhow::Result;
use std::any::Any;

/// Trait for loading model weights from various file formats.
///
/// Implementations handle format-specific details (SafeTensors, GGUF) while
/// providing a unified interface for weight access.
///
/// # Object Safety
///
/// This trait is object-safe, allowing use as `Box<dyn WeightLoader>`.
/// For the callback-based consumption pattern, use [`ModelWeights::with_raw_tensor`].
///
/// # Implementors
///
/// - [`SafeTensorsLoader`]: Loads `.safetensors` files
/// - [`GgufLoader`]: Loads `.gguf` files with quantized weights
pub trait WeightLoader: Send + Sync {
    /// Gets a raw tensor view by name.
    ///
    /// Returns a borrowed view into mmap'd memory. The view should be
    /// consumed immediately. Prefer [`ModelWeights::with_raw_tensor`]
    /// which enforces immediate consumption via callback.
    fn get_raw(&self, name: &str) -> Result<TensorView<'_>>;

    /// Checks if a tensor with the given name exists.
    fn contains(&self, name: &str) -> bool;

    /// Gets a string from metadata (GGUF only).
    fn get_string(&self, _key: &str) -> Option<&str> {
        None
    }

    /// Gets a u32 from metadata (GGUF only).
    fn get_u32(&self, _key: &str) -> Option<u32> {
        None
    }

    /// Gets a f32 from metadata (GGUF only).
    fn get_f32(&self, _key: &str) -> Option<f32> {
        None
    }

    /// Returns true if this loader has embedded metadata.
    fn has_metadata(&self) -> bool {
        false
    }

    /// Downcast to concrete type.
    fn as_any(&self) -> &dyn Any;
}