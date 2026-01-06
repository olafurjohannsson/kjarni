//! This module provides the `ModelWeights` struct, a unified, high-performance
//! loader for model weights, abstracting away the underlying file format.

use std::any::Any;

use crate::tensor::TensorView;
use anyhow::{Context, Result};

pub mod gguf_loader;
pub mod model_weights;
pub mod safetensors_loader;

pub use gguf_block_reorder::gguf_block_group_for_row;
pub use model_weights::ModelWeights;

pub use model_weights::cast_or_copy;
pub use model_weights::raw_to_typed;
mod gguf_block_reorder;
pub use gguf_block_reorder::raw_to_typed_gguf;

/// A trait for a model weight file loader.
/// This allows `ModelWeights` to be agnostic to the file format (safetensors, gguf, etc.).
pub trait WeightLoader {
    /// Gets a raw, untyped view of a tensor's bytes.
    fn get_raw(&self, name: &str) -> Result<TensorView<'_>>;

    /// Checks if a tensor with the given name exists.
    fn contains(&self, name: &str) -> bool;

    fn get_string(&self, _key: &str) -> Option<&str> {
        None
    }
    fn get_u32(&self, _key: &str) -> Option<u32> {
        None
    }
    fn get_f32(&self, _key: &str) -> Option<f32> {
        None
    }

    fn has_metadata(&self) -> bool {
        false
    }

    fn as_any(&self) -> &dyn Any;
}

#[cfg(test)]
mod tests;
