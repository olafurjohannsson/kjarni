//! This module provides the `ModelWeights` struct, a unified, high-performance
//! loader for model weights, abstracting away the underlying file format.

use crate::tensor::TensorView;
use anyhow::{Context, Result};

mod gguf_loader;
mod model_weights;
mod safetensors_loader;

pub use model_weights::ModelWeights;
pub use model_weights::raw_to_typed;
pub use model_weights::cast_or_copy;
mod gguf_block_reorder;
pub use gguf_block_reorder::{
    raw_to_typed_gguf,
    raw_to_typed_no_reorder
};


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
}

#[cfg(test)]
mod tests;
