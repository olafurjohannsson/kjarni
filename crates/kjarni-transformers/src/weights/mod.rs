//! This module provides the `ModelWeights` struct, a unified, high-performance
//! loader for model weights, abstracting away the underlying file format.

use crate::tensor::{DType, RawTensor, TypedCpuTensor};
use anyhow::{anyhow, Context, Result};
use half::bf16;
use ndarray::{ArrayD, IxDyn};
use std::fs;
use std::path::Path;

mod gguf_loader;
mod safetensors_loader;
mod model_weights;

use gguf_loader::GgufLoader;
use safetensors_loader::SafeTensorsLoader;

pub use model_weights::ModelWeights;

/// A trait for a model weight file loader.
/// This allows `ModelWeights` to be agnostic to the file format (safetensors, gguf, etc.).
trait WeightLoader {
    /// Gets a raw, untyped view of a tensor's bytes.
    fn get_raw(&self, name: &str) -> Result<RawTensor<'_>>;

    /// Checks if a tensor with the given name exists.
    fn contains(&self, name: &str) -> bool;
}
