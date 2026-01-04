//! High-level model traits and pretrained model metadata.
//!
//! This module provides the user-facing abstractions for working with language
//! models in the Kjarni inference engine. It defines traits for model operations,
//! configuration structures, and a registry of supported pretrained models.
//!
//! # Overview
//!
//! The module is organized into two main components:
//!
//! - [`base`] — Core traits and types for language model inference
//! - [`registry`] — Pretrained model metadata and download utilities
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::models::{LanguageModel, ModelType};
//!
//! // Access model metadata from the registry
//! let info = ModelType::Llama3_2_1B_Instruct.info();
//! println!("Model: {}", info.description);
//!
//! // Models implement the LanguageModel trait for inference
//! let tokens = model.tokenize("Hello, world!")?;
//! ```
//!
//! # See Also
//!
//! - [`crate::traits::InferenceModel`] — Low-level inference trait
//! - [`crate::tensor`] — Tensor types and operations

pub mod base;
pub mod registry;

// Re-export commonly used items
pub use base::LanguageModel;

pub use registry::{
    download_model_files,
    format_params,
    format_size,
    get_default_cache_dir,
    ModelArchitecture,
    ModelInfo,
    ModelPaths,
    ModelTask,
    ModelType,
};

#[cfg(test)]
mod tests;