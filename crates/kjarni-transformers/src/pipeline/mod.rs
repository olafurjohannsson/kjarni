//! High-level inference pipelines for transformer models.
//!
//! This module provides ready-to-use inference pipelines that orchestrate
//! tokenization, model execution, and generation logic. Pipelines abstract
//! away low-level details and provide a clean API for text generation and
//! encoder-decoder tasks.
//!
//! # Overview
//!
//! Two main pipeline types are provided:
//!
//! - [`DecoderPipeline`] — Autoregressive text generation (Llama, GPT-2, etc.)
//! - [`EncoderDecoderPipeline`] — Seq2Seq tasks (BART, T5 for translation/summarization)
//!
//! # Architecture
//!
//! Pipelines compose several layers:
//! 1. **Tokenizer** — Text ↔ token IDs
//! 2. **Model** — Forward pass through transformer
//! 3. **Generation logic** — Sampling, beam search, stopping criteria
//! 4. **Builder pattern** — Fluent API for configuration
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::pipeline::{DecoderPipeline, DecoderPipelineConfig};
//! use kjarni_transformers::models::registry::ModelType;
//!
//! // Build pipeline
//! let pipeline = DecoderPipeline::builder()
//!     .with_model_type(ModelType::Llama3_2_1B_Instruct)
//!     .with_config(DecoderPipelineConfig::default())
//!     .build()
//!     .await?;
//!
//! // Generate text
//! let output = pipeline.generate("Hello, how are you?", &Default::default()).await?;
//! println!("{}", output);
//! ```
//!
//! # See Also
//!
//! - [`crate::models`] — Model implementations
//! - [`crate::generation`] — Generation strategies

// kjarni-transformers/src/pipeline/mod.rs

mod decoder;
mod decoder_builder;
mod encoder_decoder;
mod encoder_decoder_builder;
mod loader;
mod cpu_factory;
pub use decoder::{DecoderPipeline, DecoderPipelineConfig};
pub use encoder_decoder::{EncoderDecoderPipeline, EncoderDecoderPipelineConfig};
pub use loader::{DecoderModelFactory, GenericLoader};
pub use decoder_builder::DecoderPipelineBuilder;
pub use cpu_factory::CpuLayerFactory;

#[cfg(test)]
mod tests;