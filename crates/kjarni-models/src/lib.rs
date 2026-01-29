//! High-level model implementations built on kjarni-transformers.
//!
//! This crate provides complete, ready-to-use transformer model implementations
//! for common NLP tasks. It builds on the low-level primitives in `kjarni-transformers`
//! to offer task-specific abstractions for embedding, classification, generation,
//! and sequence-to-sequence modeling.
//!
//! # Overview
//!
//! Model categories:
//!
//! ## Encoders
//! - [`SentenceEncoder`] — Bidirectional models for embeddings (BERT, RoBERTa, Nomic)
//! - [`CrossEncoder`] — Query-document scoring for reranking
//! - [`SequenceClassifier`] — Text classification and sentiment analysis
//!
//! ## Decoders
//! - [`models::llama`] — Llama 2/3 family (autoregressive generation)
//! - [`models::gpt2`] — GPT-2 and DistilGPT2
//! - [`models::mistral`] — Mistral 7B with sliding window attention
//! - [`models::qwen`] — Qwen/Qwen2.5 family
//!
//! ## Encoder-Decoders
//! - [`models::bart`] — BART for summarization and translation
//! - [`models::t5`] — T5/FLAN-T5 for instruction following
//!
//! # Example
//!
//! ```ignore
//! use kjarni_models::sentence_encoder::SentenceEncoder;
//! use kjarni_transformers::models::registry::ModelType;
//!
//! // Load embedding model
//! let encoder = SentenceEncoder::from_registry(
//!     ModelType::NomicEmbedText,
//!     Default::default()
//! ).await?;
//!
//! // Encode text
//! let embeddings = encoder.encode(&["Hello, world!"]).await?;
//! ```
//!
//! # Architecture
//!
//! This crate is organized as:
//! - **Task-specific APIs** (`SentenceEncoder`, `CrossEncoder`, etc.)
//! - **Model implementations** (in `models/` subdirectory)
//! - **Tokenization utilities** (`tokenizer` module)
//!
//! # See Also
//!
//! - [`kjarni_transformers`] — Low-level transformer primitives
//! - [`kjarni_transformers::models::registry`] — Model metadata and downloading

pub mod models;


pub use models::cross_encoder::CrossEncoder;
pub use models::sentence_encoder::{BertConfig, DistilBertConfig, MpnetConfig, SentenceEncoder};
pub use models::sequence_classifier::{SequenceClassifier};
// pub use text_generation::TextGenerator;
// pub use generation::Generator;

/// A callback for streaming generated tokens.
///
/// The callback receives the token ID (`u32`) and its decoded text representation (`&str`).
/// It should return `true` to continue generation or `false` to stop early.
pub type TokenCallback<'a> = Box<dyn FnMut(u32, &str) -> bool + 'a>;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(test)]
pub mod tests;

#[cfg(not(target_arch = "wasm32"))]
pub use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
pub use tokenizer::wasm::BPETokenizer;

#[cfg(test)]
mod send_sync_tests {
    use crate::{CrossEncoder, SentenceEncoder, SequenceClassifier, models::{bart::model::BartModel, gpt2::Gpt2Model, llama::LlamaModel, mistral::MistralModel, qwen::QwenModel, t5::T5Model}};
    // Compile time validation of send and sync
     const _: () = {
        const fn assert_send<T: Send>() {}
        const fn assert_sync<T: Sync>() {}
        assert_send::<CrossEncoder>();
        assert_sync::<CrossEncoder>();

        assert_send::<SentenceEncoder>();
        assert_sync::<SentenceEncoder>();

        assert_send::<SequenceClassifier>();
        assert_sync::<SequenceClassifier>();

        assert_send::<LlamaModel>();
        assert_sync::<LlamaModel>();

        assert_send::<QwenModel>();
        assert_sync::<QwenModel>();

        assert_send::<MistralModel>();
        assert_sync::<MistralModel>();

        assert_send::<T5Model>();
        assert_sync::<T5Model>();

        assert_send::<Gpt2Model>();
        assert_sync::<Gpt2Model>();

        assert_send::<BartModel>();
        assert_sync::<BartModel>();
    };

}