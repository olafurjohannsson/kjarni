//! GPT implementation using edgeTransformers
//! 
//! Provides autoregressive language models for text generation.

pub mod gptconfig;
pub mod bertconfig;
pub mod model;
pub mod gptweights;
pub mod bertweights;
pub mod tokenizer;
pub mod seq2seq;
pub mod tests;
pub mod tests2;
pub mod text_generation;
// pub mod generation_old;
pub mod sentence_encoder;
pub mod cross_encoder;
pub mod generation;

pub use sentence_encoder::SentenceEncoder;
pub use cross_encoder::CrossEncoder;

/// A callback for streaming generated tokens.
///
/// The callback receives the token ID (`u32`) and its decoded text representation (`&str`).
/// It should return `true` to continue generation or `false` to stop early.
pub type TokenCallback<'a> = Box<dyn FnMut(u32, &str) -> bool + 'a>;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-exports
// pub use gptconfig::GPTConfig;
// pub use model::{GenerativeModel, GenerativeModelType};
pub use gptweights::GPTModelWeights;
// pub use generation::{GenerationConfig, SamplingStrategy};

#[cfg(not(target_arch = "wasm32"))]
pub use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
pub use tokenizer::wasm::BPETokenizer;